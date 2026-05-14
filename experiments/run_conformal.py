#!/usr/bin/env python3
"""Run split conformal prediction evaluation across all uncertainty methods.

Uses:
  - Validation set for conformal calibration (threshold fitting)
  - Test set for coverage / set-size evaluation

Saves results to evaluation/conformal_prediction__<model>.json
and a summary figure to evaluation/figures/conformal_coverage_vs_setsize.png

Usage:
    python experiments/run_conformal.py
    python experiments/run_conformal.py --max_samples 2048 --mc_samples 20
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.load_model import get_device, load_hf_image_classifier
from uncertainty_lab.data.pcam import load_pcam_subset
from uncertainty_lab.uncertainty.conformal import SplitConformalPredictor, conformal_across_alphas
from uncertainty_lab.metrics.core import json_safe


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def get_cfg():
    with open(REPO_ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=2048, help="Max samples per split")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mc_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", type=str, default="0.05,0.10,0.20", help="Comma-separated alpha levels")
    return p.parse_args()


def _infer_probs(model, loader, device, mode: str = "confidence", mc_samples: int = 20):
    """Return (probs: N×C, labels: N) numpy arrays.

    For deep_ensemble mode, ``model`` should be a list of nn.Module objects.
    """
    all_probs, all_labels = [], []
    for batch, labels in loader:
        batch = batch.to(device)
        if mode == "mc_dropout":
            model.train()
            ps = []
            with torch.inference_mode():
                for _ in range(max(2, mc_samples)):
                    with _autocast(device):
                        ps.append(torch.softmax(model(pixel_values=batch).logits, dim=1))
            probs = torch.stack(ps).mean(0)
            model.eval()
        elif mode == "deep_ensemble":
            member_probs = []
            for m in model:
                m.eval()
                with torch.inference_mode():
                    with _autocast(device):
                        member_probs.append(torch.softmax(m(pixel_values=batch).logits, dim=1))
            probs = torch.stack(member_probs).mean(0)
        else:
            model.eval()
            with torch.inference_mode():
                with _autocast(device):
                    probs = torch.softmax(model(pixel_values=batch).logits, dim=1)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels)
    return torch.cat(all_probs).numpy().astype(np.float64), torch.cat(all_labels).numpy().astype(np.int64)


def main():
    args = parse_args()
    cfg = get_cfg()
    device = get_device()
    alphas = [float(a) for a in args.alphas.split(",")]

    model_id = cfg["model"]["model_id"]
    data_root = REPO_ROOT / cfg["data"]["root"]
    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load single model (used for confidence + mc_dropout)
    model, _, _ = load_hf_image_classifier(
        model_id=model_id, num_labels=cfg["model"]["num_labels"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
    ckpt = REPO_ROOT / "checkpoints" / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model = model.to(device)

    # Load ensemble members (used for deep_ensemble method)
    ensemble_ckpt_dir = REPO_ROOT / "checkpoints"
    ensemble_member_dirs = sorted([
        d for d in ensemble_ckpt_dir.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
    ])[:5]  # use up to 5 members
    ensemble_models = []
    for member_dir in ensemble_member_dirs:
        m, _, _ = load_hf_image_classifier(
            model_id=model_id, num_labels=cfg["model"]["num_labels"],
            dropout=cfg["model"].get("dropout", 0.1),
        )
        state = torch.load(member_dir / "best.pt", map_location="cpu", weights_only=True)
        m.load_state_dict(state.get("model_state_dict", state), strict=False)
        m = m.to(device)
        ensemble_models.append(m)
    print(f"Loaded {len(ensemble_models)} ensemble members from {ensemble_ckpt_dir}")

    # Load datasets
    print("Loading validation set (calibration)...")
    cal_ds = load_pcam_subset(data_root, split="val", transform=transform,
                              max_samples=args.max_samples, seed=args.seed)
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False)

    print("Loading test set (evaluation)...")
    test_ds = load_pcam_subset(data_root, split="test", transform=transform,
                               max_samples=args.max_samples, seed=args.seed)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    methods_config = [
        ("confidence", "confidence", model),
        ("mc_dropout", "mc_dropout", model),
        ("deep_ensemble", "deep_ensemble", ensemble_models),
    ]

    results = {}
    for method_name, infer_mode, m_obj in methods_config:
        if infer_mode == "deep_ensemble" and not ensemble_models:
            print(f"\n── {method_name.upper()} ── SKIPPED (no ensemble members found)")
            continue
        print(f"\n── {method_name.upper()} ──")
        print("  Inference on calibration set...")
        cal_probs, cal_labels = _infer_probs(m_obj, cal_loader, device, infer_mode, args.mc_samples)
        print("  Inference on test set...")
        test_probs, test_labels = _infer_probs(m_obj, test_loader, device, infer_mode, args.mc_samples)

        rows = conformal_across_alphas(test_probs, test_labels, cal_probs, cal_labels, alphas)
        results[method_name] = {
            "alphas_evaluated": rows,
            "n_cal": int(len(cal_labels)),
            "n_test": int(len(test_labels)),
        }
        for row in rows:
            print(f"  α={row['alpha']:.2f}  coverage={row['empirical_coverage']:.4f}"
                  f"  (target={row['target_coverage']:.2f})"
                  f"  avg_set_size={row['avg_prediction_set_size']:.4f}"
                  f"  singleton={row['singleton_rate']:.4f}"
                  f"  full={row['full_set_rate']:.4f}")

    # Save JSON
    out_dir = REPO_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"conformal_prediction__model-{model_id.replace('/', '-')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_safe({"model_id": model_id, "results": results}), f, indent=2)
    print(f"\nSaved: {out_path}")

    # Figure
    _plot_conformal(results, out_dir / "figures" / "conformal_coverage_setsize.png", alphas)


def _plot_conformal(results: dict, out_path: Path, alphas: list[float]):
    """Coverage vs average set size for each method and alpha level."""
    try:
        import matplotlib.pyplot as plt
        from uncertainty_lab.metrics.plots import METHOD_COLORS, _finalize

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for method, data in results.items():
            color = METHOD_COLORS.get(method)
            rows = data["alphas_evaluated"]
            targets = [r["target_coverage"] for r in rows]
            empirical = [r["empirical_coverage"] for r in rows]
            set_sizes = [r["avg_prediction_set_size"] for r in rows]
            singleton = [r["singleton_rate"] for r in rows]
            full = [r["full_set_rate"] for r in rows]

            axes[0].plot(targets, empirical, marker="o", label=method, color=color)
            axes[1].plot(targets, set_sizes, marker="o", label=f"{method} mean size", color=color)
            axes[1].plot(targets, singleton, marker="s", linestyle="--",
                         label=f"{method} singleton", color=color, alpha=0.6)

        # Diagonal reference
        t = np.linspace(0.7, 1.0, 50)
        axes[0].plot(t, t, "--", color="gray", linewidth=1, label="Ideal (coverage = target)")
        axes[0].set_xlabel("Target coverage (1 − α)")
        axes[0].set_ylabel("Empirical coverage")
        axes[0].set_title("Coverage guarantee")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)

        axes[1].set_xlabel("Target coverage (1 − α)")
        axes[1].set_ylabel("Avg prediction set size")
        axes[1].set_title("Efficiency (smaller = better)")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.2)

        fig.suptitle("Split Conformal Prediction — PCAM Test Set", fontsize=11)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _finalize(fig, out_path)
        print(f"Figure: {out_path}")
    except Exception as e:
        print(f"Figure generation skipped: {e}")


if __name__ == "__main__":
    main()

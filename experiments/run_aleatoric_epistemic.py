#!/usr/bin/env python3
"""Aleatoric vs. Epistemic uncertainty decomposition.

Uses MC Dropout member probabilities to decompose total predictive entropy into:
  - Epistemic  ≈ mutual information  (reducible — model uncertainty)
  - Aleatoric  = total - epistemic   (irreducible — data noise / ambiguity)

Output:
  evaluation/aleatoric_epistemic__<model>.json
  evaluation/figures/aleatoric_epistemic_histograms.png
  evaluation/figures/aleatoric_epistemic_scatter.png

Usage:
    python experiments/run_aleatoric_epistemic.py
    python experiments/run_aleatoric_epistemic.py --max_samples 2048 --mc_samples 30
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
from uncertainty_lab.metrics.core import disagreement_score_arrays, json_safe, safe_binary_auc, safe_binary_auprc
from uncertainty_lab.metrics.plots import (
    plot_uncertainty_decomposition,
    plot_uncertainty_decomposition_scatter,
)


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def infer_mc_dropout(model, loader, device, T: int):
    """Return (mean_logits, member_probs T×N×C, labels)."""
    model.train()
    all_member_chunks, all_labels = [], []
    with torch.inference_mode():
        for batch, labels in loader:
            batch = batch.to(device)
            ps = []
            for _ in range(T):
                with _autocast(device):
                    ps.append(torch.softmax(model(pixel_values=batch).logits, dim=1))
            all_member_chunks.append(torch.stack(ps).detach().cpu())  # (T, B, C)
            all_labels.append(labels)
    model.eval()
    member_probs = torch.cat(all_member_chunks, dim=1).numpy().astype(np.float64)  # (T, N, C)
    mean_probs = member_probs.mean(axis=0)  # (N, C)
    logits = np.log(np.clip(mean_probs, 1e-12, 1.0))
    labels = torch.cat(all_labels).numpy().astype(np.int64)
    return logits, member_probs, labels


def main():
    args = parse_args()
    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    model_id = cfg["model"]["model_id"]
    data_root = REPO_ROOT / cfg["data"]["root"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load model
    model, _, _ = load_hf_image_classifier(
        model_id=model_id, num_labels=cfg["model"]["num_labels"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
    ckpt = REPO_ROOT / "checkpoints" / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model = model.to(device)

    # Load test set
    print(f"Loading test set (max {args.max_samples} samples)...")
    test_ds = load_pcam_subset(data_root, split="test", transform=transform,
                               max_samples=args.max_samples, seed=args.seed)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  {len(test_ds)} samples, T={args.mc_samples} MC passes")

    # Run MC Dropout inference
    print("Running MC Dropout inference...")
    logits, member_probs, labels = infer_mc_dropout(model, loader, device, args.mc_samples)

    # Decompose uncertainty
    dis = disagreement_score_arrays(member_probs)
    pred_ent  = dis["predictive_entropy"]    # total
    mi        = dis["mutual_information"]     # epistemic
    aleatoric = np.maximum(0.0, pred_ent - mi)  # aleatoric

    preds = logits.argmax(axis=1)
    error = (preds != labels).astype(np.int64)
    correct = 1 - error

    # Stats
    def stats(arr):
        return {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "median": float(np.median(arr)),
            "mean_correct": float(arr[correct.astype(bool)].mean()) if correct.any() else None,
            "mean_incorrect": float(arr[error.astype(bool)].mean()) if error.any() else None,
        }

    result = {
        "model_id": model_id,
        "mc_samples": args.mc_samples,
        "n_test": len(labels),
        "accuracy": float(correct.mean()),
        "total_predictive_entropy": stats(pred_ent),
        "epistemic_mutual_information": stats(mi),
        "aleatoric": stats(aleatoric),
        "ratio_epistemic": float(mi.mean() / (pred_ent.mean() + 1e-12)),
        "ratio_aleatoric": float(aleatoric.mean() / (pred_ent.mean() + 1e-12)),
        "error_detection": {
            "total_entropy_auroc":  safe_binary_auc(error, pred_ent),
            "epistemic_mi_auroc":   safe_binary_auc(error, mi),
            "aleatoric_auroc":      safe_binary_auc(error, aleatoric),
            "total_entropy_auprc":  safe_binary_auprc(error, pred_ent),
            "epistemic_mi_auprc":   safe_binary_auprc(error, mi),
            "aleatoric_auprc":      safe_binary_auprc(error, aleatoric),
        },
        "interpretation": {
            "high_epistemic_means": "Model is uncertain due to limited data coverage — reducible with more training data or a larger model.",
            "high_aleatoric_means": "The patch is intrinsically ambiguous — even experts may disagree on the label.",
            "ratio_epistemic": "Fraction of total uncertainty that is epistemic.",
        },
    }

    # Print summary
    print("\n=== Aleatoric / Epistemic Decomposition ===")
    print(f"  Accuracy          : {result['accuracy']:.4f}")
    print(f"  Mean total entropy: {result['total_predictive_entropy']['mean']:.6f}")
    print(f"  Mean epistemic MI : {result['epistemic_mutual_information']['mean']:.6f}  ({result['ratio_epistemic']*100:.1f}%)")
    print(f"  Mean aleatoric    : {result['aleatoric']['mean']:.6f}  ({result['ratio_aleatoric']*100:.1f}%)")
    print(f"  Error det. AUROC  : total={result['error_detection']['total_entropy_auroc']:.4f}"
          f"  epistemic={result['error_detection']['epistemic_mi_auroc']:.4f}"
          f"  aleatoric={result['error_detection']['aleatoric_auroc']:.4f}")

    # Save JSON
    out_dir = REPO_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"aleatoric_epistemic__model-{model_id.replace('/', '-')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plots
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    hist_path = fig_dir / "aleatoric_epistemic_histograms.png"
    plot_uncertainty_decomposition(mi, aleatoric, error, hist_path,
                                   title=f"Aleatoric vs. Epistemic — MC Dropout (T={args.mc_samples})")
    print(f"Figure: {hist_path}")

    scatter_path = fig_dir / "aleatoric_epistemic_scatter.png"
    plot_uncertainty_decomposition_scatter(mi, aleatoric, error, scatter_path,
                                           title=f"Epistemic vs. Aleatoric per Sample (T={args.mc_samples})")
    print(f"Figure: {scatter_path}")


if __name__ == "__main__":
    main()

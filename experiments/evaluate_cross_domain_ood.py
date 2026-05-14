#!/usr/bin/env python3
"""
Cross-domain OOD evaluation: PCAM-trained model evaluated on NCT-CRC-HE-100K.

A model trained on PCAM (lymph-node tumor patches) is applied to NCT-CRC
(colorectal histology patches). NCT-CRC patches are genuine OOD — different
tissue, different scanner, different staining protocol.

This tests whether uncertainty methods produce higher uncertainty on truly
out-of-distribution images (different pathology domain), unlike synthetic
corruptions which only degrade image quality.

Output: evaluation/cross_domain_ood__<method>__<model>.json
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
from experiments.ensemble_utils import resolve_deep_ensemble_members
from uncertainty_lab.data.nct_crc import load_nct_crc_subset, NCT_CLASSES
from uncertainty_lab.data.pcam import load_pcam_subset
from uncertainty_lab.metrics.core import (
    disagreement_score_arrays,
    json_safe as _json_safe,
    safe_binary_auc,
    safe_binary_auprc,
    summarize_from_logits,
)


def _autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def get_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-domain OOD: PCAM model on NCT-CRC")
    p.add_argument("--method", default="confidence",
                   choices=["confidence", "mc_dropout", "deep_ensemble", "temperature_scaled"])
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--max_samples", type=int, default=512,
                   help="Max samples from each domain (PCAM-ID and NCT-CRC-OOD)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", type=str, default="",
                   help="PCAM-trained checkpoint run_id (default: checkpoints/best.pt)")
    p.add_argument("--ensemble_run_ids", type=str, default="")
    p.add_argument("--ensemble_size", type=int, default=2)
    p.add_argument("--nct_root", type=str, default="data/raw",
                   help="Root dir containing NCT-CRC-HE-100K/")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def _new_model(cfg: dict):
    model, _, _ = load_hf_image_classifier(
        model_id=cfg["model"]["model_id"],
        num_labels=cfg["model"]["num_labels"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
    return model


def load_models(cfg, method, run_id, ensemble_run_ids, ensemble_size):
    models = []
    if method == "deep_ensemble":
        model_id = cfg["model"]["model_id"]
        if ensemble_run_ids:
            # Load each run directly without strict hyperparameter matching
            for rid in ensemble_run_ids:
                ckpt = REPO_ROOT / "checkpoints" / rid / "best.pt"
                if not ckpt.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
                m = _new_model(cfg)
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                if "model_state_dict" in state:
                    m.load_state_dict(state["model_state_dict"], strict=False)
                    # Use the model_id from the checkpoint if available
                    saved_id = state.get("config", {}).get("model", {}).get("model_id")
                    if saved_id:
                        model_id = saved_id
                models.append(m)
        else:
            # Fall back to ensemble_utils resolver
            members = resolve_deep_ensemble_members(
                config_model_id=cfg["model"]["model_id"],
                config_dataset=cfg["data"]["dataset"],
                run_id=run_id,
                ensemble_run_ids=[],
                ensemble_size=max(2, ensemble_size),
            )
            member_model_id = members[0]["model_id"]
            ensemble_cfg = {**cfg, "model": {**cfg["model"], "model_id": member_model_id}}
            for member in members:
                m = _new_model(ensemble_cfg)
                state = torch.load(member["ckpt_path"], map_location="cpu", weights_only=True)
                if "model_state_dict" in state:
                    m.load_state_dict(state["model_state_dict"], strict=True)
                models.append(m)
            model_id = member_model_id
        if len(models) < 2:
            raise ValueError("Deep ensemble requires at least 2 members.")
        return models, model_id
    # Single model
    model_id = cfg["model"]["model_id"]
    model = _new_model(cfg)
    if run_id:
        ckpt = REPO_ROOT / "checkpoints" / run_id / "best.pt"
    else:
        ckpt = REPO_ROOT / "checkpoints" / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        try:
            saved_cfg = state.get("config", {})
            model_id = saved_cfg.get("model", {}).get("model_id", model_id) or model_id
        except Exception:
            pass
    models.append(model)
    return models, model_id


def run_inference(
    models,
    loader,
    device,
    method: str,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (logits, labels, extras) where extras may include member_probs."""
    logits_list, y_list, member_chunks = [], [], []
    effective = "confidence" if method == "temperature_scaled" else method

    for batch, labels in loader:
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)

        if effective == "mc_dropout":
            models[0].train()
            probs_stack = []
            with torch.inference_mode():
                for _ in range(max(2, mc_samples)):
                    with _autocast_context(device):
                        lg = models[0](pixel_values=batch).logits
                    probs_stack.append(torch.softmax(lg, dim=1))
            probs_t = torch.stack(probs_stack)  # (T, B, C)
            probs = probs_t.mean(0)
            logits = torch.log(probs.clamp(min=1e-12))
            member_chunks.append(probs_t.detach().cpu())
            models[0].eval()
        elif effective == "deep_ensemble":
            probs_members = []
            with torch.inference_mode():
                for m in models:
                    m.eval()
                    with _autocast_context(device):
                        lg = m(pixel_values=batch).logits
                    probs_members.append(torch.softmax(lg, dim=1))
            probs_t = torch.stack(probs_members)  # (M, B, C)
            probs = probs_t.mean(0)
            logits = torch.log(probs.clamp(min=1e-12))
            member_chunks.append(probs_t.detach().cpu())
        else:
            models[0].eval()
            with torch.inference_mode():
                with _autocast_context(device):
                    logits = models[0](pixel_values=batch).logits

        logits_list.append(logits.detach().cpu())
        y_list.append(labels.detach().cpu())

    logits_np = torch.cat(logits_list).numpy().astype(np.float64)
    y_np = torch.cat(y_list).numpy().astype(np.int64)
    extras = {}
    if member_chunks:
        extras["member_probs"] = torch.cat(member_chunks, dim=1).numpy().astype(np.float64)
    return logits_np, y_np, extras


def uncertainty_scores(logits: np.ndarray, extras: dict) -> dict[str, np.ndarray]:
    """Return a dict of uncertainty arrays (one value per sample)."""
    import torch as _t
    probs = _t.softmax(_t.tensor(logits, dtype=_t.float32), dim=1).numpy()
    conf = probs.max(axis=1)
    scores = {"one_minus_msp": 1.0 - conf}
    member_probs = extras.get("member_probs")
    if member_probs is not None:
        dis = disagreement_score_arrays(member_probs)
        if dis:
            scores["mutual_information"] = dis["mutual_information"]
            scores["predictive_entropy"] = dis["predictive_entropy"]
            scores["predictive_variance"] = dis["predictive_variance"]
    return scores


def main() -> None:
    args = parse_args()
    cfg = get_config()
    device = get_device()

    ensemble_run_ids = [r.strip() for r in args.ensemble_run_ids.split(",") if r.strip()]
    models, model_id = load_models(
        cfg, args.method, args.run_id, ensemble_run_ids, args.ensemble_size
    )
    models = [m.to(device) for m in models]

    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_root = REPO_ROOT / cfg["data"]["root"]
    nct_root = Path(args.nct_root)
    if not nct_root.is_absolute():
        nct_root = (REPO_ROOT / nct_root).resolve()

    # Temperature calibration (only for temperature_scaled method)
    temperature = 1.0
    if args.method == "temperature_scaled":
        try:
            from uncertainty_lab.metrics.core import optimize_temperature
            from uncertainty_lab.data.pcam import load_pcam_subset as _load_pcam
            print("Fitting temperature on PCAM validation set...")
            val_ds = _load_pcam(data_root, split="val", transform=transform, max_samples=512, seed=args.seed)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            val_logits_list, val_y_list = [], []
            models[0].eval()
            with torch.inference_mode():
                for batch, lbls in val_loader:
                    batch = batch.to(device)
                    val_logits_list.append(models[0](pixel_values=batch).logits.detach().cpu())
                    val_y_list.append(lbls)
            val_logits_np = torch.cat(val_logits_list).numpy().astype(np.float64)
            val_y_np = torch.cat(val_y_list).numpy().astype(np.int64)
            temperature = float(optimize_temperature(val_logits_np, val_y_np))
            print(f"  Fitted temperature T={temperature:.4f}")
        except Exception as exc:
            print(f"  Warning: temperature fitting failed ({exc}), using T=1.0")

    def _apply_temperature(logits_np: np.ndarray) -> np.ndarray:
        return logits_np / temperature if temperature != 1.0 else logits_np

    # In-distribution: PCAM test set
    print("Loading PCAM (in-distribution)...")
    pcam_ds = load_pcam_subset(
        data_root, split="test", transform=transform,
        max_samples=args.max_samples, seed=args.seed,
    )
    pcam_loader = torch.utils.data.DataLoader(
        pcam_ds, batch_size=args.batch_size, shuffle=False
    )
    print(f"  PCAM samples: {len(pcam_ds)}")

    print("Running inference on PCAM...")
    pcam_logits, pcam_y, pcam_extras = run_inference(
        models, pcam_loader, device, args.method, args.mc_samples
    )
    pcam_logits = _apply_temperature(pcam_logits)
    pcam_metrics = summarize_from_logits(pcam_logits, pcam_y, n_bins=15)
    pcam_unc = uncertainty_scores(pcam_logits, pcam_extras)

    # Out-of-distribution: NCT-CRC
    print("Loading NCT-CRC-HE-100K (cross-domain OOD)...")
    nct_ds = load_nct_crc_subset(
        nct_root, transform=transform, max_samples=args.max_samples,
        seed=args.seed, binary=True, balanced=True,
    )
    nct_loader = torch.utils.data.DataLoader(
        nct_ds, batch_size=args.batch_size, shuffle=False
    )
    print(f"  NCT-CRC samples: {len(nct_ds)}")

    print("Running inference on NCT-CRC...")
    nct_logits, nct_y, nct_extras = run_inference(
        models, nct_loader, device, args.method, args.mc_samples
    )
    nct_logits = _apply_temperature(nct_logits)
    nct_metrics = summarize_from_logits(nct_logits, nct_y, n_bins=15)
    nct_unc = uncertainty_scores(nct_logits, nct_extras)

    # OOD detection: uncertainty separation between PCAM and NCT-CRC
    # Label: 0 = ID (PCAM), 1 = OOD (NCT-CRC)
    ood_labels = np.concatenate([
        np.zeros(len(pcam_logits), dtype=np.int64),
        np.ones(len(nct_logits), dtype=np.int64),
    ])
    ood_detection: dict[str, dict] = {}
    for score_name in pcam_unc:
        if score_name in nct_unc:
            combined = np.concatenate([pcam_unc[score_name], nct_unc[score_name]])
            ood_detection[score_name] = {
                "auroc": safe_binary_auc(ood_labels, combined),
                "auprc": safe_binary_auprc(ood_labels, combined),
                "mean_id": float(pcam_unc[score_name].mean()),
                "mean_ood": float(nct_unc[score_name].mean()),
            }

    result = {
        "config": {
            "method": args.method,
            "mc_samples": args.mc_samples,
            "max_samples_per_domain": args.max_samples,
            "model_id": model_id,
            "temperature": temperature,
            "run_id": args.run_id or None,
            "id_dataset": "pcam_test",
            "ood_dataset": "nct_crc",
        },
        "id_performance": {
            "n": len(pcam_logits),
            "accuracy": pcam_metrics["predictive_performance"]["accuracy"],
            "roc_auc": pcam_metrics["predictive_performance"]["roc_auc"],
            "ece": pcam_metrics["calibration"]["ece"],
            "mean_uncertainty_1msp": float(pcam_unc["one_minus_msp"].mean()),
        },
        "ood_performance": {
            "n": len(nct_logits),
            "accuracy": nct_metrics["predictive_performance"]["accuracy"],
            "roc_auc": nct_metrics["predictive_performance"]["roc_auc"],
            "ece": nct_metrics["calibration"]["ece"],
            "mean_uncertainty_1msp": float(nct_unc["one_minus_msp"].mean()),
            "note": (
                "Accuracy/AUC here measures TUM-vs-rest on NCT-CRC using a PCAM-trained model; "
                "not expected to be meaningful — only uncertainty scores matter."
            ),
        },
        "ood_detection": ood_detection,
        "interpretation": {
            "auroc_above_0_7": "Good — model expresses higher uncertainty on OOD domain",
            "auroc_near_0_5": "Poor — uncertainty cannot distinguish ID from OOD",
            "auroc_below_0_5": "Inverted — model is MORE confident on OOD (overconfident)",
            "primary_score": max(
                ood_detection,
                key=lambda k: ood_detection[k].get("auroc") or 0.0,
                default="one_minus_msp",
            ),
        },
    }

    # Print summary
    print("\n=== Cross-Domain OOD Summary ===")
    print(f"  Method: {args.method}  |  Model: {model_id}")
    print(f"  PCAM (ID)   n={len(pcam_logits)}  acc={result['id_performance']['accuracy']:.3f}"
          f"  mean_unc={result['id_performance']['mean_uncertainty_1msp']:.3f}")
    print(f"  NCT-CRC(OOD) n={len(nct_logits)}  acc={result['ood_performance']['accuracy']:.3f}"
          f"  mean_unc={result['ood_performance']['mean_uncertainty_1msp']:.3f}")
    for name, d in ood_detection.items():
        auroc = d.get("auroc")
        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        print(f"  OOD-detect [{name}]: AUROC={auroc_str}"
              f"  mean_id={d['mean_id']:.4f}  mean_ood={d['mean_ood']:.4f}")

    # Save output
    out_dir = REPO_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out or f"cross_domain_ood__{args.method}__model-{model_id.replace('/', '-')}.json"
    out_path = out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2, allow_nan=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate predictive performance, calibration, uncertainty quality, and selective prediction.

Outputs JSON to evaluation/metrics_<method>_<split>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms
from torchvision.datasets import PCAM

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.load_model import get_device, load_hf_image_classifier
from experiments.ensemble_utils import load_run_metadata, normalize_run_id, resolve_deep_ensemble_members
from uncertainty_lab.metrics.core import (
    apply_uncertainty_thresholds,
    disagreement_score_arrays,
    fit_uncertainty_thresholds,
    fit_youden_uncertainty_threshold,
    json_safe as _json_safe,
    optimize_temperature,
    slide_level_proxy_from_probs,
    summarize_uncertainty_cohorts,
    summarize_uncertainty_scores,
    summarize_from_logits,
)
from uncertainty_lab.metrics.plots import save_reliability_plot


def _configure_cuda_backends() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _resolve_num_workers(cfg: dict) -> int:
    env_value = os.environ.get("ULAB_NUM_WORKERS")
    if env_value is not None and env_value.strip():
        return max(0, int(env_value))
    requested = int(cfg.get("data", {}).get("num_workers", 0) or 0)
    if requested > 0:
        return requested
    cpu_total = os.cpu_count() or 1
    return max(1, cpu_total - 1)


def get_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--method",
        default="confidence",
        choices=["confidence", "temperature_scaled", "mc_dropout", "deep_ensemble"],
    )
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--ensemble_size", type=int, default=2, help="Deep ensemble members")
    p.add_argument("--ensemble_run_ids", type=str, default="", help="Comma-separated run IDs for deep ensemble")
    p.add_argument("--max_samples", type=int, default=2000, help="Evaluate on at most this many samples")
    p.add_argument("--proxy_bag_size", type=int, default=16, help="PCAM patch-to-slide proxy bag size")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", type=str, default="", help="Optional checkpoints/run_*/best.pt override")
    p.add_argument(
        "--fit_temperature_on_val",
        action="store_true",
        help="Fit temperature on validation set and report calibrated metrics on target split",
    )
    p.add_argument(
        "--fit_deferral_on_val",
        action="store_true",
        help="Fit uncertainty deferral thresholds on validation split and report transfer to target split",
    )
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def _new_model(cfg: dict):
    model, _, _ = load_hf_image_classifier(
        model_id=cfg["model"]["model_id"],
        num_labels=cfg["model"]["num_labels"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
    return model


def load_models(
    cfg: dict,
    method: str,
    run_id: str,
    ensemble_run_ids: list[str],
    ensemble_size: int,
) -> tuple[list[torch.nn.Module], dict]:
    models: list[torch.nn.Module] = []
    load_method = "confidence" if method == "temperature_scaled" else method
    if load_method == "deep_ensemble":
        members = resolve_deep_ensemble_members(
            config_model_id=cfg["model"]["model_id"],
            config_dataset=cfg["data"]["dataset"],
            run_id=run_id,
            ensemble_run_ids=ensemble_run_ids,
            ensemble_size=max(2, ensemble_size),
        )
        member_model_id = members[0]["model_id"]
        ensemble_cfg = dict(cfg)
        ensemble_cfg["model"] = dict(cfg.get("model", {}))
        ensemble_cfg["model"]["model_id"] = member_model_id
        for member in members:
            m = _new_model(ensemble_cfg)
            state = torch.load(member["ckpt_path"], map_location="cpu", weights_only=True)
            if "model_state_dict" in state:
                m.load_state_dict(state["model_state_dict"], strict=True)
            models.append(m)
        return models, {
            "model_id": member_model_id,
            "ensemble_run_ids": [member["run_id"] for member in members],
        }
    model_cfg = dict(cfg)
    model_cfg["model"] = dict(cfg.get("model", {}))
    ckpt = None
    model_id = cfg["model"]["model_id"]
    if run_id:
        candidate = REPO_ROOT / "checkpoints" / run_id / "best.pt"
        if candidate.exists():
            ckpt = candidate
            try:
                meta = load_run_metadata(run_id)
                if meta.get("model_id"):
                    model_id = meta["model_id"]
            except Exception:
                pass
    else:
        default = REPO_ROOT / "checkpoints" / "best.pt"
        if default.exists():
            ckpt = default
    model_cfg["model"]["model_id"] = model_id
    model = _new_model(model_cfg)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
    models.append(model)
    return models, {"model_id": model_id, "ensemble_run_ids": [run_id] if run_id else []}


def _calibration_split_for_eval(eval_split: str) -> str:
    return "val" if eval_split != "val" else "train"


def _threshold_split_for_eval(eval_split: str) -> str:
    return "train" if eval_split != "train" else "val"


def collect_predictions(
    models: list[torch.nn.Module],
    split: str,
    transform,
    data_root: Path,
    max_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    method: str,
    mc_samples: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ds = PCAM(root=str(data_root), split=split, download=False, transform=transform)
    n_total = len(ds)
    n_use = min(max(1, max_samples), n_total)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_total, size=n_use, replace=False))
    subset = torch.utils.data.Subset(ds, indices.tolist())
    loader_kwargs = {
        "batch_size": max(1, batch_size),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = torch.utils.data.DataLoader(subset, **loader_kwargs)

    logits_list = []
    y_list = []
    member_prob_chunks = []
    first_model = models[0]
    effective_method = "confidence" if method == "temperature_scaled" else method
    for batch, labels in loader:
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if effective_method == "mc_dropout":
            # Only enable Dropout modules; keep LayerNorm etc. in eval mode.
            # Use logit-space averaging (not probability-space) to avoid Jensen's inequality:
            # mean(softmax(logits_i)) biases predictions toward class 0 for borderline samples,
            # causing ~19pp accuracy/sensitivity drop vs. softmax(mean(logits_i)).
            first_model.eval()
            for mod in first_model.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.training = True
            logits_stack = []
            with torch.inference_mode():
                for _ in range(max(2, mc_samples)):
                    with _autocast_context(device):
                        lg = first_model(pixel_values=batch).logits
                    logits_stack.append(lg)
            logits_t = torch.stack(logits_stack, dim=0)  # (T, B, C)
            probs_stack_t = torch.softmax(logits_t, dim=2)
            logits = logits_t.mean(dim=0)  # logit-space average for predictions
            member_prob_chunks.append(probs_stack_t.detach().cpu())
            first_model.eval()
        elif effective_method == "deep_ensemble":
            probs_members = []
            with torch.inference_mode():
                for m in models:
                    m.eval()
                    with _autocast_context(device):
                        logits_m = m(pixel_values=batch).logits
                    probs_members.append(torch.softmax(logits_m, dim=1))
            probs_members_t = torch.stack(probs_members, dim=0)
            probs = probs_members_t.mean(dim=0)
            logits = torch.log(probs.clamp(min=1e-12))
            member_prob_chunks.append(probs_members_t.detach().cpu())
        else:
            first_model.eval()
            with torch.inference_mode():
                with _autocast_context(device):
                    logits = first_model(pixel_values=batch).logits
        logits_list.append(logits.detach().cpu())
        y_list.append(labels.detach().cpu())

    logits_np = torch.cat(logits_list, dim=0).numpy().astype(np.float64)
    y_np = torch.cat(y_list, dim=0).numpy().astype(np.int64)
    extras: dict[str, np.ndarray] = {}
    if member_prob_chunks:
        extras["member_probs"] = torch.cat(member_prob_chunks, dim=1).numpy().astype(np.float64)
    return logits_np, y_np, extras


def build_summary(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int,
    member_probs: np.ndarray | None = None,
) -> dict:
    summary = summarize_from_logits(logits, y_true, n_bins=n_bins)
    primary_name = "uncertainty_one_minus_msp"
    primary_values = np.asarray(summary["internals"]["uncertainty_one_minus_msp"], dtype=np.float64)
    if member_probs is not None:
        disagreement_scores = disagreement_score_arrays(member_probs)
        if disagreement_scores:
            summary["uncertainty_quality"]["distributional_disagreement"] = {
                "n_members_or_passes": int(member_probs.shape[0]),
                "primary_score": "mutual_information",
                "scores": summarize_uncertainty_scores(disagreement_scores, summary["internals"]["error"]),
            }
            primary_name = "mutual_information"
            primary_values = np.asarray(disagreement_scores["mutual_information"], dtype=np.float64)
            summary["internals"]["disagreement_scores"] = disagreement_scores
    summary["internals"]["primary_uncertainty_name"] = primary_name
    summary["internals"]["primary_uncertainty"] = primary_values
    return summary


def main() -> int:
    args = parse_args()
    args.run_id = normalize_run_id(args.run_id)
    cfg = get_config()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _configure_cuda_backends()

    device = get_device()
    print(f"Using device: {device}")
    num_workers = _resolve_num_workers(cfg)
    print(f"DataLoader workers: {num_workers}")

    image_size = tuple(cfg["data"]["image_size"])
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ensemble_run_ids = [x.strip() for x in (args.ensemble_run_ids or "").split(",") if x.strip()]
    models, model_info = load_models(
        cfg,
        method=args.method,
        run_id=args.run_id,
        ensemble_run_ids=ensemble_run_ids,
        ensemble_size=max(1, args.ensemble_size),
    )
    models = [m.to(device) for m in models]
    n_bins = int(cfg.get("evaluation", {}).get("calibration_bins", 15))
    logits_eval_raw, y_eval, eval_extras = collect_predictions(
        models=models,
        split=args.split,
        transform=transform,
        data_root=REPO_ROOT / cfg["data"]["root"],
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        method=args.method,
        mc_samples=args.mc_samples,
        num_workers=num_workers,
    )
    raw_summary = build_summary(
        logits_eval_raw,
        y_eval,
        n_bins=n_bins,
        member_probs=eval_extras.get("member_probs"),
    )
    base_summary = raw_summary
    logits_eval_final = logits_eval_raw
    effective_method = "confidence" if args.method == "temperature_scaled" else args.method

    calibration_report = {"uncalibrated": raw_summary["calibration"], "temperature_scaling": None}
    calibration_split = _calibration_split_for_eval(args.split)
    needs_temperature = args.method == "temperature_scaled" or args.fit_temperature_on_val
    temperature = None
    calibration_logits = None
    calibration_y = None
    calibration_extras: dict | None = None
    if needs_temperature or args.fit_deferral_on_val:
        calibration_logits, calibration_y, calibration_extras = collect_predictions(
            models=models,
            split=calibration_split,
            transform=transform,
            data_root=REPO_ROOT / cfg["data"]["root"],
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed + 1,
            device=device,
            method=args.method,
            mc_samples=args.mc_samples,
            num_workers=num_workers,
        )
        if needs_temperature and calibration_logits is not None and calibration_y is not None:
            temperature = optimize_temperature(calibration_logits, calibration_y)
            cal_logits_eval = logits_eval_raw / temperature
            cal_summary = build_summary(cal_logits_eval, y_eval, n_bins=n_bins)
            calibration_report["temperature_scaling"] = {
                "source_split": calibration_split,
                "temperature": float(temperature),
                "calibrated": cal_summary["calibration"],
                "delta_ece": float(cal_summary["calibration"]["ece"] - raw_summary["calibration"]["ece"]),
                "delta_nll": float(cal_summary["calibration"]["nll"] - raw_summary["calibration"]["nll"]),
                "delta_brier": float(cal_summary["calibration"]["brier"] - raw_summary["calibration"]["brier"]),
            }
            if args.method == "temperature_scaled":
                logits_eval_final = cal_logits_eval
                base_summary = cal_summary
        if args.fit_deferral_on_val:
            logits_for_deferral = (
                calibration_logits / temperature
                if (temperature is not None and args.method == "temperature_scaled")
                else calibration_logits
            )
            val_summary = build_summary(
                logits_for_deferral,
                calibration_y,
                n_bins=n_bins,
                member_probs=calibration_extras.get("member_probs") if calibration_extras else None,
            )
            val_unc = np.array(val_summary["internals"]["primary_uncertainty"], dtype=np.float64)
            val_err = np.array(val_summary["internals"]["error"], dtype=np.int64)
            eval_unc = np.array(base_summary["internals"]["primary_uncertainty"], dtype=np.float64)
            eval_err = np.array(base_summary["internals"]["error"], dtype=np.int64)
            thresholds = fit_uncertainty_thresholds(val_unc, val_err, targets=[0.01, 0.02, 0.05], min_coverage=0.2)
            transfer = apply_uncertainty_thresholds(eval_unc, eval_err, thresholds)
            base_summary["selective_prediction"]["validation_fitted_deferral"] = {
                "source_split": calibration_split,
                "method": base_summary["internals"]["primary_uncertainty_name"],
                "fit_on_val": thresholds,
                "applied_on_eval": transfer,
                "reference": "Dolezal et al. 2022 style thresholding concept: defer/remove low-confidence predictions.",
            }

    threshold_split = _threshold_split_for_eval(args.split)
    logits_thr_raw, y_thr, thr_extras = collect_predictions(
        models=models,
        split=threshold_split,
        transform=transform,
        data_root=REPO_ROOT / cfg["data"]["root"],
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed + 2,
        device=device,
        method=args.method,
        mc_samples=args.mc_samples,
        num_workers=num_workers,
    )
    logits_thr = (logits_thr_raw / temperature) if (temperature is not None and args.method == "temperature_scaled") else logits_thr_raw
    thr_summary = build_summary(
        logits_thr,
        y_thr,
        n_bins=n_bins,
        member_probs=thr_extras.get("member_probs") if effective_method in ("mc_dropout", "deep_ensemble") else None,
    )
    cohort_threshold_fit = fit_youden_uncertainty_threshold(
        thr_summary["internals"]["primary_uncertainty"],
        thr_summary["internals"]["error"],
        min_coverage=0.2,
    )
    base_summary["selective_prediction"]["high_confidence_reporting"] = {
        "requested_source_split": "train",
        "source_split": threshold_split,
        "threshold_strategy": "train_only_youden_threshold" if threshold_split == "train" else "fallback_distinct_split_youden_threshold",
        "uncertainty_metric": base_summary["internals"]["primary_uncertainty_name"],
        "fit": cohort_threshold_fit,
        "cohorts": summarize_uncertainty_cohorts(
            logits_eval_final,
            y_eval,
            base_summary["internals"]["primary_uncertainty"],
            cohort_threshold_fit.get("threshold"),
            n_bins=n_bins,
            uncertainty_name=base_summary["internals"]["primary_uncertainty_name"],
        ),
        "reference": "Training-only threshold fitting inspired by clinically oriented high-confidence reporting in Dolezal et al. (2022).",
    }

    probs_eval = torch.softmax(torch.tensor(logits_eval_final, dtype=torch.float32), dim=1).numpy()
    bag_size = max(2, int(args.proxy_bag_size))
    slide_proxy = slide_level_proxy_from_probs(probs_eval, y_eval, bag_size=bag_size)

    out = {
        "config": {
            "split": args.split,
            "method": args.method,
            "mc_samples": int(args.mc_samples if args.method == "mc_dropout" else 1),
            "ensemble_size": len(models) if args.method == "deep_ensemble" else 1,
            "ensemble_run_ids": model_info.get("ensemble_run_ids", []),
            "max_samples": int(len(y_eval)),
            "dataset_size": int(len(PCAM(root=str(REPO_ROOT / cfg["data"]["root"]), split=args.split, download=False))),
            "run_id": args.run_id or None,
            "model_id": model_info.get("model_id") or cfg["model"]["model_id"],
            "fit_temperature_on_val": bool(args.fit_temperature_on_val or args.method == "temperature_scaled"),
            "fit_deferral_on_val": bool(args.fit_deferral_on_val),
            "calibration_split": calibration_split if temperature is not None else None,
            "cohort_threshold_split": threshold_split,
        },
        "predictive_performance": base_summary["predictive_performance"],
        "calibration": base_summary["calibration"],
        "uncertainty_quality": base_summary["uncertainty_quality"],
        "selective_prediction": base_summary["selective_prediction"],
        "calibration_report": calibration_report,
        "plot_data": {
            "uncertainty_one_minus_msp": np.asarray(base_summary["internals"]["uncertainty_one_minus_msp"], dtype=np.float64).tolist(),
            "entropy": np.asarray(base_summary["internals"]["entropy"], dtype=np.float64).tolist(),
            "confidence": np.asarray(base_summary["internals"]["confidence"], dtype=np.float64).tolist(),
            "primary_uncertainty": np.asarray(base_summary["internals"]["primary_uncertainty"], dtype=np.float64).tolist(),
            "primary_uncertainty_name": base_summary["internals"]["primary_uncertainty_name"],
            "correct": (1 - np.asarray(base_summary["internals"]["error"], dtype=np.int64)).astype(np.int64).tolist(),
        },
        "pathology_reporting": {
            "slide_level_proxy": {
                "dataset": "pcam",
                "proxy_type": "pcam_fixed_bag_aggregation",
                "note": "PCAM has no real slide IDs; this proxy uses deterministic fixed-size bags formed after sorting sampled patch indices by dataset order.",
                **slide_proxy,
            }
        },
        "literature_alignment": {
            "thresholding_deferral_reference": "[2] Dolezal et al. (2022) DOI: 10.1200/JCO.2022.40.16_suppl.8549",
            "temperature_scaling_reference": "Guo et al. (2017) On Calibration of Modern Neural Networks.",
            "near_far_ood_context_references": [
                "[8] Linmans et al. (2023) DOI: 10.1016/j.media.2022.102655",
                "[11] Thagaard et al. (2020) DOI: 10.1007/978-3-030-59710-8_80",
            ],
        },
    }

    # Remove internals before writing final JSON payload.
    out["predictive_performance"] = base_summary["predictive_performance"]
    out["calibration"] = base_summary["calibration"]
    out["uncertainty_quality"] = base_summary["uncertainty_quality"]
    out["selective_prediction"] = base_summary["selective_prediction"]

    if args.out:
        out_path = Path(args.out)
    else:
        out_name = f"metrics_{args.method}_{args.split}.json"
        out_path = REPO_ROOT / "evaluation" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = _json_safe(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)

    rel_plot = out_path.with_name(out_path.stem + "_reliability.png")
    save_reliability_plot(out["calibration"]["reliability_bins"], rel_plot, title=f"Reliability ({args.method}, {args.split})")
    out["calibration"]["reliability_plot_path"] = str(rel_plot)
    out = _json_safe(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)

    print(json.dumps(out["predictive_performance"], indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

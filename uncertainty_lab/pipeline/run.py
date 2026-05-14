"""Run the training and evaluation pipeline from a config dict."""
from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uncertainty_lab.config import save_config, stamp_run_dir
from uncertainty_lab.data.factory import build_eval_loader
from uncertainty_lab.metrics.core import compute_metrics_bundle, json_safe, top_k_indices
from uncertainty_lab.metrics.plots import plot_reliability, plot_risk_coverage, plot_uncertainty_histograms
from uncertainty_lab.device import resolve_device
from uncertainty_lab.models.loader import load_models_for_uncertainty
from uncertainty_lab.pipeline import train as train_mod
from uncertainty_lab.uncertainty.base import get_method


def _configure_cuda_backends() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def run_pipeline(
    config: dict,
    *,
    progress_callback: Callable[[str, float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Execute the pipeline from a config dict (see ``configs/uncertainty_lab_default.yaml``).

    Returns a result dict with ``run_dir``, paths to artifacts, and ``metrics`` (JSON-safe).

    ``progress_callback(message, fraction)`` receives ``fraction`` in ``[0, 1]`` for UI progress bars.
    ``log_callback(line)`` receives throttled batch / detail lines for logs.
    """
    run_cfg = config.setdefault("run", {})
    _rr = run_cfg.get("repo_root")
    repo_root = Path(_rr) if _rr else Path(__file__).resolve().parents[2]

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _configure_cuda_backends()

    base_runs = Path(run_cfg.get("output_base", str(repo_root / "runs")))
    if not base_runs.is_absolute():
        base_runs = (repo_root / base_runs).resolve()
    run_name = run_cfg.get("name")
    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else stamp_run_dir(base_runs, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cfg["run_dir"] = str(run_dir)

    cfg_snapshot = copy.deepcopy(config)
    save_config(cfg_snapshot, run_dir / "config.yaml")

    pipeline_mode = str(config.get("pipeline", {}).get("mode", "evaluate")).lower()
    eval_cfg = copy.deepcopy(config)

    eval_base = 0.0
    eval_span = 1.0

    def report(msg: str, t: float) -> None:
        if progress_callback is None:
            return
        x = eval_base + max(0.0, min(1.0, t)) * eval_span
        progress_callback(msg, max(0.0, min(1.0, x)))

    report("Preparing run…", 0.02)

    if pipeline_mode in ("train", "train_evaluate"):
        report("Training model…", 0.08)
        best_pt = train_mod.run_training(config, run_dir, repo_root)
        eval_cfg["model"]["source"] = "huggingface"
        eval_cfg["model"]["local_checkpoint"] = str(best_pt)
        report("Training finished.", 0.28)

    if pipeline_mode == "train":
        report("Run complete.", 1.0)
        return {
            "run_dir": str(run_dir),
            "status": "trained",
            "checkpoint": str(run_dir / "checkpoint" / "best.pt"),
        }

    if pipeline_mode == "train_evaluate":
        eval_base, eval_span = 0.32, 0.68

    # Evaluate either directly or after the optional training stage.
    report("Loading dataset…", 0.06)
    device = resolve_device(eval_cfg)
    split = str(eval_cfg.get("dataset", {}).get("eval_split", "test"))
    loader, data_meta = build_eval_loader(eval_cfg, repo_root, split=split)
    # Also load validation split for conformal calibration (best-effort)
    cal_loader = None
    try:
        cal_loader, _ = build_eval_loader(eval_cfg, repo_root, split="val")
    except Exception as _cal_exc:
        if log_callback:
            log_callback(f"[WARN] Validation loader failed — conformal prediction will be skipped: {_cal_exc}")
    if log_callback:
        log_callback(f"Dataset ready: split={split!r}, batches={len(loader)}")

    u_cfg = eval_cfg.get("uncertainty", {})
    method_id = str(u_cfg.get("method", "confidence")).lower()
    mc_samples = int(u_cfg.get("mc_dropout_n_samples", u_cfg.get("mc_samples", 30)))

    report("Loading model…", 0.14)
    models = load_models_for_uncertainty(eval_cfg, method=method_id)
    models = [m.to(device) for m in models]
    if log_callback:
        log_callback(f"Uncertainty method: {method_id}, device: {device}, MC passes: {mc_samples}")

    method = get_method(method_id)

    def on_batch(cur: int, total: int) -> None:
        if total <= 0:
            return
        t_inf = 0.22 + 0.52 * (cur / total)
        report(f"Running inference… batch {cur}/{total}", t_inf)
        if log_callback and (cur == total or total <= 12 or cur % max(1, total // 25) == 0):
            log_callback(f"Inference batch {cur}/{total}")

    logits_np, y_np, extras = method.predict_with_extras(
        models, loader, device, mc_samples=mc_samples, on_batch=on_batch
    )
    member_probs = extras.get("member_probs")  # (T, N, C) or None

    # Calibration set logits for conformal prediction
    cal_logits_np = cal_labels_np = None
    if cal_loader is not None:
        try:
            cal_logits_np, cal_labels_np, _ = method.predict_with_extras(
                models, cal_loader, device, mc_samples=mc_samples
            )
        except Exception:
            cal_logits_np = cal_labels_np = None

    report("Computing metrics…", 0.80)
    full_summary = compute_metrics_bundle(
        logits_np, y_np, eval_cfg, for_json=False,
        member_probs=member_probs,
        cal_logits=cal_logits_np, cal_labels=cal_labels_np,
    )
    metrics_for_file = compute_metrics_bundle(
        logits_np, y_np, eval_cfg, for_json=True,
        member_probs=member_probs,
        cal_logits=cal_logits_np, cal_labels=cal_labels_np,
    )
    intr = full_summary["internals"]

    ev_plots = eval_cfg.get("evaluation", {}).get("plots", {})
    plot_rel = ev_plots.get("reliability", True)
    plot_rc = ev_plots.get("risk_coverage", True)
    plot_hist = ev_plots.get("uncertainty_histograms", True)

    report("Generating plots…", 0.90)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if plot_rel:
        p = plots_dir / "reliability.png"
        plot_reliability(
            metrics_for_file["calibration"]["reliability_bins"],
            p,
            title=f"Reliability ({method_id}, {split})",
        )
        metrics_for_file["calibration"]["reliability_plot_path"] = str(p)
    if plot_rc:
        rc = full_summary["selective_prediction"]["risk_coverage_curve"]
        p = plots_dir / "risk_coverage.png"
        plot_risk_coverage(rc, p, title="Risk–coverage")
        metrics_for_file["selective_prediction"]["risk_coverage_plot_path"] = str(p)
    if plot_hist:
        # Prefer mutual information for MC dropout / deep ensemble if available
        u = intr.get("mutual_information", intr["uncertainty_one_minus_msp"])
        err = intr["error"]
        correct = 1 - err
        p = plots_dir / "uncertainty_histograms.png"
        plot_uncertainty_histograms(u, correct, p)
        metrics_for_file.setdefault("plots", {})["uncertainty_histogram_path"] = str(p)

    top_k = int(eval_cfg.get("evaluation", {}).get("top_k_uncertain", 0))
    sample_refs: dict[str, Any] = {}
    if top_k > 0:
        idx = top_k_indices(intr["uncertainty_one_minus_msp"], intr["error"], top_k, prefer_errors=True)
        sample_refs = {
            "indices": idx.tolist(),
            "uncertainty": intr["uncertainty_one_minus_msp"][idx].tolist(),
            "labels": y_np[idx].tolist(),
            "preds": intr["y_pred"][idx].tolist(),
        }
        metrics_for_file["highlighted_samples"] = sample_refs

    out_payload = {
        "run_dir": str(run_dir),
        "pipeline_mode": pipeline_mode,
        "uncertainty_method": method_id,
        "data": data_meta,
        "predictive_performance": metrics_for_file["predictive_performance"],
        "calibration": metrics_for_file["calibration"],
        "uncertainty_quality": metrics_for_file.get("uncertainty_quality"),
        "selective_prediction": metrics_for_file["selective_prediction"],
        "highlighted_samples": metrics_for_file.get("highlighted_samples"),
        "conformal_prediction": metrics_for_file.get("conformal_prediction"),
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(out_payload), f, indent=2, allow_nan=False)

    report("Run complete.", 1.0)
    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "metrics": json_safe(out_payload),
        "status": "completed",
    }


def run_benchmark(
    config: dict,
    *,
    progress_callback: Callable[[str, float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run evaluation for multiple uncertainty methods; write comparison JSON under a group folder."""
    _rr = config.get("run", {}).get("repo_root")
    repo_root = Path(_rr) if _rr else Path(__file__).resolve().parents[2]
    methods = list(config.get("benchmark", {}).get("methods", ["confidence", "mc_dropout"]))
    base_out = Path(config.get("run", {}).get("output_base", str(repo_root / "runs")))
    if not base_out.is_absolute():
        base_out = (repo_root / base_out).resolve()
    bench_root = stamp_run_dir(base_out, config.get("run", {}).get("name", "benchmark"))
    bench_root.mkdir(parents=True, exist_ok=True)
    save_config(copy.deepcopy(config), bench_root / "benchmark_config.yaml")

    rows = []
    n_m = max(len(methods), 1)
    for mi, m in enumerate(methods):
        lo = mi / n_m
        span = 1.0 / n_m

        def _wrap_cb(msg: str, frac: float, lo: float = lo, span: float = span, m: str = m) -> None:
            if progress_callback is not None:
                progress_callback(f"{m}: {msg}", lo + frac * span)

        def _wrap_log(line: str, m: str = m) -> None:
            if log_callback is not None:
                log_callback(f"[{m}] {line}")

        c = copy.deepcopy(config)
        c.setdefault("uncertainty", {})["method"] = m
        c["pipeline"] = {**(c.get("pipeline") or {}), "mode": "evaluate"}
        c["run"] = copy.deepcopy(config.get("run", {}))
        c["run"]["run_dir"] = str(bench_root / f"method_{m}")
        c["run"]["output_base"] = str(base_out)
        c["run"]["repo_root"] = str(repo_root)
        r = run_pipeline(c, progress_callback=_wrap_cb, log_callback=_wrap_log)
        met = r.get("metrics", {})
        perf = met.get("predictive_performance", {})
        cal = met.get("calibration", {})
        sel = met.get("selective_prediction", {})
        rows.append(
            {
                "method": m,
                "accuracy": perf.get("accuracy"),
                "roc_auc": perf.get("roc_auc"),
                "ece": cal.get("ece"),
                "brier": cal.get("brier"),
                "aurc": sel.get("aurc"),
                "run_dir": r.get("run_dir"),
            }
        )

    p = bench_root / "comparison.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "benchmark_root": str(bench_root)}, f, indent=2)
    if progress_callback is not None:
        progress_callback("Benchmark complete.", 1.0)
    return {"comparison_path": str(p), "benchmark_root": str(bench_root), "rows": rows}

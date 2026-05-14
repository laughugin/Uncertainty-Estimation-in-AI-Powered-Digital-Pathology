#!/usr/bin/env python3
"""
Run the evaluation pipeline matrix and write a combined summary.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--include_deep_ensemble", action="store_true")
    p.add_argument("--ensemble_size", type=int, default=2)
    p.add_argument("--ensemble_run_ids", type=str, default="", help="Comma-separated run IDs for deep ensemble")
    p.add_argument("--fit_temperature_on_val", action="store_true")
    p.add_argument("--fit_deferral_on_val", action="store_true")
    p.add_argument("--run_id", type=str, default="")
    p.add_argument("--out", type=str, default="evaluation/pipeline_summary.json")
    p.add_argument(
        "--use-uncertainty-lab",
        action="store_true",
        help="Run confidence/mc_dropout via uncertainty_lab.run_pipeline (temperature_scaled and deep_ensemble still use legacy script).",
    )
    return p.parse_args()


def run_one_lab(method: str, args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from uncertainty_lab.config import deep_merge, load_config
    from uncertainty_lab.pipeline.run import run_pipeline

    cfg = load_config(repo_root=REPO_ROOT)
    cfg = deep_merge(
        cfg,
        {
            "pipeline": {"mode": "evaluate"},
            "dataset": {
                "type": "pcam",
                "root": "data/raw",
                "eval_split": args.split,
                "max_eval_samples": args.max_samples,
                "batch_size": args.batch_size,
            },
            "uncertainty": {"method": method, "mc_dropout_n_samples": max(2, args.mc_samples)},
            "run": {"name": f"pipe_{method}_{args.split}", "repo_root": str(REPO_ROOT)},
        },
    )
    if args.run_id:
        ck = REPO_ROOT / "checkpoints" / args.run_id / "best.pt"
        if ck.is_file():
            cfg.setdefault("model", {})["local_checkpoint"] = str(ck)
    r = run_pipeline(cfg)
    dest = REPO_ROOT / "evaluation" / f"metrics_{method}_{args.split}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(r["metrics_path"]), dest)
    with open(dest, encoding="utf-8") as f:
        return json.load(f)


def run_one(method: str, args: argparse.Namespace) -> dict:
    if args.use_uncertainty_lab and method in ("confidence", "mc_dropout"):
        return run_one_lab(method, args)
    out_path = REPO_ROOT / "evaluation" / f"metrics_{method}_{args.split}.json"
    cmd = [
        sys.executable,
        "experiments/evaluate_uncertainty.py",
        "--split",
        args.split,
        "--method",
        method,
        "--max_samples",
        str(max(1, args.max_samples)),
        "--batch_size",
        str(max(1, args.batch_size)),
        "--mc_samples",
        str(max(2, args.mc_samples)),
        "--ensemble_size",
        str(max(1, args.ensemble_size)),
        "--out",
        str(out_path),
    ]
    if method == "deep_ensemble" and (args.ensemble_run_ids or "").strip():
        cmd.extend(["--ensemble_run_ids", args.ensemble_run_ids.strip()])
    if args.fit_temperature_on_val:
        cmd.append("--fit_temperature_on_val")
    if args.fit_deferral_on_val:
        cmd.append("--fit_deferral_on_val")
    if args.run_id:
        cmd.extend(["--run_id", args.run_id])

    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{method} failed:\n{r.stdout}\n{r.stderr}")
    with open(out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def main() -> int:
    args = parse_args()
    results = {}
    methods = ["confidence", "temperature_scaled", "mc_dropout"]
    if args.include_deep_ensemble:
        methods.append("deep_ensemble")
    for method in methods:
        results[method] = run_one(method, args)

    summary = {
        "config": {
            "split": args.split,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "mc_samples": args.mc_samples,
            "include_deep_ensemble": bool(args.include_deep_ensemble),
            "ensemble_size": args.ensemble_size,
            "ensemble_run_ids": [x.strip() for x in (args.ensemble_run_ids or "").split(",") if x.strip()],
            "fit_temperature_on_val": args.fit_temperature_on_val,
            "fit_deferral_on_val": args.fit_deferral_on_val,
            "run_id": args.run_id or None,
            "model_id": next((results[m].get("config", {}).get("model_id") for m in results if results[m].get("config", {}).get("model_id")), None),
        },
        "results": {
            m: {
                "config": results[m].get("config", {}),
                "predictive_performance": results[m].get("predictive_performance", {}),
                "calibration": results[m].get("calibration", {}),
                "selective_prediction": results[m].get("selective_prediction", {}),
                "uncertainty_quality": results[m].get("uncertainty_quality", {}),
                "calibration_report": results[m].get("calibration_report", {}),
                "pathology_reporting": results[m].get("pathology_reporting", {}),
            }
            for m in results
        },
    }

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved pipeline summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

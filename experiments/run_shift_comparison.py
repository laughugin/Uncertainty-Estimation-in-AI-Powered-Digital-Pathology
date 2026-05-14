#!/usr/bin/env python3
"""
Run shift/OOD comparison across uncertainty methods and write a combined summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--include_deep_ensemble", action="store_true")
    p.add_argument("--ensemble_size", type=int, default=2)
    p.add_argument("--ensemble_run_ids", type=str, default="", help="Comma-separated run IDs for deep ensemble")
    p.add_argument("--run_id", type=str, default="")
    p.add_argument("--shifts", type=str, default="id,blur,jpeg,color,noise")
    p.add_argument("--severities", type=str, default="1,3,5")
    p.add_argument("--out", type=str, default="evaluation/shift_comparison_summary.json")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(method: str, args: argparse.Namespace) -> dict:
    out_path = REPO_ROOT / "evaluation" / f"shift_ood_{method}_{args.split}.json"
    cmd = [
        sys.executable,
        "experiments/evaluate_shift_ood.py",
        "--split",
        args.split,
        "--method",
        method,
        "--mc_samples",
        str(max(2, args.mc_samples)),
        "--ensemble_size",
        str(max(1, args.ensemble_size)),
        "--max_samples",
        str(max(1, args.max_samples)),
        "--batch_size",
        str(max(1, args.batch_size)),
        "--shifts",
        args.shifts,
        "--severities",
        args.severities,
        "--out",
        str(out_path),
    ]
    if method == "deep_ensemble" and (args.ensemble_run_ids or "").strip():
        cmd.extend(["--ensemble_run_ids", args.ensemble_run_ids.strip()])
    if args.run_id:
        cmd.extend(["--run_id", args.run_id])
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{method} shift comparison failed:\n{r.stdout}\n{r.stderr}")
    return load_json(out_path)


def main() -> int:
    args = parse_args()
    methods = ["confidence", "temperature_scaled", "mc_dropout"]
    if args.include_deep_ensemble:
        methods.append("deep_ensemble")

    by_method = {method: run_one(method, args) for method in methods}

    def _mean(field: str, group: str) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for method, payload in by_method.items():
            out[method] = ((payload.get("grouped_summary", {}) or {}).get(group, {}) or {}).get(field)
        return out

    def _best(values: dict[str, float | None]) -> dict | None:
        valid = [(k, v) for k, v in values.items() if v is not None]
        if not valid:
            return None
        valid.sort(key=lambda item: item[1], reverse=True)
        return {"method": valid[0][0], "value": float(valid[0][1])}

    near_auroc = _mean("mean_ood_auroc", "near_ood")
    far_auroc = _mean("mean_ood_auroc", "far_ood")

    summary = {
        "config": {
            "split": args.split,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "mc_samples": args.mc_samples,
            "include_deep_ensemble": bool(args.include_deep_ensemble),
            "ensemble_size": args.ensemble_size,
            "ensemble_run_ids": [x.strip() for x in (args.ensemble_run_ids or "").split(",") if x.strip()],
            "run_id": args.run_id or None,
            "shifts": args.shifts,
            "severities": args.severities,
        },
        "results_by_method": {k: v.get("results", {}) for k, v in by_method.items()},
        "grouped_summary_by_method": {k: v.get("grouped_summary", {}) for k, v in by_method.items()},
        "method_comparison": {
            "near_ood_mean_auroc_by_method": near_auroc,
            "far_ood_mean_auroc_by_method": far_auroc,
            "best_near_ood_by_auroc": _best(near_auroc),
            "best_far_ood_by_auroc": _best(far_auroc),
        },
    }

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved shift comparison summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

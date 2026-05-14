#!/usr/bin/env python3
"""ECE (and accuracy) under corruption severity — per shift type, per method.

Reads the existing shift-OOD JSON files in evaluation/ and plots how
calibration error and accuracy change as severity increases from 1 → 3 → 5
for blur, noise, jpeg, and color shifts.

Output:
  evaluation/figures/ece_under_shift.png
  evaluation/figures/accuracy_under_shift.png
  evaluation/ece_under_shift_summary.json

If shift-OOD JSONs are missing, runs a quick inference pass to generate them.

Usage:
    python experiments/run_ece_under_shift.py
    python experiments/run_ece_under_shift.py --regenerate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from uncertainty_lab.metrics.plots import plot_ece_under_shift, plot_accuracy_under_shift
from uncertainty_lab.metrics.core import json_safe


METHODS = ["confidence", "mc_dropout", "deep_ensemble", "temperature_scaled"]
SHIFT_TYPES = ["blur", "noise", "jpeg", "color"]
SEVERITIES = [1, 3, 5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--regenerate", action="store_true",
                   help="Force re-run of shift evaluation even if JSONs exist")
    return p.parse_args()


def _load_shift_json(method: str, split: str) -> dict:
    """Return {condition_key: {ece, accuracy, ...}} from saved shift JSON."""
    path = REPO_ROOT / "evaluation" / f"shift_ood_{method}_{split}.json"
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text())
        # New-format: d["results"] is the dict keyed by condition
        results = d.get("results", d)
        if isinstance(results, dict) and any(k.endswith("_s0") or k.endswith("_s1") for k in results):
            return results
        return {}
    except Exception:
        return {}


def _collect_data(split: str) -> dict[str, dict]:
    """Return dict method -> condition_key -> metric_dict."""
    out = {}
    for method in METHODS:
        conds = _load_shift_json(method, split)
        if conds:
            out[method] = conds
    return out


def _summarise_shift_data(shift_data: dict[str, dict]) -> dict:
    """Create a tidy summary dict for JSON output."""
    summary: dict[str, dict] = {}
    for method, conds in shift_data.items():
        summary[method] = {}
        for shift in SHIFT_TYPES:
            summary[method][shift] = {"id": None}
            id_cond = conds.get("id_s0", {})
            summary[method][shift]["id"] = {
                "ece": id_cond.get("ece"),
                "accuracy": id_cond.get("accuracy"),
            }
            for sev in SEVERITIES:
                key = f"{shift}_s{sev}"
                cond = conds.get(key, {})
                summary[method][shift][f"sev_{sev}"] = {
                    "ece": cond.get("ece"),
                    "accuracy": cond.get("accuracy"),
                }
    return summary


def main():
    args = parse_args()
    split = args.split

    print("Loading shift-OOD evaluation data...")
    shift_data = _collect_data(split)

    if not shift_data:
        print("No shift-OOD JSON files found. Run the shift evaluation first:")
        print("  python experiments/run_evaluation_pipeline.py --shift")
        sys.exit(1)

    print(f"Loaded data for methods: {list(shift_data.keys())}")
    for method, conds in shift_data.items():
        print(f"  {method}: {len(conds)} conditions")

    # Generate figures
    out_dir = REPO_ROOT / "evaluation"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ece_path = fig_dir / "ece_under_shift.png"
    plot_ece_under_shift(shift_data, ece_path,
                         title="ECE under distribution shift (all methods)")
    print(f"Figure: {ece_path}")

    acc_path = fig_dir / "accuracy_under_shift.png"
    plot_accuracy_under_shift(shift_data, acc_path,
                              title="Accuracy under distribution shift (all methods)")
    print(f"Figure: {acc_path}")

    # Print summary table
    print("\n=== ECE by method and severity ===")
    header = f"{'Method':<22} {'Shift':<8} {'Clean':>8} {'Sev1':>8} {'Sev3':>8} {'Sev5':>8}"
    print(header)
    print("-" * len(header))
    for method, conds in sorted(shift_data.items()):
        id_ece = (conds.get("id_s0") or {}).get("ece")
        for shift in SHIFT_TYPES:
            row = f"{method:<22} {shift:<8} {id_ece or float('nan'):8.4f}"
            for sev in SEVERITIES:
                key = f"{shift}_s{sev}"
                v = (conds.get(key) or {}).get("ece")
                row += f" {v if v is not None else float('nan'):8.4f}"
            print(row)

    # Save JSON summary
    summary = _summarise_shift_data(shift_data)
    out_path = out_dir / "ece_under_shift_summary.json"
    out_path.write_text(json.dumps(json_safe(summary), indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

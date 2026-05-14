#!/usr/bin/env python3
"""CLI: run | evaluate | compare | benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_run(ns: argparse.Namespace) -> int:
    from uncertainty_lab.config import load_config
    from uncertainty_lab.pipeline.run import run_pipeline

    root = _repo_root()
    cfg = load_config(ns.config, repo_root=root)
    if ns.mode:
        cfg.setdefault("pipeline", {})["mode"] = ns.mode
    if ns.method:
        cfg.setdefault("uncertainty", {})["method"] = ns.method
    if ns.dataset_type:
        cfg.setdefault("dataset", {})["type"] = ns.dataset_type
    if ns.data_root:
        cfg.setdefault("dataset", {})["root"] = ns.data_root
    if ns.model_id:
        cfg.setdefault("model", {})["model_id"] = ns.model_id
    if ns.checkpoint:
        cfg.setdefault("model", {})["local_checkpoint"] = ns.checkpoint
        cfg.setdefault("model", {})["source"] = "huggingface"
    r = run_pipeline(cfg)
    print(json.dumps({"run_dir": r.get("run_dir"), "metrics_path": r.get("metrics_path"), "status": r.get("status")}, indent=2))
    return 0


def cmd_evaluate(ns: argparse.Namespace) -> int:
    p = Path(ns.run_dir) / "metrics.json"
    if not p.is_file():
        print(f"Missing {p}", file=sys.stderr)
        return 1
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    print(json.dumps(m, indent=2))
    return 0


def cmd_compare(ns: argparse.Namespace) -> int:
    rows = []
    for d in ns.run_dirs:
        p = Path(d) / "metrics.json"
        if not p.is_file():
            print(f"Skip (no metrics): {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        perf = m.get("predictive_performance", {})
        cal = m.get("calibration", {})
        sel = m.get("selective_prediction", {})
        rows.append(
            {
                "run_dir": str(d),
                "accuracy": perf.get("accuracy"),
                "roc_auc": perf.get("roc_auc"),
                "ece": cal.get("ece"),
                "brier": cal.get("brier"),
                "aurc": sel.get("aurc"),
            }
        )
    print(json.dumps({"comparison": rows}, indent=2))
    return 0


def cmd_benchmark(ns: argparse.Namespace) -> int:
    from uncertainty_lab.config import load_config
    from uncertainty_lab.pipeline.run import run_benchmark

    root = _repo_root()
    cfg = load_config(ns.config, repo_root=root)
    if ns.methods:
        cfg.setdefault("benchmark", {})["methods"] = [x.strip() for x in ns.methods.split(",") if x.strip()]
    r = run_benchmark(cfg)
    print(json.dumps(r, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="uncertainty-lab", description="Uncertainty Lab CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline from config")
    p_run.add_argument("--config", "-c", type=str, default=None, help="YAML config (defaults merged from configs/uncertainty_lab_default.yaml)")
    p_run.add_argument("--mode", choices=["evaluate", "train", "train_evaluate"], default=None)
    p_run.add_argument("--method", type=str, default=None, help="Uncertainty method")
    p_run.add_argument("--dataset-type", type=str, default=None, dest="dataset_type")
    p_run.add_argument("--data-root", type=str, default=None, dest="data_root")
    p_run.add_argument("--model-id", type=str, default=None, dest="model_id")
    p_run.add_argument("--checkpoint", type=str, default=None, help="Optional .pt to load after HF init")
    p_run.set_defaults(func=cmd_run)

    p_ev = sub.add_parser("evaluate", help="Print metrics.json from a run directory")
    p_ev.add_argument("run_dir", type=str)
    p_ev.set_defaults(func=cmd_evaluate)

    p_co = sub.add_parser("compare", help="Compare metrics across run directories")
    p_co.add_argument("run_dirs", nargs="+", type=str)
    p_co.set_defaults(func=cmd_compare)

    p_b = sub.add_parser("benchmark", help="Run multiple uncertainty methods (evaluate only)")
    p_b.add_argument("--config", "-c", type=str, default=None)
    p_b.add_argument("--methods", type=str, default=None, help="Comma-separated: confidence,mc_dropout,deep_ensemble")
    p_b.set_defaults(func=cmd_benchmark)

    ns = ap.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())

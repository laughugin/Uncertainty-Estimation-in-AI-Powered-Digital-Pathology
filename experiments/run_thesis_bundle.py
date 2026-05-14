#!/usr/bin/env python3
"""
Run full thesis bundle:
1) Core uncertainty pipeline (confidence + temperature_scaled + mc_dropout)
2) Shift/OOD pipeline per method
3) Combined summary JSON + method comparison
"""
from __future__ import annotations

import argparse
import json
import os
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
    p.add_argument("--fit_temperature_on_val", action="store_true")
    p.add_argument("--fit_deferral_on_val", action="store_true")
    p.add_argument("--run_id", type=str, default="")
    p.add_argument("--shift_severities", type=str, default="1,3,5")
    p.add_argument("--out", type=str, default="evaluation/thesis_bundle_summary.json")
    return p.parse_args()


def emit_progress(progress: float, stage: str, message: str) -> None:
    print(
        "__PROGRESS__" + json.dumps(
            {
                "progress": max(0.0, min(1.0, float(progress))),
                "stage": stage,
                "message": message,
            }
        ),
        flush=True,
    )


def run_cmd(cmd: list[str], *, stage: str, start_progress: float, end_progress: float, message: str) -> str:
    emit_progress(start_progress, stage, message)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: list[str] = []
    if proc.stdout is not None:
        for raw in proc.stdout:
            line = raw.rstrip()
            lines.append(raw)
            if line:
                print(line, flush=True)
    ret = proc.wait()
    log = "".join(lines)
    if ret != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n{log}")
    emit_progress(end_progress, stage, f"Completed: {message}")
    return log


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    emit_progress(0.02, "prepare", "Preparing thesis bundle evaluation.")

    pipe_out = REPO_ROOT / "evaluation" / "pipeline_summary.json"
    shift_out = REPO_ROOT / "evaluation" / f"shift_ood_{args.split}.json"

    cmd_pipe = [
        sys.executable,
        "experiments/run_evaluation_pipeline.py",
        "--split",
        args.split,
        "--max_samples",
        str(max(1, args.max_samples)),
        "--batch_size",
        str(max(1, args.batch_size)),
        "--mc_samples",
        str(max(2, args.mc_samples)),
        "--ensemble_size",
        str(max(1, args.ensemble_size)),
        "--out",
        str(pipe_out.relative_to(REPO_ROOT)),
    ]
    if args.fit_temperature_on_val:
        cmd_pipe.append("--fit_temperature_on_val")
    if args.fit_deferral_on_val:
        cmd_pipe.append("--fit_deferral_on_val")
    if args.include_deep_ensemble:
        cmd_pipe.append("--include_deep_ensemble")
    if (args.ensemble_run_ids or "").strip():
        cmd_pipe.extend(["--ensemble_run_ids", args.ensemble_run_ids.strip()])
    if args.run_id:
        cmd_pipe.extend(["--run_id", args.run_id])
    log_pipe = run_cmd(
        cmd_pipe,
        stage="core_pipeline",
        start_progress=0.08,
        end_progress=0.42,
        message="Running core uncertainty evaluation pipeline.",
    )

    pipe = load_json(pipe_out)
    shift_methods = ["confidence", "temperature_scaled", "mc_dropout"] + (["deep_ensemble"] if args.include_deep_ensemble else [])
    shift_by_method = {}
    n_shift = max(len(shift_methods), 1)
    for mi, method in enumerate(shift_methods):
        out_m = REPO_ROOT / "evaluation" / f"shift_ood_{method}_{args.split}.json"
        cmd_shift = [
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
            "--severities",
            args.shift_severities,
            "--out",
            str(out_m),
        ]
        if method == "deep_ensemble" and (args.ensemble_run_ids or "").strip():
            cmd_shift.extend(["--ensemble_run_ids", args.ensemble_run_ids.strip()])
        if args.run_id:
            cmd_shift.extend(["--run_id", args.run_id])
        lo = 0.45 + (0.45 * mi / n_shift)
        hi = 0.45 + (0.45 * (mi + 1) / n_shift)
        run_cmd(
            cmd_shift,
            stage=f"shift_{method}",
            start_progress=lo,
            end_progress=hi,
            message=f"Running shift/OOD evaluation for {method}.",
        )
        shift_by_method[method] = load_json(out_m)

    # Keep legacy single-shift output for compatibility (default to mc_dropout if available).
    shift = shift_by_method.get("mc_dropout") or shift_by_method.get("confidence") or {}
    with open(shift_out, "w", encoding="utf-8") as f:
        json.dump(shift, f, indent=2)

    def _mean(field: str, group: str):
        out = {}
        for m, payload in shift_by_method.items():
            g = (payload.get("grouped_summary", {}) or {}).get(group, {}) or {}
            out[m] = g.get(field)
        return out

    near_ood = _mean("mean_ood_auroc", "near_ood")
    far_ood = _mean("mean_ood_auroc", "far_ood")

    def _best(d: dict):
        vals = [(k, v) for k, v in d.items() if v is not None]
        if not vals:
            return None
        vals.sort(key=lambda x: x[1], reverse=True)
        return {"method": vals[0][0], "value": float(vals[0][1])}

    shift_method_comparison = {
        "near_ood_mean_auroc_by_method": near_ood,
        "far_ood_mean_auroc_by_method": far_ood,
        "best_near_ood_by_auroc": _best(near_ood),
        "best_far_ood_by_auroc": _best(far_ood),
        "references": [
            "[8] Linmans et al. (2023) DOI: 10.1016/j.media.2022.102655",
            "[11] Thagaard et al. (2020) DOI: 10.1007/978-3-030-59710-8_80",
        ],
        "note": "Near-vs-far OOD method ranking is reported explicitly, as highlighted in digital pathology uncertainty literature.",
    }

    summary = {
        "config": {
            "split": args.split,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "mc_samples": args.mc_samples,
            "fit_temperature_on_val": args.fit_temperature_on_val,
            "fit_deferral_on_val": args.fit_deferral_on_val,
            "include_deep_ensemble": args.include_deep_ensemble,
            "ensemble_size": args.ensemble_size,
            "ensemble_run_ids": [x.strip() for x in (args.ensemble_run_ids or "").split(",") if x.strip()],
            "shift_severities": args.shift_severities,
            "run_id": args.run_id or None,
        },
        "outputs": {
            "pipeline_summary": str(pipe_out.relative_to(REPO_ROOT)),
            "shift_summary": str(shift_out.relative_to(REPO_ROOT)),
        },
        "pipeline": pipe.get("results", {}),
        "shift_ood": shift.get("results", {}),
        "shift_ood_by_method": {k: v.get("results", {}) for k, v in shift_by_method.items()},
        "shift_ood_grouped_by_method": {k: v.get("grouped_summary", {}) for k, v in shift_by_method.items()},
        "shift_method_comparison": shift_method_comparison,
        "literature_alignment": {
            "thresholding_reference": "[2] Dolezal et al. (2022) DOI: 10.1200/JCO.2022.40.16_suppl.8549",
            "near_far_ood_references": [
                "[8] Linmans et al. (2023) DOI: 10.1016/j.media.2022.102655",
                "[11] Thagaard et al. (2020) DOI: 10.1007/978-3-030-59710-8_80",
            ],
        },
    }

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emit_progress(0.94, "summarize", "Building combined thesis summary.")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    emit_progress(1.0, "done", "Thesis bundle finished.")
    print(f"Saved thesis bundle: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

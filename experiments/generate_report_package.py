#!/usr/bin/env python3
"""Generate a thesis-grade report package from a saved bundle JSON."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

from uncertainty_lab.metrics.plots import (
    plot_benchmark_summary,
    plot_confusion_matrix,
    plot_error_detection_curves,
    plot_histogram_from_values,
    plot_predictive_performance_ranking,
    plot_predictive_performance_thresholded,
    plot_reliability,
    plot_reliability_overlay,
    plot_risk_coverage,
    plot_risk_coverage_overlay,
    plot_shift_condition_bars,
    plot_temperature_scaling,
    plot_uncertainty_histograms,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a report package from thesis bundle JSON")
    p.add_argument("--bundle", required=True, help="Path to thesis bundle summary JSON")
    p.add_argument("--out-dir", required=True, help="Destination directory for report package")
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(REPO_ROOT))


def _load_reference_catalog() -> dict:
    path = REPO_ROOT / "references" / "reference_catalog.json"
    if not path.exists():
        return {"ordered_literature": [], "foundation_references": [], "method_reference_map": {}}
    return _load_json(path)


def _metrics_rows(bundle: dict) -> list[dict]:
    rows = []
    for method, payload in sorted((bundle.get("pipeline") or {}).items()):
        perf = payload.get("predictive_performance", {}) or {}
        cal = payload.get("calibration", {}) or {}
        sel = payload.get("selective_prediction", {}) or {}
        rows.append(
            {
                "method": method,
                "accuracy": perf.get("accuracy"),
                "roc_auc": perf.get("roc_auc"),
                "pr_auc": perf.get("pr_auc"),
                "sensitivity": perf.get("sensitivity"),
                "specificity": perf.get("specificity"),
                "ece": cal.get("ece"),
                "brier": cal.get("brier"),
                "aurc": sel.get("aurc"),
            }
        )
    return rows


def _method_explanations() -> dict[str, dict[str, object]]:
    catalog = _load_reference_catalog()
    ordered = {item.get("number"): item for item in catalog.get("ordered_literature", [])}
    foundation = {item.get("key"): item for item in catalog.get("foundation_references", [])}
    method_map = catalog.get("method_reference_map", {}) or {}

    def refs_for(method: str, foundation_keys: list[str] | None = None) -> tuple[str, list[dict]]:
        entries: list[dict] = []
        for number in method_map.get(method, []):
            ref = ordered.get(number)
            if ref:
                entries.append(
                    {
                        "label": f"[{ref.get('number')}] {ref.get('short')}",
                        "citation": ref.get("citation"),
                        "summary": ref.get("summary"),
                    }
                )
        for key in foundation_keys or []:
            ref = foundation.get(key)
            if ref:
                entries.append(
                    {
                        "label": ref.get("label"),
                        "citation": ref.get("citation"),
                        "summary": "Foundational method paper used directly for this implementation family.",
                    }
                )
        text = "; ".join(entry["label"] for entry in entries) if entries else ""
        return text, entries

    confidence_text, confidence_entries = refs_for("confidence")
    temp_text, temp_entries = refs_for("temperature_scaling", foundation_keys=["guo2017calibration"])
    mc_text, mc_entries = refs_for("mc_dropout", foundation_keys=["gal2016dropout"])
    ensemble_text, ensemble_entries = refs_for("deep_ensemble", foundation_keys=["lakshminarayanan2017simple"])
    return {
        "confidence": {
            "what": "Maximum Softmax Probability (MSP) baseline.",
            "why": "Provides a crude confidence reference point and shows how overconfident a plain softmax score can be.",
            "interpretation": "Useful as a weak baseline only; strong performance here does not guarantee trustworthy uncertainty.",
            "references": confidence_text,
            "reference_entries": confidence_entries,
        },
        "temperature_scaled": {
            "what": "Maximum Softmax Probability baseline after post-hoc temperature scaling.",
            "why": "Provides the strongest cheap calibrated baseline before comparing against stochastic or ensemble-based uncertainty methods.",
            "interpretation": "Improvement here mainly reflects better in-distribution calibration; it should not be assumed to fix shift/OOD behavior by itself.",
            "references": temp_text,
            "reference_entries": temp_entries,
        },
        "mc_dropout": {
            "what": "Stochastic forward passes with dropout enabled at test time.",
            "why": "Approximates Bayesian uncertainty and gives a practical epistemic-uncertainty baseline for limited-data pathology tasks.",
            "interpretation": "Better uncertainty quality should appear in calibration, error detection, and deferral behavior, not only in ROC/PR metrics.",
            "references": mc_text,
            "reference_entries": mc_entries,
        },
        "deep_ensemble": {
            "what": "Average prediction over multiple independently trained models.",
            "why": "Acts as a strong empirical uncertainty baseline and is especially relevant for robustness under shift/OOD.",
            "interpretation": "Particularly informative in shift/OOD and calibration comparisons, often stronger than single-model confidence.",
            "references": ensemble_text,
            "reference_entries": ensemble_entries,
        },
    }


def main() -> int:
    args = parse_args()
    bundle_path = Path(args.bundle).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    fig_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"
    shift_dir = out_dir / "shift"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    shift_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_json(bundle_path)
    outputs = dict(bundle.get("outputs") or {})
    methods = sorted(list((bundle.get("pipeline") or {}).keys()))

    copied_metrics: dict[str, str] = {}
    metrics_by_method: dict[str, dict] = {}
    rel_bins_by_method: dict[str, list[dict]] = {}
    rc_by_method: dict[str, list[dict]] = {}

    for method in methods:
        rel_path = outputs.get("detailed_metrics_by_method", {}).get(method)
        if not rel_path:
            continue
        src = (REPO_ROOT / rel_path).resolve()
        dst = metrics_dir / f"{method}.json"
        copied_metrics[method] = _copy(src, dst)
        metrics = _load_json(src)
        metrics_by_method[method] = metrics
        rel_bins_by_method[method] = metrics.get("calibration", {}).get("reliability_bins", [])
        rc_by_method[method] = metrics.get("selective_prediction", {}).get("risk_coverage_curve", [])

    copied_shift: dict[str, str] = {}
    for method, rel_path in sorted((outputs.get("shift_by_method") or {}).items()):
        src = (REPO_ROOT / rel_path).resolve()
        dst = shift_dir / f"{method}.json"
        copied_shift[method] = _copy(src, dst)

    figures: dict[str, str] = {}
    rows = _metrics_rows(bundle)
    if rows:
        p = fig_dir / "benchmark_summary_metrics.png"
        plot_benchmark_summary(rows, p, title="Benchmark metrics across uncertainty methods")
        if p.exists():
            figures["benchmark_summary_metrics"] = str(p.relative_to(REPO_ROOT))

        p = fig_dir / "predictive_performance_thresholded.png"
        plot_predictive_performance_thresholded(rows, p, title="Predictive performance: thresholded metrics")
        if p.exists():
            figures["predictive_performance_thresholded"] = str(p.relative_to(REPO_ROOT))

        p = fig_dir / "predictive_performance_ranking.png"
        plot_predictive_performance_ranking(rows, p, title="Predictive performance: ranking metrics")
        if p.exists():
            figures["predictive_performance_ranking"] = str(p.relative_to(REPO_ROOT))

    if rel_bins_by_method:
        p = fig_dir / "benchmark_reliability_comparison.png"
        plot_reliability_overlay(rel_bins_by_method, p, title="Reliability diagrams by method")
        if p.exists():
            figures["benchmark_reliability_comparison"] = str(p.relative_to(REPO_ROOT))

    if rc_by_method:
        p = fig_dir / "benchmark_risk_coverage_comparison.png"
        plot_risk_coverage_overlay(rc_by_method, p, title="Risk-coverage curves by method")
        if p.exists():
            figures["benchmark_risk_coverage_comparison"] = str(p.relative_to(REPO_ROOT))

    shift_by_method = bundle.get("shift_ood_by_method", {}) or {}
    if shift_by_method:
        for metric_key, name in [
            ("ood_detection_auroc", "shift_detail_ood_auroc"),
            ("ood_detection_auprc", "shift_detail_ood_auprc"),
            ("accuracy", "shift_detail_accuracy"),
            ("ece", "shift_detail_ece"),
        ]:
            p = fig_dir / f"{name}.png"
            plot_shift_condition_bars(shift_by_method, metric_key, p, title=f"Per-condition shift detail: {metric_key}")
            if p.exists():
                figures[name] = str(p.relative_to(REPO_ROOT))

    explanations = _method_explanations()
    method_manifests: dict[str, dict] = {}
    for method, metrics in metrics_by_method.items():
        m_figs: dict[str, str] = {}
        perf = metrics.get("predictive_performance", {}) or {}
        cal = metrics.get("calibration", {}) or {}
        uq = metrics.get("uncertainty_quality", {}) or {}
        sel = metrics.get("selective_prediction", {}) or {}
        rep = metrics.get("calibration_report", {}) or {}
        plot_data = metrics.get("plot_data", {}) or {}

        rel_p = fig_dir / f"{method}_reliability.png"
        plot_reliability(cal.get("reliability_bins", []), rel_p, title=f"Reliability ({method})")
        if rel_p.exists():
            m_figs["reliability"] = str(rel_p.relative_to(REPO_ROOT))

        rc_p = fig_dir / f"{method}_risk_coverage.png"
        plot_risk_coverage(sel.get("risk_coverage_curve", []), rc_p, title=f"Risk-coverage ({method})")
        if rc_p.exists():
            m_figs["risk_coverage"] = str(rc_p.relative_to(REPO_ROOT))

        cm_p = fig_dir / f"{method}_confusion_matrix.png"
        plot_confusion_matrix(perf.get("confusion_matrix", {}), cm_p, title=f"Confusion matrix ({method})")
        if cm_p.exists():
            m_figs["confusion_matrix"] = str(cm_p.relative_to(REPO_ROOT))

        roc_p = fig_dir / f"{method}_error_roc.png"
        plot_error_detection_curves(
            (uq.get("error_detection_entropy", {}) or {}).get("roc_curve", []),
            (uq.get("error_detection_one_minus_msp", {}) or {}).get("roc_curve", []),
            roc_p,
            curve_type="roc",
            title=f"Error-detection ROC ({method})",
        )
        if roc_p.exists():
            m_figs["error_roc"] = str(roc_p.relative_to(REPO_ROOT))

        pr_p = fig_dir / f"{method}_error_pr.png"
        plot_error_detection_curves(
            (uq.get("error_detection_entropy", {}) or {}).get("pr_curve", []),
            (uq.get("error_detection_one_minus_msp", {}) or {}).get("pr_curve", []),
            pr_p,
            curve_type="pr",
            title=f"Error-detection PR ({method})",
        )
        if pr_p.exists():
            m_figs["error_pr"] = str(pr_p.relative_to(REPO_ROOT))

        correct = np.asarray(plot_data.get("correct", []), dtype=np.int64)
        uncertainty = np.asarray(plot_data.get("uncertainty_one_minus_msp", []), dtype=np.float64)
        entropy = np.asarray(plot_data.get("entropy", []), dtype=np.float64)
        if uncertainty.size and correct.size:
            uh_p = fig_dir / f"{method}_uncertainty_histogram.png"
            plot_uncertainty_histograms(uncertainty, correct, uh_p, title="Uncertainty (1 - MSP)")
            if uh_p.exists():
                m_figs["uncertainty_histogram"] = str(uh_p.relative_to(REPO_ROOT))
        if entropy.size and correct.size:
            eh_p = fig_dir / f"{method}_entropy_histogram.png"
            plot_histogram_from_values(entropy, correct, eh_p, xlabel="Entropy", title=f"Entropy distribution ({method})")
            if eh_p.exists():
                m_figs["entropy_histogram"] = str(eh_p.relative_to(REPO_ROOT))

        ts_p = fig_dir / f"{method}_temperature_scaling.png"
        plot_temperature_scaling(rep, ts_p, title=f"Calibration before/after ({method})")
        if ts_p.exists():
            m_figs["temperature_scaling"] = str(ts_p.relative_to(REPO_ROOT))

        method_manifests[method] = {
            "metrics_path": copied_metrics.get(method),
            "figures": m_figs,
            "explanation": explanations.get(method, {}),
        }

    manifest = {
        "bundle_path": str(bundle_path.relative_to(REPO_ROOT)),
        "report_dir": str(out_dir.relative_to(REPO_ROOT)),
        "config": bundle.get("config", {}),
        "methods": methods,
        "summary_figures": figures,
        "method_reports": method_manifests,
        "copied_metrics": copied_metrics,
        "copied_shift": copied_shift,
        "notes": {
            "purpose": "Shared program-generated evaluation report package for web display and later thesis reuse.",
            "comment": "LaTeX should consume these saved program outputs rather than rebuild figures manually.",
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"manifest": str(manifest_path.relative_to(REPO_ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Matplotlib plots for uncertainty and calibration."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DPI = 220
METHOD_COLORS = {
    "confidence": "#4C78A8",
    "temperature_scaled": "#F58518",
    "mc_dropout": "#54A24B",
    "deep_ensemble": "#E45756",
}


def _finalize(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(rel_bins: list[dict], out_path: Path, title: str = "Reliability diagram") -> None:
    xs, ys = [], []
    for b in rel_bins:
        if b["count"] > 0 and b["conf"] is not None and b["acc"] is not None:
            xs.append(float(b["conf"]))
            ys.append(float(b["acc"]))
    fig = plt.figure(figsize=(5.2, 4.2))
    plt.plot([0, 1], [0, 1], "--", linewidth=1, color="gray")
    if xs:
        plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(alpha=0.2)
    _finalize(fig, out_path)


save_reliability_plot = plot_reliability


def plot_risk_coverage(curve: list[dict], out_path: Path, title: str = "Risk–coverage") -> None:
    if not curve:
        return
    cov = [p["coverage"] for p in curve]
    risk = [p["risk"] for p in curve]
    fig = plt.figure(figsize=(5.2, 4.2))
    plt.plot(cov, risk, linewidth=1.5)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (error rate among kept)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, max(0.05, max(risk) * 1.05))
    plt.grid(alpha=0.2)
    _finalize(fig, out_path)


def plot_uncertainty_histograms(
    uncertainty: np.ndarray,
    correct_mask: np.ndarray,
    out_path: Path,
    title: str = "Uncertainty (1 − max prob)",
) -> None:
    u = uncertainty.astype(np.float64)
    fig = plt.figure(figsize=(5.2, 4.2))
    c = correct_mask.astype(bool)
    if c.any():
        plt.hist(u[c], bins=30, alpha=0.6, label="Correct", density=True)
    if (~c).any():
        plt.hist(u[~c], bins=30, alpha=0.6, label="Incorrect", density=True)
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.legend()
    plt.title("Uncertainty distribution")
    plt.grid(alpha=0.15)
    _finalize(fig, out_path)


def plot_histogram_from_values(
    values: np.ndarray,
    correct_mask: np.ndarray,
    out_path: Path,
    *,
    xlabel: str,
    title: str,
    bins: int = 30,
) -> None:
    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(correct_mask).astype(bool)
    if x.size == 0:
        return
    fig = plt.figure(figsize=(5.2, 4.2))
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi <= lo:
        hi = lo + 1e-6
    hist_bins = np.linspace(lo, hi, max(5, bins))
    if c.any():
        plt.hist(x[c], bins=hist_bins, alpha=0.6, label="Correct", density=True)
    if (~c).any():
        plt.hist(x[~c], bins=hist_bins, alpha=0.6, label="Incorrect", density=True)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.15)
    _finalize(fig, out_path)


def plot_reliability_overlay(
    rel_bins_by_method: dict[str, list[dict]],
    out_path: Path,
    title: str = "Reliability comparison",
) -> None:
    fig = plt.figure(figsize=(5.8, 4.6))
    plt.plot([0, 1], [0, 1], "--", linewidth=1, color="gray", label="Ideal")
    for method, rel_bins in rel_bins_by_method.items():
        xs, ys = [], []
        for b in rel_bins:
            if b["count"] > 0 and b["conf"] is not None and b["acc"] is not None:
                xs.append(float(b["conf"]))
                ys.append(float(b["acc"]))
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    _finalize(fig, out_path)


def plot_risk_coverage_overlay(
    curves_by_method: dict[str, list[dict]],
    out_path: Path,
    title: str = "Risk-coverage comparison",
) -> None:
    fig = plt.figure(figsize=(5.8, 4.6))
    ymax = 0.05
    has_any = False
    for method, curve in curves_by_method.items():
        if not curve:
            continue
        cov = [p["coverage"] for p in curve]
        risk = [p["risk"] for p in curve]
        ymax = max(ymax, max(risk) * 1.05)
        plt.plot(cov, risk, linewidth=1.5, label=method)
        has_any = True
    if not has_any:
        return
    plt.xlabel("Coverage")
    plt.ylabel("Risk (error rate among kept)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, ymax)
    plt.legend()
    plt.grid(alpha=0.2)
    _finalize(fig, out_path)


def plot_benchmark_summary(
    rows: list[dict],
    out_path: Path,
    title: str = "Uncertainty benchmark summary",
) -> None:
    if not rows:
        return
    metrics = [
        ("accuracy", "Accuracy", False),
        ("roc_auc", "ROC AUC", False),
        ("pr_auc", "PR AUC", False),
        ("sensitivity", "Sensitivity", False),
        ("specificity", "Specificity", False),
        ("ece", "ECE", True),
        ("brier", "Brier", True),
        ("aurc", "AURC", True),
    ]
    methods = [str(r.get("method", "method")) for r in rows]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    axes = axes.flatten()
    for ax, (key, label, lower_better) in zip(axes, metrics, strict=False):
        vals = [float(r.get(key)) if r.get(key) is not None else np.nan for r in rows]
        bars = ax.bar(x, vals, color="#4C78A8")
        ax.set_xticks(x, methods, rotation=15)
        ax.set_title(f"{label} ({'lower' if lower_better else 'higher'} is better)")
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            vmax = max(finite_vals)
            ax.set_ylim(0, vmax * 1.18 if vmax > 0 else 1.0)
        for bar, val in zip(bars, vals, strict=False):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(title)
    _finalize(fig, out_path)


def _plot_metric_dotplot(
    rows: list[dict],
    metrics: list[tuple[str, str]],
    out_path: Path,
    *,
    title: str,
    subtitle: str | None = None,
    clamp_unit_interval: bool = False,
) -> None:
    if not rows or not metrics:
        return
    methods = [str(r.get("method", "method")) for r in rows]
    n_methods = len(methods)
    y_base = np.arange(len(metrics), dtype=np.float64)
    offsets = np.linspace(-0.24, 0.24, n_methods) if n_methods > 1 else np.array([0.0], dtype=np.float64)

    all_vals = []
    for row in rows:
        for key, _ in metrics:
            val = row.get(key)
            if val is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            if np.isfinite(fv):
                all_vals.append(fv)
    if not all_vals:
        return

    lo = min(all_vals)
    hi = max(all_vals)
    span = hi - lo
    pad = max(0.02 if clamp_unit_interval else 0.03, span * 0.12)
    lo = lo - pad
    hi = hi + pad
    if clamp_unit_interval:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if hi - lo < 0.08:
            mid = 0.5 * (hi + lo)
            lo = max(0.0, mid - 0.04)
            hi = min(1.0, mid + 0.04)

    fig, ax = plt.subplots(figsize=(8.6, 4.8 if len(metrics) <= 3 else 5.6))
    for yi in y_base:
        ax.axhline(yi, color="#E6E6E6", linewidth=0.9, zorder=0)

    for mi, row in enumerate(rows):
        method = str(row.get("method", "method"))
        color = METHOD_COLORS.get(method, "#4C78A8")
        xs = []
        ys = []
        for gi, (key, _) in enumerate(metrics):
            val = row.get(key)
            if val is None:
                continue
            fv = float(val)
            if not np.isfinite(fv):
                continue
            y = y_base[gi] + offsets[mi]
            xs.append(fv)
            ys.append(y)
            ax.scatter(fv, y, s=48, color=color, edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(
                min(hi - 0.001, fv + 0.004),
                y,
                f"{fv:.3f}",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
            )
        if xs:
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.9, label=method.replace("_", " "))

    ax.set_yticks(y_base, [label for _, label in metrics])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Score")
    ax.set_title(title)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555", va="bottom")
    ax.grid(axis="x", alpha=0.18)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", frameon=False)
    _finalize(fig, out_path)


def plot_predictive_performance_thresholded(
    rows: list[dict],
    out_path: Path,
    title: str = "Predictive performance: thresholded classification metrics",
) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1", "F1"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="These metrics depend on the chosen decision threshold.",
        clamp_unit_interval=True,
    )


def plot_predictive_performance_ranking(
    rows: list[dict],
    out_path: Path,
    title: str = "Predictive performance: ranking metrics",
) -> None:
    metrics = [
        ("roc_auc", "ROC AUC"),
        ("pr_auc", "PR AUC"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="These metrics evaluate ranking quality across all thresholds.",
        clamp_unit_interval=True,
    )


def plot_calibration_metrics_summary(
    rows: list[dict],
    out_path: Path,
    title: str = "Calibration comparison",
) -> None:
    metrics = [
        ("ece", "ECE"),
        ("nll", "NLL"),
        ("brier", "Brier"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="Lower values are better; this summarizes numerical calibration rather than class ranking.",
        clamp_unit_interval=False,
    )


def plot_uncertainty_primary_summary(
    rows: list[dict],
    out_path: Path,
    title: str = "Primary uncertainty score comparison",
) -> None:
    metrics = [
        ("primary_auroc", "Primary AUROC"),
        ("primary_auprc", "Primary AUPRC"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="Each method is evaluated by its own designated uncertainty score.",
        clamp_unit_interval=True,
    )


def plot_uncertainty_common_summary(
    rows: list[dict],
    out_path: Path,
    title: str = "Common uncertainty score comparison",
) -> None:
    metrics = [
        ("entropy_auroc", "Entropy AUROC"),
        ("entropy_auprc", "Entropy AUPRC"),
        ("msp_auroc", "1-MSP AUROC"),
        ("msp_auprc", "1-MSP AUPRC"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="These shared scores allow a like-for-like comparison across all methods.",
        clamp_unit_interval=True,
    )


def plot_high_confidence_accuracy_coverage(
    rows: list[dict],
    out_path: Path,
    title: str = "High-confidence operating point",
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    has_any = False
    for row in rows:
        cov = row.get("coverage")
        acc = row.get("accuracy")
        if cov is None or acc is None:
            continue
        x = float(cov)
        y = float(acc)
        method = str(row.get("method", "method"))
        color = METHOD_COLORS.get(method, "#4C78A8")
        ax.scatter(x, y, s=70, color=color, edgecolor="white", linewidth=0.8, zorder=3, label=method.replace("_", " "))
        ax.text(min(0.995, x + 0.012), min(0.995, y + 0.004), method.replace("_", " "), color=color, fontsize=8)
        has_any = True
    if not has_any:
        plt.close(fig)
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy in retained high-confidence cohort")
    ax.set_title(title)
    ax.grid(alpha=0.18)
    _finalize(fig, out_path)


def plot_pathology_proxy_summary(
    rows: list[dict],
    out_path: Path,
    title: str = "Pathology proxy reporting comparison",
) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC AUC"),
        ("pr_auc", "PR AUC"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="Pseudo-slide aggregation metrics for the pathology-style reporting proxy.",
        clamp_unit_interval=True,
    )


def plot_shift_detection_summary(
    rows: list[dict],
    out_path: Path,
    title: str = "Shift / OOD detection summary",
) -> None:
    metrics = [
        ("near_ood_auroc", "Near-OOD AUROC"),
        ("near_ood_auprc", "Near-OOD AUPRC"),
        ("far_ood_auroc", "Far-OOD AUROC"),
        ("far_ood_auprc", "Far-OOD AUPRC"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="Higher values indicate stronger discrimination between in-distribution and shifted samples.",
        clamp_unit_interval=True,
    )


def plot_shift_robustness_summary(
    rows: list[dict],
    out_path: Path,
    title: str = "Shift robustness summary",
) -> None:
    metrics = [
        ("near_accuracy", "Near-OOD accuracy"),
        ("far_accuracy", "Far-OOD accuracy"),
        ("near_ece", "Near-OOD ECE"),
        ("far_ece", "Far-OOD ECE"),
    ]
    _plot_metric_dotplot(
        rows,
        metrics,
        out_path,
        title=title,
        subtitle="Accuracy should stay high while calibration error should stay low under stronger shift.",
        clamp_unit_interval=True,
    )


def plot_confusion_matrix(cm: dict, out_path: Path, title: str = "Confusion matrix") -> None:
    vals = np.array(
        [
            [float(cm.get("tn", 0)), float(cm.get("fp", 0))],
            [float(cm.get("fn", 0)), float(cm.get("tp", 0))],
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(vals, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(vals[i, j])}", ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finalize(fig, out_path)


def plot_error_detection_curves(
    entropy_curve: list[dict],
    msp_curve: list[dict],
    out_path: Path,
    *,
    curve_type: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
        if entropy_curve:
            ax.plot([float(p["fpr"]) for p in entropy_curve], [float(p["tpr"]) for p in entropy_curve], linewidth=1.8, label="Entropy")
        if msp_curve:
            ax.plot([float(p["fpr"]) for p in msp_curve], [float(p["tpr"]) for p in msp_curve], linewidth=1.8, label="1 - MSP")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        if entropy_curve:
            ax.plot([float(p["recall"]) for p in entropy_curve], [float(p["precision"]) for p in entropy_curve], linewidth=1.8, label="Entropy")
        if msp_curve:
            ax.plot([float(p["recall"]) for p in msp_curve], [float(p["precision"]) for p in msp_curve], linewidth=1.8, label="1 - MSP")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    _finalize(fig, out_path)


def plot_temperature_scaling(report: dict, out_path: Path, title: str = "Calibration before/after temperature scaling") -> None:
    uncal = report.get("uncalibrated", {}) or {}
    temp = report.get("temperature_scaling", {}) or {}
    cal = temp.get("calibrated", {}) or {}
    labels = ["ECE", "NLL", "Brier"]
    before = [float(uncal.get("ece", 0.0)), float(uncal.get("nll", 0.0)), float(uncal.get("brier", 0.0))]
    after = [float(cal.get("ece", 0.0)), float(cal.get("nll", 0.0)), float(cal.get("brier", 0.0))]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.bar(x - width / 2, before, width=width, label="Before")
    if cal:
        ax.bar(x + width / 2, after, width=width, label="After")
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _finalize(fig, out_path)


def plot_shift_condition_bars(
    shift_by_method: dict[str, dict],
    metric_key: str,
    out_path: Path,
    title: str,
) -> None:
    methods = sorted(list(shift_by_method.keys()))
    condition_set: set[str] = set()
    for payload in shift_by_method.values():
        condition_set.update(k for k in payload.keys() if k != "id_s0")
    conditions = sorted(list(condition_set))
    if not methods or not conditions:
        return
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = np.arange(len(conditions))
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        vals = [float((shift_by_method.get(method, {}).get(cond, {}) or {}).get(metric_key, np.nan)) for cond in conditions]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, label=method)
    ax.set_xticks(x, conditions, rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    _finalize(fig, out_path)


# Aleatoric / Epistemic decomposition

def plot_uncertainty_decomposition(
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    error: np.ndarray,
    out_path: Path,
    title: str = "Aleatoric vs. Epistemic Uncertainty",
) -> None:
    """2-panel figure: histograms of epistemic and aleatoric uncertainty,
    split by correct / incorrect predictions."""
    ep = np.asarray(epistemic, dtype=np.float64)
    al = np.asarray(aleatoric, dtype=np.float64)
    err = np.asarray(error, dtype=bool)
    correct = ~err

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    def _hist_panel(ax, vals, label, color_correct="#4C78A8", color_wrong="#E45756"):
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if hi <= lo:
            hi = lo + 1e-9
        bins = np.linspace(lo, hi, 35)
        if correct.any():
            ax.hist(vals[correct], bins=bins, alpha=0.65, label="Correct", density=True, color=color_correct)
        if err.any():
            ax.hist(vals[err], bins=bins, alpha=0.65, label="Incorrect", density=True, color=color_wrong)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)
        mean_c = vals[correct].mean() if correct.any() else float("nan")
        mean_w = vals[err].mean() if err.any() else float("nan")
        ax.set_title(f"{label}\ncorrect μ={mean_c:.4f}  incorrect μ={mean_w:.4f}", fontsize=9)

    _hist_panel(axes[0], ep, "Epistemic (Mutual Information)")
    _hist_panel(axes[1], al, "Aleatoric (PE − MI)")
    fig.suptitle(title, fontsize=11)
    _finalize(fig, out_path)


def plot_uncertainty_decomposition_scatter(
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    error: np.ndarray,
    out_path: Path,
    title: str = "Epistemic vs. Aleatoric per Sample",
) -> None:
    """Scatter plot: each point is one sample; axes are epistemic and aleatoric,
    colour-coded by correct / incorrect."""
    ep = np.asarray(epistemic, dtype=np.float64)
    al = np.asarray(aleatoric, dtype=np.float64)
    err = np.asarray(error, dtype=bool)
    correct = ~err

    # Subsample for large datasets
    n = len(ep)
    idx = np.arange(n)
    if n > 2000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 2000, replace=False)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    c_mask = correct[idx]
    e_mask = err[idx]
    if c_mask.any():
        ax.scatter(ep[idx][c_mask], al[idx][c_mask], s=12, alpha=0.4,
                   label="Correct", color="#4C78A8")
    if e_mask.any():
        ax.scatter(ep[idx][e_mask], al[idx][e_mask], s=18, alpha=0.6,
                   label="Incorrect", color="#E45756", marker="x")
    ax.set_xlabel("Epistemic uncertainty (MI)")
    ax.set_ylabel("Aleatoric uncertainty (PE − MI)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.15)
    _finalize(fig, out_path)


# ECE under corruption severity

def plot_ece_under_shift(
    shift_data_by_method: dict[str, dict],
    out_path: Path,
    title: str = "Calibration (ECE) under distribution shift",
    metric_key: str = "ece",
) -> None:
    """Line plot: x = severity (0=clean, 1, 3, 5), y = ECE, one line per method.

    Args:
        shift_data_by_method: dict method → dict condition_key → {ece, accuracy, ...}
            e.g. {"confidence": {"id_s0": {"ece": 0.07, ...}, "blur_s1": {...}, ...}}
    """
    SHIFT_TYPES = ["blur", "noise", "jpeg", "color"]
    SEVERITIES = [0, 1, 3, 5]   # 0 = in-distribution
    SEV_LABELS = {0: "Clean", 1: "Sev 1\n(mild)", 3: "Sev 3\n(med)", 5: "Sev 5\n(severe)"}

    n_shifts = len(SHIFT_TYPES)
    fig, axes = plt.subplots(1, n_shifts, figsize=(13, 4.2), sharey=True)
    if n_shifts == 1:
        axes = [axes]

    methods = sorted(shift_data_by_method.keys())
    for si, shift in enumerate(SHIFT_TYPES):
        ax = axes[si]
        for method in methods:
            data = shift_data_by_method.get(method, {})
            color = METHOD_COLORS.get(method)
            ys = []
            xs_valid = []
            for sev in SEVERITIES:
                key = "id_s0" if sev == 0 else f"{shift}_s{sev}"
                cond = data.get(key, {})
                val = cond.get(metric_key) if cond else None
                if val is not None:
                    ys.append(float(val))
                    xs_valid.append(sev)
            if ys:
                ax.plot(
                    range(len(xs_valid)), ys,
                    marker="o", linewidth=1.8, markersize=6,
                    label=method, color=color,
                )
        ax.set_title(shift.capitalize(), fontsize=10)
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels([SEV_LABELS[s] for s in SEVERITIES], fontsize=8)
        ax.grid(alpha=0.18)
        if si == 0:
            ax.set_ylabel(metric_key.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=11)
    _finalize(fig, out_path)


def plot_accuracy_under_shift(
    shift_data_by_method: dict[str, dict],
    out_path: Path,
    title: str = "Accuracy under distribution shift",
) -> None:
    """Same layout as ECE-under-shift but for accuracy degradation."""
    plot_ece_under_shift(
        shift_data_by_method, out_path, title=title, metric_key="accuracy"
    )

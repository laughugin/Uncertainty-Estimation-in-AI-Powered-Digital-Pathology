"""
Local web UI for the Uncertainty Estimation in Digital Pathology project.
Run: python web/app.py   or   flask --app web.app run
"""
from pathlib import Path
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import yaml
import numpy as np

# Project root (parent of web/)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, render_template, send_file, jsonify, request, Response
from flask_sock import Sock
from experiments.ensemble_utils import list_deep_ensemble_candidates, load_run_metadata, normalize_run_id
from uncertainty_lab.metrics.plots import (
    plot_calibration_metrics_summary,
    plot_high_confidence_accuracy_coverage,
    plot_pathology_proxy_summary,
    plot_predictive_performance_ranking,
    plot_predictive_performance_thresholded,
    plot_reliability_overlay,
    plot_risk_coverage_overlay,
    plot_shift_condition_bars,
    plot_shift_detection_summary,
    plot_shift_robustness_summary,
    plot_uncertainty_common_summary,
    plot_uncertainty_primary_summary,
)
from web.evaluation_methods import build_evaluation_method_map, build_evaluation_methods

app = Flask(__name__, static_folder="static", template_folder="templates")
sock = Sock(app)
app.config["REPO_ROOT"] = REPO_ROOT
app.config["RUN_TIMEOUT"] = 600  # seconds for run tasks

MODEL_CATALOG = [
    {"id": "google/vit-base-patch16-224", "name": "ViT-Base (Patch16-224)", "input_size": [224, 224]},
    {"id": "microsoft/beit-base-patch16-224", "name": "BEiT-Base (Patch16-224)", "input_size": [224, 224]},
    {"id": "facebook/deit-base-patch16-224", "name": "DeiT-Base (Patch16-224)", "input_size": [224, 224]},
]

# Lazy-loaded datasets per split (avoid loading all at startup)
_pcam_cache = {}


def load_reference_catalog():
    """Load the canonical ordered reference catalog for web pages and reports."""
    path = REPO_ROOT / "references" / "reference_catalog.json"
    if not path.exists():
        return {"ordered_literature": [], "foundation_references": [], "method_reference_map": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pcam(split: str):
    """Get or create PCAM dataset for split. Raises if data missing."""
    global _pcam_cache
    if split not in _pcam_cache:
        from torchvision.datasets import PCAM
        root = REPO_ROOT / "data" / "raw"
        _pcam_cache[split] = PCAM(root=str(root), split=split, download=False)
    return _pcam_cache[split]


def get_dataset(dataset_id: str, split: str):
    """
    Central dataset router.
    Current implementation supports only `pcam` (thesis scaffold for future datasets).
    """
    dataset_id = (dataset_id or "pcam").strip().lower()
    if dataset_id != "pcam":
        raise ValueError(f"Unsupported dataset: {dataset_id}. Only 'pcam' is supported.")
    return get_pcam(split)


def get_config():
    """Load default config YAML."""
    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _slugify(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "default"


def _evaluation_identity(run_id: str | None = None, model_id: str | None = None) -> dict:
    normalized_run_id = normalize_run_id(run_id)
    run_value = normalized_run_id or "default"
    model_value = model_id or _resolve_model_id_for_run(run_id=run_id)
    return {
        "run_id": None if run_value == "default" else run_value,
        "model_id": model_value,
        "tag": f"run-{_slugify(run_value)}__model-{_slugify(model_value)}",
    }


def _evaluation_output_path(kind: str, split: str, *, method: str | None = None, run_id: str | None = None, model_id: str | None = None) -> Path:
    ident = _evaluation_identity(run_id=run_id, model_id=model_id)
    base = REPO_ROOT / "evaluation"
    base.mkdir(parents=True, exist_ok=True)
    if kind == "metrics":
        return base / f"metrics__{_slugify(method or 'unknown')}__{_slugify(split)}__{ident['tag']}.json"
    if kind == "shift":
        return base / f"shift-ood__{_slugify(method or 'unknown')}__{_slugify(split)}__{ident['tag']}.json"
    if kind == "pipeline_summary":
        return base / f"pipeline-summary__{_slugify(split)}__{ident['tag']}.json"
    if kind == "shift_summary":
        return base / f"shift-summary__{_slugify(split)}__{ident['tag']}.json"
    if kind == "bundle_summary":
        return base / f"thesis-bundle-summary__{_slugify(split)}__{ident['tag']}.json"
    if kind == "section_compare":
        return base / f"section-compare__{_slugify(method or 'unknown')}__{_slugify(split)}__{ident['tag']}.json"
    raise ValueError(f"Unknown evaluation output kind: {kind}")


def _report_package_dir(split: str, *, run_id: str | None = None, model_id: str | None = None) -> Path:
    ident = _evaluation_identity(run_id=run_id, model_id=model_id)
    base = REPO_ROOT / "evaluation" / "report_packages"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_slugify(split)}__{ident['tag']}"


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(REPO_ROOT))


def _infer_result_metadata(payload: dict) -> dict:
    kind = "unknown"
    methods = []
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    section = config.get("section")
    if isinstance(payload, dict) and "pipeline" in payload and "shift_ood" in payload:
        kind = "bundle"
        methods = sorted(list((payload.get("pipeline") or {}).keys()))
    elif isinstance(payload, dict) and "results_by_method" in payload and "grouped_summary_by_method" in payload:
        kind = "section_compare"
        methods = sorted(list((payload.get("grouped_summary_by_method") or {}).keys()))
        section = section or "shift"
    elif isinstance(payload, dict) and "results" in payload and isinstance(payload.get("results"), dict):
        result_map = payload.get("results") or {}
        if any(isinstance(v, dict) and "predictive_performance" in v for v in result_map.values()):
            kind = "section_compare"
            methods = sorted(list(result_map.keys()))
    elif isinstance(payload, dict) and "predictive_performance" in payload and "calibration" in payload:
        kind = "metrics"
        if config.get("method"):
            methods = [config.get("method")]
    elif isinstance(payload, dict) and "results" in payload and "grouped_summary" in payload:
        kind = "shift"
        if config.get("method"):
            methods = [config.get("method")]
    return {
        "kind": kind,
        "run_id": config.get("run_id"),
        "model_id": config.get("model_id"),
        "split": config.get("split"),
        "methods": methods,
        "section": section,
    }


def _persist_bundle_artifacts(payload: dict, *, run_id: str | None, split: str, bundle_out_path: Path) -> dict:
    result = dict(payload or {})
    config = dict(result.get("config") or {})
    model_id = _resolve_model_id_for_run(run_id=run_id)
    config["model_id"] = model_id
    config["run_id"] = run_id or None
    result["config"] = config

    outputs = dict(result.get("outputs") or {})
    detailed_metrics = {}
    shift_by_method = {}
    pipeline = result.get("pipeline") or {}
    shift_grouped = result.get("shift_ood_grouped_by_method") or {}

    generic_pipeline = REPO_ROOT / "evaluation" / "pipeline_summary.json"
    generic_shift_summary = REPO_ROOT / "evaluation" / f"shift_ood_{split}.json"
    copied_pipeline = _copy_if_exists(generic_pipeline, _evaluation_output_path("pipeline_summary", split, run_id=run_id, model_id=model_id))
    copied_shift_summary = _copy_if_exists(generic_shift_summary, _evaluation_output_path("shift_summary", split, run_id=run_id, model_id=model_id))
    if copied_pipeline:
        outputs["pipeline_summary"] = copied_pipeline
    if copied_shift_summary:
        outputs["shift_summary"] = copied_shift_summary

    for method in sorted(list(pipeline.keys())):
        src = REPO_ROOT / "evaluation" / f"metrics_{method}_{split}.json"
        copied = _copy_if_exists(src, _evaluation_output_path("metrics", split, method=method, run_id=run_id, model_id=model_id))
        if copied:
            detailed_metrics[method] = copied
    for method in sorted(list(shift_grouped.keys())):
        src = REPO_ROOT / "evaluation" / f"shift_ood_{method}_{split}.json"
        copied = _copy_if_exists(src, _evaluation_output_path("shift", split, method=method, run_id=run_id, model_id=model_id))
        if copied:
            shift_by_method[method] = copied

    outputs["bundle_summary"] = str(bundle_out_path.relative_to(REPO_ROOT))
    outputs["detailed_metrics_by_method"] = detailed_metrics
    outputs["shift_by_method"] = shift_by_method

    report_dir = _report_package_dir(split, run_id=run_id, model_id=model_id)
    report_manifest = report_dir / "manifest.json"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "experiments/generate_report_package.py",
                "--bundle",
                str(bundle_out_path.relative_to(REPO_ROOT)),
                "--out-dir",
                str(report_dir.relative_to(REPO_ROOT)),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=max(60, app.config.get("RUN_TIMEOUT", 600)),
        )
        if report_manifest.exists():
            outputs["report_manifest"] = str(report_manifest.relative_to(REPO_ROOT))
            outputs["report_dir"] = str(report_dir.relative_to(REPO_ROOT))
    except Exception as e:
        outputs["report_error"] = str(e)

    result["outputs"] = outputs

    with open(bundle_out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def _section_performance_rows(payload: dict) -> list[dict]:
    rows = []
    for method, block in sorted((payload.get("results") or {}).items()):
        perf = block.get("predictive_performance", {}) or {}
        rows.append(
            {
                "method": method,
                "accuracy": perf.get("accuracy"),
                "balanced_accuracy": perf.get("balanced_accuracy"),
                "f1": perf.get("f1"),
                "roc_auc": perf.get("roc_auc"),
                "pr_auc": perf.get("pr_auc"),
                "sensitivity": perf.get("sensitivity"),
                "specificity": perf.get("specificity"),
            }
        )
    return rows


def _section_calibration_rows(payload: dict) -> list[dict]:
    rows = []
    for method, block in sorted((payload.get("results") or {}).items()):
        cal = block.get("calibration", {}) or {}
        temp = (block.get("calibration_report", {}) or {}).get("temperature_scaling", {}) or {}
        rows.append(
            {
                "method": method,
                "ece": cal.get("ece"),
                "nll": cal.get("nll"),
                "brier": cal.get("brier"),
                "calibrated_ece": ((temp.get("calibrated") or {}).get("ece")),
            }
        )
    return rows


def _section_reliability_bins(payload: dict) -> dict[str, list[dict]]:
    return {
        method: ((block.get("calibration", {}) or {}).get("reliability_bins") or [])
        for method, block in sorted((payload.get("results") or {}).items())
    }


def _section_uncertainty_rows(payload: dict) -> list[dict]:
    rows = []
    for method, block in sorted((payload.get("results") or {}).items()):
        uq = block.get("uncertainty_quality", {}) or {}
        primary_name = "1 - MSP"
        primary_block = (uq.get("error_detection_one_minus_msp") or {})
        disagreement = uq.get("distributional_disagreement", {}) or {}
        if disagreement:
            primary_name = str(disagreement.get("primary_score") or primary_name)
            scores = disagreement.get("scores") or {}
            primary_block = ((scores.get(primary_name) or {}).get("error_detection") or primary_block)
        rows.append(
            {
                "method": method,
                "primary_name": primary_name,
                "primary_auroc": primary_block.get("auroc"),
                "primary_auprc": primary_block.get("auprc"),
                "entropy_auroc": ((uq.get("error_detection_entropy") or {}).get("auroc")),
                "entropy_auprc": ((uq.get("error_detection_entropy") or {}).get("auprc")),
                "msp_auroc": ((uq.get("error_detection_one_minus_msp") or {}).get("auroc")),
                "msp_auprc": ((uq.get("error_detection_one_minus_msp") or {}).get("auprc")),
            }
        )
    return rows


def _section_selective_curves(payload: dict) -> dict[str, list[dict]]:
    return {
        method: ((block.get("selective_prediction", {}) or {}).get("risk_coverage_curve") or [])
        for method, block in sorted((payload.get("results") or {}).items())
    }


def _section_high_confidence_rows(payload: dict) -> list[dict]:
    rows = []
    for method, block in sorted((payload.get("results") or {}).items()):
        reporting = ((block.get("selective_prediction", {}) or {}).get("high_confidence_reporting") or {})
        high = ((reporting.get("cohorts") or {}).get("high_confidence") or {})
        perf = high.get("predictive_performance", {}) or {}
        rows.append(
            {
                "method": method,
                "coverage": high.get("coverage"),
                "accuracy": perf.get("accuracy"),
            }
        )
    return rows


def _section_pathology_rows(payload: dict) -> list[dict]:
    rows = []
    for method, block in sorted((payload.get("results") or {}).items()):
        proxy = ((block.get("pathology_reporting", {}) or {}).get("slide_level_proxy") or {})
        rows.append(
            {
                "method": method,
                "accuracy": proxy.get("accuracy"),
                "roc_auc": proxy.get("roc_auc"),
                "pr_auc": proxy.get("pr_auc"),
            }
        )
    return rows


def _section_shift_rows(payload: dict) -> list[dict]:
    rows = []
    grouped = payload.get("grouped_summary_by_method") or {}
    for method, block in sorted(grouped.items()):
        near = (block.get("near_ood") or {})
        far = (block.get("far_ood") or {})
        rows.append(
            {
                "method": method,
                "near_ood_auroc": near.get("mean_ood_auroc"),
                "near_ood_auprc": near.get("mean_ood_auprc"),
                "far_ood_auroc": far.get("mean_ood_auroc"),
                "far_ood_auprc": far.get("mean_ood_auprc"),
                "near_accuracy": near.get("mean_accuracy"),
                "far_accuracy": far.get("mean_accuracy"),
                "near_ece": near.get("mean_ece"),
                "far_ece": far.get("mean_ece"),
            }
        )
    return rows


def _persist_section_artifacts(payload: dict, *, section: str, result_out_path: Path) -> dict:
    result = dict(payload or {})
    outputs = dict(result.get("outputs") or {})
    fig_dir = REPO_ROOT / "evaluation" / "section_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if section == "performance":
        rows = _section_performance_rows(result)
        thresholded_path = fig_dir / f"{result_out_path.stem}__thresholded.png"
        ranking_path = fig_dir / f"{result_out_path.stem}__ranking.png"
        plot_predictive_performance_thresholded(rows, thresholded_path, title="Predictive performance: thresholded metrics")
        plot_predictive_performance_ranking(rows, ranking_path, title="Predictive performance: ranking metrics")
        outputs["section_figures"] = {
            "performance_thresholded": str(thresholded_path.relative_to(REPO_ROOT)) if thresholded_path.exists() else None,
            "performance_ranking": str(ranking_path.relative_to(REPO_ROOT)) if ranking_path.exists() else None,
        }
    elif section == "calibration":
        rows = _section_calibration_rows(result)
        metrics_path = fig_dir / f"{result_out_path.stem}__calibration_metrics.png"
        reliability_path = fig_dir / f"{result_out_path.stem}__calibration_reliability.png"
        plot_calibration_metrics_summary(rows, metrics_path, title="Calibration comparison: numerical metrics")
        plot_reliability_overlay(_section_reliability_bins(result), reliability_path, title="Calibration comparison: reliability diagrams")
        outputs["section_figures"] = {
            "calibration_metrics": str(metrics_path.relative_to(REPO_ROOT)) if metrics_path.exists() else None,
            "calibration_reliability": str(reliability_path.relative_to(REPO_ROOT)) if reliability_path.exists() else None,
        }
    elif section == "uncertainty":
        rows = _section_uncertainty_rows(result)
        primary_path = fig_dir / f"{result_out_path.stem}__uncertainty_primary.png"
        common_path = fig_dir / f"{result_out_path.stem}__uncertainty_common.png"
        plot_uncertainty_primary_summary(rows, primary_path, title="Uncertainty quality: method-specific primary score")
        plot_uncertainty_common_summary(rows, common_path, title="Uncertainty quality: shared score comparison")
        outputs["section_figures"] = {
            "uncertainty_primary": str(primary_path.relative_to(REPO_ROOT)) if primary_path.exists() else None,
            "uncertainty_common": str(common_path.relative_to(REPO_ROOT)) if common_path.exists() else None,
        }
    elif section == "selective":
        rc_path = fig_dir / f"{result_out_path.stem}__selective_risk_coverage.png"
        hc_path = fig_dir / f"{result_out_path.stem}__selective_high_confidence.png"
        plot_risk_coverage_overlay(_section_selective_curves(result), rc_path, title="Selective prediction: risk-coverage curves")
        plot_high_confidence_accuracy_coverage(_section_high_confidence_rows(result), hc_path, title="Selective prediction: high-confidence coverage vs accuracy")
        outputs["section_figures"] = {
            "selective_risk_coverage": str(rc_path.relative_to(REPO_ROOT)) if rc_path.exists() else None,
            "selective_high_confidence": str(hc_path.relative_to(REPO_ROOT)) if hc_path.exists() else None,
        }
    elif section == "pathology":
        rows = _section_pathology_rows(result)
        pathology_path = fig_dir / f"{result_out_path.stem}__pathology_proxy.png"
        plot_pathology_proxy_summary(rows, pathology_path, title="Pathology proxy reporting comparison")
        outputs["section_figures"] = {
            "pathology_proxy": str(pathology_path.relative_to(REPO_ROOT)) if pathology_path.exists() else None,
        }
    elif section == "shift":
        rows = _section_shift_rows(result)
        detect_path = fig_dir / f"{result_out_path.stem}__shift_detection.png"
        robust_path = fig_dir / f"{result_out_path.stem}__shift_robustness.png"
        detail_path = fig_dir / f"{result_out_path.stem}__shift_detail_ood_auroc.png"
        plot_shift_detection_summary(rows, detect_path, title="Shift / OOD summary: detection metrics")
        plot_shift_robustness_summary(rows, robust_path, title="Shift / OOD summary: predictive robustness")
        plot_shift_condition_bars((result.get("results_by_method") or {}), "ood_detection_auroc", detail_path, title="Shift / OOD detail: per-condition OOD AUROC")
        outputs["section_figures"] = {
            "shift_detection": str(detect_path.relative_to(REPO_ROOT)) if detect_path.exists() else None,
            "shift_robustness": str(robust_path.relative_to(REPO_ROOT)) if robust_path.exists() else None,
            "shift_detail_ood_auroc": str(detail_path.relative_to(REPO_ROOT)) if detail_path.exists() else None,
        }
    result["outputs"] = outputs
    return result


def _safe_persist_section_artifacts(payload: dict, *, section: str, result_out_path: Path) -> dict:
    try:
        return _persist_section_artifacts(payload, section=section, result_out_path=result_out_path)
    except Exception as e:
        result = dict(payload or {})
        outputs = dict(result.get("outputs") or {})
        outputs["section_artifact_error"] = str(e)
        result["outputs"] = outputs
        return result


@app.route("/")
def index():
    return render_template("index.html", config=get_config())


@app.route("/project")
def project():
    config = get_config()
    return render_template("project.html", config=config, references=load_reference_catalog())


@app.route("/dataset")
def dataset_page():
    config = get_config()
    return render_template("dataset.html", config=config)


@app.route("/run")
def run_page():
    tasks = {k: v[1] for k, v in RUN_TASKS.items()}
    return render_template("run.html", config=get_config(), run_tasks=tasks)


@app.route("/evaluate")
def evaluate_page():
    references = load_reference_catalog()
    return render_template(
        "evaluate.html",
        config=get_config(),
        references=references,
        evaluation_methods=build_evaluation_methods(references),
        evaluation_methods_by_id=build_evaluation_method_map(references),
    )


@app.route("/setup")
def setup_page():
    """Dataset setup and browsing workspace."""
    return render_template("setup.html", config=get_config())


@app.route("/api/dataset/<split>/info")
def api_dataset_info(split):
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        dataset_id = request.args.get("dataset", "pcam")
        ds = get_dataset(dataset_id, split)
        return jsonify({"split": split, "size": len(ds)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/<split>/sample/<int:idx>")
def api_dataset_sample(split, idx):
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        dataset_id = request.args.get("dataset", "pcam")
        ds = get_dataset(dataset_id, split)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of range"}), 404
        img, label = ds[idx]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/<split>/sample/<int:idx>/corrupted")
def api_dataset_sample_corrupted(split, idx):
    """Return a sample image with a corruption applied.

    Query params:
        shift   : id | blur | noise | jpeg | color  (default: id)
        severity: 1-5 (default: 3)
        size    : output pixel size (default: 224)
        dataset : dataset id (default: pcam)
    """
    from PIL import ImageEnhance, ImageFilter
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        dataset_id = request.args.get("dataset", "pcam")
        shift = request.args.get("shift", "id")
        severity = max(1, min(5, int(request.args.get("severity", 3))))
        size = max(32, min(512, int(request.args.get("size", 224))))

        ds = get_dataset(dataset_id, split)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of range"}), 404
        img, _ = ds[idx]
        img = img.convert("RGB").resize((size, size))

        if shift == "blur":
            radius = [0.5, 1.0, 1.5, 2.0, 2.5][severity - 1]
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        elif shift == "noise":
            import numpy as _np
            arr = _np.asarray(img).astype(_np.float32)
            sigma = [6, 12, 18, 24, 30][severity - 1]
            noisy = arr + _np.random.default_rng(severity * 7).normal(0.0, sigma, arr.shape).astype(_np.float32)
            from PIL import Image as _PIL
            img = _PIL.fromarray(_np.clip(noisy, 0, 255).astype(_np.uint8))
        elif shift == "jpeg":
            scale = [0.95, 0.85, 0.75, 0.6, 0.45][severity - 1]
            w, h = img.size
            w2, h2 = max(8, int(w * scale)), max(8, int(h * scale))
            img = img.resize((w2, h2)).resize((w, h))
        elif shift == "color":
            color = [0.95, 0.9, 0.8, 0.7, 0.6][severity - 1]
            contrast = [1.05, 1.1, 1.15, 1.2, 1.25][severity - 1]
            img = ImageEnhance.Color(img).enhance(color)
            img = ImageEnhance.Contrast(img).enhance(contrast)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/corruption-grid")
def api_corruption_grid():
    """Return the pre-generated corruption preview grid PNG."""
    grid_path = REPO_ROOT / "evaluation" / "corruption_preview.png"
    if not grid_path.exists():
        # Generate on demand
        try:
            from experiments.generate_corruption_preview import generate_preview
            generate_preview(grid_path, n_samples=3, seed=42)
        except Exception as e:
            return jsonify({"error": f"Could not generate grid: {e}"}), 500
    return send_file(str(grid_path), mimetype="image/png")


@app.route("/api/dataset/<split>/label/<int:idx>")
def api_dataset_label(split, idx):
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        dataset_id = request.args.get("dataset", "pcam")
        ds = get_dataset(dataset_id, split)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of range"}), 404
        _, label = ds[idx]
        return jsonify({"idx": idx, "label": int(label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Cached per-split stats (size, class counts) — computed from H5 labels (fast)
_dataset_stats_cache = {}

_PCAM_Y_FILES = {
    "train": "camelyonpatch_level_2_split_train_y.h5",
    "val": "camelyonpatch_level_2_split_valid_y.h5",
    "test": "camelyonpatch_level_2_split_test_y.h5",
}


_pcam_label_cache = {}


def _pcam_labels_for_split(split: str):
    """
    Load PCAM H5 label array once per split (tiny: ~262k labels).
    Returns dict with:
      - y: array shape (N,) with labels {0,1}
      - indices0 / indices1: arrays of indices for each class
      - size: N
    """
    global _pcam_label_cache
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split: {split}")
    if split in _pcam_label_cache:
        return _pcam_label_cache[split]

    import h5py

    base = REPO_ROOT / "data" / "raw" / "pcam"
    path = base / _PCAM_Y_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"PCAM H5 label file not found: {path}")

    with h5py.File(path, "r") as f:
        y = f["y"][:]
        y = y.ravel()

    # Ensure compact dtype for faster indexing; labels are 0/1.
    y = y.astype(np.int64, copy=False)
    indices0 = np.flatnonzero(y == 0).astype(np.int64, copy=False)
    indices1 = np.flatnonzero(y == 1).astype(np.int64, copy=False)

    info = {"y": y, "indices0": indices0, "indices1": indices1, "size": int(y.shape[0])}
    _pcam_label_cache[split] = info
    return info


def _dataset_stats_from_h5(split: str):
    """Read label counts directly from PCAM H5 file (fast, no Python loop)."""
    if split not in ("train", "val", "test"):
        return None
    import h5py
    base = REPO_ROOT / "data" / "raw" / "pcam"
    path = base / _PCAM_Y_FILES[split]
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        y = f["y"][:]  # shape (N, 1, 1, 1)
        y = y.ravel()
        n = len(y)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
    return {
        "split": split,
        "size": n,
        "count_normal": n0,
        "count_metastasis": n1,
        "ratio_normal": round(n0 / n, 4) if n else 0,
        "ratio_metastasis": round(n1 / n, 4) if n else 0,
    }


@app.route("/api/dataset/<split>/stats")
def api_dataset_stats(split):
    """Return dataset statistics: size, count_normal (0), count_metastasis (1). Read from H5 (fast)."""
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    dataset_id = request.args.get("dataset", "pcam")
    if (dataset_id or "").strip().lower() != "pcam":
        return jsonify({"error": "Dataset stats only implemented for 'pcam' in this repo."}), 501
    global _dataset_stats_cache
    cache_key = (dataset_id, split)
    if cache_key not in _dataset_stats_cache:
        try:
            st = _dataset_stats_from_h5(split)
            if st is None:
                ds = get_dataset(dataset_id, split)
                n = len(ds)
                n0 = sum(1 for i in range(n) if ds[i][1] == 0)
                n1 = n - n0
                st = {
                    "split": split,
                    "size": n,
                    "count_normal": n0,
                    "count_metastasis": n1,
                    "ratio_normal": round(n0 / n, 4) if n else 0,
                    "ratio_metastasis": round(n1 / n, 4) if n else 0,
                }
            _dataset_stats_cache[cache_key] = st
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify(_dataset_stats_cache[cache_key])


@app.route("/api/dataset/verify")
def api_dataset_verify():
    """Verify that the configured dataset is fully labeled: for each split, size = count_normal + count_metastasis."""
    dataset_id = request.args.get("dataset", "pcam")
    if (dataset_id or "").strip().lower() != "pcam":
        return jsonify({"error": "Dataset verification only implemented for 'pcam' in this repo."}), 501
    result = {"datasets": [], "all_labeled": True}
    for split in ("train", "val", "test"):
        try:
            st = _dataset_stats_from_h5(split)
            if st is None:
                ds = get_dataset(dataset_id, split)
                n = len(ds)
                n0 = sum(1 for i in range(n) if ds[i][1] == 0)
                n1 = n - n0
                st = {"split": split, "size": n, "count_normal": n0, "count_metastasis": n1}
            total_labeled = st["count_normal"] + st["count_metastasis"]
            all_labeled = total_labeled == st["size"] and st["size"] > 0
            result["datasets"].append({
                "split": split,
                "size": st["size"],
                "count_normal": st["count_normal"],
                "count_metastasis": st["count_metastasis"],
                "all_labeled": all_labeled,
                "label_domain": [0, 1],
            })
            if not all_labeled:
                result["all_labeled"] = False
        except Exception as e:
            result["datasets"].append({"split": split, "error": str(e)})
            result["all_labeled"] = False
    return jsonify(result)


@app.route("/api/datasets")
def api_datasets():
    """
    Return datasets for UI.

    Fields:
    - can_browse: Dataset preview/stats implemented in Dataset tab
    - can_train: Dataset selection implemented for training runs
    """
    cfg = get_config()
    dataset_id = (cfg.get("data") or {}).get("dataset", "pcam")
    # PCAM sizes from official splits (thesis defaults)
    # Notes for thesis UI:
    # - Browsing/preview is implemented only for PCAM (patches stored as H5).
    # - Additional trusted datasets may be downloadable but not yet previewable.
    available = [
        {
            "id": "pcam",
            "name": "Patch Camelyon (PCAM)",
            "description": "Binary metastasis detection; 96×96 patches from CAMELYON16.",
            "splits": ["train", "val", "test"],
            "max_train": 262144,
            "max_val": 32768,
            "max_test": 32768,
            "can_browse": True,
            "can_train": True,
            "download_url": "https://patchcamelyon.grand-challenge.org/Download/",
            "resource_url": "https://patchcamelyon.grand-challenge.org/Download/",
        },
        {
            "id": "nct_crc_he_100k",
            "name": "NCT-CRC-HE-100K",
            "description": "Trusted colorectal patch dataset (download available). Browsing/training not implemented yet.",
            "splits": ["train", "val", "test"],
            "can_browse": False,
            "can_train": False,
            "download_url": "https://zenodo.org/record/1214456",
            "resource_url": "https://zenodo.org/record/1214456",
        },
    ]
    return jsonify({"datasets": available, "default": dataset_id})


@app.route("/api/models")
def api_models():
    """Return available model IDs for comparative experiments."""
    cfg = get_config()
    default_model = ((cfg.get("model") or {}).get("model_id")) or "google/vit-base-patch16-224"
    return jsonify({"models": MODEL_CATALOG, "default": default_model})


@app.route("/api/dataset/<split>/indices")
def api_dataset_indices(split):
    """Return indices for browsing, optionally filtered by label. Query: label=all|0|1, offset=0, limit=24."""
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    dataset_id = request.args.get("dataset", "pcam")
    ds = get_dataset(dataset_id, split)
    label_arg = request.args.get("label", "all")
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(500, max(1, int(request.args.get("limit", 24))))
    except ValueError:
        offset, limit = 0, 24
    try:
        size = len(ds)
        if label_arg == "all":
            indices = list(range(offset, min(offset + limit, size)))
        else:
            target = int(label_arg)
            if target not in (0, 1):
                return jsonify({"error": "label must be 0, 1, or all"}), 400
            indices = []
            skipped = 0
            for i in range(min(size, 500000)):
                _, lab = ds[i]
                if int(lab) != target:
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                indices.append(i)
                if len(indices) >= limit:
                    break
        return jsonify({"split": split, "indices": indices, "size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/<split>/samples")
def api_dataset_samples(split):
    """
    Return browseable samples in a single call.
    This is used by the Dataset tab to avoid N+1 label fetches.
    Query: dataset=<id>&label=all|0|1&offset=0&limit=24
    """
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400

    dataset_id = request.args.get("dataset", "pcam")
    try:
        label_arg = request.args.get("label", "all")
        mode = (request.args.get("mode", "offset") or "offset").strip().lower()
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(500, max(1, int(request.args.get("limit", 24))))
        random_n = int(request.args.get("random_n", limit))
        random_n = min(500, max(1, random_n))
    except ValueError:
        return jsonify({"error": "Invalid offset/limit"}), 400

    try:
        dataset_id_norm = (dataset_id or "").strip().lower()
        if dataset_id_norm != "pcam":
            return jsonify({"error": f"Dataset browse only implemented for 'pcam' (requested: {dataset_id})"}), 501

        # PCAM: use cached H5 label array for fast indexing + random sampling.
        pc = _pcam_labels_for_split(split)
        y = pc["y"]
        size = pc["size"]
        if size <= 0:
            return jsonify({"split": split, "dataset": dataset_id, "items": [], "size": 0})

        if label_arg == "all":
            if mode == "random":
                k = min(random_n, size)
                rng = np.random.default_rng()
                sel = rng.choice(size, size=k, replace=False)
                items = [{"idx": int(i), "label": int(y[int(i)])} for i in sel]
            else:
                start = min(offset, size)
                end = min(offset + limit, size)
                sel = np.arange(start, end, dtype=np.int64)
                items = [{"idx": int(i), "label": int(y[int(i)])} for i in sel]
        else:
            target = int(label_arg)
            if target not in (0, 1):
                return jsonify({"error": "label must be 0, 1, or all"}), 400
            pool = pc["indices0"] if target == 0 else pc["indices1"]
            pool_size = int(pool.shape[0])

            if pool_size <= 0:
                return jsonify({"split": split, "dataset": dataset_id, "items": [], "size": size})

            if mode == "random":
                k = min(random_n, pool_size)
                rng = np.random.default_rng()
                sel = rng.choice(pool, size=k, replace=False)
                items = [{"idx": int(i), "label": target} for i in sel]
            else:
                start = min(offset, pool_size)
                end = min(offset + limit, pool_size)
                sel = pool[start:end]
                items = [{"idx": int(i), "label": target} for i in sel]

        return jsonify({"split": split, "dataset": dataset_id, "items": items, "size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/project/structure")
def api_project_structure():
    """Return tree of key dirs/files for project management view."""
    root = REPO_ROOT
    ignore = {"venv", "__pycache__", ".git", ".h5", ".gz"}
    structure = []

    def walk(path: Path, prefix: str = ""):
        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            return
        dirs = [e for e in entries if e.is_dir() and e.name not in ignore]
        files = [e for e in entries if e.is_file() and not e.suffix in (".h5", ".gz")]
        for d in dirs:
            rel = d.relative_to(root)
            structure.append({"path": str(rel), "type": "dir"})
            if len([x for x in structure if x["path"].startswith(str(rel))]) < 50:
                walk(d, prefix + "  ")
        for f in files[:100]:  # cap files per dir
            structure.append({"path": str(f.relative_to(root)), "type": "file"})

    for top in [
        "configs",
        "data",
        "models",
        "uncertainty",
        "uncertainty_lab",
        "evaluation",
        "experiments",
        "web",
        "scripts",
    ]:
        p = root / top
        if p.exists():
            structure.append({"path": top, "type": "dir"})
            if p.is_dir():
                walk(p)
    return jsonify({"root": str(root), "entries": structure})


@app.route("/api/config")
def api_config():
    return jsonify(get_config())


@app.route("/api/device")
def api_device():
    """Return where the model runs: local (on this machine), and device (cuda/cpu)."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        name = None
        if cuda:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        return jsonify({
            "model_location": "local",
            "device": device,
            "device_name": name,
            "note": "Model is downloaded once from Hugging Face, then all training and inference run locally on this machine. GPU = fast, CPU = slower.",
        })
    except Exception as e:
        return jsonify({"model_location": "local", "device": "unknown", "error": str(e)})


# ---------- Training with live log (SSE) ----------
_train_state = {
    "process": None,
    "queue": None,
    "thread": None,
    "epochs_total": None,
    "buffer": [],  # list[{seq:int, text:str}] ring buffer for UI replay
    "next_seq": 0,
}
_train_lock = threading.Lock()


def _train_reader(process, queue):
    """
    Read training stdout line-by-line and push into:
      - queue: for SSE streaming
      - buffer: for UI replay when user navigates away/back
    """
    try:
        for raw_line in iter(process.stdout.readline, ""):
            text = raw_line.replace("\n", " ").strip()
            if not text:
                continue
            with _train_lock:
                seq = _train_state["next_seq"]
                _train_state["next_seq"] = seq + 1
                _train_state["buffer"].append({"seq": seq, "text": text})
                # Cap buffer size to keep memory bounded.
                if len(_train_state["buffer"]) > 2000:
                    _train_state["buffer"] = _train_state["buffer"][-2000:]
            queue.put({"seq": seq, "text": text})
    except Exception:
        pass
    finally:
        queue.put(None)  # sentinel


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    """Start training in background. Body: { dataset, model_id, epochs, n_train, n_val, lr, batch_size }."""
    global _train_state
    if _train_state["process"] is not None and _train_state["process"].poll() is None:
        return jsonify({"ok": False, "error": "Training already running"}), 400
    data = request.get_json() or {}
    cfg = get_config()
    cmd = [sys.executable, "experiments/train.py"]
    if data.get("dataset") is not None:
        cmd.extend(["--dataset", str(data["dataset"]).strip()])
    if data.get("model_id") is not None and str(data.get("model_id")).strip():
        cmd.extend(["--model_id", str(data["model_id"]).strip()])
    if data.get("epochs") is not None:
        epochs_total = int(data.get("epochs"))
        cmd.extend(["--epochs", str(epochs_total)])
    else:
        epochs_total = int(((cfg or {}).get("train") or {}).get("epochs", 3))
    if data.get("n_train") is not None:
        cmd.extend(["--n_train", str(int(data["n_train"]))])
    if data.get("n_val") is not None:
        cmd.extend(["--n_val", str(int(data["n_val"]))])
    if data.get("lr") is not None:
        cmd.extend(["--lr", str(float(data["lr"]))])
    if data.get("batch_size") is not None:
        cmd.extend(["--batch_size", str(int(data["batch_size"]))])
    try:
        import queue as q
        proc_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        que = q.Queue()
        th = threading.Thread(target=_train_reader, args=(proc, que), daemon=True)
        th.start()
        with _train_lock:
            _train_state["process"] = proc
            _train_state["queue"] = que
            _train_state["thread"] = th
            _train_state["epochs_total"] = epochs_total
            _train_state["buffer"] = []
            _train_state["next_seq"] = 0
        return jsonify({"ok": True, "task_id": "train", "status": "started"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/train/status")
def api_train_status():
    """Return whether training is running and recent log buffer for UI replay."""
    with _train_lock:
        proc = _train_state.get("process")
        running = proc is not None and proc.poll() is None
        epochs_total = _train_state.get("epochs_total")
        buf = list(_train_state.get("buffer") or [])
        last_seq = buf[-1]["seq"] if buf else -1
    return jsonify({
        "running": running,
        "epochs_total": epochs_total,
        "buffer": buf[-200:],  # keep status response small
        "last_seq": last_seq,
    })


@app.route("/api/train/runs")
def api_train_runs():
    """List saved training runs (checkpoints/run_*) with their metrics."""
    base = REPO_ROOT / "checkpoints"
    if not base.exists():
        return jsonify({"runs": []})
    runs = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        metrics_file = d / "metrics.json"
        if not metrics_file.exists():
            runs.append({"run_id": d.name, "run_dir": str(d), "best_val_acc": None})
            continue
        try:
            with open(metrics_file) as f:
                m = json.load(f)
            runs.append({
                "run_id": m.get("run_id", d.name),
                "run_dir": m.get("run_dir", str(d)),
                "model_id": m.get("model_id"),
                "epochs": m.get("epochs"),
                "dataset": m.get("dataset"),
                "n_train": m.get("n_train"),
                "n_val": m.get("n_val"),
                "lr": m.get("lr"),
                "batch_size": m.get("batch_size"),
                "best_val_acc": m.get("best_val_acc"),
                "best_epoch": m.get("best_epoch"),
                "history": m.get("history", []),
            })
        except Exception:
            runs.append({"run_id": d.name, "run_dir": str(d), "best_val_acc": None})
    return jsonify({"runs": runs})


@app.route("/api/evaluate/deep-ensemble/candidates")
def api_deep_ensemble_candidates():
    """List compatible runs that can form a scientifically valid deep ensemble."""
    run_id = (request.args.get("run_id") or "").strip()
    try:
        ensemble_size = int(request.args.get("ensemble_size", 2))
    except Exception:
        ensemble_size = 2

    cfg = get_config()
    model_id = _resolve_model_id_for_run(run_id=run_id) if run_id else cfg.get("model", {}).get("model_id", "google/vit-base-patch16-224")
    dataset = cfg.get("data", {}).get("dataset", "pcam")

    try:
        payload = list_deep_ensemble_candidates(
            config_model_id=model_id,
            config_dataset=dataset,
            run_id=run_id,
            ensemble_size=max(2, ensemble_size),
        )
        payload["ok"] = True
        payload["requested_ensemble_size"] = max(2, ensemble_size)
        return jsonify(payload)
    except Exception as e:
        return jsonify(
            {
                "ok": False,
                "error": str(e),
                "requested_ensemble_size": max(2, ensemble_size),
                "recipe": {"model_id": model_id, "dataset": dataset},
                "candidates": [],
                "groups": [],
            }
        )


@app.route("/api/train/stream")
def api_train_stream():
    """Server-Sent Events: stream training log lines until process ends."""
    from_seq = int(request.args.get("from_seq", -1))
    def generate(from_seq=from_seq):
        queue = _train_state.get("queue")
        process = _train_state.get("process")
        if queue is None or process is None:
            yield "data: {\"type\":\"done\",\"text\":\"[No training running]\"}\n\n"
            return
        while True:
            try:
                item = queue.get(timeout=2)
                if item is None:
                    yield 'data: {"type":"done","text":"[DONE]"}\n\n'
                    break
                seq = item.get("seq", -1)
                text = item.get("text", "")
                if seq > from_seq and text:
                    # JSON payload for robustness on the client.
                    yield 'data: {"type":"log","seq":%d,"text":%s}\n\n' % (seq, json.dumps(text))
            except Exception:
                if process.poll() is not None:
                    yield 'data: {"type":"done","text":"[DONE]"}\n\n'
                    break
                yield ": keepalive\n\n"
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- Blind test: random sample, predict, reveal label ----------
_model_cache = None


def _resolve_model_id_for_run(run_id=None):
    """Resolve model id from run metrics or fallback to default config."""
    cfg = get_config()
    model_id = cfg.get("model", {}).get("model_id", "google/vit-base-patch16-224")
    run_id = normalize_run_id(run_id)
    if not run_id:
        return model_id
    try:
        meta = load_run_metadata(run_id)
        if meta.get("model_id"):
            return meta["model_id"]
    except Exception:
        pass
    metrics_path = REPO_ROOT / "checkpoints" / run_id / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            model_id = m.get("model_id") or model_id
        except Exception:
            pass
    return model_id


def _predict_probs(model, x, method="mc_dropout", mc_samples=30):
    """
    Return class probabilities and uncertainty fields.
    method: mc_dropout (preferred) | confidence
    """
    import torch
    method = (method or "mc_dropout").strip().lower()
    if method not in ("confidence", "mc_dropout"):
        method = "mc_dropout"

    if method == "mc_dropout":
        # Enable dropout stochasticity at test time.
        model.train()
        T = max(2, min(100, int(mc_samples)))
        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                logits = model(pixel_values=x).logits
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
        probs_stack = torch.stack(probs_list, dim=0)  # (T, B, C)
        probs_mean = probs_stack.mean(dim=0)
        # Epistemic proxy: mean predictive variance over classes.
        uncertainty_var = probs_stack.var(dim=0).mean(dim=1)  # (B,)
        # Predictive entropy.
        entropy = -(probs_mean * (probs_mean.clamp(min=1e-12)).log()).sum(dim=1)  # (B,)
        model.eval()
        return probs_mean, {"uncertainty_var": uncertainty_var, "entropy": entropy, "mc_samples": T}

    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=x).logits
        probs = torch.softmax(logits, dim=1)
    entropy = -(probs * (probs.clamp(min=1e-12)).log()).sum(dim=1)
    return probs, {"uncertainty_var": None, "entropy": entropy, "mc_samples": 1}


def _compute_calibration_metrics(results, n_bins=15):
    """
    Compute ECE and reliability bins.
    results items require: confidence, correct.
    """
    n_bins = max(2, min(50, int(n_bins)))
    if not results:
        return {"ece": 0.0, "bins": []}

    bins = []
    ece = 0.0
    total = float(len(results))

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Right-closed on last bin.
        in_bin = []
        for r in results:
            c = float(r["confidence"])
            if (c >= lo and c < hi) or (b == n_bins - 1 and c <= hi):
                in_bin.append(r)
        if not in_bin:
            bins.append({"bin": b, "lo": round(lo, 6), "hi": round(hi, 6), "count": 0, "acc": None, "conf": None})
            continue
        acc = sum(1.0 if x["correct"] else 0.0 for x in in_bin) / len(in_bin)
        conf = sum(float(x["confidence"]) for x in in_bin) / len(in_bin)
        frac = len(in_bin) / total
        ece += abs(acc - conf) * frac
        bins.append({"bin": b, "lo": round(lo, 6), "hi": round(hi, 6), "count": len(in_bin), "acc": round(acc, 6), "conf": round(conf, 6)})

    return {"ece": round(float(ece), 6), "bins": bins}


def _get_model_for_inference(run_id=None, model_id_override=None):
    """Load model from run checkpoint or HF. run_id e.g. 'run_20250315_120000'. Cached by run/model."""
    global _model_cache
    model_id = model_id_override or _resolve_model_id_for_run(run_id=run_id)
    cache_key = f"{run_id or 'default'}::{model_id}"
    if _model_cache is not None:
        if getattr(_model_cache, "_run_id", None) == cache_key:
            return _model_cache
        _model_cache = None
    import torch
    from models.load_model import load_hf_image_classifier, get_device
    cfg = get_config()
    model, _, _ = load_hf_image_classifier(
        model_id=model_id,
        num_labels=cfg.get("model", {}).get("num_labels", 2),
    )
    if run_id:
        ckpt = REPO_ROOT / "checkpoints" / run_id / "best.pt"
    else:
        ckpt = REPO_ROOT / "checkpoints" / "best.pt"
    if not ckpt.exists() and not run_id:
        # Fallback: latest run
        base = REPO_ROOT / "checkpoints"
        if base.exists():
            run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
            if run_dirs and (run_dirs[0] / "best.pt").exists():
                ckpt = run_dirs[0] / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
    device = get_device()
    model = model.to(device)
    model.eval()
    _model_cache = model
    _model_cache._run_id = cache_key
    return model


@app.route("/api/evaluate/random")
def api_evaluate_random():
    """Return a random patch index for blind evaluation.

    Query:
      - split: train|val|test (default: test)
    """
    try:
        split = (request.args.get("split", "test") or "test").strip().lower()
        if split not in ("train", "val", "test"):
            return jsonify({"error": "Invalid split"}), 400

        ds = get_pcam(split)
        n = len(ds)
        if n == 0:
            return jsonify({"error": "Test set empty"}), 500
        idx = random.randint(0, n - 1)
        return jsonify({"split": split, "idx": idx, "size": n})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/batch")
def api_evaluate_batch():
    """
    Run inference on N random patches and return results + summary.

    Query:
      - split: train|val|test (default: test)
      - n: number of random patches (default: 10, max: 50)
      - run_id: optional training run_id to load from checkpoints/run_*/best.pt
      - method: confidence|temperature_scaled|mc_dropout
      - mc_samples: number of stochastic passes for MC dropout
      - calibration_bins: number of ECE bins
    """
    split = (request.args.get("split", "test") or "test").strip().lower()
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400

    run_id = request.args.get("run_id") or None
    method = (request.args.get("method", "mc_dropout") or "mc_dropout").strip().lower()
    try:
        mc_samples = int(request.args.get("mc_samples", 30))
    except ValueError:
        mc_samples = 30
    try:
        calibration_bins = int(request.args.get("calibration_bins", 15))
    except ValueError:
        calibration_bins = 15

    try:
        n = int(request.args.get("n", 10))
    except ValueError:
        n = 10
    n = max(1, min(50, n))

    try:
        ds = get_pcam(split)
        size = len(ds)
        if size == 0:
            return jsonify({"error": f"{split} set empty"}), 500

        if n >= size:
            indices = list(range(size))
        else:
            indices = random.sample(range(size), n)

        import torch
        from torchvision import transforms

        # Load model once per batch.
        model = _get_model_for_inference(run_id=run_id)
        device = next(model.parameters()).device
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        results = []
        correct = 0
        confs_correct = []
        confs_incorrect = []

        with torch.no_grad():
            for idx in indices:
                img, label = ds[idx]
                # Ensure RGB for image models.
                if hasattr(img, "convert"):
                    img = img.convert("RGB")

                x = transform(img).unsqueeze(0).to(device)
                probs_t, extra = _predict_probs(model, x, method=method, mc_samples=mc_samples)
                probs = probs_t[0].cpu().numpy()

                pred = int(probs.argmax())
                conf = float(probs[pred])
                gt = int(label)
                is_correct = (pred == gt)
                if is_correct:
                    correct += 1
                    confs_correct.append(conf)
                else:
                    confs_incorrect.append(conf)

                label_name = "Metastasis" if pred == 1 else "Normal"

                results.append({
                    "idx": int(idx),
                    "gt": gt,
                    "pred": pred,
                    "pred_label_name": label_name,
                    "p_normal": float(probs[0]),
                    "p_metastasis": float(probs[1]),
                    "confidence": round(conf, 6),
                    "uncertainty": round(float(1.0 - conf), 6),
                    "entropy": round(float(extra["entropy"][0].cpu().item()), 6),
                    "uncertainty_var": None if extra["uncertainty_var"] is None else round(float(extra["uncertainty_var"][0].cpu().item()), 6),
                    "correct": is_correct,
                })

        mean_conf = float(sum([r["confidence"] for r in results]) / len(results)) if results else 0.0
        acc = correct / len(results) if results else 0.0
        mean_conf_correct = float(sum(confs_correct) / len(confs_correct)) if confs_correct else None
        mean_conf_incorrect = float(sum(confs_incorrect) / len(confs_incorrect)) if confs_incorrect else None

        calib = _compute_calibration_metrics(results, n_bins=calibration_bins)

        return jsonify({
            "split": split,
            "n": len(results),
            "run_id": run_id,
            "method": method,
            "mc_samples": max(2, min(100, int(mc_samples))) if method == "mc_dropout" else 1,
            "summary": {
                "accuracy": round(acc, 6),
                "mean_confidence": round(mean_conf, 6),
                "mean_confidence_correct": None if mean_conf_correct is None else round(mean_conf_correct, 6),
                "mean_confidence_incorrect": None if mean_conf_incorrect is None else round(mean_conf_incorrect, 6),
                "ece": calib["ece"],
            },
            "calibration": calib,
            "results": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/predict")
def api_evaluate_predict():
    """Run model on one sample; return prediction and uncertainty fields."""
    split = request.args.get("split", "test")
    run_id = request.args.get("run_id") or None
    method = (request.args.get("method", "mc_dropout") or "mc_dropout").strip().lower()
    try:
        mc_samples = int(request.args.get("mc_samples", 30))
    except ValueError:
        mc_samples = 30
    try:
        idx = int(request.args.get("idx", 0))
    except ValueError:
        return jsonify({"error": "Invalid idx"}), 400
    if split not in ("train", "val", "test"):
        return jsonify({"error": "Invalid split"}), 400
    try:
        import torch
        from torchvision import transforms
        ds = get_pcam(split)
        if idx < 0 or idx >= len(ds):
            return jsonify({"error": "Index out of range"}), 404
        img, _ = ds[idx]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        x = transform(img).unsqueeze(0)
        model = _get_model_for_inference(run_id=run_id)
        device = next(model.parameters()).device
        x = x.to(device)
        probs_t, extra = _predict_probs(model, x, method=method, mc_samples=mc_samples)
        prob = probs_t[0].cpu().numpy()
        pred = int(prob.argmax())
        conf = float(prob[pred])
        label_name = "Metastasis" if pred == 1 else "Normal"
        return jsonify({
            "split": split,
            "idx": idx,
            "pred": pred,
            "prob": round(conf, 4),
            "uncertainty": round(float(1.0 - conf), 4),
            "entropy": round(float(extra["entropy"][0].cpu().item()), 6),
            "uncertainty_var": None if extra["uncertainty_var"] is None else round(float(extra["uncertainty_var"][0].cpu().item()), 6),
            "method": method,
            "mc_samples": int(extra.get("mc_samples") or 1),
            "label_name": label_name,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate/full", methods=["POST"])
def api_evaluate_full():
    """
    Run full thesis evaluation pipeline via experiments/evaluate_uncertainty.py.
    Body:
      - split: train|val|test
      - method: confidence|temperature_scaled|mc_dropout
      - mc_samples: int
      - max_samples: int
      - batch_size: int
      - run_id: optional
      - fit_temperature_on_val: bool
      - fit_deferral_on_val: bool
    """
    data = request.get_json() or {}
    split = (data.get("split") or "test").strip().lower()
    method = (data.get("method") or "mc_dropout").strip().lower()
    if split not in ("train", "val", "test"):
        return jsonify({"ok": False, "error": "Invalid split"}), 400
    if method not in ("confidence", "temperature_scaled", "mc_dropout", "deep_ensemble"):
        return jsonify({"ok": False, "error": "Invalid method"}), 400

    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    try:
        max_samples = int(data.get("max_samples", 2000))
    except Exception:
        max_samples = 2000
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        seed = int(data.get("seed", 42))
    except Exception:
        seed = 42
    try:
        ensemble_size = int(data.get("ensemble_size", 2))
    except Exception:
        ensemble_size = 3
    ensemble_run_ids = ",".join([str(x).strip() for x in (data.get("ensemble_run_ids") or []) if str(x).strip()])

    run_id = normalize_run_id(data.get("run_id"))
    fit_temp = bool(data.get("fit_temperature_on_val", False))
    fit_deferral = bool(data.get("fit_deferral_on_val", True))
    model_id = _resolve_model_id_for_run(run_id=run_id)
    out_path = _evaluation_output_path("metrics", split, method=method, run_id=run_id, model_id=model_id)

    cmd = [
        sys.executable,
        "experiments/evaluate_uncertainty.py",
        "--split",
        split,
        "--method",
        method,
        "--mc_samples",
        str(max(2, min(100, mc_samples))),
        "--ensemble_size",
        str(max(1, ensemble_size)),
        "--max_samples",
        str(max(1, max_samples)),
        "--batch_size",
        str(max(1, batch_size)),
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]
    if method == "deep_ensemble" and ensemble_run_ids:
        cmd.extend(["--ensemble_run_ids", ensemble_run_ids])
    if run_id:
        cmd.extend(["--run_id", run_id])
    if fit_temp:
        cmd.append("--fit_temperature_on_val")
    if fit_deferral:
        cmd.append("--fit_deferral_on_val")

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(120, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Evaluation script failed", "log": log}), 500
        if not out_path.exists():
            return jsonify({"ok": False, "error": f"Expected output not found: {out_path}", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return jsonify(
            {
                "ok": True,
                "returncode": 0,
                "output_path": str(out_path.relative_to(REPO_ROOT)),
                "result": payload,
                "log": log,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Evaluation timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/api/evaluate/shift", methods=["POST"])
def api_evaluate_shift():
    """Run synthetic shift/OOD evaluation pipeline."""
    data = request.get_json() or {}
    split = (data.get("split") or "test").strip().lower()
    if split not in ("train", "val", "test"):
        return jsonify({"ok": False, "error": "Invalid split"}), 400
    try:
        max_samples = int(data.get("max_samples", 512))
    except Exception:
        max_samples = 512
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        seed = int(data.get("seed", 42))
    except Exception:
        seed = 42

    shifts = (data.get("shifts") or "id,blur,jpeg,color,noise").strip()
    severities = (data.get("severities") or "1,3,5").strip()
    method = (data.get("method") or "mc_dropout").strip().lower()
    if method not in ("confidence", "temperature_scaled", "mc_dropout", "deep_ensemble"):
        return jsonify({"ok": False, "error": "Invalid method"}), 400
    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    try:
        ensemble_size = int(data.get("ensemble_size", 2))
    except Exception:
        ensemble_size = 3
    ensemble_run_ids = ",".join([str(x).strip() for x in (data.get("ensemble_run_ids") or []) if str(x).strip()])
    run_id = normalize_run_id(data.get("run_id"))
    model_id = _resolve_model_id_for_run(run_id=run_id)
    out_path = _evaluation_output_path("shift", split, method=method, run_id=run_id, model_id=model_id)

    cmd = [
        sys.executable,
        "experiments/evaluate_shift_ood.py",
        "--split",
        split,
        "--method",
        method,
        "--mc_samples",
        str(max(2, mc_samples)),
        "--ensemble_size",
        str(max(1, ensemble_size)),
        "--max_samples",
        str(max(1, max_samples)),
        "--batch_size",
        str(max(1, batch_size)),
        "--seed",
        str(seed),
        "--shifts",
        shifts,
        "--severities",
        severities,
        "--out",
        str(out_path),
    ]
    if method == "deep_ensemble" and ensemble_run_ids:
        cmd.extend(["--ensemble_run_ids", ensemble_run_ids])
    if run_id:
        cmd.extend(["--run_id", run_id])

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(120, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Shift evaluation failed", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return jsonify(
            {
                "ok": True,
                "returncode": 0,
                "output_path": str(out_path.relative_to(REPO_ROOT)),
                "result": payload,
                "log": log,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Shift evaluation timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/api/evaluate/compare/core", methods=["POST"])
def api_evaluate_compare_core():
    """Run comparison across methods for core evaluation sections."""
    data = request.get_json() or {}
    split = (data.get("split") or "test").strip().lower()
    section = (data.get("section") or "performance").strip().lower()
    if section not in {"performance", "calibration", "uncertainty", "selective", "pathology"}:
        return jsonify({"ok": False, "error": "Invalid core section"}), 400
    if split not in ("train", "val", "test"):
        return jsonify({"ok": False, "error": "Invalid split"}), 400
    try:
        max_samples = int(data.get("max_samples", 2000))
    except Exception:
        max_samples = 2000
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    try:
        ensemble_size = int(data.get("ensemble_size", 2))
    except Exception:
        ensemble_size = 2
    run_id = normalize_run_id(data.get("run_id"))
    fit_temp = bool(data.get("fit_temperature_on_val", False))
    fit_deferral = bool(data.get("fit_deferral_on_val", True))
    include_deep_ensemble = bool(data.get("include_deep_ensemble", True))
    ensemble_run_ids = ",".join([str(x).strip() for x in (data.get("ensemble_run_ids") or []) if str(x).strip()])
    model_id = _resolve_model_id_for_run(run_id=run_id)
    out_path = _evaluation_output_path("section_compare", split, method=section, run_id=run_id, model_id=model_id)

    cmd = [
        sys.executable,
        "experiments/run_evaluation_pipeline.py",
        "--split",
        split,
        "--max_samples",
        str(max(1, max_samples)),
        "--batch_size",
        str(max(1, batch_size)),
        "--mc_samples",
        str(max(2, mc_samples)),
        "--ensemble_size",
        str(max(1, ensemble_size)),
        "--out",
        str(out_path),
    ]
    if include_deep_ensemble:
        cmd.append("--include_deep_ensemble")
    if ensemble_run_ids:
        cmd.extend(["--ensemble_run_ids", ensemble_run_ids])
    if run_id:
        cmd.extend(["--run_id", run_id])
    if fit_temp:
        cmd.append("--fit_temperature_on_val")
    if fit_deferral:
        cmd.append("--fit_deferral_on_val")

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(180, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Core comparison failed", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("config", {})
        payload["config"]["section"] = section
        payload["config"]["run_id"] = run_id or None
        payload["config"]["model_id"] = model_id
        payload["config"]["split"] = split
        payload = _safe_persist_section_artifacts(payload, section=section, result_out_path=out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return jsonify({"ok": True, "returncode": 0, "output_path": str(out_path.relative_to(REPO_ROOT)), "result": payload, "log": log})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Core comparison timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/api/evaluate/compare/shift", methods=["POST"])
def api_evaluate_compare_shift():
    """Run comparison across methods for shift/OOD section."""
    data = request.get_json() or {}
    split = (data.get("split") or "test").strip().lower()
    section = "shift"
    if split not in ("train", "val", "test"):
        return jsonify({"ok": False, "error": "Invalid split"}), 400
    try:
        max_samples = int(data.get("max_samples", 512))
    except Exception:
        max_samples = 512
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    try:
        ensemble_size = int(data.get("ensemble_size", 2))
    except Exception:
        ensemble_size = 2
    shifts = (data.get("shifts") or "id,blur,jpeg,color,noise").strip()
    severities = (data.get("severities") or "1,3,5").strip()
    run_id = normalize_run_id(data.get("run_id"))
    include_deep_ensemble = bool(data.get("include_deep_ensemble", True))
    ensemble_run_ids = ",".join([str(x).strip() for x in (data.get("ensemble_run_ids") or []) if str(x).strip()])
    model_id = _resolve_model_id_for_run(run_id=run_id)
    out_path = _evaluation_output_path("section_compare", split, method=section, run_id=run_id, model_id=model_id)

    cmd = [
        sys.executable,
        "experiments/run_shift_comparison.py",
        "--split",
        split,
        "--max_samples",
        str(max(1, max_samples)),
        "--batch_size",
        str(max(1, batch_size)),
        "--mc_samples",
        str(max(2, mc_samples)),
        "--ensemble_size",
        str(max(1, ensemble_size)),
        "--shifts",
        shifts,
        "--severities",
        severities,
        "--out",
        str(out_path.relative_to(REPO_ROOT)),
    ]
    if include_deep_ensemble:
        cmd.append("--include_deep_ensemble")
    if ensemble_run_ids:
        cmd.extend(["--ensemble_run_ids", ensemble_run_ids])
    if run_id:
        cmd.extend(["--run_id", run_id])

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(180, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Shift comparison failed", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("config", {})
        payload["config"]["section"] = section
        payload["config"]["run_id"] = run_id or None
        payload["config"]["model_id"] = model_id
        payload["config"]["split"] = split
        payload = _persist_section_artifacts(payload, section=section, result_out_path=out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return jsonify({"ok": True, "returncode": 0, "output_path": str(out_path.relative_to(REPO_ROOT)), "result": payload, "log": log})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Shift comparison timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


def _parse_thesis_bundle_request(data: dict) -> tuple[dict, str]:
    split = (data.get("split") or "test").strip().lower()
    if split not in ("train", "val", "test"):
        raise ValueError("Invalid split")
    try:
        max_samples = int(data.get("max_samples", 512))
    except Exception:
        max_samples = 512
    try:
        batch_size = int(data.get("batch_size", 64))
    except Exception:
        batch_size = 64
    try:
        mc_samples = int(data.get("mc_samples", 30))
    except Exception:
        mc_samples = 30
    run_id = normalize_run_id(data.get("run_id"))
    fit_temp = bool(data.get("fit_temperature_on_val", False))
    fit_deferral = bool(data.get("fit_deferral_on_val", True))
    include_deep_ensemble = bool(data.get("include_deep_ensemble", True))
    try:
        ensemble_size = int(data.get("ensemble_size", 2))
    except Exception:
        ensemble_size = 3
    ensemble_run_ids = ",".join([str(x).strip() for x in (data.get("ensemble_run_ids") or []) if str(x).strip()])
    shift_severities = (data.get("shift_severities") or "1,3,5").strip()
    model_id = _resolve_model_id_for_run(run_id=run_id)
    out_path = _evaluation_output_path("bundle_summary", split, run_id=run_id, model_id=model_id)
    payload = {
        "split": split,
        "max_samples": max(1, max_samples),
        "batch_size": max(1, batch_size),
        "mc_samples": max(2, mc_samples),
        "run_id": run_id,
        "fit_temperature_on_val": fit_temp,
        "fit_deferral_on_val": fit_deferral,
        "include_deep_ensemble": include_deep_ensemble,
        "ensemble_size": max(1, ensemble_size),
        "ensemble_run_ids": ensemble_run_ids,
        "shift_severities": shift_severities,
        "model_id": model_id,
    }
    return payload, str(out_path)


def _build_thesis_bundle_cmd(payload: dict, out_path: str) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/run_thesis_bundle.py",
        "--split",
        payload["split"],
        "--max_samples",
        str(payload["max_samples"]),
        "--batch_size",
        str(payload["batch_size"]),
        "--mc_samples",
        str(payload["mc_samples"]),
        "--ensemble_size",
        str(payload["ensemble_size"]),
        "--shift_severities",
        payload["shift_severities"],
        "--out",
        str(Path(out_path).relative_to(REPO_ROOT)),
    ]
    if payload["fit_temperature_on_val"]:
        cmd.append("--fit_temperature_on_val")
    if payload["fit_deferral_on_val"]:
        cmd.append("--fit_deferral_on_val")
    if payload["include_deep_ensemble"]:
        cmd.append("--include_deep_ensemble")
    if payload.get("ensemble_run_ids"):
        cmd.extend(["--ensemble_run_ids", payload["ensemble_run_ids"]])
    if payload["run_id"]:
        cmd.extend(["--run_id", payload["run_id"]])
    return cmd


@app.route("/api/evaluate/all", methods=["POST"])
def api_evaluate_all():
    """Run complete thesis bundle (core + shift) and return combined summary."""
    data = request.get_json() or {}
    try:
        payload, out_path = _parse_thesis_bundle_request(data)
        cmd = _build_thesis_bundle_cmd(payload, out_path)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=max(180, app.config.get("RUN_TIMEOUT", 600)),
        )
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return jsonify({"ok": False, "returncode": r.returncode, "error": "Thesis bundle failed", "log": log}), 500
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload = _persist_bundle_artifacts(payload, run_id=payload.get("config", {}).get("run_id"), split=payload.get("config", {}).get("split") or payload.get("config", {}).get("eval_split") or payload.get("config", {}).get("target_split") or "test", bundle_out_path=Path(out_path))
        return jsonify(
            {
                "ok": True,
                "returncode": 0,
                "output_path": str(out_path.relative_to(REPO_ROOT)),
                "result": payload,
                "log": log,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Thesis bundle timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@sock.route("/ws/evaluate/all")
def ws_evaluate_all(ws):
    """WebSocket: run complete thesis bundle and stream progress/logs live."""
    proc = None
    try:
        raw = ws.receive()
        data = json.loads(raw or "{}")
        payload, out_path = _parse_thesis_bundle_request(data)
        cmd = _build_thesis_bundle_cmd(payload, out_path)
        ws.send(json.dumps({"type": "status", "message": "Starting thesis bundle evaluation."}))
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.stdout is not None:
            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                if line.startswith("__PROGRESS__"):
                    try:
                        payload_msg = json.loads(line[len("__PROGRESS__") :])
                    except Exception:
                        payload_msg = {"progress": None, "stage": "unknown", "message": line}
                    payload_msg["type"] = "progress"
                    ws.send(json.dumps(payload_msg))
                else:
                    ws.send(json.dumps({"type": "log", "text": line}))
        ret = proc.wait(timeout=max(180, app.config.get("RUN_TIMEOUT", 600)))
        if ret != 0:
            ws.send(json.dumps({"type": "error", "message": f"Thesis bundle failed with exit code {ret}."}))
            return
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        result = _persist_bundle_artifacts(result, run_id=payload.get("run_id"), split=payload.get("split") or "test", bundle_out_path=Path(out_path))
        ws.send(
            json.dumps(
                {
                    "type": "result",
                    "output_path": str(Path(out_path).relative_to(REPO_ROOT)),
                    "result": result,
                }
            )
        )
        ws.send(json.dumps({"type": "done", "message": "Evaluation finished."}))
    except Exception as e:
        try:
            ws.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


@app.route("/api/evaluate/results")
def api_evaluate_results():
    """
    List saved evaluation JSON files and optionally load one.
    Query:
      - action: list | load
      - name: file name (required for action=load), e.g. metrics_confidence_test.json
    """
    action = (request.args.get("action") or "list").strip().lower()
    base = REPO_ROOT / "evaluation"
    base.mkdir(parents=True, exist_ok=True)

    if action == "list":
        kind_filter = (request.args.get("kind") or "").strip().lower()
        run_filter = (request.args.get("run_id") or "").strip()
        split_filter = (request.args.get("split") or "").strip().lower()
        section_filter = (request.args.get("section") or "").strip().lower()
        files = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in files:
            try:
                st = p.stat()
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                meta = _infer_result_metadata(payload)
                if kind_filter and meta.get("kind") != kind_filter:
                    continue
                if run_filter and (meta.get("run_id") or "") != run_filter:
                    continue
                if split_filter and (meta.get("split") or "") != split_filter:
                    continue
                if section_filter and (meta.get("section") or "").strip().lower() != section_filter:
                    continue
                preview = {}
                if meta.get("kind") == "bundle" and isinstance(payload, dict):
                    outputs = payload.get("outputs") or {}
                    manifest_rel = outputs.get("report_manifest")
                    preview = {
                        "has_report_manifest": bool(manifest_rel),
                        "report_manifest": manifest_rel,
                        "summary_figures": [],
                        "method_reports": [],
                    }
                    if manifest_rel:
                        manifest_path = (REPO_ROOT / manifest_rel).resolve()
                        if manifest_path.is_file():
                            try:
                                with open(manifest_path, "r", encoding="utf-8") as mf:
                                    manifest = json.load(mf)
                                summary_figures = manifest.get("summary_figures") or {}
                                preview["summary_figures"] = sorted([key for key, value in summary_figures.items() if value])
                                preview["method_reports"] = sorted(list((manifest.get("method_reports") or {}).keys()))
                            except Exception:
                                pass
                out.append(
                    {
                        "name": p.name,
                        "path": str(p.relative_to(REPO_ROOT)),
                        "size_bytes": int(st.st_size),
                        "modified_ts": int(st.st_mtime),
                        "metadata": meta,
                        "preview": preview,
                    }
                )
            except Exception:
                continue
        return jsonify({"ok": True, "files": out})

    if action == "load":
        name = (request.args.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Missing 'name' for load action"}), 400
        # basic filename hardening
        if "/" in name or "\\" in name or not name.endswith(".json"):
            return jsonify({"ok": False, "error": "Invalid file name"}), 400
        path = base / name
        if not path.exists():
            return jsonify({"ok": False, "error": f"File not found: {name}"}), 404
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            meta = _infer_result_metadata(payload)
            if meta.get("kind") == "bundle":
                outputs = payload.get("outputs") or {}
                if not outputs.get("report_manifest"):
                    payload = _persist_bundle_artifacts(
                        payload,
                        run_id=meta.get("run_id"),
                        split=meta.get("split") or "test",
                        bundle_out_path=path,
                    )
                    meta = _infer_result_metadata(payload)
            if meta.get("kind") == "section_compare":
                section = (meta.get("section") or "").strip().lower()
                outputs = payload.get("outputs") or {}
                figs = outputs.get("section_figures") or {}
                if section and not figs:
                    payload = _persist_section_artifacts(payload, section=section, result_out_path=path)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    meta = _infer_result_metadata(payload)
            return jsonify({"ok": True, "name": name, "path": str(path.relative_to(REPO_ROOT)), "metadata": meta, "result": payload})
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to read JSON: {e}"}), 500

    return jsonify({"ok": False, "error": "Unknown action. Use list or load."}), 400


@app.route("/api/evaluate/artifact")
def api_evaluate_artifact():
    rel_path = (request.args.get("path") or "").strip()
    if not rel_path:
        return jsonify({"ok": False, "error": "Missing 'path' query parameter"}), 400
    target = (REPO_ROOT / rel_path).resolve()
    eval_root = (REPO_ROOT / "evaluation").resolve()
    try:
        target.relative_to(eval_root)
    except Exception:
        return jsonify({"ok": False, "error": "Artifact path must stay inside evaluation/"}), 400
    if not target.exists() or not target.is_file():
        return jsonify({"ok": False, "error": "Artifact not found"}), 404
    return send_file(target)


# Allowed run tasks (command list; first arg is Python, relative paths to REPO_ROOT)
def _run_cmd(py_args):
    return [sys.executable] + py_args


RUN_TASKS = {
    "check_setup": (
        _run_cmd(["-c", "import torch; import torchvision; from torchvision.datasets import PCAM; print('OK: torch', torch.__version__, 'PCAM available')"]),
        "Verify environment: PyTorch, torchvision, PCAM dataset availability.",
    ),
    "download_pcam": (
        _run_cmd(["data/download_datasets.py", "--root", "data/raw", "--dataset", "pcam"]),
        "Download Patch Camelyon (train/val/test) to data/raw.",
    ),
    "download_nct_crc_he_100k": (
        _run_cmd(["data/download_datasets.py", "--root", "data/raw", "--dataset", "nct_crc_he_100k"]),
        "Download NCT-CRC-HE-100K (Zenodo zip) to data/raw/nct_crc_he_100k.",
    ),
    "cache_model": (
        _run_cmd(["models/load_model.py", "--model_id", "google/vit-base-patch16-224"]),
        "Download and cache Hugging Face ViT model for binary classification.",
    ),
    "uncertainty_lab_check": (
        _run_cmd(
            [
                "-c",
                "import uncertainty_lab; from uncertainty_lab.pipeline.run import run_pipeline; "
                "print('OK uncertainty_lab', getattr(uncertainty_lab, '__version__', '?'))",
            ]
        ),
        "Verify the pipeline package import and run entry point.",
    ),
}
# Training is started via POST /api/train/start and streamed via /api/train/stream (see Run page).


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run a named task (download_pcam, cache_model, check_setup). Returns log."""
    data = request.get_json() or {}
    task = data.get("task")
    if task not in RUN_TASKS:
        return jsonify({"ok": False, "error": "Unknown task", "allowed": list(RUN_TASKS.keys())}), 400
    cmd = RUN_TASKS[task][0]
    if task == "cache_model":
        model_id = (data.get("model_id") or "").strip()
        if model_id:
            cmd = _run_cmd(["models/load_model.py", "--model_id", model_id])
    cwd = str(REPO_ROOT)
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=app.config.get("RUN_TIMEOUT", 600),
        )
        out = (r.stdout or "") + (r.stderr or "")
        return jsonify({
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "log": out or "(no output)",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Timeout", "log": "Task timed out."}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "log": ""}), 500


@app.route("/api/evaluate/conformal-results")
def api_conformal_results():
    import glob
    files = sorted(glob.glob(str(REPO_ROOT / "evaluation" / "conformal_prediction__*.json")))
    if not files:
        return jsonify({"error": "No conformal results found. Run experiments/run_conformal.py first."}), 404
    data = json.loads(Path(files[-1]).read_text())
    return jsonify(data)


@app.route("/api/evaluate/aleatoric-epistemic")
def api_aleatoric_epistemic():
    import glob
    files = sorted(glob.glob(str(REPO_ROOT / "evaluation" / "aleatoric_epistemic__*.json")))
    if not files:
        return jsonify({"error": "No aleatoric/epistemic results found. Run experiments/run_aleatoric_epistemic.py first."}), 404
    data = json.loads(Path(files[-1]).read_text())
    return jsonify(data)


@app.route("/api/evaluate/ece-under-shift")
def api_ece_under_shift():
    path = REPO_ROOT / "evaluation" / "ece_under_shift_summary.json"
    if not path.exists():
        return jsonify({"error": "No ECE shift data found. Run experiments/run_ece_under_shift.py first."}), 404
    return jsonify(json.loads(path.read_text()))


@app.route("/api/evaluate/cross-domain-ood")
def api_cross_domain_ood():
    import glob as _glob
    files = sorted(_glob.glob(str(REPO_ROOT / "evaluation" / "cross_domain_ood__*.json")))
    if not files:
        return jsonify({"error": "No cross-domain OOD results. Run experiments/evaluate_cross_domain_ood.py first."}), 404
    results = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        method = d["config"]["method"]
        results[method] = {
            "pcam_accuracy": d["id_performance"]["accuracy"],
            "pcam_mean_unc": d["id_performance"]["mean_uncertainty_1msp"],
            "nct_accuracy": d["ood_performance"]["accuracy"],
            "nct_mean_unc": d["ood_performance"]["mean_uncertainty_1msp"],
            "ood_detection": d["ood_detection"],
            "temperature": d["config"].get("temperature", 1.0),
        }
    return jsonify({"results": results})


@app.route("/api/evaluate/figures/<path:filename>")
def api_evaluate_figure(filename):
    fig_dir = REPO_ROOT / "evaluation" / "figures"
    safe = fig_dir / filename
    if not str(safe).startswith(str(fig_dir)) or not safe.exists():
        return jsonify({"error": "Figure not found"}), 404
    return send_file(str(safe), mimetype="image/png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=5000, help="Port")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

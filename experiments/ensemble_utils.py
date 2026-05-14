from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_MODEL_IDS = {
    "vit": "google/vit-base-patch16-224",
    "beit": "microsoft/beit-base-patch16-224",
    "deit": "facebook/deit-base-patch16-224",
}

RUN_NAME_RE = re.compile(
    r"^run_(?P<dataset>[a-z0-9_]+)_(?P<model_slug>[a-z0-9-]+)_e(?P<epochs>\d+)_nt(?P<n_train>\d+)_nv(?P<n_val>\d+)_bs(?P<batch_size>\d+)_lr(?P<lr>[0-9eE\.-]+)_(?P<stamp>\d{8}_\d{6})$"
)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_id_from_slug(slug: str | None) -> str | None:
    if not slug:
        return None
    slug = slug.strip().lower()
    if slug.startswith("google-vit-"):
        return "google/vit-base-patch16-224"
    if slug.startswith("microsoft-beit-"):
        return "microsoft/beit-base-patch16-224"
    if slug.startswith("facebook-deit-"):
        return "facebook/deit-base-patch16-224"
    return None


def _infer_model_id_from_checkpoint(ckpt_path: Path) -> str | None:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = state.get("model_state_dict") or {}
        keys = list(state_dict.keys())
    except Exception:
        return None
    if any(key.startswith("beit.") for key in keys):
        return SUPPORTED_MODEL_IDS["beit"]
    if any(key.startswith("deit.") for key in keys):
        return SUPPORTED_MODEL_IDS["deit"]
    if any(key.startswith("vit.") for key in keys):
        return SUPPORTED_MODEL_IDS["vit"]
    return None


def _parse_run_name(run_name: str) -> dict:
    match = RUN_NAME_RE.match(run_name)
    if not match:
        return {}
    data = match.groupdict()
    return {
        "dataset": data["dataset"],
        "model_id": _model_id_from_slug(data["model_slug"]),
        "epochs": int(data["epochs"]),
        "n_train": int(data["n_train"]),
        "n_val": int(data["n_val"]),
        "batch_size": int(data["batch_size"]),
        "lr": float(data["lr"]),
    }


def normalize_run_id(run_id: str | None) -> str:
    value = (run_id or "").strip()
    if value in {"", "default", "run-default"}:
        return ""
    return value


def load_run_metadata(run_id: str) -> dict:
    run_dir = REPO_ROOT / "checkpoints" / run_id
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found for run '{run_id}'. Expected: {ckpt_path}")

    metrics_path = run_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    parsed = _parse_run_name(run_id)

    model_id = metrics.get("model_id") or parsed.get("model_id") or _infer_model_id_from_checkpoint(ckpt_path)
    dataset = metrics.get("dataset") or parsed.get("dataset")
    meta = {
        "run_id": run_id,
        "run_dir": run_dir,
        "ckpt_path": ckpt_path,
        "model_id": model_id,
        "dataset": dataset,
        "epochs": metrics.get("epochs", parsed.get("epochs")),
        "n_train": metrics.get("n_train", parsed.get("n_train")),
        "n_val": metrics.get("n_val", parsed.get("n_val")),
        "lr": metrics.get("lr", parsed.get("lr")),
        "batch_size": metrics.get("batch_size", parsed.get("batch_size")),
        "best_val_acc": metrics.get("best_val_acc"),
    }
    return meta


def _recipe_signature(meta: dict) -> tuple:
    return (
        meta.get("model_id"),
        meta.get("dataset"),
        meta.get("epochs"),
        meta.get("n_train"),
        meta.get("n_val"),
        meta.get("lr"),
        meta.get("batch_size"),
    )


def _validate_metadata_complete(meta: dict, context: str) -> None:
    missing = [key for key in ("model_id", "dataset", "epochs", "n_train", "n_val", "lr", "batch_size") if meta.get(key) is None]
    if missing:
        raise ValueError(
            f"{context} is missing metadata required for a scientifically valid deep ensemble: {', '.join(missing)}. "
            f"Add complete run metadata or pass explicit compatible ensemble members."
        )


def _all_candidate_runs() -> list[dict]:
    base = REPO_ROOT / "checkpoints"
    candidates = []
    if not base.exists():
        return candidates
    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            continue
        try:
            candidates.append(load_run_metadata(run_dir.name))
        except Exception:
            continue
    return candidates


def _format_run(meta: dict) -> str:
    return (
        f"{meta['run_id']} (model={meta.get('model_id')}, dataset={meta.get('dataset')}, "
        f"epochs={meta.get('epochs')}, n_train={meta.get('n_train')}, n_val={meta.get('n_val')}, "
        f"lr={meta.get('lr')}, batch_size={meta.get('batch_size')})"
    )


def resolve_deep_ensemble_members(
    *,
    config_model_id: str,
    config_dataset: str,
    run_id: str,
    ensemble_run_ids: list[str],
    ensemble_size: int,
) -> list[dict]:
    if ensemble_size < 2 and not ensemble_run_ids:
        raise ValueError("Deep ensemble must use at least 2 members.")

    if ensemble_run_ids:
        unique_ids = []
        seen = set()
        for rid in ensemble_run_ids:
            if rid in seen:
                continue
            seen.add(rid)
            unique_ids.append(rid)
        if len(unique_ids) < 2:
            raise ValueError("Deep ensemble must use at least 2 distinct run IDs.")
        members = [load_run_metadata(rid) for rid in unique_ids]
        for meta in members:
            _validate_metadata_complete(meta, f"Run '{meta['run_id']}'")
        signatures = {_recipe_signature(meta) for meta in members}
        if len(signatures) != 1:
            import os
            if os.environ.get("FORCE_ENSEMBLE", "0") != "1":
                raise ValueError(
                    "Explicit deep ensemble members are not compatible. All members must share "
                    "model_id, dataset, epochs, n_train, n_val, lr, and batch_size.\n"
                    + "\n".join(_format_run(meta) for meta in members)
                )
            # FORCE_ENSEMBLE=1: skip compatibility check (for diverse ensembles)
            import warnings
            warnings.warn("FORCE_ENSEMBLE=1: combining runs with different hyperparameters. Ensemble diversity may vary.")
        return members

    anchor_id = normalize_run_id(run_id)
    if anchor_id:
        anchor = load_run_metadata(anchor_id)
        _validate_metadata_complete(anchor, f"Run '{anchor_id}'")
        target_signature = _recipe_signature(anchor)
        candidates = [meta for meta in _all_candidate_runs() if _recipe_signature(meta) == target_signature]
        candidates.sort(key=lambda meta: (meta["run_id"] != anchor_id, -(meta.get("best_val_acc") or -1.0), meta["run_id"]))
    else:
        candidates = []
        for meta in _all_candidate_runs():
            if meta.get("model_id") != config_model_id or meta.get("dataset") != config_dataset:
                continue
            try:
                _validate_metadata_complete(meta, f"Run '{meta['run_id']}'")
            except ValueError:
                continue
            candidates.append(meta)
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for meta in candidates:
            grouped[_recipe_signature(meta)].append(meta)
        valid_groups = [group for group in grouped.values() if len(group) >= ensemble_size]
        if not valid_groups:
            raise ValueError(
                f"No valid deep ensemble group found for model={config_model_id} and dataset={config_dataset}. "
                "A valid group requires at least "
                f"{ensemble_size} runs with the same model_id, dataset, epochs, n_train, n_val, lr, and batch_size."
            )
        valid_groups.sort(
            key=lambda group: (
                -len(group),
                -max((meta.get("best_val_acc") or -1.0) for meta in group),
                max(meta["run_id"] for meta in group),
            )
        )
        candidates = sorted(valid_groups[0], key=lambda meta: (-(meta.get("best_val_acc") or -1.0), meta["run_id"]))

    if len(candidates) < ensemble_size:
        raise ValueError(
            f"Only found {len(candidates)} compatible run(s) for deep ensemble, but {ensemble_size} were requested. "
            "Train more matching runs or pass explicit ensemble_run_ids.\n"
            + "\n".join(_format_run(meta) for meta in candidates)
        )
    return candidates[:ensemble_size]


def list_deep_ensemble_candidates(
    *,
    config_model_id: str,
    config_dataset: str,
    run_id: str = "",
    ensemble_size: int = 2,
) -> dict:
    ensemble_size = max(2, int(ensemble_size))

    def _serialize_meta(meta: dict) -> dict:
        return {
            "run_id": meta.get("run_id"),
            "model_id": meta.get("model_id"),
            "dataset": meta.get("dataset"),
            "epochs": meta.get("epochs"),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
            "lr": meta.get("lr"),
            "batch_size": meta.get("batch_size"),
            "best_val_acc": meta.get("best_val_acc"),
        }

    anchor_id = normalize_run_id(run_id)
    if anchor_id:
        anchor = load_run_metadata(anchor_id)
        _validate_metadata_complete(anchor, f"Run '{anchor_id}'")
        signature = _recipe_signature(anchor)
        candidates = [meta for meta in _all_candidate_runs() if _recipe_signature(meta) == signature]
        candidates.sort(key=lambda meta: (meta["run_id"] != anchor_id, -(meta.get("best_val_acc") or -1.0), meta["run_id"]))
        return {
            "mode": "anchored",
            "anchor": _serialize_meta(anchor),
            "recipe": {
                "model_id": anchor.get("model_id"),
                "dataset": anchor.get("dataset"),
                "epochs": anchor.get("epochs"),
                "n_train": anchor.get("n_train"),
                "n_val": anchor.get("n_val"),
                "lr": anchor.get("lr"),
                "batch_size": anchor.get("batch_size"),
            },
            "recommended_run_ids": [meta["run_id"] for meta in candidates[:ensemble_size]],
            "candidates": [_serialize_meta(meta) for meta in candidates],
        }

    candidates = []
    for meta in _all_candidate_runs():
        if meta.get("model_id") != config_model_id or meta.get("dataset") != config_dataset:
            continue
        try:
            _validate_metadata_complete(meta, f"Run '{meta['run_id']}'")
        except ValueError:
            continue
        candidates.append(meta)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for meta in candidates:
        grouped[_recipe_signature(meta)].append(meta)

    groups = []
    for signature, members in grouped.items():
        members_sorted = sorted(members, key=lambda meta: (-(meta.get("best_val_acc") or -1.0), meta["run_id"]))
        sample = members_sorted[0]
        groups.append(
            {
                "recipe": {
                    "model_id": sample.get("model_id"),
                    "dataset": sample.get("dataset"),
                    "epochs": sample.get("epochs"),
                    "n_train": sample.get("n_train"),
                    "n_val": sample.get("n_val"),
                    "lr": sample.get("lr"),
                    "batch_size": sample.get("batch_size"),
                },
                "n_candidates": len(members_sorted),
                "recommended_run_ids": [meta["run_id"] for meta in members_sorted[:ensemble_size]],
                "candidates": [_serialize_meta(meta) for meta in members_sorted],
            }
        )

    groups.sort(
        key=lambda group: (
            -group["n_candidates"],
            -max((member.get("best_val_acc") or -1.0) for member in group["candidates"]),
            group["recipe"]["model_id"] or "",
        )
    )

    top_candidates = groups[0]["candidates"] if groups else []
    top_recommended = groups[0]["recommended_run_ids"] if groups else []
    return {
        "mode": "config",
        "anchor": None,
        "recipe": {"model_id": config_model_id, "dataset": config_dataset},
        "recommended_run_ids": top_recommended,
        "candidates": top_candidates,
        "groups": groups,
    }

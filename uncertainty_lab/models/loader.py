"""Load Hugging Face image classifiers or local .pt checkpoints (ViT-style)."""
from __future__ import annotations

from pathlib import Path

import torch

from uncertainty_lab.models.hf import load_hf_image_classifier


def create_hf_model(cfg: dict) -> torch.nn.Module:
    m_cfg = cfg["model"]
    model, _, _ = load_hf_image_classifier(
        model_id=m_cfg["model_id"],
        num_labels=int(m_cfg.get("num_labels", 2)),
        cache_dir=Path(m_cfg["cache_dir"]) if m_cfg.get("cache_dir") else None,
        dropout=float(m_cfg.get("dropout", 0.1)),
    )
    return model


def _load_state_into(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    elif isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def load_models_for_uncertainty(cfg: dict, method: str) -> list[torch.nn.Module]:
    m_cfg = cfg["model"]
    source = str(m_cfg.get("source", "huggingface")).lower()
    models: list[torch.nn.Module] = []

    if method == "deep_ensemble":
        paths: list[Path] = []
        for p in m_cfg.get("ensemble_checkpoints", []) or []:
            paths.append(Path(p).expanduser().resolve())
        if not paths:
            raise ValueError("deep_ensemble requires model.ensemble_checkpoints (list of .pt paths)")
        for ckpt in paths:
            if not ckpt.is_file():
                raise FileNotFoundError(f"Ensemble checkpoint not found: {ckpt}")
            m = create_hf_model(cfg)
            _load_state_into(m, ckpt)
            models.append(m)
        return models

    if source == "local":
        ckpt = Path(m_cfg["local_checkpoint"]).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"local_checkpoint not found: {ckpt}")
        m = create_hf_model(cfg)
        _load_state_into(m, ckpt)
        models.append(m)
        return models

    m = create_hf_model(cfg)
    ckpt_path = m_cfg.get("local_checkpoint")
    if ckpt_path:
        p = Path(ckpt_path).expanduser().resolve()
        if p.is_file():
            _load_state_into(m, p)
    models.append(m)
    return models

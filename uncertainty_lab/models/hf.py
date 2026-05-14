"""Hugging Face ViT-style image classifiers (binary); used by Uncertainty Lab and re-exported from ``models.load_model``."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hf_image_classifier(
    model_id: str = "google/vit-base-patch16-224",
    num_labels: int = 2,
    cache_dir: Optional[Path] = None,
    dropout: float = 0.1,
) -> tuple:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    def _rewrite_dropout_modules(module: torch.nn.Module, p: float) -> None:
        for child in module.modules():
            if isinstance(child, torch.nn.Dropout):
                child.p = p

    cache_dir = str(cache_dir) if cache_dir else None
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    if hasattr(processor, "size") and processor.size:
        image_size = (processor.size["height"], processor.size["width"])
    else:
        image_size = (224, 224)

    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    if hasattr(model, "classifier") and hasattr(model.classifier, "dropout"):
        model.classifier.dropout.p = dropout
    if hasattr(model, "config") and hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = dropout
    if hasattr(model, "config") and hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = dropout
    _rewrite_dropout_modules(model, dropout)
    model.eval()
    return model, processor, image_size


def download_model_only(model_id: str, cache_dir: Optional[Path] = None) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model_id, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(path)

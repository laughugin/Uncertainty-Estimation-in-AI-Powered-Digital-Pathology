"""Pluggable uncertainty methods (binary classification)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class UncertaintyMethod(ABC):
    """Each method produces averaged logits per sample (N, num_classes) for downstream metrics."""

    method_id: str = "base"

    @abstractmethod
    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return logits (N, C), labels (N,) as float64 / int64 numpy.

        ``on_batch(current, total_batches)`` is invoked after each batch (1-based index).
        """

    def predict_with_extras(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Like predict_logits but also returns an extras dict.

        Subclasses may override to supply ``member_probs`` (T, N, C) for proper
        predictive-entropy / mutual-information uncertainty scores.
        Default returns an empty extras dict.
        """
        logits, y = self.predict_logits(models, loader, device, mc_samples, on_batch=on_batch)
        return logits, y, {}


class ConfidenceMethod(UncertaintyMethod):
    method_id = "confidence"

    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = models[0]
        m.eval()
        logits_list, y_list = [], []
        n_batches = len(loader)
        with torch.inference_mode():
            for bi, (batch, labels) in enumerate(loader, start=1):
                batch = batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if batch.dim() == 3:
                    batch = batch.unsqueeze(0)
                with _autocast_context(device):
                    logits = m(pixel_values=batch).logits
                logits_list.append(logits.detach().cpu())
                y_list.append(labels.detach().cpu())
                if on_batch is not None:
                    on_batch(bi, n_batches)
        return _cat_logits_labels(logits_list, y_list)


class MCDropoutMethod(UncertaintyMethod):
    method_id = "mc_dropout"

    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        logits, y, _ = self.predict_with_extras(
            models, loader, device, mc_samples, on_batch=on_batch
        )
        return logits, y

    def predict_with_extras(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        m = models[0]
        T = max(2, int(mc_samples))
        logits_list, y_list, member_chunks = [], [], []
        n_batches = len(loader)
        for bi, (batch, labels) in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            # Enable only Dropout modules, keep everything else (LayerNorm etc.) in eval mode.
            # Averaging in logit space (not probability space) avoids Jensen's inequality bias:
            # mean(softmax(logits_i)) is systematically lower for the minority class than
            # softmax(mean(logits_i)), causing ~19pp accuracy drop in probability-space averaging.
            m.eval()
            for mod in m.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.training = True
            logits_stack = []
            with torch.inference_mode():
                for _ in range(T):
                    with _autocast_context(device):
                        lg = m(pixel_values=batch).logits
                    logits_stack.append(lg)
            logits_t = torch.stack(logits_stack, dim=0)  # (T, B, C)
            # Store per-pass probabilities for entropy/MI uncertainty metrics
            probs_stack_t = torch.softmax(logits_t, dim=2)
            # Use logit-space averaging for predictions (avoids Jensen's inequality)
            logits = logits_t.mean(dim=0)
            # probs used only for uncertainty metrics, not for predictions
            m.eval()
            logits_list.append(logits.detach().cpu())
            y_list.append(labels.detach().cpu())
            member_chunks.append(probs_stack_t.detach().cpu())
            if on_batch is not None:
                on_batch(bi, n_batches)
        logits_np, y_np = _cat_logits_labels(logits_list, y_list)
        # member_probs: (T, N, C) — per-pass probabilities for entropy/MI computation
        member_probs = torch.cat(member_chunks, dim=1).numpy().astype(np.float64)
        return logits_np, y_np, {"member_probs": member_probs}


class DeepEnsembleMethod(UncertaintyMethod):
    method_id = "deep_ensemble"

    def predict_logits(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        logits, y, _ = self.predict_with_extras(
            models, loader, device, mc_samples, on_batch=on_batch
        )
        return logits, y

    def predict_with_extras(
        self,
        models: list[torch.nn.Module],
        loader: DataLoader,
        device: torch.device,
        mc_samples: int = 30,
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        logits_list, y_list, member_chunks = [], [], []
        n_batches = len(loader)
        for bi, (batch, labels) in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)
            probs_members = []
            with torch.inference_mode():
                for m in models:
                    m.eval()
                    with _autocast_context(device):
                        lg = m(pixel_values=batch).logits
                    probs_members.append(torch.softmax(lg, dim=1))
            probs_stack_t = torch.stack(probs_members, dim=0)  # (M, B, C)
            probs = probs_stack_t.mean(dim=0)
            logits = torch.log(probs.clamp(min=1e-12))
            logits_list.append(logits.detach().cpu())
            y_list.append(labels.detach().cpu())
            member_chunks.append(probs_stack_t.detach().cpu())
            if on_batch is not None:
                on_batch(bi, n_batches)
        logits_np, y_np = _cat_logits_labels(logits_list, y_list)
        member_probs = torch.cat(member_chunks, dim=1).numpy().astype(np.float64)
        return logits_np, y_np, {"member_probs": member_probs}


def _cat_logits_labels(logits_list, y_list) -> tuple[np.ndarray, np.ndarray]:
    logits_np = torch.cat(logits_list, dim=0).numpy().astype(np.float64)
    y_np = torch.cat(y_list, dim=0).numpy().astype(np.int64)
    return logits_np, y_np


_METHODS: dict[str, type[UncertaintyMethod]] = {
    "confidence": ConfidenceMethod,
    "mc_dropout": MCDropoutMethod,
    "deep_ensemble": DeepEnsembleMethod,
}


def get_method(method_id: str) -> UncertaintyMethod:
    key = str(method_id).lower().strip()
    if key not in _METHODS:
        raise ValueError(f"Unknown uncertainty method '{method_id}'. Choose from {list(_METHODS)}")
    return _METHODS[key]()

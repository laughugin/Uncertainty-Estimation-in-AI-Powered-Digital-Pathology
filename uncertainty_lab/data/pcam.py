"""PCAM dataset wrapper (torchvision)."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Subset
from torchvision.datasets import PCAM


def load_pcam_subset(
    root: Path | str,
    split: str,
    transform,
    max_samples: int | None,
    seed: int,
) -> Subset:
    root = Path(root)
    full = PCAM(root=str(root), split=split, download=False, transform=transform)
    n_total = len(full)
    n_use = n_total if max_samples is None else min(max(1, max_samples), n_total)
    import numpy as np

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_use, replace=False).tolist()
    return Subset(full, indices)

"""Binary image folder dataset: root/class0 and root/class1 (names configurable)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BinaryFolderDataset(Dataset):
    """
    Expects ``root`` with exactly two class subdirectories.
    Labels are 0 and 1 in the order of ``class_dirs`` (sorted if None).
    """

    IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(
        self,
        root: Path | str,
        class_dirs: list[str] | None = None,
        transform: Callable | None = None,
    ):
        self.root = Path(root).resolve()
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        subdirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if class_dirs:
            dirs = [self.root / d for d in class_dirs]
            for d in dirs:
                if not d.is_dir():
                    raise FileNotFoundError(f"Class folder missing: {d}")
        else:
            if len(subdirs) != 2:
                raise ValueError(
                    f"Binary folder dataset needs exactly 2 subfolders under {self.root}, found {len(subdirs)}"
                )
            dirs = subdirs

        self.class_dirs = [d.name for d in dirs]
        self.samples: list[tuple[Path, int]] = []
        for label, d in enumerate(dirs):
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in self.IMG_EXT:
                    self.samples.append((p, label))

        if len(self.samples) < 2:
            raise ValueError(f"Need at least 2 images in {self.root}, found {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            x = self.transform(img)
        else:
            raise ValueError("transform is required for BinaryFolderDataset")
        return x, int(y)


def stratified_indices(labels: list[int], seed: int, val_frac: float, test_frac: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.array(labels, dtype=np.int64)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for c in (0, 1):
        cls = np.where(labels == c)[0]
        rng.shuffle(cls)
        n = len(cls)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        n_test = min(max(0, n_test), n - 1) if n > 1 else 0
        n_val = min(max(0, n_val), max(0, n - n_test - 1))
        te = cls[:n_test]
        va = cls[n_test : n_test + n_val]
        tr = cls[n_test + n_val :]
        test_idx.extend(te.tolist())
        val_idx.extend(va.tolist())
        train_idx.extend(tr.tolist())
    return {
        "train": np.array(train_idx, dtype=np.int64),
        "val": np.array(val_idx, dtype=np.int64),
        "test": np.array(test_idx, dtype=np.int64),
    }

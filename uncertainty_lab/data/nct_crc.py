"""NCT-CRC-HE-100K colorectal histology dataset loader.

Dataset: https://zenodo.org/record/1214456
  100,000 non-overlapping image patches (224x224 px) from H&E-stained histological slides.
  9 tissue classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM

Binary mode:
  label=1 → TUM (colorectal adenocarcinoma epithelium) — analogous to PCAM positive class
  label=0 → all other tissue classes

OOD / cross-domain mode:
  All samples from NCT-CRC are treated as a single OOD domain relative to PCAM-trained models.
  Label assignment still uses the binary mapping so OOD-detection AUROC can be computed
  alongside class-level accuracy.

Expected directory layout after download:
  <root>/NCT-CRC-HE-100K/
    ADI/  BACK/  DEB/  LYM/  MUC/  MUS/  NORM/  STR/  TUM/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset

# Canonical 9-class label mapping (alphabetical → int)
NCT_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
NCT_CLASS_TO_IDX = {c: i for i, c in enumerate(NCT_CLASSES)}

# Binary mapping: TUM=1, everything else=0
NCT_BINARY_POSITIVE = {"TUM"}


class NCTCRCDataset(Dataset):
    """Loads NCT-CRC-HE-100K as either 9-class or binary (TUM vs rest).

    Parameters
    ----------
    root : Path
        Parent directory containing the ``NCT-CRC-HE-100K`` folder.
    binary : bool
        If True, labels are 0/1 (TUM vs non-TUM). If False, labels are 0-8 (9-class).
    transform : callable, optional
        torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        binary: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        root = Path(root)
        dataset_dir = root / "NCT-CRC-HE-100K"
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"NCT-CRC-HE-100K directory not found at {dataset_dir}.\n"
                "Run: python data/download_nct_crc.py --root <root> to download."
            )
        self.dataset_dir = dataset_dir
        self.binary = binary
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._class_counts: dict[str, int] = {}

        for class_name in sorted(os.listdir(dataset_dir)):
            class_dir = dataset_dir / class_name
            if not class_dir.is_dir() or class_name not in NCT_CLASS_TO_IDX:
                continue
            if binary:
                label = 1 if class_name in NCT_BINARY_POSITIVE else 0
            else:
                label = NCT_CLASS_TO_IDX[class_name]
            count = 0
            for fname in sorted(class_dir.iterdir()):
                if fname.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    self.samples.append((fname, label))
                    count += 1
            self._class_counts[class_name] = count

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(self._class_counts)

    @property
    def n_classes(self) -> int:
        return 2 if self.binary else len(NCT_CLASSES)

    @property
    def labels(self) -> list[int]:
        return [s[1] for s in self.samples]


def load_nct_crc_subset(
    root: Path,
    *,
    transform,
    max_samples: int | None = None,
    seed: int = 42,
    binary: bool = True,
    balanced: bool = True,
) -> Dataset:
    """Return a (possibly subsampled, optionally balanced) NCT-CRC dataset.

    Parameters
    ----------
    root : Path
        Parent directory containing ``NCT-CRC-HE-100K``.
    transform : torchvision transform
    max_samples : int or None
        If set, randomly subsample to at most this many images.
    seed : int
        Random seed for reproducible subsampling.
    binary : bool
        Binary (TUM vs rest) or 9-class mode.
    balanced : bool
        If True and ``max_samples`` is set, draw equal numbers from each class.
    """
    ds = NCTCRCDataset(root, binary=binary, transform=transform)
    n = len(ds)
    if max_samples is None or max_samples >= n:
        return ds

    rng = np.random.default_rng(seed)
    labels = np.array(ds.labels, dtype=np.int64)
    n_classes = ds.n_classes

    if balanced:
        per_class = max(1, max_samples // n_classes)
        chosen: list[int] = []
        for c in range(n_classes):
            class_idx = np.where(labels == c)[0]
            k = min(per_class, len(class_idx))
            chosen.extend(rng.choice(class_idx, size=k, replace=False).tolist())
        chosen_arr = np.array(chosen[:max_samples])
    else:
        chosen_arr = rng.choice(n, size=max_samples, replace=False)

    return Subset(ds, chosen_arr.tolist())

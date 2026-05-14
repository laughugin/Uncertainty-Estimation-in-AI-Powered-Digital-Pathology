"""Binary classification dataset from CSV: image paths and 0/1 labels."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from uncertainty_lab.data.folder import stratified_indices


def _resolve_path(base_dir: Path | None, cell: str) -> Path:
    p = Path(cell.strip().strip('"').strip("'"))
    if p.is_absolute():
        return p.resolve()
    if base_dir is None:
        return p.resolve()
    return (base_dir / p).resolve()


def read_csv_samples(
    csv_path: Path,
    path_column: str = "path",
    label_column: str = "label",
    base_dir: Path | None = None,
) -> list[tuple[Path, int]]:
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if base_dir is None:
        base_dir = csv_path.parent
    else:
        base_dir = Path(base_dir).resolve()

    samples: list[tuple[Path, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row")
        pc, lc = path_column.strip(), label_column.strip()
        if pc not in reader.fieldnames or lc not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns {pc!r} and {lc!r}; got {reader.fieldnames}")
        for row in reader:
            raw_p = row.get(pc, "").strip()
            if not raw_p:
                continue
            path = _resolve_path(base_dir, raw_p)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            y = int(float(row[lc].strip()))
            if y not in (0, 1):
                raise ValueError(f"Label must be 0 or 1, got {y} for {path}")
            samples.append((path, y))
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 rows in {csv_path}")
    return samples


class BinaryCSVDataset(Dataset):
    """CSV with columns for path and binary label (see ``read_csv_samples``)."""

    def __init__(
        self,
        csv_path: Path | str,
        path_column: str = "path",
        label_column: str = "label",
        base_dir: Path | str | None = None,
        transform: Callable | None = None,
    ):
        csv_path = Path(csv_path).resolve()
        bd = Path(base_dir).resolve() if base_dir else None
        self.samples = read_csv_samples(csv_path, path_column, label_column, bd)
        self.transform = transform
        if self.transform is None:
            raise ValueError("transform is required for BinaryCSVDataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(y)


def build_csv_splits(cfg: dict, repo_root: Path) -> dict[str, np.ndarray]:
    if "_csv_splits" in cfg:
        return cfg["_csv_splits"]
    ds_cfg = cfg["dataset"]
    csv_path = Path(ds_cfg["csv_path"])
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()
    path_col = str(ds_cfg.get("path_column", "path"))
    label_col = str(ds_cfg.get("label_column", "label"))
    bd_raw = ds_cfg.get("csv_base_dir")
    base_dir = Path(bd_raw).resolve() if bd_raw else None
    if bd_raw and not Path(bd_raw).is_absolute():
        base_dir = (repo_root / bd_raw).resolve()

    samples = read_csv_samples(csv_path, path_col, label_col, base_dir)
    labels = [s[1] for s in samples]
    seed = int(ds_cfg.get("seed", cfg.get("seed", 42)))
    val_frac = float(ds_cfg.get("val_fraction", 0.15))
    test_frac = float(ds_cfg.get("test_fraction", 0.15))
    splits = stratified_indices(labels, seed=seed, val_frac=val_frac, test_frac=test_frac)
    cfg["_csv_splits"] = splits
    cfg["_csv_path"] = str(csv_path)
    cfg["_csv_n"] = len(samples)
    return splits

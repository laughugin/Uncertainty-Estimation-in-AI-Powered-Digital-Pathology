"""Build PyTorch DataLoaders from configuration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import PCAM

from uncertainty_lab.data.csv_dataset import BinaryCSVDataset, build_csv_splits
from uncertainty_lab.data.folder import BinaryFolderDataset, stratified_indices
from uncertainty_lab.data.nct_crc import load_nct_crc_subset
from uncertainty_lab.data.pcam import load_pcam_subset


def imagenet_transform(image_size: tuple[int, int]):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_dataset_kind(cfg: dict) -> str:
    return str(cfg.get("dataset", {}).get("type", "pcam")).lower()


def _resolve_num_workers(ds_cfg: dict[str, Any]) -> int:
    env_value = os.environ.get("ULAB_NUM_WORKERS")
    if env_value is not None and env_value.strip():
        return max(0, int(env_value))
    requested = int(ds_cfg.get("num_workers", 0) or 0)
    if requested > 0:
        return requested
    cpu_total = os.cpu_count() or 1
    return max(1, cpu_total - 1)


def _loader_kwargs(
    *,
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": max(1, batch_size),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, prefetch_factor)
    return kwargs


def build_folder_splits(cfg: dict, repo_root: Path) -> dict[str, np.ndarray]:
    """Return index arrays for train/val/test for folder dataset (cached on cfg['_folder_splits'])."""
    if "_folder_splits" in cfg:
        return cfg["_folder_splits"]
    ds_cfg = cfg["dataset"]
    root = Path(ds_cfg["root"])
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    class_dirs = ds_cfg.get("class_dirs")  # optional list of two dir names
    tmp = BinaryFolderDataset(root, class_dirs=class_dirs, transform=transforms.ToTensor())
    labels = [tmp.samples[i][1] for i in range(len(tmp))]
    seed = int(ds_cfg.get("seed", cfg.get("seed", 42)))
    val_frac = float(ds_cfg.get("val_fraction", 0.15))
    test_frac = float(ds_cfg.get("test_fraction", 0.15))
    splits = stratified_indices(labels, seed=seed, val_frac=val_frac, test_frac=test_frac)
    cfg["_folder_splits"] = splits
    cfg["_folder_root"] = str(root)
    cfg["_folder_class_dirs"] = tmp.class_dirs
    return splits


def build_eval_loader(cfg: dict, repo_root: Path, split: str | None = None) -> tuple[DataLoader, dict[str, Any]]:
    """
    Build a single DataLoader for evaluation and metadata for the run output.
    """
    ds_cfg = cfg["dataset"]
    split = split or str(ds_cfg.get("eval_split", "test"))
    image_size = tuple(ds_cfg.get("image_size", [224, 224]))
    transform = imagenet_transform(image_size)
    batch_size = int(ds_cfg.get("batch_size", 32))
    max_samples = ds_cfg.get("max_eval_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    seed = int(ds_cfg.get("seed", cfg.get("seed", 42)))
    nw = _resolve_num_workers(ds_cfg)
    pin = torch.cuda.is_available()
    prefetch_factor = int(ds_cfg.get("prefetch_factor", 4) or 4)

    kind = get_dataset_kind(cfg)
    meta: dict[str, Any] = {"dataset_type": kind, "split": split}

    if kind == "pcam":
        root = Path(ds_cfg["root"])
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        loader_ds = load_pcam_subset(root, split=split, transform=transform, max_samples=max_samples, seed=seed)
        meta["n_samples"] = len(loader_ds)
        meta["root"] = str(root)
    elif kind == "folder":
        root = Path(ds_cfg["root"])
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        class_dirs = ds_cfg.get("class_dirs")
        full = BinaryFolderDataset(root, class_dirs=class_dirs, transform=transform)
        splits = build_folder_splits(cfg, repo_root)
        if split not in splits:
            raise ValueError(f"Unknown split '{split}' for folder dataset (use train, val, test)")
        idx = splits[split]
        if max_samples is not None and len(idx) > max_samples:
            rng = np.random.default_rng(seed + 17)
            idx = rng.choice(idx, size=max_samples, replace=False)
        loader_ds = Subset(full, idx.tolist())
        meta["n_samples"] = len(loader_ds)
        meta["root"] = str(root)
        meta["class_dirs"] = full.class_dirs
    elif kind == "nct_crc":
        root = Path(ds_cfg["root"])
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        binary = bool(ds_cfg.get("binary", True))
        balanced = bool(ds_cfg.get("balanced_subsample", True))
        loader_ds = load_nct_crc_subset(
            root,
            transform=transform,
            max_samples=max_samples,
            seed=seed,
            binary=binary,
            balanced=balanced,
        )
        meta["n_samples"] = len(loader_ds)
        meta["root"] = str(root)
        meta["binary"] = binary
    elif kind == "csv":
        ds_cfg = cfg["dataset"]
        csv_path = Path(ds_cfg["csv_path"])
        if not csv_path.is_absolute():
            csv_path = (repo_root / csv_path).resolve()
        path_col = str(ds_cfg.get("path_column", "path"))
        label_col = str(ds_cfg.get("label_column", "label"))
        bd_raw = ds_cfg.get("csv_base_dir")
        base_dir = None
        if bd_raw:
            base_dir = Path(bd_raw)
            if not base_dir.is_absolute():
                base_dir = (repo_root / base_dir).resolve()
            else:
                base_dir = base_dir.resolve()
        splits = build_csv_splits(cfg, repo_root)
        if split not in splits:
            raise ValueError(f"Unknown split '{split}' for csv dataset (use train, val, test)")
        idx = splits[split]
        if max_samples is not None and len(idx) > max_samples:
            rng = np.random.default_rng(seed + 17)
            idx = rng.choice(idx, size=max_samples, replace=False)
        full = BinaryCSVDataset(csv_path, path_col, label_col, base_dir, transform=transform)
        loader_ds = Subset(full, idx.tolist())
        meta["n_samples"] = len(loader_ds)
        meta["csv_path"] = str(csv_path)
    else:
        raise ValueError(f"Unsupported dataset.type: {kind}")

    loader = DataLoader(
        **_loader_kwargs(
            dataset=loader_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            prefetch_factor=prefetch_factor,
        )
    )
    return loader, meta


def build_train_val_loaders(cfg: dict, repo_root: Path) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Train and validation loaders for fine-tuning."""
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg.get("train", {})
    image_size = tuple(ds_cfg.get("image_size", [224, 224]))
    transform = imagenet_transform(image_size)
    batch_size = int(ds_cfg.get("batch_size", 32))
    nw = _resolve_num_workers(ds_cfg)
    pin = torch.cuda.is_available()
    prefetch_factor = int(ds_cfg.get("prefetch_factor", 4) or 4)
    seed = int(ds_cfg.get("seed", cfg.get("seed", 42)))
    kind = get_dataset_kind(cfg)
    meta: dict[str, Any] = {"dataset_type": kind}

    n_train = tr_cfg.get("n_train")
    n_val = tr_cfg.get("n_val")

    if kind == "pcam":
        root = Path(ds_cfg["root"])
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        train_full = PCAM(root=str(root), split="train", download=False, transform=transform)
        val_full = PCAM(root=str(root), split="val", download=False, transform=transform)
        nt = len(train_full) if n_train is None else min(int(n_train), len(train_full))
        nv = len(val_full) if n_val is None else min(int(n_val), len(val_full))
        train_ds = Subset(train_full, range(nt))
        val_ds = Subset(val_full, range(nv))
        meta.update({"n_train": nt, "n_val": nv, "root": str(root)})
    elif kind == "folder":
        root = Path(ds_cfg["root"])
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        class_dirs = ds_cfg.get("class_dirs")
        full = BinaryFolderDataset(root, class_dirs=class_dirs, transform=transform)
        splits = build_folder_splits(cfg, repo_root)
        tr_idx = splits["train"].tolist()
        va_idx = splits["val"].tolist()
        if n_train is not None:
            tr_idx = tr_idx[: int(n_train)]
        if n_val is not None:
            va_idx = va_idx[: int(n_val)]
        train_ds = Subset(full, tr_idx)
        val_ds = Subset(full, va_idx)
        meta.update({"n_train": len(train_ds), "n_val": len(val_ds), "root": str(root)})
    elif kind == "csv":
        ds_cfg = cfg["dataset"]
        csv_path = Path(ds_cfg["csv_path"])
        if not csv_path.is_absolute():
            csv_path = (repo_root / csv_path).resolve()
        path_col = str(ds_cfg.get("path_column", "path"))
        label_col = str(ds_cfg.get("label_column", "label"))
        bd_raw = ds_cfg.get("csv_base_dir")
        base_dir = None
        if bd_raw:
            base_dir = Path(bd_raw)
            if not base_dir.is_absolute():
                base_dir = (repo_root / base_dir).resolve()
            else:
                base_dir = base_dir.resolve()
        splits = build_csv_splits(cfg, repo_root)
        tr_idx = splits["train"].tolist()
        va_idx = splits["val"].tolist()
        if n_train is not None:
            tr_idx = tr_idx[: int(n_train)]
        if n_val is not None:
            va_idx = va_idx[: int(n_val)]
        full = BinaryCSVDataset(csv_path, path_col, label_col, base_dir, transform=transform)
        train_ds = Subset(full, tr_idx)
        val_ds = Subset(full, va_idx)
        meta.update({"n_train": len(train_ds), "n_val": len(val_ds), "csv_path": str(csv_path)})
    else:
        raise ValueError(f"Unsupported dataset.type for training: {kind}")

    train_loader = DataLoader(
        **_loader_kwargs(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            prefetch_factor=prefetch_factor,
        )
    )
    val_loader = DataLoader(
        **_loader_kwargs(
            dataset=val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            prefetch_factor=prefetch_factor,
        )
    )
    return train_loader, val_loader, meta

#!/usr/bin/env python3
"""
Download and prepare datasets for digital pathology uncertainty estimation.

Thesis defaults:
- PCAM (Patch Camelyon) via torchvision (patch dataset, binary labels).

Additional trusted datasets can be downloaded for future experiments even if
they are not yet supported by the current training/evaluation pipeline.
"""
from pathlib import Path
import argparse
import os
import urllib.request
import zipfile


def download_pcam(root: Path, splits: tuple = ("train", "val", "test")) -> None:
    """Download PCAM dataset into root/pcam. Requires torchvision>=0.25, h5py, gdown."""
    try:
        from torchvision.datasets import PCAM
    except ImportError as e:
        raise ImportError(
            "PCAM requires torchvision>=0.25. Install with: pip install torchvision>=0.25 h5py gdown"
        ) from e

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"Downloading PCAM split: {split} ...")
        try:
            PCAM(root=str(root), split=split, download=True)
        except Exception as e:
            print(f"Warning: {split} download failed: {e}")
            print("  You can retry or download manually from:")
            print("  https://patchcamelyon.grand-challenge.org/Download/")
            raise
    print("PCAM download complete.")


def _download_url(url: str, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists and is non-empty, assume it's already downloaded.
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Using cached download: {out_path}")
        return

    print(f"Downloading: {url}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    if tmp_path.exists():
        tmp_path.unlink()

    with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as f:
        # Stream to disk to avoid large memory use.
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    os.replace(tmp_path, out_path)
    print(f"Downloaded to: {out_path}")


def download_nct_crc_he_100k(root: Path) -> None:
    """
    Download NCT-CRC-HE-100K (patch dataset) from Zenodo.

    Note: This repo currently does not implement a dataset loader for NCT yet,
    but downloading/extracting the data is useful for later thesis steps.
    """
    root = Path(root).resolve()
    out_dir = root / "nct_crc_he_100k"
    done_marker = out_dir / ".download_complete"
    out_dir.mkdir(parents=True, exist_ok=True)

    if done_marker.exists():
        print("NCT-CRC-HE-100K already downloaded (marker present).")
        return

    # Zenodo provides direct file links that are scriptable.
    zip_url = "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip"
    zip_path = out_dir / "NCT-CRC-HE-100K.zip"

    _download_url(zip_url, zip_path)

    # Extract if extracted folder not present.
    with zipfile.ZipFile(zip_path, "r") as zf:
        top_level_names = {n.split("/")[0] for n in zf.namelist() if n.strip()}
    # Typical top-level dir is "NCT-CRC-HE-100K"
    extracted_dirs = [out_dir / name for name in top_level_names if name and name != ""]
    extracted_ok = any(d.exists() for d in extracted_dirs)

    if not extracted_ok:
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        print("Extraction complete.")
    else:
        print("Extraction seems already done; skipping.")

    done_marker.write_text("ok\n")


def main():
    parser = argparse.ArgumentParser(description="Download pathology datasets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Root directory for dataset storage",
    )
    parser.add_argument(
        "--dataset",
        choices=["pcam", "nct_crc_he_100k", "all"],
        default="pcam",
        help="Dataset to download",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="PCAM splits to download",
    )
    args = parser.parse_args()

    if args.dataset in ("pcam", "all"):
        download_pcam(args.root, tuple(args.splits))

    if args.dataset in ("nct_crc_he_100k", "all"):
        download_nct_crc_he_100k(args.root)

    print(f"Data directory: {args.root.resolve()}")


if __name__ == "__main__":
    main()

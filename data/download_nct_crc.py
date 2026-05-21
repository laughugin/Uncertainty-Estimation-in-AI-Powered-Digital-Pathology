#!/usr/bin/env python3
"""Download NCT-CRC-HE-100K colorectal histology dataset from Zenodo.

Dataset: https://zenodo.org/record/1214456
  Kather et al., "100,000 histological images of human colorectal cancer and healthy tissue"
  ~785 MB zip archive, 100,000 PNG patches (224x224 px), 9 tissue classes.

Usage:
    python data/download_nct_crc.py --root data/raw
    python data/download_nct_crc.py --root data/raw --skip-verify
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip"
ZIP_FILENAME = "NCT-CRC-HE-100K.zip"
EXPECTED_CLASSES = {"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"}


def download_file(url: str, dest: Path) -> None:
    try:
        import urllib.request

        print(f"Downloading {url}")
        print(f"  → {dest}")
        print("  (this may take several minutes — ~785 MB)")

        def _progress(count, block_size, total_size):
            if total_size <= 0:
                return
            pct = min(100.0, count * block_size / total_size * 100)
            mb_done = count * block_size / 1e6
            mb_total = total_size / 1e6
            print(f"\r  {pct:5.1f}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            "\nManual download instructions:\n"
            f"  1. Go to {url}\n"
            f"  2. Save the file as: {dest}\n"
            f"  3. Re-run this script with --skip-download",
            file=sys.stderr,
        )
        sys.exit(1)


def verify_structure(extract_dir: Path) -> bool:
    dataset_dir = extract_dir / "NCT-CRC-HE-100K"
    if not dataset_dir.is_dir():
        return False
    found = {d.name for d in dataset_dir.iterdir() if d.is_dir()}
    missing = EXPECTED_CLASSES - found
    if missing:
        print(f"Warning: missing class directories: {missing}")
        return False
    total = sum(len(list(d.iterdir())) for d in dataset_dir.iterdir() if d.is_dir())
    print(f"  Found {total:,} image files across {len(found)} class directories.")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Download NCT-CRC-HE-100K dataset")
    p.add_argument("--root", default="data/raw", help="Destination directory (default: data/raw)")
    p.add_argument("--skip-download", action="store_true", help="Skip download, only extract")
    p.add_argument("--skip-verify", action="store_true", help="Skip post-extraction verification")
    p.add_argument("--keep-zip", action="store_true", help="Keep .zip after extraction")
    args = p.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / ZIP_FILENAME
    dataset_dir = root / "NCT-CRC-HE-100K"

    if dataset_dir.is_dir() and not args.skip_verify:
        print(f"Dataset directory already exists: {dataset_dir}")
        if verify_structure(root):
            print("Structure looks correct — skipping download and extraction.")
            return
        print("Structure check failed — re-downloading.")

    if not args.skip_download:
        if zip_path.exists():
            print(f"ZIP already present: {zip_path} — skipping download.")
        else:
            download_file(ZENODO_URL, zip_path)
    else:
        if not zip_path.exists():
            print(f"--skip-download set but {zip_path} not found.", file=sys.stderr)
            sys.exit(1)

    print(f"Extracting {zip_path} → {root} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    print("Extraction complete.")

    if not args.skip_verify:
        if verify_structure(root):
            print("Verification passed.")
        else:
            print("Verification failed — check the extracted directory.", file=sys.stderr)
            sys.exit(1)

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"Removed {zip_path}")

    print(f"\nDataset ready at: {dataset_dir}")
    print("Use in config:  dataset.type: nct_crc  dataset.root: data/raw")


if __name__ == "__main__":
    main()

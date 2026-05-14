#!/usr/bin/env python3
"""
Load (and optionally download) image classification models from Hugging Face.
Suitable for pathology vs. non-pathology binary classification.

Implementation lives in ``uncertainty_lab.models.hf`` for a single source of truth.
"""
from uncertainty_lab.models.hf import download_model_only, get_device, load_hf_image_classifier

__all__ = ["get_device", "load_hf_image_classifier", "download_model_only"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/vit-base-patch16-224", help="Hugging Face model ID")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Cache directory for downloads")
    parser.add_argument("--download_only", action="store_true", help="Only download, do not load")
    args = parser.parse_args()

    if args.download_only:
        path = download_model_only(args.model_id, args.cache_dir)
        print(f"Cached to: {path}")
    else:
        model, processor, size = load_hf_image_classifier(args.model_id, cache_dir=args.cache_dir)
        print(f"Model: {args.model_id}")
        print(f"Image size: {size}")
        print(f"Processor: {type(processor).__name__}")
        nparams = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {nparams:,}")

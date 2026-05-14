#!/usr/bin/env python3
"""
Train the image classifier on PCAM. Reads configs/default.yaml.
CLI overrides: --epochs, --n_train, --n_val, --lr, --batch_size.
Saves to checkpoints/run_<timestamp>/ with best.pt and metrics.json.

Note: When training on CUDA, DataLoader workers default to 0 because forking
worker processes after the parent has initialized CUDA often crashes or deadlocks
(PyTorch/Linux). Override with TRAIN_NUM_WORKERS if you use spawn start method.
"""
import argparse
import json
import os
import re
import sys
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import PCAM
from torchvision import transforms
import yaml


def _configure_cuda_backends() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _resolve_num_workers(cfg: dict) -> int:
    nw_env = os.environ.get("TRAIN_NUM_WORKERS")
    if nw_env is not None and str(nw_env).strip() != "":
        return max(0, int(nw_env))
    requested = int(cfg["data"].get("num_workers", 0) or 0)
    if requested > 0:
        return requested
    cpu_total = os.cpu_count() or 1
    return max(1, cpu_total - 1)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def get_config():
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="pcam", choices=["pcam"], help="Dataset to use for training")
    p.add_argument("--model_id", type=str, default=None, help="Hugging Face image model id")
    p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    p.add_argument("--n_train", type=int, default=None, help="Max training samples (subset)")
    p.add_argument("--n_val", type=int, default=None, help="Max validation samples (subset)")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=None, help="Batch size")
    p.add_argument("--run_dir", type=str, default=None, help="Output run dir (default: checkpoints/run_<timestamp>)")
    p.add_argument(
        "--uncertainty-lab-config",
        type=str,
        default=None,
        help="If set, train via uncertainty_lab from this YAML (merged with configs/uncertainty_lab_default.yaml) and exit.",
    )
    return p.parse_args()


def _slug_model_id(model_id: str) -> str:
    """Create short filesystem-safe slug from HF model id."""
    if not model_id:
        return "model"
    slug = model_id.strip().lower().replace("/", "-")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    # Keep names readable but bounded.
    return slug[:36].strip("-") or "model"


def _fmt_lr(lr: float) -> str:
    """Format learning rate compactly for run names."""
    try:
        return f"{float(lr):.0e}".replace("+0", "").replace("-0", "-")
    except Exception:
        return "lr"


def _write_metrics(run_dir: Path, payload: dict) -> None:
    """Atomically write metrics JSON so partial progress survives crashes/kill."""
    tmp = run_dir / "metrics.json.tmp"
    out = run_dir / "metrics.json"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out)


def main():
    # Line-buffer stdout when redirected (e.g. nohup ... > train.log).
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    if args.uncertainty_lab_config:
        from uncertainty_lab.config import load_config
        from uncertainty_lab.pipeline.run import run_pipeline

        lab_cfg = load_config(args.uncertainty_lab_config, repo_root=REPO_ROOT)
        lab_cfg.setdefault("pipeline", {})["mode"] = "train"
        lab_cfg.setdefault("run", {})["repo_root"] = str(REPO_ROOT)
        r = run_pipeline(lab_cfg)
        print(json.dumps(r, indent=2))
        return

    cfg = get_config()
    # Override from CLI
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.n_train is not None:
        cfg["_n_train"] = args.n_train
    if args.n_val is not None:
        cfg["_n_val"] = args.n_val
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.model_id is not None and str(args.model_id).strip():
        cfg["model"]["model_id"] = str(args.model_id).strip()

    torch.manual_seed(cfg.get("seed", 42))
    _configure_cuda_backends()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data (dataset selection: currently only PCAM)
    dataset_id = args.dataset or cfg.get("data", {}).get("dataset", "pcam")
    if dataset_id != "pcam":
        raise ValueError(f"Unsupported dataset: {dataset_id}. Only 'pcam' is supported.")
    root = REPO_ROOT / cfg["data"]["root"]
    image_size = tuple(cfg["data"]["image_size"])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds_full = PCAM(root=str(root), split="train", download=False, transform=transform)
    val_ds_full = PCAM(root=str(root), split="val", download=False, transform=transform)
    n_train = cfg.get("_n_train")
    n_val = cfg.get("_n_val")
    if n_train is None:
        n_train = min(len(train_ds_full), 10_000)
    else:
        n_train = min(n_train, len(train_ds_full))
    if n_val is None:
        n_val = min(len(val_ds_full), 2_000)
    else:
        n_val = min(n_val, len(val_ds_full))
    train_ds = Subset(train_ds_full, range(n_train))
    val_ds = Subset(val_ds_full, range(n_val))
    print(f"Dataset: {dataset_id}. Train samples: {n_train}, Val samples: {n_val}")

    # Run output dir (named by key hyperparams for easier experiment tracking)
    base_ckpt = REPO_ROOT / cfg["train"]["checkpoint_dir"]
    base_ckpt.mkdir(parents=True, exist_ok=True)
    epochs = int(cfg["train"]["epochs"])
    bs = int(cfg["data"]["batch_size"])
    lr = float(cfg["train"]["lr"])
    model_slug = _slug_model_id(cfg["model"]["model_id"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_name = (
        f"run_{dataset_id}_{model_slug}"
        f"_e{epochs}_nt{n_train}_nv{n_val}_bs{bs}_lr{_fmt_lr(lr)}_{ts}"
    )
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = base_ckpt / auto_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    num_workers = _resolve_num_workers(cfg)
    pin_mem = device.type == "cuda"
    use_amp = device.type == "cuda"
    prefetch_factor = max(2, int(cfg["data"].get("prefetch_factor", 4) or 4))
    print(f"DataLoader num_workers={num_workers} pin_memory={pin_mem} amp={use_amp} device={device}")

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs = {"persistent_workers": True, "prefetch_factor": prefetch_factor}
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        **loader_kwargs,
    )

    # Model
    from models.load_model import load_hf_image_classifier

    metrics_base = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "dataset": dataset_id,
        "model_id": cfg["model"]["model_id"],
        "epochs": epochs,
        "n_train": n_train,
        "n_val": n_val,
        "lr": cfg["train"]["lr"],
        "batch_size": cfg["data"]["batch_size"],
        "num_workers": num_workers,
        "device": str(device),
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "best_val_acc": 0.0,
        "best_epoch": None,
        "history": [],
        "error": None,
    }
    _write_metrics(run_dir, metrics_base)

    try:
        model, _, _ = load_hf_image_classifier(
            model_id=cfg["model"]["model_id"],
            num_labels=cfg["model"]["num_labels"],
            dropout=cfg["model"].get("dropout", 0.1),
        )
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        epochs = cfg["train"]["epochs"]
        best_acc = 0.0
        history = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (pixel_values, labels) in enumerate(train_loader):
                pixel_values = pixel_values.to(device, non_blocking=pin_mem)
                labels = labels.to(device, non_blocking=pin_mem)
                if pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                opt.zero_grad(set_to_none=True)
                with _autocast_context(device, use_amp):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else criterion(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                train_loss += loss.item()
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1} batch {batch_idx+1} loss {loss.item():.4f}", flush=True)

            model.eval()
            correct = 0
            total = 0
            with torch.inference_mode():
                for pixel_values, labels in val_loader:
                    pixel_values = pixel_values.to(device, non_blocking=pin_mem)
                    labels = labels.to(device, non_blocking=pin_mem)
                    if pixel_values.dim() == 3:
                        pixel_values = pixel_values.unsqueeze(0)
                    with _autocast_context(device, use_amp):
                        out = model(pixel_values=pixel_values)
                    pred = out.logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total if total else 0
            avg_loss = train_loss / len(train_loader)
            history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 4), "val_acc": round(acc, 4)})
            print(f"Epoch {epoch+1} train_loss {avg_loss:.4f} val_acc {acc:.4f}", flush=True)

            # Last epoch checkpoint (always recoverable)
            last_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_acc": acc,
                "config": cfg,
            }
            torch.save(last_state, run_dir / "last.pt")

            if acc > best_acc:
                best_acc = acc
                torch.save(last_state, run_dir / "best.pt")
                print(f"  Saved best checkpoint (acc {acc:.4f})", flush=True)

            _write_metrics(
                run_dir,
                {
                    **metrics_base,
                    "status": "running",
                    "best_val_acc": round(best_acc, 4),
                    "best_epoch": max(
                        (h["epoch"] for h in history if h["val_acc"] == round(best_acc, 4)),
                        default=None,
                    ),
                    "history": history,
                    "last_epoch_completed": epoch + 1,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

        # Save final metrics for this run
        metrics = {
            **metrics_base,
            "status": "completed",
            "best_val_acc": round(best_acc, 4),
            "best_epoch": max(
                (h["epoch"] for h in history if h["val_acc"] == round(best_acc, 4)),
                default=epochs,
            ),
            "history": history,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_metrics(run_dir, metrics)
        print(f"Done. Best val accuracy: {best_acc:.4f}. Run dir: {run_dir}", flush=True)
        print(f"Metrics saved to {run_dir / 'metrics.json'}", flush=True)
    except Exception as e:
        err_payload = {
            **metrics_base,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "failed_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_metrics(run_dir, err_payload)
        print(f"Training failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

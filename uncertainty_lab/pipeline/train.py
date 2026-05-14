"""Fine-tune HF image classifier (binary); saves best.pt compatible with evaluation."""
from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from uncertainty_lab.data.factory import build_train_val_loaders
from uncertainty_lab.device import resolve_device
from uncertainty_lab.models.loader import create_hf_model


def _configure_cuda_backends() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _write_metrics(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def run_training(config: dict, run_dir: Path, repo_root: Path) -> Path:
    """Train model; write ``run_dir/checkpoint/best.pt`` and ``run_dir/train_metrics.json``."""
    _configure_cuda_backends()
    device = resolve_device(config)
    train_loader, val_loader, meta = build_train_val_loaders(config, repo_root)
    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tr = config.get("train", {})
    epochs = int(tr.get("epochs", 3))
    lr = float(tr.get("lr", 2e-5))

    model = create_hf_model(config)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    pin_mem = device.type == "cuda"
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics_base = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": meta,
        "epochs": epochs,
        "device": str(device),
    }
    _write_metrics(run_dir / "train_metrics.json", metrics_base)

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

        model.eval()
        correct = total = 0
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
        acc = correct / total if total else 0.0
        avg_loss = train_loss / max(1, len(train_loader))
        history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 4), "val_acc": round(acc, 4)})

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_acc": acc,
            "config": config,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if acc > best_acc:
            best_acc = acc
            torch.save(state, ckpt_dir / "best.pt")

    _write_metrics(
        run_dir / "train_metrics.json",
        {
            **metrics_base,
            "status": "completed",
            "best_val_acc": round(best_acc, 4),
            "history": history,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return ckpt_dir / "best.pt"

#!/usr/bin/env python3
"""
Run inference on random test samples and compare prediction vs ground truth.
Outputs JSON: list of {idx, pred, prob, label, correct}.
Use either trained checkpoint (checkpoints/best.pt) or pretrained HF model.
"""
import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torchvision.datasets import PCAM
from torchvision import transforms
import yaml


def get_config():
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def load_model_and_processor(cfg, device):
    """Load model from checkpoint if exists, else from HF. Return model, processor, image_size."""
    from models.load_model import load_hf_image_classifier, get_device
    model, processor, image_size = load_hf_image_classifier(
        model_id=cfg["model"]["model_id"],
        num_labels=cfg["model"]["num_labels"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
    checkpoint_dir = REPO_ROOT / cfg["train"]["checkpoint_dir"]
    ckpt = checkpoint_dir / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, processor, image_size


def main():
    cfg = get_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = cfg.get("evaluation", {}).get("random_test_n", 24)
    n = min(n, 100)

    root = REPO_ROOT / cfg["data"]["root"]
    image_size = tuple(cfg["data"]["image_size"])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = PCAM(root=str(root), split="test", download=False, transform=transform)
    size = len(test_ds)
    indices = random.sample(range(size), min(n, size))

    model, _, _ = load_model_and_processor(cfg, device)
    results = []
    with torch.no_grad():
        for idx in indices:
            img, label = test_ds[idx]
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(device)
            out = model(pixel_values=img)
            logits = out.logits
            probs = torch.softmax(logits, dim=1)
            prob, pred = probs.max(dim=1)
            pred = pred.item()
            prob = prob.item()
            label = int(label)
            correct = pred == label
            results.append({
                "idx": idx,
                "pred": pred,
                "prob": round(prob, 4),
                "label": label,
                "correct": correct,
            })

    out_path = REPO_ROOT / "evaluation" / "random_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n": len(results), "results": results}, f, indent=2)
    print(json.dumps({"n": len(results), "results": results}, indent=2))
    print(f"Saved to {out_path}", file=sys.stderr)
    correct_count = sum(1 for r in results if r["correct"])
    print(f"Accuracy on this sample: {correct_count}/{len(results)} = {100*correct_count/len(results):.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()

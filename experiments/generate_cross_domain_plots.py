"""
Generate reliability diagrams and risk-coverage curves for cross-domain evaluation.
Runs inference on PCAM test and NCT-CRC, plots both domains side by side per method.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import random, h5py, os

REPO_ROOT   = Path(__file__).resolve().parent.parent
CHECKPOINT  = REPO_ROOT / "checkpoints" / "best.pt"
PCAM_ROOT   = REPO_ROOT / "data" / "raw" / "pcam"
NCT_ROOT    = REPO_ROOT / "data" / "raw" / "NCT-CRC-HE-100K"
OUT_FIG     = REPO_ROOT / "thesis" / "figures" / "cross_domain_reliability_rc.png"
N_SAMPLES   = 256
N_MC        = 10
BATCH_SIZE  = 64
SEED        = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(dropout_train=False):
    from uncertainty_lab.models.hf import load_hf_image_classifier
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model, _, _ = load_hf_image_classifier("google/vit-base-patch16-224", num_labels=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    if dropout_train:
        model.train()
        for m in model.modules():
            if not isinstance(m, torch.nn.Dropout):
                m.eval()
    else:
        model.eval()
    return model


# ── Data ─────────────────────────────────────────────────────────────────────

def load_pcam(n):
    xf = PCAM_ROOT / "camelyonpatch_level_2_split_test_x.h5"
    yf = PCAM_ROOT / "camelyonpatch_level_2_split_test_y.h5"
    with h5py.File(xf) as fx, h5py.File(yf) as fy:
        total = len(fx["x"])
        idx = sorted(random.sample(range(total), n))
        imgs   = [Image.fromarray(fx["x"][i]) for i in idx]
        labels = [int(fy["y"][i].flat[0]) for i in idx]
    return imgs, labels


def load_nct(n):
    """NCT-CRC: label 1 = TUM (tumour), everything else = 0. Balanced 50/50."""
    all_paths = list(NCT_ROOT.rglob("*.tif")) + list(NCT_ROOT.rglob("*.png"))
    tum_paths  = [p for p in all_paths if "TUM" in str(p).upper()]
    rest_paths = [p for p in all_paths if "TUM" not in str(p).upper()]
    per_class = n // 2
    sel_tum  = random.sample(tum_paths,  min(per_class, len(tum_paths)))
    sel_rest = random.sample(rest_paths, min(per_class, len(rest_paths)))
    sel = sel_tum + sel_rest
    imgs   = [Image.open(p).convert("RGB") for p in sel]
    labels = [1 if "TUM" in str(p).upper() else 0 for p in sel]
    return imgs, labels


def preprocess(imgs, processor):
    return processor(images=imgs, return_tensors="pt")


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_confidence(model, imgs, processor):
    probs_all = []
    for i in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[i:i+BATCH_SIZE]
        inputs = {k: v.to(DEVICE) for k, v in preprocess(batch, processor).items()}
        logits = model(**inputs).logits
        probs_all.append(torch.softmax(logits, dim=-1).cpu().numpy())
        print(f"  {min(i+BATCH_SIZE, len(imgs))}/{len(imgs)}", end="\r")
    print()
    return np.vstack(probs_all)   # (N, 2)


def run_mc_dropout(model, imgs, processor, T=N_MC):
    """Run T stochastic passes; return mean softmax probs."""
    all_passes = []
    for t in range(T):
        probs = run_confidence(model, imgs, processor)
        all_passes.append(probs)
    return np.mean(all_passes, axis=0)   # (N, 2)


# ── Metrics ───────────────────────────────────────────────────────────────────

def reliability_bins(probs, labels, n_bins=10):
    """Return (mean_conf, mean_acc, counts) per bin."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == np.array(labels)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    mean_conf, mean_acc, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        mean_conf.append(confs[mask].mean())
        mean_acc.append(correct[mask].mean())
        counts.append(mask.sum())
    return np.array(mean_conf), np.array(mean_acc), np.array(counts)


def risk_coverage(probs, labels):
    """Return (coverage, risk) arrays for threshold sweep."""
    uncertainty = 1 - probs.max(axis=1)
    preds   = probs.argmax(axis=1)
    correct = (preds == np.array(labels))
    order   = np.argsort(uncertainty)           # most confident first
    coverages, risks = [], []
    for k in range(10, len(order) + 1, max(1, len(order) // 100)):
        sel = order[:k]
        coverages.append(k / len(order))
        risks.append(1 - correct[sel].mean())
    return np.array(coverages), np.array(risks)


# ── Plot ─────────────────────────────────────────────────────────────────────

COLORS  = {"PCAM": "#2166ac", "NCT-CRC": "#d6604d"}
METHODS = ["Confidence", "MC Dropout"]

def plot_all(results):
    """
    results: dict method -> dict domain -> {"probs": ..., "labels": ...}
    """
    n_methods = len(METHODS)
    fig = plt.figure(figsize=(12, 4.5 * n_methods))
    gs  = gridspec.GridSpec(n_methods, 2, hspace=0.45, wspace=0.35)

    for row, method in enumerate(METHODS):
        # ── Reliability diagram ─────────────────────────────────────────────
        ax_rel = fig.add_subplot(gs[row, 0])
        ax_rel.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        for domain, style in [("PCAM", "-"), ("NCT-CRC", "--")]:
            d = results[method][domain]
            mc, ma, _ = reliability_bins(d["probs"], d["labels"])
            ax_rel.plot(mc, ma, style, color=COLORS[domain],
                        marker="o", ms=4, lw=1.5, label=domain)
        ax_rel.set_xlim(0, 1); ax_rel.set_ylim(0, 1)
        ax_rel.set_xlabel("Mean confidence", fontsize=10)
        ax_rel.set_ylabel("Fraction correct", fontsize=10)
        ax_rel.set_title(f"{method} — Reliability diagram", fontsize=11)
        ax_rel.legend(fontsize=9)

        # ── Risk-coverage curve ─────────────────────────────────────────────
        ax_rc = fig.add_subplot(gs[row, 1])
        for domain, style in [("PCAM", "-"), ("NCT-CRC", "--")]:
            d = results[method][domain]
            cov, risk = risk_coverage(d["probs"], d["labels"])
            ax_rc.plot(cov, risk, style, color=COLORS[domain],
                       lw=1.5, label=domain)
        ax_rc.set_xlim(0, 1); ax_rc.set_ylim(bottom=0)
        ax_rc.set_xlabel("Coverage", fontsize=10)
        ax_rc.set_ylabel("Risk (error rate)", fontsize=10)
        ax_rc.set_title(f"{method} — Risk-coverage curve", fontsize=11)
        ax_rc.legend(fontsize=9)

    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    print(f"Saved → {OUT_FIG}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    print("Loading PCAM test images …")
    pcam_imgs, pcam_labels = load_pcam(N_SAMPLES)
    print("Loading NCT-CRC images …")
    nct_imgs,  nct_labels  = load_nct(N_SAMPLES)

    results = {m: {} for m in METHODS}

    # ── Confidence ────────────────────────────────────────────────────────────
    print("\n=== Confidence ===")
    model_det = load_model(dropout_train=False)
    for domain, imgs, labels in [("PCAM", pcam_imgs, pcam_labels),
                                  ("NCT-CRC", nct_imgs, nct_labels)]:
        print(f"  {domain} …")
        probs = run_confidence(model_det, imgs, processor)
        results["Confidence"][domain] = {"probs": probs, "labels": labels}

    # ── MC Dropout ────────────────────────────────────────────────────────────
    print("\n=== MC Dropout ===")
    model_mc = load_model(dropout_train=True)
    for domain, imgs, labels in [("PCAM", pcam_imgs, pcam_labels),
                                  ("NCT-CRC", nct_imgs, nct_labels)]:
        print(f"  {domain} (T={N_MC} passes) …")
        probs = run_mc_dropout(model_mc, imgs, processor)
        results["MC Dropout"][domain] = {"probs": probs, "labels": labels}

    plot_all(results)

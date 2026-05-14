#!/usr/bin/env python3
"""
Generate a visual preview grid showing all corruption types and severity levels
applied to real PCAM sample images.

Produces: evaluation/corruption_preview.png
  - Rows: original + 4 corruption types (blur, jpeg, noise, color)
  - Columns: severity 1, 3, 5 (mild → severe)
  - Shows N sample patches side-by-side for each condition
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Corruption functions (mirrors evaluate_shift_ood.py)

def apply_shift(img: Image.Image, shift: str, severity: int) -> Image.Image:
    s = max(1, min(5, int(severity)))
    if shift == "id":
        return img
    if shift == "blur":
        radius = [0.5, 1.0, 1.5, 2.0, 2.5][s - 1]
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if shift == "noise":
        arr = np.asarray(img).astype(np.float32)
        rng = np.random.default_rng(s * 7)
        sigma = [6, 12, 18, 24, 30][s - 1]
        noisy = arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    if shift == "jpeg":
        scale = [0.95, 0.85, 0.75, 0.6, 0.45][s - 1]
        w, h = img.size
        w2, h2 = max(8, int(w * scale)), max(8, int(h * scale))
        return img.resize((w2, h2), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if shift == "color":
        color = [0.95, 0.9, 0.8, 0.7, 0.6][s - 1]
        contrast = [1.05, 1.1, 1.15, 1.2, 1.25][s - 1]
        out = ImageEnhance.Color(img).enhance(color)
        return ImageEnhance.Contrast(out).enhance(contrast)
    return img


# Layout constants

SHIFTS = ["id", "blur", "jpeg", "noise", "color"]
SHIFT_LABELS = {
    "id":    "Original",
    "blur":  "Blur",
    "jpeg":  "JPEG",
    "noise": "Noise",
    "color": "Color shift",
}
SEVERITIES = [1, 3, 5]
SEVERITY_LABELS = {1: "Severity 1\n(mild)", 3: "Severity 3\n(medium)", 5: "Severity 5\n(severe)"}
N_SAMPLES = 3          # patches per cell
PATCH_SIZE = 96        # display size per patch (px)
GAP = 4                # gap between patches within a cell
CELL_PAD = 8           # padding inside each cell
HEADER_H = 48          # column header height
ROW_LABEL_W = 110      # row label width
BG_COLOR = (30, 30, 35)
HEADER_COLOR = (50, 50, 60)
CELL_BG_EVEN = (42, 42, 52)
CELL_BG_ODD  = (38, 38, 48)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (100, 160, 255)


def _try_font(size: int) -> ImageFont.ImageFont:
    for name in ["DejaVuSans", "LiberationSans", "Arial", "Helvetica"]:
        for ext in [".ttf", "-Regular.ttf"]:
            for base in ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype",
                         "/usr/share/fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]:
                p = Path(base) / (name + ext)
                if p.exists():
                    try:
                        return ImageFont.truetype(str(p), size)
                    except Exception:
                        pass
    return ImageFont.load_default()


def load_pcam_samples(n: int, seed: int = 42) -> list[Image.Image]:
    """Load n random PCAM test patches as PIL Images."""
    from torchvision.datasets import PCAM
    ds = PCAM(root=str(REPO_ROOT / "data" / "raw"), split="test", download=False, transform=None)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n, replace=False)
    imgs = []
    for i in indices:
        img, _ = ds[int(i)]
        if not isinstance(img, Image.Image):
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img)
        imgs.append(img.convert("RGB").resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS))
    return imgs


def build_cell(imgs: list[Image.Image]) -> Image.Image:
    """Arrange N patches in a row within a single cell."""
    w = N_SAMPLES * PATCH_SIZE + (N_SAMPLES - 1) * GAP + 2 * CELL_PAD
    h = PATCH_SIZE + 2 * CELL_PAD
    cell = Image.new("RGB", (w, h), BG_COLOR)
    for k, im in enumerate(imgs):
        x = CELL_PAD + k * (PATCH_SIZE + GAP)
        cell.paste(im, (x, CELL_PAD))
    return cell


def draw_text_centered(draw: ImageDraw.ImageDraw, text: str, bbox, font, color=TEXT_COLOR):
    x0, y0, x1, y1 = bbox
    lines = text.strip().split("\n")
    line_h = font.getbbox("A")[3] + 4
    total_h = len(lines) * line_h
    y_start = y0 + (y1 - y0 - total_h) // 2
    for i, line in enumerate(lines):
        tw = font.getbbox(line)[2]
        x = x0 + (x1 - x0 - tw) // 2
        draw.text((x, y_start + i * line_h), line, font=font, fill=color)


def generate_preview(
    out_path: Path,
    n_samples: int = N_SAMPLES,
    seed: int = 42,
) -> None:
    print("Loading PCAM sample patches...")
    samples = load_pcam_samples(n_samples, seed=seed)

    cell_w = n_samples * PATCH_SIZE + (n_samples - 1) * GAP + 2 * CELL_PAD
    cell_h = PATCH_SIZE + 2 * CELL_PAD

    # Grid: rows = shifts (original + 4 corruptions), cols = severities (3)
    # Original row spans all severities with the same image (no severity axis)
    n_shift_rows = len(SHIFTS)   # 5 (id + 4 corruptions)
    n_sev_cols = len(SEVERITIES) # 3

    total_w = ROW_LABEL_W + n_sev_cols * cell_w + (n_sev_cols - 1) * 1
    total_h = HEADER_H + n_shift_rows * cell_h + (n_shift_rows - 1) * 1

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    font_header = _try_font(14)
    font_row    = _try_font(13)
    font_small  = _try_font(11)

    # Column headers
    for ci, sev in enumerate(SEVERITIES):
        x0 = ROW_LABEL_W + ci * (cell_w + 1)
        x1 = x0 + cell_w
        draw.rectangle([x0, 0, x1, HEADER_H], fill=HEADER_COLOR)
        label = SEVERITY_LABELS[sev] if sev != 1 or True else "Original"
        if ci == 0:
            label = "Severity 1\n(mild)"
        draw_text_centered(draw, label, (x0, 0, x1, HEADER_H), font_header, ACCENT_COLOR)

    # Row labels + cells
    for ri, shift in enumerate(SHIFTS):
        y0 = HEADER_H + ri * (cell_h + 1)
        y1 = y0 + cell_h
        row_bg = CELL_BG_EVEN if ri % 2 == 0 else CELL_BG_ODD
        # Row label
        draw.rectangle([0, y0, ROW_LABEL_W - 1, y1], fill=HEADER_COLOR)
        draw_text_centered(draw, SHIFT_LABELS[shift], (0, y0, ROW_LABEL_W, y1),
                           font_row, ACCENT_COLOR if shift == "id" else TEXT_COLOR)

        for ci, sev in enumerate(SEVERITIES):
            x0 = ROW_LABEL_W + ci * (cell_w + 1)
            # For Original row, all columns show the same unmodified patches
            effective_sev = sev if shift != "id" else 0
            corrupted = [apply_shift(img, shift, effective_sev) for img in samples]
            cell_img = build_cell(corrupted)
            # Light cell background tint
            bg = Image.new("RGB", cell_img.size, row_bg)
            canvas.paste(bg, (x0, y0))
            canvas.paste(cell_img, (x0, y0))

    # Thin grid lines
    line_color = (60, 60, 70)
    for ci in range(1, n_sev_cols):
        x = ROW_LABEL_W + ci * (cell_w + 1) - 1
        draw.line([(x, 0), (x, total_h)], fill=line_color, width=1)
    for ri in range(1, n_shift_rows):
        y = HEADER_H + ri * (cell_h + 1) - 1
        draw.line([(0, y), (total_w, y)], fill=line_color, width=1)
    draw.line([(ROW_LABEL_W - 1, 0), (ROW_LABEL_W - 1, total_h)], fill=line_color, width=1)
    draw.line([(0, HEADER_H - 1), (total_w, HEADER_H - 1)], fill=line_color, width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"Saved: {out_path}  ({total_w}x{total_h} px)")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate corruption type/severity preview grid")
    p.add_argument("--out", default="evaluation/corruption_preview.png")
    p.add_argument("--n_samples", type=int, default=3,
                   help="Number of PCAM patches to show per cell (default: 3)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    generate_preview(out_path, n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()

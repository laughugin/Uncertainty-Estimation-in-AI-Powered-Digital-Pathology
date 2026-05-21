# Uncertainty Estimation in AI-Powered Digital Pathology

Code and experiments for a diploma thesis comparing uncertainty estimation methods on binary histopathology classification. The model is a Vision Transformer fine-tuned on the PatchCamelyon dataset (tumor / normal lymph node patches); four methods are evaluated: a confidence baseline, MC Dropout, a deep ensemble, and temperature scaling.

A browser-based web interface is included for interactive exploration of results.

---

## Setup

Requires Python 3.10+ and a GPU (CPU works but is slow for MC Dropout).

```bash
git clone https://github.com/laughugin/Uncertainty-Estimation-in-AI-Powered-Digital-Pathology.git
cd Uncertainty-Estimation-in-AI-Powered-Digital-Pathology
./setup_env.sh
source venv/bin/activate
```

Or manually without the script:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Data

**PatchCamelyon** (~1.3 GB, downloaded automatically):

```bash
python data/download_datasets.py --root data/raw --dataset pcam
```

**NCT-CRC-HE-100K** (used for cross-domain OOD evaluation only) — download from [Zenodo](https://zenodo.org/record/1214456) and place the folder at `data/raw/NCT-CRC-HE-100K/`.

---

## Training

```bash
python experiments/train.py --config configs/default.yaml
```

The best checkpoint is saved to `checkpoints/best.pt`. On a single GPU this takes around 30–60 minutes for the default 2-epoch run. If the loss oscillates rather than converging, check that the learning rate in the config is `1e-5` (not `1e-4`).

---

## Evaluation

Run the full pipeline (all four methods, all metrics) with:

```bash
python experiments/run_evaluation_pipeline.py
```

Or run individual analyses separately:

```bash
# ECE under synthetic distribution shift
python experiments/run_ece_under_shift.py

# Conformal prediction (split conformal, α = 0.05 / 0.10 / 0.20)
python experiments/run_conformal.py

# Aleatoric / epistemic decomposition via MC Dropout
python experiments/run_aleatoric_epistemic.py

# Cross-domain OOD evaluation against NCT-CRC-HE-100K
python experiments/evaluate_cross_domain_ood.py --method confidence
python experiments/evaluate_cross_domain_ood.py --method mc_dropout
```

Results are written to `evaluation/` as JSON files and PNG figures.

---

## Web interface

```bash
python web/app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The UI lets you browse dataset samples, launch training with live log output, and explore all evaluation results — reliability diagrams, risk-coverage curves, conformal prediction coverage, aleatoric/epistemic decomposition, and cross-domain OOD detection.

---

## Docker

Docker is the easiest way to run the web interface without touching your Python environment.

**Build and start:**

```bash
docker compose up --build
```

Open [http://localhost:5000](http://localhost:5000).

The compose file mounts four host directories into the container so nothing large is baked into the image and everything survives a rebuild:

| Host path | Container path | Purpose |
|---|---|---|
| `./data/raw` | `/app/data/raw` | PCAM / NCT-CRC dataset files |
| `./checkpoints` | `/app/checkpoints` | Trained model weights |
| `./evaluation` | `/app/evaluation` | Generated JSON results and figures |
| *(named volume)* | `/root/.cache/huggingface` | ViT weights downloaded from Hub |

**Download data inside the container** (or download to `./data/raw` on the host first):

```bash
docker compose run --rm web python data/download_datasets.py --root data/raw --dataset pcam
```

**Run evaluation scripts** against existing checkpoints:

```bash
docker compose run --rm web python experiments/run_evaluation_pipeline.py
```

To run on a GPU, add a `deploy` block to `docker-compose.yml`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Project structure

```
uncertainty_lab/    core package — models, metrics, uncertainty methods, pipeline
experiments/        training and evaluation entry points
web/                Flask web UI and templates
configs/            YAML configuration files
evaluation/         generated results (JSON + figures)
thesis/             LaTeX source of the diploma thesis
references/         BibTeX bibliography
data/               dataset download helpers (raw data not tracked)
checkpoints/        saved model weights (not tracked)
```

---

## Requirements

- Python 3.10+
- PyTorch 2.x (GPU strongly recommended for MC Dropout with T=30 passes)
- ~3 GB disk space for PCAM
- ~350 MB for the ViT checkpoint (downloaded automatically on first run via Hugging Face)

For theoretical background, experimental methodology, and results, see the thesis in `thesis/`.

# Pathology AI Uncertainty Estimation

This repository contains a thesis-focused toolkit for studying predictive uncertainty in digital pathology image classification. It combines experiment scripts, a reusable Python package, a small web UI, and thesis materials in one place.

---

## 1. Overview

The project focuses on **uncertainty estimation** for binary pathology decisions. It does **not** introduce a new model architecture. Instead, it aims to:

- Run existing image classification models, mainly from Hugging Face.
- Use public pathology datasets with a clear binary task, primarily PCAM.
- Compare established uncertainty methods such as confidence scores, MC dropout, and deep ensembles.
- Produce reproducible outputs that can be reported directly in a thesis.

The main goal is a clean, reproducible workflow for training, evaluation, and uncertainty analysis.

---

## Getting Started

**1. Set up the environment**  
On Debian/Ubuntu, install venv if needed: `sudo apt install python3-venv`. Then:

```bash
./setup_env.sh
source venv/bin/activate
```

Or install dependencies directly: `pip install -r requirements.txt`.

**2. Download the PCAM dataset** (~1-2 GB; train/val/test):

```bash
python data/download_datasets.py --root data/raw --dataset pcam
```

**3. Download and cache a model** (this also happens automatically on first run):

```bash
python models/load_model.py --model_id google/vit-base-patch16-224
```

Main config: `configs/default.yaml`. Local data lives under `data/raw/`. Downloaded model weights are stored in the Hugging Face cache (for example `~/.cache/huggingface/`).

**Model execution is local.** The model is downloaded once, after which training and inference run on your machine. A GPU is strongly preferred, but CPU execution is supported.

**4. Start the web UI (optional)**  
From the project root with the virtual environment activated:
```bash
pip install -r requirements.txt   # ensures Flask and deps are installed
python web/app.py --host 127.0.0.1 --port 5000
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000). The UI includes a dashboard, dataset setup and browsing tools, run controls, and evaluation views.

**5. Use the installable toolkit**  
`uncertainty_lab/` is the reusable core package. It supports PCAM, folder-based datasets, and CSV manifests; Hugging Face models or local `.pt` checkpoints; and multiple uncertainty methods. Metrics and plots are written under `runs/`.

```bash
pip install -e .
uncertainty-lab run --help
uncertainty-lab run -c configs/uncertainty_lab_default.yaml --method mc_dropout
uncertainty-lab benchmark --methods confidence,mc_dropout
streamlit run uncertainty_lab/streamlit_app.py
```

- Default config: `configs/uncertainty_lab_default.yaml`.
- Legacy thesis scripts still work and share metric logic with `uncertainty_lab.metrics`.
- Training can still be launched via `experiments/train.py`.
- Batch evaluation can be launched via `experiments/run_evaluation_pipeline.py`.

---

## Docker

The project can also be run with Docker and Docker Compose:

```bash
# Build and run the web UI
docker compose up --build
```

- The UI is available at `http://localhost:5000`.
- Data can be mounted from the host or downloaded through the UI.
- The compose setup persists `./data` and the Hugging Face cache across restarts.

To move the project to another machine, copy the repository without local artifacts such as `venv/`, `data/raw/`, `runs/`, and `checkpoints/`, then run `docker compose up --build` on the new system.

---

## 2. Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Model selection** | Identify and document Hugging Face (or equivalent) models for pathology vs. non-pathology classification; run inference and baseline metrics. |
| 2 | **Dataset selection** | Choose public digital pathology datasets (WSI or patches) with labels suitable for binary pathology decision; document access, licensing, and preprocessing. |
| 3 | **Uncertainty methods** | Implement or integrate standard uncertainty-estimation methods and apply them to the selected models and data. |
| 4 | **Evaluation** | Evaluate uncertainty quality (calibration, discrimination, utility in deferral/screening) and report results in a thesis with correct methodology and citations. |

---

## 3. Project Plan (Phases)

### Phase 1 — Literature and model survey
- Review digital pathology classification and uncertainty estimation in medical imaging.
- Search Hugging Face Hub (and related repos) for: histopathology, whole-slide image (WSI), patch-based classification, binary pathology detection.
- Shortlist 1–3 models by criteria: availability, documentation, compatibility with our data format, and reported performance where available.
- Document model cards, licenses, and citation requirements.

### Phase 2 — Dataset selection and preparation
- List candidate public datasets (see §4).
- For each: check license, download procedure, label format (slide-level or patch-level), and whether it matches “pathology vs. non-pathology” (or a well-defined proxy).
- Choose primary (and optionally secondary) dataset(s); define train/validation/test splits and any preprocessing (tiling, normalization, resolution).
- Produce a small, versioned preprocessing pipeline and document it for reproducibility.

### Phase 3 — Baseline and uncertainty pipeline
- Run selected model(s) on the chosen dataset(s); report accuracy, sensitivity, specificity, and basic ROC/PR curves as baseline.
- Integrate or implement chosen uncertainty-estimation methods (see §5).
- Ensure all experiments are reproducible (seeds, configs, code versioning).

### Phase 4 — Uncertainty evaluation and thesis writing
- Evaluate uncertainty using agreed metrics (calibration, ECE, reliability diagrams, possibly deferral/screening curves).
- Compare methods and document limitations.
- Write thesis: problem definition, related work, datasets, methods, experiments, results, discussion, conclusion, and references.

---

## 4. Candidate Datasets

Datasets below are typical in digital pathology research. Final choice must respect **license and intended use**; always verify current terms before use.

| Dataset | Type | Task / labels | Notes |
|--------|------|----------------|-------|
| **CAMELYON16** | WSI (breast) | Metastasis detection (binary) | Standard benchmark; lymph node metastases. |
| **CAMELYON17** | WSI (breast) | Metastasis detection, multi-center | Extension of CAMELYON16. |
| **TCGA** (e.g. via NIH Genomic Data Commons) | WSI (multi-cancer) | Diagnosis / subtype labels | Many cancer types; access and preprocessing are non-trivial. |
| **NCT-CRC-HE-100K** | Patches (H&E) | Tissue type classification | Patch-level; can be adapted to pathology vs. normal if class definitions are clear. |
| **PCAM** (Patch Camelyon) | Patches | Metastasis (binary, from CAMELYON16) | Directly binary; widely used, good for patch-based models. |
| **Lizard** | Patches / annotations | Colon, multiple cell/tissue labels | Can be used for binary tasks if a clear pathology criterion is defined. |

**Recommendation for thesis:** Start with **PCAM** or **CAMELYON16** (or both) for a clear, binary pathology task and existing benchmarks; add one more dataset if needed for robustness or discussion.

---

## 5. Uncertainty Estimation Methods

The following are **established methods** suitable for thesis work; they can be applied to existing classifiers without changing their core architecture.

### 5.1 Maximum softmax probability (confidence)
- **Quantity:** \( u = 1 - \max_c p(y=c|x) \).
- **Use:** Baseline “confidence” as a simple uncertainty proxy; no extra computation.
- **Limitation:** Often overconfident; not calibrated.

### 5.2 Monte Carlo (MC) Dropout
- **Idea:** Keep dropout enabled at test time; run \(T\) forward passes; treat outputs as a sample from an approximate posterior.
- **Uncertainty:** Predictive variance (or entropy) over the \(T\) predictions.
- **Use:** Model uncertainty; no retraining if the model already uses dropout.
- **Reference:** Gal & Ghahramani, “Dropout as a Bayesian approximation…” (2016).

### 5.3 Deep ensembles
- **Idea:** Train \(M\) identical models with different random seeds (and/or data splits); aggregate predictions.
- **Uncertainty:** Variance or entropy across ensemble members.
- **Use:** Well-calibrated uncertainty; higher compute and storage.
- **Reference:** Lakshminarayanan et al., “Simple and scalable predictive uncertainty estimation using deep ensembles” (NeurIPS 2017).

### 5.4 Conformal prediction (CP)
- **Idea:** Use a calibration set to build prediction sets/intervals with a guaranteed coverage level (e.g. 90% or 95%).
- **Use:** Set-valued predictions with a formal guarantee; good for “when to defer” or flagging.
- **Reference:** Vovk et al.; recent variants (e.g. RAPS, CQR) for classification/regression.

### 5.5 Conformalised MC (e.g. MC-CP)
- **Idea:** Combine MC dropout (or similar) with conformal calibration to get both uncertainty estimates and coverage guarantees.
- **Use:** More efficient and often better than MC dropout alone; suitable for safety-critical or high-stakes settings.
- **Reference:** E.g. “Robust Uncertainty Quantification Using Conformalised Monte Carlo Prediction” (AAAI); code: e.g. MC-CP repositories.

### 5.6 Temperature scaling / Platt scaling
- **Idea:** Post-hoc calibration of softmax outputs using a validation set.
- **Use:** Improve calibration of a single model; does not add “model uncertainty,” only better probability estimates.

**Suggested scope for thesis:**  
Implement at least: (1) **confidence baseline**, (2) **MC Dropout** (if the HF model supports dropout), (3) **one of** Deep ensemble or Conformal prediction (or MC-CP). Compare them in terms of calibration (e.g. ECE), reliability diagrams, and optionally deferral/screening performance.

---

## 6. Evaluation of Uncertainty

- **Calibration:** Expected Calibration Error (ECE), Maximum Calibration Error (MCE), reliability diagrams.
- **Discrimination:** Separation of correct vs. incorrect predictions by uncertainty (e.g. AUC of uncertainty vs. error).
- **Utility:** Accuracy–coverage or accuracy–rejection curves when deferring high-uncertainty cases (if applicable).

All metrics and experimental setup (splits, seeds, hyperparameters) must be clearly defined in the thesis for reproducibility.

---

## 7. Deliverables (Thesis-Oriented)

1. **Codebase:** Data loading, model inference (Hugging Face or equivalent), uncertainty methods, evaluation scripts; README with environment and run instructions.
2. **Experiments:** Configs and (where feasible) logs or summaries for each setting (model, dataset, uncertainty method).
3. **Thesis document:** Introduction, related work, problem formulation, datasets, methods (models + uncertainty), experiments, results, discussion, conclusion, references.
4. **Reproducibility:** Requirements file, fixed seeds, and clear train/validation/test split description.

---

## 8. Repository Structure

```
diplom/
├── README.md                     # Project overview and setup
├── pyproject.toml                # Installable package metadata
├── requirements.txt              # Python dependencies
├── configs/                      # Default and experiment config files
├── data/                         # Dataset download helpers and local data root
├── models/                       # Model loading helpers used by legacy scripts and web UI
├── uncertainty_lab/              # Main package: pipeline, metrics, CLI, Streamlit app
├── experiments/                  # Training and evaluation entry points
├── evaluation/                   # Generated evaluation JSON summaries
├── runs/                         # Generated Uncertainty Lab run artifacts
├── web/                          # Flask UI and templates
├── scripts/                      # Utility and environment helper scripts
└── thesis/                       # Thesis sources and supporting material
```

The root `uncertainty/` package was a legacy stub and is no longer part of the active code path.

---

## 9. References

Canonical references now live under `references/`. The thesis bibliography and the web UI both use that directory as the source of truth.

Key implementation references:

- **Model:** Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. Implementation: Hugging Face `google/vit-base-patch16-224`.
- **Dataset:** Veeling et al., *Rotation Equivariant CNNs for Digital Pathology*, MICCAI 2018; Patch Camelyon Grand Challenge. PCAM is fully labeled (binary: 0 = normal, 1 = metastasis).
- **Uncertainty (thesis):** Gal & Ghahramani (2016, ICML); Lakshminarayanan et al. (2017, NeurIPS); Vovk et al. (2005).

---

This README is the single project guide for the repository.

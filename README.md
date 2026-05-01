# melanoma-abc-hybrid

Reproducible evaluation framework for dermoscopic melanoma classification integrating
segmentation-derived ABC descriptors, deep convolutional embeddings, calibration analysis,
decision-curve evaluation, and dual-level explainability.

This repository accompanies the manuscript:
"A Reproducible Within-Corpus Audit Framework for Melanoma Classification Integrating Segmentation, Calibration, Decision-Curve Analysis, and Explainability"

---

## Overview

The pipeline implements:

- Lesion-level split (HAM10000 by `lesion_id`)
- Segmentation benchmarking (ISIC 2018 Task 1; Dice, IoU, failure rate)
- Handcrafted ABC descriptors (asymmetry, border, HSV, GLCM)
- Deep baseline (EfficientNet-B0)
- Hybrid model (ABC + EfficientNet embeddings via XGBoost)
- Secondary ISIC 2018 Task 3 melanoma-label evaluation on the same HAM10000/ISIC 2018 training image corpus
- Ablation experiments
- Lesion-clustered bootstrap (95% CIs)
- Calibration analysis (Brier, slope/intercept, reliability diagrams)
- Post-hoc recalibration (Platt, isotonic)
- Decision-curve analysis (net benefit)
- Explainability:
  - SHAP (tree-based models)
  - Grad-CAM (deep baseline)
- Reproducible paper tables and figures under `results/`

The framework explicitly separates:
- Discrimination
- Calibration
- Decision utility
- Explainability

---

## Labeling Protocol

### HAM10000
Binary classification:
- Positive = melanoma (`dx == mel`)
- Negative = all other diagnoses

Metadata must include:
- image_id
- lesion_id
- dx

### ISIC 2018 Task 3
Binary labels file:

image_id,label

Where:
- label = 1 -> melanoma
- label = 0 -> non-melanoma

The local ISIC 2018 Task 3 files used with this repository overlap HAM10000 by
10,015/10,015 image identifiers. They are therefore treated as a secondary
within-corpus melanoma-label evaluation, not as independent external validation.

---

## Data Placement (NOT included in repository)

Due to size and licensing constraints, datasets must be obtained separately.

Expected structure:


data/raw/HAM10000/images/*
data/raw/HAM10000/metadata.csv

data/raw/ISIC2018/Task1/images/*
data/raw/ISIC2018/Task1/masks/*.png

data/raw/ISIC2018/Task3/images/*
data/raw/ISIC2018/Task3/labels.csv


Datasets:
- HAM10000: https://doi.org/10.1038/sdata.2018.161
- ISIC 2018: https://challenge.isic-archive.com/

---

## Environment Setup (PowerShell)


python -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


---

## Execution Order


python -m experiments.01_make_split
python -m experiments.02_segment_isic_task1
python -m experiments.03_segment_ham
python -m experiments.04_extract_abc
python -m experiments.05_train_deep_baseline
python -m experiments.06_extract_embeddings
python -m experiments.07_train_handcrafted
python -m experiments.08_train_hybrid
python -m experiments.09_external_validation
python -m experiments.10_ablation
python -m experiments.11_explainability
python -m experiments.12_build_paper_artifacts


Optional:

python -m experiments.13_build_split_comparison


---

## Outputs

Generated artifacts:

- `results/tables/*.csv`
- `results/figures/*.png`
- `results/runs/*` (model checkpoints and prediction files)
- `results/audit/*` (calibration and prediction exports)

No raw datasets are stored in this repository.

---

## Reproducibility

- Global seed: 42
- Fixed hyperparameters (see `config.yaml`)
- Lesion-clustered bootstrap for uncertainty estimation
- Deterministic split generation

---

## License

See LICENSE file.

---

## Citation

See CITATION.cff or cite the associated manuscript.

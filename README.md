# melanoma-abc-hybrid

Reproducible evaluation framework for dermoscopic melanoma classification integrating
segmentation-derived ABC descriptors, deep convolutional embeddings, calibration analysis,
decision-curve evaluation, and dual-level explainability.

This repository accompanies the manuscript:
"A Structured Evaluation Framework for Dermoscopic Classification Integrating Segmentation, Calibration, and Explainability"

---

## Overview

The pipeline implements:

- Lesion-level split (HAM10000 by `lesion_id`)
- Segmentation benchmarking (ISIC 2018 Task 1; Dice, IoU, failure rate)
- Handcrafted ABC descriptors (asymmetry, border, HSV, GLCM)
- Deep baseline (EfficientNet-B0)
- Hybrid model (ABC + EfficientNet embeddings via XGBoost)
- External validation (train: HAM10000 → test: ISIC 2018 Task 3)
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
- Malignant = {mel, bcc, akiec}
- Benign = all other classes

Metadata must include:
- image_id
- lesion_id
- dx

### ISIC 2018 Task 3
Binary labels file:

image_id,label

Where:
- label = 1 → malignant
- label = 0 → benign

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


python experiments\01_make_split.py
python experiments\02_segment_isic_task1.py
python experiments\03_segment_ham.py
python experiments\04_extract_abc.py
python experiments\05_train_deep_baseline.py
python experiments\06_extract_embeddings.py
python experiments\07_train_handcrafted.py
python experiments\08_train_hybrid.py
python experiments\09_external_validation.py
python experiments\10_ablation.py
python experiments\11_explainability.py
python experiments\12_build_paper_artifacts.py


Optional:

python experiments\13_build_split_comparison.py


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
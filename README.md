# melanoma-abc-hybrid

Reproducibility repository for the manuscript **"Lesion-Aware External Evaluation of Dermoscopic Melanoma Classifiers under Dataset Shift."**

The current release evaluates dermoscopic melanoma classifiers across HAM10000 development data and an external BCN20000 cohort while accounting for repeated images from the same lesion. The revised analysis uses lesion-clustered uncertainty, source-derived operating thresholds, lesion-mean sensitivity analyses, segmentation sensitivity checks, reference-standard restrictions, and saliency localization diagnostics.

## Current scientific scope

Three model families are evaluated:

- **Handcrafted model:** asymmetry, border, HSV color, and GLCM texture descriptors with XGBoost.
- **Deep model:** end-to-end EfficientNet-B0 melanoma probability.
- **Hybrid model:** XGBoost on EfficientNet embeddings combined with handcrafted descriptors.

A post hoc embeddings-only XGBoost ablation estimates the incremental discrimination associated with the handcrafted descriptors.

The current external analysis reports image-level discrimination with lesion-clustered confidence intervals, lesion-mean discrimination, source-derived operating thresholds applied unchanged to BCN20000, lesion-mean operating-point sensitivity analyses, reference-standard restrictions, critical anatomic subgroups, segmentation sensitivity, and Grad-CAM localization diagnostics.

The revised manuscript does **not** use BCN20000 outcomes to choose operating thresholds. Decision-curve and descriptor-conditioned correction analyses from earlier development versions are not part of the current paper.

## Main external findings

On 12,413 BCN20000 images from 3,576 lesions, external ROC-AUC was approximately 0.692 for the handcrafted model, 0.713 for the deep model, and 0.761 for the hybrid model. The hybrid-deep AUC difference was approximately +0.048. The embeddings-only XGBoost ablation reached essentially the same external AUC as the hybrid model, so the revised paper does not attribute the ranking gain to explicit handcrafted-descriptor fusion.

At the source target specificity of 0.90, transported image-level sensitivity was approximately 0.161, 0.302, and 0.326 for the handcrafted, deep, and hybrid models. The hybrid-deep sensitivity difference had a confidence interval spanning zero. A lesion-mean operating analysis reached the same conclusion.

## Repository layout

- `src/` — reusable data, modeling, evaluation, segmentation, and explainability code.
- `experiments/` — training and evaluation entry points. The current external audit entry point is `23_final_external_audit.py`.
- `data/splits/` — deterministic split definitions and external identifier audit outputs.
- `data/derived/` — distributable derived features and embeddings where permitted.
- `results/final/` — tables used by the revised manuscript.
- `results/figures/` — figures and saliency diagnostics.

Raw HAM10000, BCN20000, and ISIC image data are not redistributed. Obtain them from their original sources and place them according to `config.yaml` and `data_instructions.md`.

## Reproducing the current external audit

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiments\23_final_external_audit.py
```

The final audit expects locally generated model prediction files under `results/runs/` and BCN20000 metadata at the configured dataset path. It writes current primary external tables to `results/final/`.

## Analysis safeguards in the current release

- HAM10000 train/validation/test separation is by `lesion_id`.
- BCN20000 repeated images are linked through `lesion_id` for cluster resampling.
- External confidence intervals resample lesions as clusters.
- Source operating thresholds are estimated from HAM10000 validation data only.
- BCN20000 labels never determine the primary operating thresholds.
- A lesion-mean analysis gives each external lesion one prediction.
- The embeddings-only ablation is labeled post hoc.
- Error-descriptor analyses are treated as associations and repeated under alternative segmentation.
- Grad-CAM localization is exploratory and is not interpreted as proof of a specific background artifact.

## Previous releases

Releases `v1.0.0` and `1.0.1` archive earlier development states. Their decision-curve, BCN-derived fixed-specificity, and ABC-fusion claims are superseded by the current analysis. Use the latest release for the manuscript reported here.

## Citation and archive

The manuscript-aligned archival release is **v2.0.0 — Lesion-aware external evaluation**.

- Zenodo DOI: **10.5281/zenodo.23130026**
- GitHub release: https://github.com/squareshorts/melanoma-abc-hybrid/releases/tag/v2.0.0
- Zenodo record: https://doi.org/10.5281/zenodo.23130026

Please cite the version-specific Zenodo DOI when referring to the code and derived outputs used for the revised manuscript.

## License

MIT. See `LICENSE`.

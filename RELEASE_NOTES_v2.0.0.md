# v2.0.0 — Lesion-aware external evaluation

This major release aligns the repository with the revised manuscript:

**Lesion-Aware External Evaluation of Dermoscopic Melanoma Classifiers under Dataset Shift**

## Main methodological corrections

- External inference now accounts for repeated BCN20000 images from the same lesion.
- The evaluated external set contains 12,413 images from 3,576 lesions.
- Confidence intervals for image-level external performance use lesion-cluster resampling.
- A complementary lesion-mean analysis gives each lesion one prediction.
- Operating thresholds are selected from HAM10000 validation data and applied unchanged to BCN20000.
- Lesion-mean source-threshold analyses are included.
- The former "ABC-only" model is described as a handcrafted descriptor model because its predictors include asymmetry, border, HSV color, and GLCM texture features.
- The embeddings-only XGBoost comparison is explicitly post hoc and shows essentially no incremental external AUC from adding handcrafted descriptors.
- False-negative descriptor profiling uses the transported source-derived hybrid threshold and is repeated with Level Set and Otsu segmentation.
- Histopathology-only and critical-anatomic-site sensitivity analyses are included.
- Grad-CAM findings are limited to localization behavior; no specific background artifact is inferred.

## Superseded material

Earlier development versions included BCN-derived fixed-specificity operating points, decision-curve analyses on the external score scale, descriptor-conditioned score correction, threshold-0.5 error profiling, and stronger claims for explicit ABC fusion. Those analyses and claims are not part of the v2 manuscript evidence and were removed from the current branch. They remain available through Git history and pre-v2 archival releases.

## Current outputs

The manuscript-aligned derived results are under `results/final/`.

The current external audit entry point is:

```
experiments/23_final_external_audit.py
```

Raw dermoscopic datasets are not redistributed.

# Threshold-transport evaluation: supplementary reproducibility files

These files support the melanoma decision-threshold transport analysis
reported in "External transport of melanoma decision thresholds: a
lesion-level evaluation" (18 October 2026).

## Public baseline archive

Earlier model-development and lesion-aware external-evaluation code and
outputs were published in version 2.0.0:
https://doi.org/10.5281/zenodo.23130026
https://github.com/squareshorts/melanoma-abc-hybrid/releases/tag/v2.0.0

Version 2.0.0 predates the current threshold-transport calculations. The
current supplementary files therefore have a separate release scope and
should receive a new archival DOI when published.

## Included material

- `inputs/four_model_known_identity_corrected/`: per-image prediction
  probabilities for HAM validation, HAM test, and BCN, with lesion keys and
  melanoma labels, after published lesion-identity corrections.
- `results/four_model_corrected/`: selected source cutoffs, lesion- and
  image-level operating estimates, discrimination, paired comparisons,
  transport differences, and bootstrap output summaries (the primary results).
- `inputs/four_model_original/` and `results/four_model_original/`: companion
  analyses with the original lesion grouping for the identity sensitivity check.
- `analysis/frozen_score_transport.py`: reproduces source threshold selection,
  transported operating rates and 2,000-replicate lesion-bootstrap intervals.
- `analysis/test_frozen_score_transport.py`: synthetic numerical tests.
- `comparator/`: model and exact prediction exports for the embeddings-only
  XGBoost head, with the training-feature matrix and reconstruction script.
- `provenance/outputs/`: derived lesion grouping and identity crosswalk checks.

These files contain derived scores, lesion IDs, and reference labels. Original
HAM10000 and BCN20000 image files are not redistributed. The CNN's original
training execution and checkpoint selection are not reproduced by the score-
level analysis; uncertainty intervals condition on the previously fitted models.

## Run the score-level evaluation

Requirements: Python, numpy, pandas. For synthetic tests use scikit-learn.
`requirements_reanalysis.txt` records the software used in the analysis.
Run from this archive's root directory (the output folder should not exist):

```bash
python analysis/frozen_score_transport.py \
  --ham-validation inputs/four_model_known_identity_corrected/ham_validation.csv \
  --ham-test inputs/four_model_known_identity_corrected/ham_test.csv \
  --bcn-external inputs/four_model_known_identity_corrected/bcn_external.csv \
  --models Handcrafted Deep Hybrid EmbeddingsOnly_Reconstructed \
  --output-dir reproduction/primary \
  --bootstrap 2000 --seed 20261018
```

Reproducing the original-identifier sensitivity analysis uses the inputs in
`inputs/four_model_original/` and a different output folder. The four-model
source predictions and output tables were verified independently against the
threshold-selection definitions used in the article.

To run numerical unit tests:

```bash
python -m unittest discover -s analysis -p test_frozen_score_transport.py -v
```

## Scope

The outputs provide a direct, independent rerun of the score-based threshold
transport and interval calculations. Reproducing the original end-to-end CNN
training requires the raw public images and source environment described in
v2.0.0. The included embeddings-only training matrix and XGBoost model permit
independent inspection of the added classifier head without distributing images.

DOI for this supplementary version: not yet assigned. Do not cite the earlier
v2.0.0 DOI as though it contains these updated files.

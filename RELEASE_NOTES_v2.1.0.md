# v2.1.0 — External transport of melanoma decision thresholds

Accompanies **External transport of melanoma decision thresholds: a lesion-level evaluation**.

## Included when the threshold-transport package is imported

- Predictions, lesion identifiers and melanoma labels for HAM validation, HAM test and BCN external evaluations.
- Identity-corrected and original-lesion-group analyses for four classifiers: descriptor-only XGBoost, EfficientNet-B0, combined XGBoost and embeddings-only XGBoost.
- Source-selected sensitivity and specificity thresholds, transported operating characteristics and discrimination summaries.
- Paired lesion-level bootstrap estimates and re-selection of source thresholds.
- Model and reproducing scripts for the embeddings-only XGBoost comparator, including the archived training embedding tensor.
- Metadata crosswalks, checksums, requirements and nine numerical unit tests.

The archive excludes raw dermoscopic images. The analysis conditions on previously fitted EfficientNet and other model weights; it does not retrain the original CNN.

## Required before publishing GitHub release / triggering Zenodo

The 65-file archive MUST be committed at `reproducibility/threshold_transport_v2_1_0/` on the branch to be tagged. Use `tools/import_threshold_transport.ps1` with the supplied ZIP. Verify the data files exist in GitHub, **then** manually publish a new v2.1.0 release at that commit. The Zenodo integration archives the repository source snapshot; separate GitHub release assets do not substitute for committed data.

Prior release: v2.0.0, DOI 10.5281/zenodo.23130026. Cite the *new* Zenodo version-specific DOI for the v2.1.0 analysis.

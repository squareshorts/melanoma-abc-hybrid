# Historical embeddings-only classifier reconstruction

This folder holds one explicitly labeled reconstruction, `EmbeddingsOnly_Reconstructed`, of the historical embeddings-only XGBoost comparison. It uses the archived EfficientNet-B0 embeddings and does not retrain the CNN.

## Specification recovered before fitting

The original `embeddings_only_check.py` was recovered from `melanoma_reanalysis_results.zip`. It explicitly specifies:

| Parameter | Value |
|---|---:|
| Trees | 600 |
| Maximum depth | 4 |
| Learning rate | 0.03 |
| Row subsampling | 0.9 |
| Column subsampling per tree | 0.9 |
| L2 regularization | 1.0 |
| Evaluation metric | logloss |
| Random state | 0 |
| Worker threads | 4 |

The training rows follow the original HAM metadata order after selecting the original train lesion IDs, retaining distinct `image_id`/`y_true` rows, and filtering to available embeddings. The feature tensor retains its original float32 dtype. There are 7,000 rows, 1,280 embedding features and 774 melanoma images.

The historical model helper did not set `random_state`, but the recovered embeddings-only script removes that ambiguity by explicitly specifying zero. This reconstruction follows that script, not the later five-seed classifier-head comparison.

## Execution and provenance

`reconstruct_embeddings_only.py` writes its specification, ordered training rows and exact selected training tensor before fitting. It fits one XGBoost head and saves the native model plus its hash before reading the external evaluation cohort or comparing performance. No external result selects a seed, hyperparameter, model, or additional fit.

XGBoost 3.2.0 matches the recovered later audit environment lock. The exact runtime of the original October 3 ablation was not separately recovered. The current NumPy, pandas, scikit-learn and SciPy versions are recorded in the specification and pinned in `requirements_reconstruction.txt`. The original source script, helper, YAML configuration and later environment lock are retained in `inputs/`; `source_manifest.json` identifies those recovered sources by SHA-256.

The `vendor/` directory is a temporary local package installation and is not needed in the scientific deliverable. Install the pinned requirements in a separate environment to rerun.

## Outputs

- `outputs/predeclared_specification.json`: immutable specification written before the fit, including source and tensor hashes.
- `outputs/training_rows.csv`, `training_features_float32.npy` and `training_labels.npy`: the exact ordered training input.
- `outputs/embeddings_only_reconstructed.ubj`: stable native fitted model.
- `outputs/booster_configuration.json` and `model_frozen_manifest.json`: full fitted configuration and model identity recorded before external evaluation.
- `outputs/predictions_ham_val.csv`, `predictions_ham_test.csv` and `predictions_bcn_external.csv`: per-image scores with image and lesion IDs, labels, embedding-row indices, partition and the explicit reconstructed-model name.
- `outputs/reconstructed_performance.csv`: descriptive AUC, average precision and Brier score.
- `outputs/reproduction_comparison.json`: comparison with the archived external AUC, evaluated after model and scores were saved.
- `outputs/final_artifact_manifest.json`: fitted-model, score-file and evaluation-input hashes.

The single fit reproduced the archived external AUC exactly: **0.7613159017096941**. Combined XGBoost had AUC **0.760814299086658** on the same rows, giving Combined minus reconstructed embeddings-only **−0.0005016026230361126**. This reproduces the historical external summary statistic. It does not prove row-wise identity to original embeddings-only predictions that were not archived, so the reconstructed label is retained.

The reconstructed HAM validation AUC is **0.9241699959873690** and HAM test AUC is **0.8845377004739590**. The latter does not reproduce the manuscript's separate rounded embeddings-only internal figure of 0.888. The recovered historical ablation script evaluated only BCN; it did not establish that internal row. Use the actual reconstructed internal results when reporting this model and do not attribute the external reproduction to every prior result.

`outputs/export_verification.json` records a focused check of every score export: expected sample counts, unique image IDs and finite probabilities in [0,1]. Reloading the native model reproduced all exported validation probabilities to a maximum absolute difference of 2.22 × 10⁻¹⁶ from decimal CSV parsing. Verification required no additional training fit.

These score exports preserve the original validation and test populations. The separately documented known-duplicate exclusions in `../provenance/outputs/ham_known_duplicate_validation_test_exclusions.csv` should be applied only in a labeled sensitivity analysis; they are not silently removed here. The model's original training partition is preserved.

## Reproduce once in a fresh output directory

```text
python reconstruct_embeddings_only.py --bundle /path/to/melanoma_reanalysis_bundle --historical-results /path/to/melanoma_reanalysis_results --output /path/to/new_reconstruction_output
```

The script refuses to overwrite a nonempty output directory. Source images are not needed. The original archived embedding arrays, embedding ID tables, HAM metadata/split and external evaluation table are needed for score regeneration. Their identities are recorded in the manifests.

# Lesion-aware external validation of melanoma AI: tutorial code

This directory contains a Python score-level tutorial for evaluating binary melanoma classifiers on repeated lesion images. Its working example uses preserved, previously fitted HAM10000 development scores and BCN20000 external scores. No dermoscopic image processing, classifier refitting, or CNN retraining takes place.

## Requirements

Python 3.11 is tested. Install the pinned numerical packages in `tutorial/requirements.txt`. Model-score files are already present in the public repository's `reproducibility/threshold_transport_v2_1_0/` directory. External source images are not required.

## Run from the repository root

```bash
python -m pip install -r tutorial/requirements.txt
python tutorial/worked_example.py --repo . --out tutorial_results --bootstrap 250 --seed 20261110
TUTORIAL_RESULT_DIR=tutorial_results python -m unittest discover -s tutorial/tests -p 'test_worked_example.py' -v
```

On Windows PowerShell use `$env:TUTORIAL_RESULT_DIR = 'tutorial_results'` before invoking the final test command. The Python script can be run with a larger bootstrap replicate count; 250 is the bounded teaching example. The archive's 2,000-replicate analyses remain the reference for reported uncertainty intervals.

## Expected input contract

Files in the source directory `reproducibility/threshold_transport_v2_1_0/inputs/four_model_known_identity_corrected/` must include `ham_validation.csv`, `ham_test.csv`, and `bcn_external.csv`. A known-identity comparison also reads `inputs/four_model_original/ham_test.csv`.

Each score CSV has `image_id`, `lesion_id`, `y_true` (binary melanoma status), `Handcrafted`, `Deep`, and `Hybrid` columns, with one row per image. The script verifies distinct image IDs, complete labels, finite scores, and consistent diagnosis within lesions before evaluating the data. The class columns refer to three previously fitted classifier families; to adapt the tutorial to another application, replace these named columns consistently and supply aligned source/external score files.

## Outputs

- `threshold_transport.csv` records source-only score cutoff selection at sensitivity and specificity targets and external operating results at image and lesion level.
- `roc_transport_lesion.pdf` plots the external lesion ROC with source-derived operating points.
- `bootstrap_comparison.csv` and `bootstrap_comparison.pdf` show independent image versus whole-lesion bootstrap intervals for the same image-weighted AUC. Dots indicate full-sample AUC estimates.
- `paired_bootstrap.csv` distinguishes uncertainty conditional on fixed cutoffs from independent source-lesion cutoff reselection.
- `duplicate_sensitivity.csv` compares original and known-identity-corrected HAM held-out partitions using the existing fitted scores.
- `score_distributions.csv` and `calibration_deciles.csv` summarize class-conditional score changes and descriptive calibration.
- `run_summary.json` records the seed, cohort counts, and scope.

## Scope and provenance

The archival data and model-score analysis come from repository commit `85e5140999a4c41764130f80a98c025f6f447e01`. The authoritative threshold-transport script and 2,000-resample outputs are in `reproducibility/threshold_transport_v2_1_0/analysis/` and `results/four_model_corrected/`. The histopathology-restricted sensitivity analysis cited in the manuscript derives from the older `results/final/table_histopathology_only_external.csv` at that same commit and uses the original lesion grouping.

The 18 known cross-identifier HAM10000 duplicate pairs are drawn from the published audit by Abhishek K, Jain A, Hamarneh G, *Scientific Data* 12:196 (2025), DOI 10.1038/s41597-025-04382-5. The 18-pair original file is `DermaMNIST/HAM10000_DuplicateConfirmation/AnalysesOutputFiles/newly_discovered_duplicates.csv` in the authors' public repository. It has Git blob SHA `49dcfe1d5873a65a3dc302f5dc189f59c0650d40`, matching the source checksum recorded by this repository. Two separate near-duplicate pairs suspected to have been imaged at different times are not treated as confirmed same-lesion corrections.

## Important limitations

The score-level implementation combines established validation procedures and does not establish a novel clinical algorithm. Changing the external threshold on labeled test cases gives an optimistically estimated within-sample adapted rule, which needs independent validation. Bootstrap intervals condition on the previously fitted models and available lesion IDs. The synthetic test suite validates protocol behavior and input guards; its reference-output checks are regression tests and are not independent confirmation of the original CNN training or cohort provenance.

The repository contains no patient-identifying records or copyrighted input images. The source metadata and score exports remain with their original public archive.
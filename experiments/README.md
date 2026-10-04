# Experiment entry points

The retained numbered scripts cover model development, segmentation, feature extraction, prediction generation, and explainability components that remain relevant to the v2 pipeline.

For BCN20000:

- `09b_audit_bcn20000_overlap.py` performs the identifier/exact perceptual-hash overlap audit.
- `09c_extract_bcn20000_features.py` creates the external handcrafted descriptors and deep embeddings.
- `09d_bcn20000_external_validation.py` generates predictions only.
- `23_final_external_audit.py` is the authoritative external inference entry point for the current manuscript.

The v2 external audit selects operating thresholds from HAM10000 validation data and applies them unchanged to BCN20000. External uncertainty is lesion-clustered and lesion-mean sensitivity analyses are included.

Exploratory scripts used by pre-v2 manuscripts for decision curves, descriptor-conditioned score correction, threshold-0.5 error profiling, and subgroup fairness were removed from the current branch. They remain recoverable from Git history and archived pre-v2 releases.

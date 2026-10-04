# Experiment entry points

Scripts `01_*` through `18_*` contain the original training, feature extraction, ablation, and explainability pipeline.

The current manuscript's corrected external inference is implemented in:

- `23_final_external_audit.py`

The current audit selects operating thresholds from HAM10000 validation data and applies them unchanged to BCN20000. External uncertainty is lesion-clustered, and a lesion-mean sensitivity analysis is included.

Historical resubmission scripts that selected thresholds from BCN20000 outcomes or profiled errors at an arbitrary probability threshold were removed from the current branch. They remain recoverable from repository history and archived pre-v2 releases.

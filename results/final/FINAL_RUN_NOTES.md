# Final analysis notes

Core external results use the fitted models from repository commit `cb6b164a3962c654599c16a22377dcc79a7f96a4`, followed by reviewer-driven reanalysis of saved predictions and BCN20000 lesion metadata.

Final manuscript inference settings:

- Primary external image-level performance: lesion-cluster bootstrap.
- Lesion-mean discrimination and paired AUC contrasts: lesion-level bootstrap.
- Source-derived operating thresholds: selected on HAM10000 validation and applied unchanged to BCN20000.
- Lesion-mean operating thresholds: source validation predictions and external predictions averaged within lesion before threshold analysis.
- FN descriptor mask-sensitivity contrasts: lesion-cluster bootstrap.
- Critical anatomical subgroup intervals: lesion-cluster bootstrap.
- Embeddings-only paired AUC comparison: post hoc.

Decision-curve outputs, BCN-derived fixed-specificity thresholds, and threshold-0.5 FN summaries are superseded and are not part of the final manuscript evidence.

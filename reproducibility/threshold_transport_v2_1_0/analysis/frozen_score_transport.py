#!/usr/bin/env python3
"""Frozen-score, lesion-aware threshold transport; no model fitting.

Inputs are three wide CSVs with image_id, lesion_id, y_true and explicitly named
model probability columns. Source thresholds are selected on HAM validation
only. The primary unit is the arithmetic mean probability per lesion; the
secondary unit retains image decisions and resamples whole lesions.

All decisions use score >= threshold. Candidate cutoffs are distinct observed
validation scores plus +inf (all source predictions negative). For a minimum
specificity target, maximize sensitivity, then specificity, then the cutoff.
For a minimum sensitivity target, maximize specificity, then sensitivity, then
the cutoff. Whole score ties are included; constraints use integer percentages.
The same rule is used in the observed data and every source bootstrap sample.

Bootstrap intervals are percentile intervals, conditional on the saved fitted
models. A fixed-cutoff interval evaluates the archived decision rule selected
from this validation set. A source-reestimated interval additionally represents
uncertainty in re-estimating that rule; it is a different inferential target.
Neither interval includes CNN retraining or checkpoint-selection uncertainty.

This program deliberately performs no calibration, model fitting, imputation,
prediction-based exclusion, or selection using HAM test or BCN outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGETS = (("specificity", 80), ("specificity", 90), ("specificity", 95),
           ("sensitivity", 90), ("sensitivity", 95))
COHORTS = ("ham_validation", "ham_test", "bcn_external")
UNITS = ("lesion_mean", "image_cluster")
MODES = ("fixed_cutoff", "source_reestimated")
DISCRIMINATION = ("roc_auc", "average_precision", "brier")
RATES = ("sensitivity", "specificity", "positive_fraction")
REQUIRED = ("image_id", "lesion_id", "y_true")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class WeightedCurve:
    """Descending tied-score groups, with weighted counts at every cutoff."""

    scores: np.ndarray
    tp: np.ndarray  # Prefix count, including zero at index zero.
    fp: np.ndarray
    active: np.ndarray  # Groups represented in this bootstrap sample.

    @property
    def positives(self) -> float:
        return float(self.tp[-1])

    @property
    def negatives(self) -> float:
        return float(self.fp[-1])

    def choose(self, target_type: str, target_percent: int) -> float:
        """Select using only this source sample and explicitly defined ties."""
        if self.positives <= 0 or self.negatives <= 0:
            raise ValueError("Threshold selection requires both outcome classes")
        indices = np.r_[0, np.flatnonzero(self.active) + 1]
        tp, tn = self.tp[indices], self.negatives - self.fp[indices]
        if target_type == "specificity":
            keep = tn * 100 >= target_percent * self.negatives
            first, second = tp, tn
        elif target_type == "sensitivity":
            keep = tp * 100 >= target_percent * self.positives
            first, second = tn, tp
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        eligible = np.flatnonzero(keep)
        eligible = eligible[first[eligible] == first[eligible].max()]
        eligible = eligible[second[eligible] == second[eligible].max()]
        # Cutoffs are in descending order, so first also implements last tie rule.
        index = int(indices[eligible[0]])
        return math.inf if index == 0 else float(self.scores[index - 1])

    def thresholds(self) -> np.ndarray:
        return np.array([self.choose(kind, pct) for kind, pct in TARGETS])

    def operating(self, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return rates [Se, Sp, positive fraction] and [TP,FN,TN,FP]."""
        indices = np.searchsorted(-self.scores, -np.asarray(thresholds), side="right")
        tp, fp = self.tp[indices], self.fp[indices]
        fn, tn = self.positives - tp, self.negatives - fp
        rates = np.column_stack((tp / self.positives, tn / self.negatives,
                                 (tp + fp) / (self.positives + self.negatives)))
        counts = np.column_stack((tp, fn, tn, fp))
        return rates, counts

    def metrics(self) -> np.ndarray:
        p, n = self.positives, self.negatives
        if p <= 0 or n <= 0:
            raise ValueError("Discrimination requires both outcome classes")
        dp, dn = np.diff(self.tp), np.diff(self.fp)
        auc = np.sum(dp * (n - self.fp[1:] + 0.5 * dn)) / (p * n)
        precision = np.divide(self.tp[1:], self.tp[1:] + self.fp[1:],
                              out=np.zeros_like(dp),
                              where=(self.tp[1:] + self.fp[1:]) > 0)
        ap = np.sum((dp / p) * precision)
        brier = np.sum(dp * (1 - self.scores) ** 2 + dn * self.scores ** 2) / (p + n)
        return np.array([auc, ap, brier])


@dataclass
class ScoreOrder:
    scores: np.ndarray
    sorted_labels: np.ndarray
    sorted_clusters: np.ndarray
    group_starts: np.ndarray

    @classmethod
    def from_arrays(cls, scores: np.ndarray, labels: np.ndarray,
                    clusters: np.ndarray) -> "ScoreOrder":
        order = np.argsort(-scores, kind="stable")
        ordered_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
        return cls(ordered_scores[starts], labels[order], clusters[order], starts)

    def curve(self, cluster_weights: np.ndarray) -> WeightedCurve:
        weights = cluster_weights[self.sorted_clusters]
        p = np.add.reduceat(weights * self.sorted_labels, self.group_starts)
        n = np.add.reduceat(weights * (1 - self.sorted_labels), self.group_starts)
        return WeightedCurve(self.scores, np.r_[0., np.cumsum(p, dtype=float)],
                             np.r_[0., np.cumsum(n, dtype=float)], (p + n) > 0)


@dataclass
class Cohort:
    name: str
    path: Path
    frame: pd.DataFrame
    model_names: tuple[str, ...]
    lesion_ids: np.ndarray
    lesion_labels: np.ndarray
    image_clusters: np.ndarray
    lesion_scores: np.ndarray
    orders: dict[str, list[ScoreOrder]]

    @classmethod
    def read(cls, path: Path, name: str, model_names: Iterable[str]) -> "Cohort":
        models = tuple(model_names)
        if not models or len(set(models)) != len(models) or set(models) & set(REQUIRED):
            raise ValueError("Provide distinct model column names, excluding identifier/label columns")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        if len(header) != len(set(header)):
            raise ValueError(f"{name}: duplicate CSV column names")
        missing = set(REQUIRED + models) - set(header)
        if missing:
            raise ValueError(f"{name}: required columns absent: {sorted(missing)}")
        frame = pd.read_csv(path, dtype={"image_id": str, "lesion_id": str},
                            keep_default_na=False)
        if frame.empty:
            raise ValueError(f"{name}: empty cohort")
        for column in ("image_id", "lesion_id"):
            values = frame[column]
            if values.isna().any() or (values.str.len() == 0).any():
                raise ValueError(f"{name}: missing {column}")
            if (values != values.str.strip()).any():
                raise ValueError(f"{name}: whitespace in {column}; correct provenance explicitly")
        if frame.image_id.duplicated().any():
            raise ValueError(f"{name}: duplicate image_id")
        labels = pd.to_numeric(frame.y_true, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.isin(labels, [0., 1.]).all():
            raise ValueError(f"{name}: y_true must contain only 0 and 1")
        frame["y_true"] = labels.astype(np.int8)
        for model in models:
            values = pd.to_numeric(frame[model], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: missing/nonfinite scores in {model}")
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"{name}: {model} probabilities outside [0,1]; no clipping performed")
            frame[model] = values
        if frame.groupby("lesion_id", sort=False).y_true.nunique().max() != 1:
            raise ValueError(f"{name}: conflicting binary labels within a lesion")
        if frame.y_true.nunique() != 2:
            raise ValueError(f"{name}: both outcome classes are required")
        frame = frame.sort_values(["lesion_id", "image_id"], kind="stable").reset_index(drop=True)
        clusters, lesion_ids = pd.factorize(frame.lesion_id, sort=True)
        labels = frame.y_true.to_numpy()
        scores = frame.loc[:, models].to_numpy(dtype=float)
        counts = np.bincount(clusters)
        lesion_scores = np.column_stack([
            np.bincount(clusters, weights=scores[:, j]) / counts for j in range(len(models))])
        lesion_labels = frame.groupby("lesion_id", sort=True).y_true.first().to_numpy()
        orders = {
            "image_cluster": [ScoreOrder.from_arrays(scores[:, j], labels, clusters)
                              for j in range(len(models))],
            "lesion_mean": [ScoreOrder.from_arrays(lesion_scores[:, j], lesion_labels,
                                                    np.arange(len(lesion_ids)))
                            for j in range(len(models))],
        }
        return cls(name, path, frame, models, np.asarray(lesion_ids), lesion_labels,
                   clusters, lesion_scores, orders)

    @property
    def n_lesions(self) -> int:
        return len(self.lesion_ids)

    def has_both_classes(self, weights: np.ndarray) -> bool:
        return bool(np.dot(weights, self.lesion_labels) > 0 and
                    np.dot(weights, 1 - self.lesion_labels) > 0)

    def summary(self) -> dict:
        counts = np.bincount(self.image_clusters)
        images = len(self.frame)
        positive_images = int(self.frame.y_true.sum())
        positive_lesions = int(self.lesion_labels.sum())
        p_image, p_lesion = positive_images / images, positive_lesions / self.n_lesions
        return {"cohort": self.name, "images": images, "lesions": self.n_lesions,
                "melanoma_images": positive_images, "melanoma_lesions": positive_lesions,
                "image_prevalence": p_image, "lesion_prevalence": p_lesion,
                "image_prevalence_only_brier": p_image * (1 - p_image),
                "lesion_prevalence_only_brier": p_lesion * (1 - p_lesion),
                "median_images_per_lesion": float(np.median(counts)),
                "maximum_images_per_lesion": int(counts.max()),
                "multiple_image_lesions": int((counts > 1).sum()),
                "mean_images_per_melanoma_lesion": positive_images / positive_lesions,
                "mean_images_per_nonmelanoma_lesion":
                    (images - positive_images) / (self.n_lesions - positive_lesions)}


def validate_cohort_separation(cohorts: list[Cohort]) -> None:
    for a, b in itertools.combinations(cohorts, 2):
        if set(a.frame.image_id) & set(b.frame.image_id):
            raise ValueError(f"Shared image IDs between {a.name} and {b.name}")
    if set(cohorts[0].lesion_ids) & set(cohorts[1].lesion_ids):
        raise ValueError("Shared HAM lesion IDs between validation and independent test")


def interval(values: np.ndarray) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap endpoint; no silent dropping is permitted")
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return float(low), float(high)


def threshold_kind(value: float) -> str:
    return "all_source_negative" if math.isinf(value) else "observed_source_score"


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def run_analysis(cohorts: list[Cohort], output_dir: Path, bootstrap: int = 2000,
                 seed: int = 20261018, max_attempts: int | None = None) -> dict:
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if tuple(c.name for c in cohorts) != COHORTS:
        raise ValueError(f"Cohorts must be supplied in this order: {COHORTS}")
    if any(c.model_names != cohorts[0].model_names for c in cohorts):
        raise ValueError("Model names and order must match across cohorts")
    validate_cohort_separation(cohorts)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty to prevent mixing analysis runs")
    models = cohorts[0].model_names
    n_models, n_targets = len(models), len(TARGETS)
    input_hashes = {c.name: file_hash(c.path) for c in cohorts}
    points = {}
    selected = np.empty((len(UNITS), n_models, n_targets))
    source_roc_rows, threshold_rows = [], []
    point_metrics = np.empty((3, 2, n_models, 3))
    point_rates = np.empty((3, 2, n_models, n_targets, 3))
    point_counts = np.empty((3, 2, n_models, n_targets, 4))
    for ci, cohort in enumerate(cohorts):
        for ui, unit in enumerate(UNITS):
            for mi, model in enumerate(models):
                curve = cohort.orders[unit][mi].curve(np.ones(cohort.n_lesions, dtype=int))
                points[ci, ui, mi] = curve
                point_metrics[ci, ui, mi] = curve.metrics()
                if ci == 0:
                    selected[ui, mi] = curve.thresholds()
                rates, counts = curve.operating(selected[ui, mi])
                point_rates[ci, ui, mi], point_counts[ci, ui, mi] = rates, counts
                if ci != 0:
                    continue
                for ki, (kind, percent) in enumerate(TARGETS):
                    threshold_rows.append({"unit": unit, "model": model,
                        "target_type": kind, "target": percent / 100,
                        "threshold": selected[ui, mi, ki],
                        "threshold_kind": threshold_kind(selected[ui, mi, ki]),
                        "source_sensitivity": rates[ki, 0], "source_specificity": rates[ki, 1],
                        **dict(zip(("source_tp", "source_fn", "source_tn", "source_fp"),
                                   counts[ki].astype(int)))})
                for gi, cutoff in enumerate(np.r_[math.inf, curve.scores]):
                    source_roc_rows.append({"unit": unit, "model": model, "threshold": cutoff,
                        "threshold_kind": threshold_kind(cutoff), "tp": int(curve.tp[gi]),
                        "fp": int(curve.fp[gi]), "sensitivity": curve.tp[gi] / curve.positives,
                        "specificity": 1 - curve.fp[gi] / curve.negatives,
                        "selected_for": "|".join(f"{k}_{p}" for ii, (k, p) in enumerate(TARGETS)
                                                 if cutoff == selected[ui, mi, ii])})

    metric_boot = np.empty((bootstrap, 3, 2, n_models, 3))
    rate_boot = np.empty((bootstrap, 2, 2, 2, n_models, n_targets, 3))
    threshold_boot = np.empty((bootstrap, 2, n_models, n_targets))
    sequences = np.random.SeedSequence(seed).spawn(3)
    generators = [np.random.default_rng(s) for s in sequences]
    attempts, valid = 0, 0
    rejected = {c.name: 0 for c in cohorts}
    limit = max_attempts if max_attempts is not None else max(100, bootstrap * 100)
    while valid < bootstrap:
        attempts += 1
        if attempts > limit:
            raise RuntimeError(f"Unable to obtain {bootstrap} valid paired replicates within "
                               f"{limit} attempts; valid={valid}, degenerate={rejected}")
        weights = [np.bincount(rng.integers(0, c.n_lesions, size=c.n_lesions),
                               minlength=c.n_lesions)
                   for c, rng in zip(cohorts, generators)]
        invalid = [not c.has_both_classes(w) for c, w in zip(cohorts, weights)]
        if any(invalid):
            for c, bad in zip(cohorts, invalid):
                rejected[c.name] += int(bad)
            continue
        for ui, unit in enumerate(UNITS):
            for mi in range(n_models):
                curves = [c.orders[unit][mi].curve(w) for c, w in zip(cohorts, weights)]
                cutoffs = curves[0].thresholds()
                threshold_boot[valid, ui, mi] = cutoffs
                for ci, curve in enumerate(curves):
                    metric_boot[valid, ci, ui, mi] = curve.metrics()
                    if ci > 0:
                        rate_boot[valid, 0, ci - 1, ui, mi] = curve.operating(selected[ui, mi])[0]
                        rate_boot[valid, 1, ci - 1, ui, mi] = curve.operating(cutoffs)[0]
        valid += 1
        if valid % 250 == 0 or valid == bootstrap:
            print(f"Bootstrap: {valid}/{bootstrap} valid paired replicates", flush=True)

    discrimination_rows, operating_rows = [], []
    paired_discrimination, paired_operating = [], []
    transfer_operating, transfer_discrimination = [], []
    for ci, cohort in enumerate(cohorts):
        for ui, unit in enumerate(UNITS):
            for mi, model in enumerate(models):
                for di, metric in enumerate(DISCRIMINATION):
                    low, high = interval(metric_boot[:, ci, ui, mi, di])
                    discrimination_rows.append({"cohort": cohort.name, "unit": unit, "model": model,
                        "metric": metric, "estimate": point_metrics[ci, ui, mi, di],
                        "ci_low": low, "ci_high": high, "bootstrap_replicates": bootstrap,
                        "source_apparent": ci == 0})
                for ki, (kind, percent) in enumerate(TARGETS):
                    counts = point_counts[ci, ui, mi, ki].astype(int)
                    for mode_i, mode in enumerate(("source_apparent",) if ci == 0 else MODES):
                        row = {"cohort": cohort.name, "unit": unit, "model": model,
                               "target_type": kind, "target": percent / 100,
                               "threshold": selected[ui, mi, ki], "uncertainty_mode": mode,
                               "bootstrap_replicates": 0 if ci == 0 else bootstrap,
                               **dict(zip(("tp", "fn", "tn", "fp"), counts))}
                        for ri, rate in enumerate(RATES):
                            row[rate] = point_rates[ci, ui, mi, ki, ri]
                            low, high = (math.nan, math.nan) if ci == 0 else interval(
                                rate_boot[:, mode_i, ci - 1, ui, mi, ki, ri])
                            row[rate + "_ci_low"], row[rate + "_ci_high"] = low, high
                        row["misses_per_1000_melanomas"] = 1000 * (1 - row["sensitivity"])
                        row["false_positives_per_1000_nonmelanomas"] = 1000 * (1 - row["specificity"])
                        operating_rows.append(row)
            for ai, bi in itertools.combinations(range(n_models), 2):
                base = {"cohort": cohort.name, "unit": unit,
                        "model_a": models[ai], "model_b": models[bi], "direction": "a_minus_b"}
                for di, metric in enumerate(DISCRIMINATION):
                    low, high = interval(metric_boot[:, ci, ui, ai, di] - metric_boot[:, ci, ui, bi, di])
                    paired_discrimination.append({**base, "metric": metric,
                        "difference": point_metrics[ci, ui, ai, di] - point_metrics[ci, ui, bi, di],
                        "ci_low": low, "ci_high": high, "bootstrap_replicates": bootstrap})
                if ci == 0:
                    continue
                for ki, (kind, percent) in enumerate(TARGETS):
                    for mode_i, mode in enumerate(MODES):
                        for ri, rate in enumerate(RATES):
                            samples = (rate_boot[:, mode_i, ci - 1, ui, ai, ki, ri] -
                                       rate_boot[:, mode_i, ci - 1, ui, bi, ki, ri])
                            low, high = interval(samples)
                            paired_operating.append({**base, "target_type": kind, "target": percent / 100,
                                "uncertainty_mode": mode, "metric": rate,
                                "difference": point_rates[ci, ui, ai, ki, ri] - point_rates[ci, ui, bi, ki, ri],
                                "ci_low": low, "ci_high": high, "bootstrap_replicates": bootstrap})
    for ui, unit in enumerate(UNITS):
        for mi, model in enumerate(models):
            base = {"unit": unit, "model": model, "direction": "bcn_external_minus_ham_test"}
            for di, metric in enumerate(DISCRIMINATION):
                low, high = interval(metric_boot[:, 2, ui, mi, di] - metric_boot[:, 1, ui, mi, di])
                transfer_discrimination.append({**base, "metric": metric,
                    "difference": point_metrics[2, ui, mi, di] - point_metrics[1, ui, mi, di],
                    "ci_low": low, "ci_high": high, "bootstrap_replicates": bootstrap})
            for ki, (kind, percent) in enumerate(TARGETS):
                for mode_i, mode in enumerate(MODES):
                    for ri, rate in enumerate(RATES):
                        low, high = interval(rate_boot[:, mode_i, 1, ui, mi, ki, ri] -
                                             rate_boot[:, mode_i, 0, ui, mi, ki, ri])
                        transfer_operating.append({**base, "target_type": kind, "target": percent / 100,
                            "uncertainty_mode": mode, "metric": rate,
                            "difference": point_rates[2, ui, mi, ki, ri] - point_rates[1, ui, mi, ki, ri],
                            "ci_low": low, "ci_high": high, "bootstrap_replicates": bootstrap})
    threshold_diagnostics = []
    for ui, unit in enumerate(UNITS):
        for mi, model in enumerate(models):
            for ki, (kind, percent) in enumerate(TARGETS):
                samples = threshold_boot[:, ui, mi, ki]
                finite = samples[np.isfinite(samples)]
                low, high = interval(finite)
                threshold_diagnostics.append({"unit": unit, "model": model,
                    "target_type": kind, "target": percent / 100, "observed_threshold": selected[ui, mi, ki],
                    "valid_replicates": bootstrap, "all_source_negative_replicates": int(np.isinf(samples).sum()),
                    "distinct_cutoffs": len(np.unique(samples)),
                    "finite_threshold_ci_low": low, "finite_threshold_ci_high": high,
                    "finite_threshold_median": float(np.median(finite)) if len(finite) else math.nan,
                    "note": "Finite-cutoff quantiles exclude +inf; +inf count is reported explicitly"})
    outputs = {
        "cohort_summary.csv": [c.summary() for c in cohorts], "source_roc.csv": source_roc_rows,
        "thresholds.csv": threshold_rows, "discrimination.csv": discrimination_rows,
        "operating_performance.csv": operating_rows, "paired_operating_deltas.csv": paired_operating,
        "paired_discrimination_deltas.csv": paired_discrimination,
        "transfer_deltas.csv": transfer_operating, "transfer_discrimination_deltas.csv": transfer_discrimination,
        "bootstrap_diagnostics.csv": threshold_diagnostics,
    }
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)
    if any(file_hash(c.path) != input_hashes[c.name] for c in cohorts):
        raise RuntimeError("An input file changed during execution")
    manifest = {
        "status": "complete" if bootstrap >= 1000 else "complete_exploratory_bootstrap" if bootstrap else "point_estimates_only",
        "engine": str(Path(__file__).resolve()), "engine_sha256": file_hash(Path(__file__)),
        "models": list(models), "score_definition": "Melanoma probability; larger scores predict melanoma",
        "positive_predicate": "score >= threshold", "primary_unit": "lesion_mean",
        "secondary_unit": "image_cluster", "aggregation": "Arithmetic mean of all available image probabilities within a lesion",
        "threshold_source": "ham_validation only", "threshold_candidates": "Distinct source scores plus +inf",
        "specificity_rule": "Maximize sensitivity subject to target specificity, then maximize specificity, then cutoff",
        "sensitivity_rule": "Maximize specificity subject to target sensitivity, then maximize sensitivity, then cutoff",
        "targets": [{"type": k, "value": p / 100} for k, p in TARGETS],
        "bootstrap_requested": bootstrap, "bootstrap_valid": valid, "bootstrap_attempts": attempts,
        "bootstrap_rejected_joint_draws": attempts - valid, "degenerate_draws_by_cohort": rejected,
        "bootstrap_seed": seed, "bootstrap_method": "Ordinary nonparametric lesion resampling with replacement",
        "bootstrap_pairing": "Same lesion multiplicities across models and analysis units; independent cohorts; matched target draws across uncertainty modes",
        "bootstrap_degenerate_handling": "Reject and redraw whole paired replicate if any cohort contains only one outcome class",
        "confidence_interval": "Pointwise 95% percentile, NumPy linear quantiles; no multiplicity adjustment",
        "scope": "Conditional on supplied fitted scores; no training, checkpoint-selection or patient-cluster uncertainty",
        "interpretation": {"fixed_cutoff": "Sampling uncertainty for this source-selected decision rule",
                           "source_reestimated": "Additional uncertainty in re-estimating cutoffs from another source sample",
                           "ham_validation": "Apparent source estimates; validation selected checkpoint and thresholds",
                           "ap_and_brier": "Depend on case mix/prevalence; cross-cohort differences are descriptive",
                           "null_brier": "Descriptive constant equal to observed cohort prevalence, not an independently trained predictor",
                           "positive_fraction": "Fraction in these retrospective cohorts, not a population referral estimate"},
        "provenance_scope": "Input identity and internal joins verified; upstream official-dataset mapping must be documented separately",
        "checks": {"complete_common_model_scores": True, "unique_image_ids": True,
                   "consistent_binary_lesion_labels": True, "validation_test_lesion_disjoint": True,
                   "cross_cohort_image_ids_disjoint": True, "no_imputation_or_row_exclusion": True,
                   "input_hashes_unchanged": True, "train_partition_checked": False,
                   "official_external_partition_verified_by_engine": False},
        "inputs": [{"cohort": c.name, "path": str(c.path.resolve()), "sha256": input_hashes[c.name],
                    "summary": c.summary()} for c in cohorts],
        "outputs": {name: {"rows": len(rows), "sha256": file_hash(output_dir / name)}
                    for name, rows in outputs.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ham-validation", type=Path, required=True)
    parser.add_argument("--ham-test", type=Path, required=True)
    parser.add_argument("--bcn-external", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--max-bootstrap-attempts", type=int)
    args = parser.parse_args(argv)
    # Refusal must not append even a failure marker to a completed analysis.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print("Analysis failed: output directory must be empty; existing files were untouched",
              file=sys.stderr)
        return 1
    try:
        paths = (args.ham_validation, args.ham_test, args.bcn_external)
        cohorts = [Cohort.read(path, name, args.models) for name, path in zip(COHORTS, paths)]
        result = run_analysis(cohorts, args.output_dir, args.bootstrap, args.seed, args.max_bootstrap_attempts)
    except (ValueError, RuntimeError, OSError, pd.errors.ParserError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = args.output_dir / "failure.json"
        if not failure.exists():
            failure.write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n")
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "bootstrap_valid": result["bootstrap_valid"],
                      "output_dir": str(args.output_dir.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

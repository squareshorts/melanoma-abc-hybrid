"""Tests of statistical semantics and strict inputs, using synthetic scores only."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from frozen_score_transport import (
    COHORTS, Cohort, ScoreOrder, run_analysis, validate_cohort_separation,
)


class FrozenTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def cohort(self, name, rows, models=("model_a", "model_b"), suffix=""):
        frame = pd.DataFrame(rows, columns=("image_id", "lesion_id", "y_true") + models)
        path = self.root / f"{name}{suffix}.csv"
        frame.to_csv(path, index=False)
        return Cohort.read(path, name, models)

    def ordinary_cohorts(self, alternate_external=False, suffix=""):
        cohorts = []
        for ci, name in enumerate(COHORTS):
            rows = []
            for lesion, (label, scores) in enumerate(((1, (.9, .3)), (0, (.7,)),
                                                      (1, (.75,)), (0, (.1, .2)))):
                for image, score in enumerate(scores):
                    if ci == 2 and alternate_external:
                        score = 1 - score
                    rows.append((f"{ci}_{lesion}_{image}", f"{ci}_L{lesion}", label,
                                 score, .8 * score + .1))
            cohorts.append(self.cohort(name, rows, suffix=suffix))
        return cohorts

    def test_whole_score_ties_and_lexicographic_cutoffs(self):
        scores = np.array([.9, .9, .8, .7, .7, .2])
        labels = np.array([1, 0, 1, 0, 1, 0])
        curve = ScoreOrder.from_arrays(scores, labels, np.arange(6)).curve(np.ones(6, int))
        self.assertTrue(math.isinf(curve.choose("specificity", 80)))
        self.assertEqual(curve.choose("specificity", 50), .8)
        self.assertEqual(curve.choose("sensitivity", 90), .7)
        rates, counts = curve.operating(np.array([.9, np.nextafter(.9, math.inf)]))
        np.testing.assert_array_equal(counts[0], [1, 2, 2, 1])
        np.testing.assert_array_equal(counts[1], [0, 3, 3, 0])
        self.assertEqual(rates[0, 0], 1 / 3)
        # A lower cutoff adding only false positives cannot win a tied-Se choice.
        scores = np.array([.95, .9, .85, .7])
        labels = np.array([1, 0, 0, 1])
        curve = ScoreOrder.from_arrays(scores, labels, np.arange(4)).curve(np.ones(4, int))
        self.assertEqual(curve.choose("specificity", 50), .95)

    def test_weighted_metrics_match_explicit_cluster_replication(self):
        scores = np.array([.8, .2, .8, .3, .7, .7])
        labels = np.array([1, 1, 0, 0, 1, 1])
        clusters = np.array([0, 0, 1, 1, 2, 2])
        weights = np.array([1, 3, 2])
        curve = ScoreOrder.from_arrays(scores, labels, clusters).curve(weights)
        indices = np.repeat(np.arange(6), weights[clusters])
        expected = [roc_auc_score(labels[indices], scores[indices]),
                    average_precision_score(labels[indices], scores[indices]),
                    brier_score_loss(labels[indices], scores[indices])]
        np.testing.assert_allclose(curve.metrics(), expected, atol=1e-14)
        for cutoff in [.8, .7, 0., math.inf]:
            pred = scores[indices] >= cutoff
            y = labels[indices]
            expected_counts = [sum(pred & (y == 1)), sum(~pred & (y == 1)),
                               sum(~pred & (y == 0)), sum(pred & (y == 0))]
            np.testing.assert_array_equal(curve.operating(np.array([cutoff]))[1][0], expected_counts)

    def test_zero_weight_groups_do_not_become_empirical_source_cutoffs(self):
        scores = np.array([.95, .9, .6, .2, .1])
        labels = np.array([1, 1, 0, 0, 1])
        weights = np.array([0, 2, 1, 3, 1])
        curve = ScoreOrder.from_arrays(scores, labels, np.arange(5)).curve(weights)
        copied = np.repeat(np.arange(5), weights)
        repeated_curve = ScoreOrder.from_arrays(scores[copied], labels[copied],
                                                np.arange(len(copied))).curve(np.ones(len(copied), int))
        np.testing.assert_array_equal(curve.thresholds(), repeated_curve.thresholds())

    def test_lesion_mean_and_cluster_resampling_have_distinct_units(self):
        c = self.ordinary_cohorts()[0]
        weights = np.array([2, 1, 0, 3])
        image_curve = c.orders["image_cluster"][0].curve(weights)
        lesion_curve = c.orders["lesion_mean"][0].curve(weights)
        self.assertEqual(image_curve.positives + image_curve.negatives,
                         int(np.sum(weights[c.image_clusters])))
        self.assertEqual(lesion_curve.positives + lesion_curve.negatives, int(weights.sum()))
        ordinary = np.ones(c.n_lesions, int)
        expected_mean = c.frame.groupby("lesion_id").model_a.mean().to_numpy()
        np.testing.assert_allclose(c.lesion_scores[:, 0], expected_mean)
        self.assertAlmostEqual(c.orders["lesion_mean"][0].curve(ordinary).metrics()[0],
                               roc_auc_score(c.lesion_labels, expected_mean))
        self.assertNotEqual(c.orders["lesion_mean"][0].curve(ordinary).metrics()[0],
                            c.orders["image_cluster"][0].curve(ordinary).metrics()[0])

    def test_all_equal_scores_have_defined_extreme_decisions(self):
        curve = ScoreOrder.from_arrays(np.full(4, .4), np.array([1, 0, 1, 0]),
                                        np.arange(4)).curve(np.ones(4, int))
        self.assertTrue(math.isinf(curve.choose("specificity", 90)))
        self.assertEqual(curve.choose("sensitivity", 95), .4)
        np.testing.assert_allclose(curve.metrics(), [.5, .5, .26])

    def test_thresholds_use_validation_only_and_runs_preserve_input_hashes(self):
        first = self.ordinary_cohorts(suffix="_first")
        second = self.ordinary_cohorts(alternate_external=True, suffix="_second")
        a, b = self.root / "run_a", self.root / "run_b"
        ma = run_analysis(first, a, bootstrap=12, seed=4)
        mb = run_analysis(second, b, bootstrap=12, seed=4)
        self.assertEqual((a / "thresholds.csv").read_bytes(), (b / "thresholds.csv").read_bytes())
        self.assertEqual((a / "source_roc.csv").read_bytes(), (b / "source_roc.csv").read_bytes())
        self.assertNotEqual((a / "operating_performance.csv").read_bytes(),
                            (b / "operating_performance.csv").read_bytes())
        self.assertTrue(ma["checks"]["input_hashes_unchanged"])
        self.assertTrue(mb["checks"]["input_hashes_unchanged"])
        metrics = pd.read_csv(a / "operating_performance.csv")
        self.assertEqual(set(metrics.cohort), set(COHORTS))
        target = metrics[metrics.cohort != "ham_validation"]
        self.assertTrue(target.sensitivity_ci_low.notna().all())
        self.assertTrue(target.specificity_ci_low.notna().all())

    def test_degenerate_resamples_are_retried_as_complete_paired_draws(self):
        cohorts = []
        for ci, name in enumerate(COHORTS):
            cohorts.append(self.cohort(name, [(f"{ci}_p", f"{ci}_p", 1, .8, .6),
                                               (f"{ci}_n", f"{ci}_n", 0, .2, .4)]))
        manifest = run_analysis(cohorts, self.root / "bootstrap", bootstrap=24, seed=31)
        self.assertEqual(manifest["bootstrap_valid"], 24)
        self.assertGreater(manifest["bootstrap_attempts"], 24)
        self.assertEqual(manifest["bootstrap_rejected_joint_draws"],
                         manifest["bootstrap_attempts"] - 24)
        self.assertTrue(all(v > 0 for v in manifest["degenerate_draws_by_cohort"].values()))
        with self.assertRaisesRegex(RuntimeError, "Unable to obtain"):
            run_analysis(cohorts, self.root / "too_few_attempts", bootstrap=24,
                         seed=31, max_attempts=1)

    def test_invalid_inputs_are_not_silently_repaired(self):
        valid = [("a", "p", 1, .8, .7), ("b", "n", 0, .1, .2)]
        invalid = {
            "duplicate_image": [valid[0], ("a", "n", 0, .1, .2)],
            "conflicting_lesion": [valid[0], ("b", "p", 0, .1, .2)],
            "missing_probability": [valid[0], ("b", "n", 0, "", .2)],
            "invalid_probability": [valid[0], ("b", "n", 0, 1.01, .2)],
            "invalid_label": [valid[0], ("b", "n", 2, .1, .2)],
            "missing_id": [valid[0], ("", "n", 0, .1, .2)],
            "single_class": [valid[0], ("b", "n", 1, .1, .2)],
        }
        for name, rows in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.cohort(name, rows)
        path = self.root / "duplicate_columns.csv"
        path.write_text("image_id,lesion_id,y_true,model_a,model_a\na,p,1,.8,.7\n")
        with self.assertRaisesRegex(ValueError, "duplicate CSV column"):
            Cohort.read(path, "invalid", ["model_a"])

    def test_source_test_separation_and_no_run_overwriting(self):
        cohorts = self.ordinary_cohorts()
        cohorts[1].frame.loc[0, "image_id"] = cohorts[0].frame.iloc[0].image_id
        with self.assertRaisesRegex(ValueError, "Shared image IDs"):
            validate_cohort_separation(cohorts)
        cohorts = self.ordinary_cohorts()
        cohorts[1].lesion_ids[0] = cohorts[0].lesion_ids[0]
        with self.assertRaisesRegex(ValueError, "Shared HAM lesion"):
            validate_cohort_separation(cohorts)
        cohorts = self.ordinary_cohorts()
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("Existing analysis")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            run_analysis(cohorts, output, bootstrap=0)
        self.assertEqual(marker.read_text(), "Existing analysis")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from aigc_detector.metrics import (
    average_precision,
    basic_binary_classification_metrics,
    roc_auc,
)
from aigc_detector.selection import checkpoint_selection_score


class MetricTests(unittest.TestCase):
    def test_auc_and_average_precision_known_values(self):
        labels = np.asarray([0, 1, 0, 1, 1, 0, 0, 1])
        scores = np.asarray([0.1, 0.8, 0.3, 0.7, 0.4, 0.2, 0.6, 0.9])
        self.assertAlmostEqual(roc_auc(labels, scores), 0.9375)
        self.assertAlmostEqual(average_precision(labels, scores), 0.95)

    def test_binary_metrics_include_expected_keys(self):
        metrics = basic_binary_classification_metrics(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.4, 0.6, 0.9],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_source_balanced_selection_does_not_hide_weak_domain(self):
        metrics = {
            "roc_auc": 0.98,
            "balanced_accuracy": 0.9,
            "source_auc_birdy654": 0.99,
            "source_auc_external_pilot": 0.81,
        }
        self.assertAlmostEqual(
            checkpoint_selection_score(metrics, "mean_source_auc"), 0.90
        )
        self.assertAlmostEqual(checkpoint_selection_score(metrics, "roc_auc"), 0.98)


if __name__ == "__main__":
    unittest.main()

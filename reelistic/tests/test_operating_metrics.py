import unittest

import numpy as np

from aigc_detector.metrics import (
    basic_binary_classification_metrics,
    binary_classification_metrics,
    operating_point_metrics,
    roc_curve_points,
)


class OperatingMetricTests(unittest.TestCase):
    def test_perfect_classifier_operating_points(self):
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.r_[np.linspace(0.0, 0.49, 100), np.linspace(0.5, 1.0, 100)]
        result = operating_point_metrics(labels, scores)
        self.assertEqual(result["tpr_at_1pct_fpr"], 1.0)
        self.assertEqual(result["actual_fpr_at_1pct_target"], 0.0)
        self.assertEqual(result["fpr_at_99pct_tpr"], 0.0)

    def test_tied_scores_are_thresholded_as_one_group(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        curve = roc_curve_points(labels, scores)
        self.assertEqual(curve["fpr"], [0.0, 0.5, 1.0])
        self.assertEqual(curve["tpr"], [0.0, 0.5, 1.0])

    def test_curves_are_optional(self):
        basic = binary_classification_metrics([0, 1], [0.1, 0.9])
        detailed = binary_classification_metrics(
            [0, 1], [0.1, 0.9], include_curves=True
        )
        self.assertNotIn("roc_curve", basic)
        self.assertIn("roc_curve", detailed)
        self.assertIn("tpr_at_1pct_fpr", basic)

    def test_basic_metrics_are_the_shared_subset(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.6, 0.9]
        basic = basic_binary_classification_metrics(labels, scores)
        detailed = binary_classification_metrics(labels, scores)
        for key in ("accuracy", "balanced_accuracy", "f1", "average_precision", "roc_auc"):
            self.assertEqual(basic[key], detailed[key])
        self.assertNotIn("ece", basic)


if __name__ == "__main__":
    unittest.main()

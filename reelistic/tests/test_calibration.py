import unittest

import numpy as np

from aigc_detector.calibration import (
    balanced_group_indices,
    balanced_indices,
    fit_bounded_temperature,
)


class CalibrationTests(unittest.TestCase):
    def test_balanced_indices_use_equal_class_counts(self):
        labels = np.asarray([0, 0, 1, 1, 1, 1])
        selected = balanced_indices(labels)
        self.assertEqual(labels[selected].tolist().count(0), 2)
        self.assertEqual(labels[selected].tolist().count(1), 2)

    def test_group_balancing_uses_equal_source_class_counts(self):
        labels = np.asarray([0, 0, 1, 1, 0, 1, 1, 1])
        groups = np.asarray(["large"] * 4 + ["small"] * 4)
        selected = balanced_group_indices(labels, groups)
        pairs = [(groups[index], labels[index]) for index in selected]
        self.assertEqual(pairs.count(("large", 0)), 1)
        self.assertEqual(pairs.count(("large", 1)), 1)
        self.assertEqual(pairs.count(("small", 0)), 1)
        self.assertEqual(pairs.count(("small", 1)), 1)

    def test_accepted_temperature_improves_log_loss(self):
        logits = np.asarray([[5, 0], [4, 0], [0, 2], [2, 0]], dtype=float)
        labels = np.asarray([0, 0, 1, 1])
        report = fit_bounded_temperature(logits, labels)
        self.assertEqual(report["status"], "accepted")
        self.assertGreaterEqual(report["temperature"], 0.05)
        self.assertLessEqual(report["temperature"], 10.0)
        self.assertLess(report["after"]["log_loss"], report["before"]["log_loss"])

    def test_boundary_solution_is_rejected(self):
        logits = np.asarray([[1, 0], [2, 0], [0, 1], [0, 2]], dtype=float)
        labels = np.asarray([0, 0, 1, 1])
        report = fit_bounded_temperature(logits, labels)
        self.assertEqual(report["status"], "boundary_rejected")
        self.assertEqual(report["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from aigc_detector.selection import (
    checkpoint_selection_components,
    checkpoint_selection_score,
)


class CheckpointSelectionTests(unittest.TestCase):
    def test_robust_score_penalizes_weakest_source_and_negative_fusion_gap(self):
        metrics = {
            "roc_auc": 0.98,
            "balanced_accuracy": 0.90,
            "source_auc_birdy654": 0.99,
            "source_auc_external_pilot": 0.81,
            "fusion_minus_semantic_auc_birdy654": 0.01,
            "fusion_minus_semantic_auc_external_pilot": -0.02,
        }
        components = checkpoint_selection_components(metrics)
        self.assertAlmostEqual(components["mean_source_auc"], 0.90)
        self.assertAlmostEqual(components["weakest_source_auc"], 0.81)
        self.assertAlmostEqual(components["fusion_gap_penalty"], 0.02)
        self.assertAlmostEqual(
            checkpoint_selection_score(metrics, "robust_source_auc"),
            0.5 * 0.90 + 0.5 * 0.81 - 0.25 * 0.02,
        )

    def test_balanced_candidate_beats_same_mean_with_collapsed_source(self):
        balanced = {
            "balanced_accuracy": 0.9,
            "source_auc_a": 0.90,
            "source_auc_b": 0.90,
        }
        collapsed = {
            "balanced_accuracy": 0.9,
            "source_auc_a": 0.99,
            "source_auc_b": 0.81,
        }
        self.assertGreater(
            checkpoint_selection_score(balanced, "robust_source_auc"),
            checkpoint_selection_score(collapsed, "robust_source_auc"),
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            checkpoint_selection_score({"balanced_accuracy": 0.5}, "unknown")


if __name__ == "__main__":
    unittest.main()

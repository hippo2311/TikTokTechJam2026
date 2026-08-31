from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from aigc_detector.data.augmentations import (
    StochasticCompressionAugment,
    downsample_to_explicit_size,
    real_jpeg_roundtrip,
)
from aigc_detector.data.handcrafted import compute_quality_features
from aigc_detector.metrics import binary_classification_metrics
from aigc_detector.models.ensemble import AIGCDetectionEnsemble
from aigc_detector.models.fusion_head import (
    QualityAwareFusion,
    branch_disagreement_feature,
)
from aigc_detector.models.noise_branch import NoiseBranch
from aigc_detector.models.semantic_branch import SEMANTIC_BACKBONE_ALIASES, SemanticBranch


class RealisticAugmentationTests(unittest.TestCase):
    def test_clean_path_and_real_jpeg_preserve_expected_shape(self):
        image = Image.new("RGB", (32, 32), color=(64, 128, 192))
        tensor = StochasticCompressionAugment(128, clean_probability=1.0)(image)
        self.assertEqual(tuple(tensor.shape), (3, 128, 128))
        jpeg = real_jpeg_roundtrip(image, quality=50)
        self.assertEqual(jpeg.size, image.size)
        self.assertEqual(jpeg.mode, "RGB")

    def test_explicit_downsampling_can_reach_true_eight_pixels(self):
        image = Image.new("RGB", (32, 32))
        resize_calls = []
        original_resize = Image.Image.resize

        def recording_resize(instance, size, *args, **kwargs):
            resize_calls.append(size)
            return original_resize(instance, size, *args, **kwargs)

        with patch("random.choice", return_value=8), patch.object(
            Image.Image, "resize", recording_resize
        ):
            result = downsample_to_explicit_size(image)
        self.assertIn((8, 8), resize_calls)
        self.assertEqual(result.size, (32, 32))


class ImprovedArchitectureTests(unittest.TestCase):
    def test_explicit_disagreement_is_bounded_and_zero_initialized(self):
        identical = torch.tensor([[[0.0, 2.0]] * 4])
        conflicting = torch.tensor(
            [[[0.0, 5.0], [5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]]
        )
        self.assertAlmostEqual(float(branch_disagreement_feature(identical)), 0.0)
        self.assertGreater(float(branch_disagreement_feature(conflicting)), 0.9)

        baseline = QualityAwareFusion(explicit_disagreement=False).eval()
        candidate = QualityAwareFusion(explicit_disagreement=True).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(torch.count_nonzero(candidate.disagreement_gate.weight) == 0)
        self.assertTrue(torch.count_nonzero(candidate.disagreement_gate.bias) == 0)

    def test_clip_b32_alias_uses_clip_input_and_normalization(self):
        branch = SemanticBranch(backbone_name="clip_vit_b32", pretrained=False).eval()
        self.assertEqual(branch.backbone_name, "vit_base_patch32_clip_224.openai")
        self.assertEqual(branch.input_size, 224)
        self.assertAlmostEqual(float(branch.image_mean.flatten()[0]), 0.48145466, places=5)
        with torch.inference_mode():
            outputs = branch(torch.rand(1, 3, 128, 128))
        self.assertEqual(tuple(outputs["logits"].shape), (1, 2))

    def test_clip_l14_alias_is_available_without_loading_large_weights(self):
        self.assertEqual(
            SEMANTIC_BACKBONE_ALIASES["clip_vit_l14"],
            "vit_large_patch14_clip_224.openai",
        )

    def test_quality_features_and_improved_noise_are_finite(self):
        images = torch.rand(2, 3, 128, 128)
        quality = compute_quality_features(images)
        noise = NoiseBranch(version="improved")(images)
        self.assertEqual(tuple(quality.shape), (2, 8))
        self.assertEqual(tuple(noise["logits"].shape), (2, 2))
        self.assertTrue(torch.isfinite(quality).all())
        self.assertTrue(torch.isfinite(noise["logits"]).all())

    def test_quality_gate_is_soft_and_can_mask_noise(self):
        model = AIGCDetectionEnsemble(
            semantic_pretrained=False,
            texture_pretrained=False,
            quality_aware_fusion=True,
            noise_version="improved",
            noise_enabled=False,
        ).eval()
        with torch.inference_mode():
            outputs = model(torch.rand(2, 3, 128, 128))
        self.assertEqual(tuple(outputs.gate_weights.shape), (2, 4))
        self.assertTrue(torch.allclose(outputs.gate_weights.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.all(outputs.gate_weights[:, 2] == 0))


class RobustnessMetricTests(unittest.TestCase):
    def test_calibration_metrics_are_reported(self):
        metrics = binary_classification_metrics(
            np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9])
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIn("ece", metrics)
        self.assertIn("brier", metrics)
        self.assertLess(metrics["ece"], 0.2)


if __name__ == "__main__":
    unittest.main()

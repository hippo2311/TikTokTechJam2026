import unittest

import torch
import torch.nn as nn

from aigc_detector.models.ensemble import EnsembleOutputs, TemperatureScaler
from aigc_detector.models.seed_ensemble import SharedBackboneSeedEnsemble
from aigc_detector.param_budget import shared_seed_ensemble_breakdown


class ToySemantic(nn.Module):
    def __init__(self, probe_bias: float) -> None:
        super().__init__()
        self.backbone = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.backbone.weight, 2.0)
        self.probe = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.probe.weight.copy_(torch.tensor([[0.0], [probe_bias]]))
        self.backbone_name = "toy"
        self.extract_calls = 0

    def extract_features(self, images):
        self.extract_calls += 1
        return self.backbone(images)

    def classify_features(self, features):
        return {"features": features, "logits": self.probe(features)}


class ToySeed(nn.Module):
    def __init__(self, bias: float) -> None:
        super().__init__()
        self.semantic_branch = ToySemantic(bias)
        self.seed_weight = nn.Parameter(torch.tensor(bias))
        self.temperature_scaler = TemperatureScaler()

    def forward(self, images, apply_temperature=False, semantic_features=None):
        semantic = self.semantic_branch.classify_features(semantic_features)
        logits = semantic["logits"] + torch.stack(
            [torch.zeros_like(self.seed_weight), self.seed_weight]
        )
        zeros = torch.zeros_like(logits)
        return EnsembleOutputs(
            logits=logits,
            texture_logits=zeros,
            frequency_logits=zeros,
            noise_logits=zeros,
            semantic_logits=semantic["logits"],
            texture_projections=zeros,
            handcrafted_features=zeros,
            quality_features=zeros,
            gate_weights=torch.full((images.shape[0], 4), 0.25),
        )


class SharedSeedEnsembleTests(unittest.TestCase):
    def test_averages_raw_logits_and_runs_backbone_once(self):
        seeds = [ToySeed(1.0), ToySeed(2.0), ToySeed(3.0)]
        model = SharedBackboneSeedEnsemble(seeds)
        model.temperature_scaler.temperature.data.fill_(2.0)
        output = model(torch.ones(2, 1), apply_temperature=True)

        # Raw positive logits are [3, 6, 9], mean 6, then T=2 -> 3.
        torch.testing.assert_close(output.logits[:, 1], torch.full((2,), 3.0))
        self.assertEqual(seeds[0].semantic_branch.extract_calls, 1)
        self.assertEqual(seeds[1].semantic_branch.extract_calls, 0)
        self.assertEqual(seeds[2].semantic_branch.extract_calls, 0)
        self.assertIs(
            seeds[0].semantic_branch.backbone,
            seeds[2].semantic_branch.backbone,
        )

    def test_parameter_accounting_deduplicates_shared_backbone(self):
        seeds = [ToySeed(1.0), ToySeed(2.0), ToySeed(3.0)]
        shared = seeds[0].semantic_branch.backbone
        seeds[1].semantic_branch.backbone = shared
        seeds[2].semantic_branch.backbone = shared
        breakdown = shared_seed_ensemble_breakdown(seeds)
        self.assertLess(
            breakdown["total_unique_parameters"],
            breakdown["naive_unshared_parameters"],
        )
        self.assertEqual(breakdown["shared_semantic_backbone"], 1)


if __name__ == "__main__":
    unittest.main()

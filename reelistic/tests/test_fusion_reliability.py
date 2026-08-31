from __future__ import annotations

import unittest

import torch

from cluster.train_cluster import fusion_reliability_penalty


class FusionReliabilityTests(unittest.TestCase):
    def test_penalty_only_activates_when_fusion_is_worse(self):
        labels = torch.tensor([0, 1])
        semantic = torch.tensor([[4.0, -1.0], [-1.0, 4.0]])
        worse_fusion = torch.zeros(2, 2, requires_grad=True)
        penalty = fusion_reliability_penalty(worse_fusion, semantic, labels)
        self.assertGreater(float(penalty), 0.0)
        penalty.backward()
        self.assertIsNotNone(worse_fusion.grad)

        better_fusion = torch.tensor([[8.0, -2.0], [-2.0, 8.0]])
        self.assertAlmostEqual(
            float(fusion_reliability_penalty(better_fusion, semantic, labels)),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()

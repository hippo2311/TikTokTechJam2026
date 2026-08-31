from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aigc_detector.param_budget import print_branch_params


class FusionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_classes: int = 2,
        semantic_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.semantic_weight = semantic_weight
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        print_branch_params("fusion", self)

    def forward(
        self,
        texture_logits: torch.Tensor,
        frequency_logits: torch.Tensor,
        noise_logits: torch.Tensor,
        semantic_logits: torch.Tensor,
        handcrafted_features: torch.Tensor,
    ) -> torch.Tensor:
        semantic_logits = semantic_logits * self.semantic_weight
        fusion_input = torch.cat(
            [texture_logits, frequency_logits, noise_logits, semantic_logits, handcrafted_features],
            dim=1,
        )
        return self.net(fusion_input)


class QualityAwareFusion(nn.Module):
    """Soft per-image branch routing driven by logits and quality features."""

    def __init__(
        self,
        num_branches: int = 4,
        num_classes: int = 2,
        quality_dim: int = 8,
        hidden_dim: int = 48,
        branch_dropout: float = 0.15,
        explicit_disagreement: bool = False,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.branch_dropout = branch_dropout
        self.explicit_disagreement = explicit_disagreement
        self.quality_norm = nn.LayerNorm(quality_dim)
        gate_input_dim = num_branches * num_classes + quality_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, num_branches),
        )
        self.logit_scales = nn.Parameter(torch.zeros(num_branches))
        if explicit_disagreement:
            self.disagreement_gate = nn.Linear(1, num_branches)
            nn.init.zeros_(self.disagreement_gate.weight)
            nn.init.zeros_(self.disagreement_gate.bias)
        else:
            self.disagreement_gate = None
        print_branch_params("quality_fusion", self)

    def forward(
        self,
        texture_logits: torch.Tensor,
        frequency_logits: torch.Tensor,
        noise_logits: torch.Tensor,
        semantic_logits: torch.Tensor,
        quality_features: torch.Tensor,
        available_branches: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch_logits = torch.stack(
            [texture_logits, frequency_logits, noise_logits, semantic_logits], dim=1
        )
        normalized_quality = self.quality_norm(quality_features)
        gate_input = torch.cat([branch_logits.flatten(1), normalized_quality], dim=1)
        gate_logits = self.gate(gate_input)
        if self.disagreement_gate is not None:
            disagreement = branch_disagreement_feature(branch_logits)
            gate_logits = gate_logits + self.disagreement_gate(disagreement)

        if available_branches is not None:
            available = available_branches.to(
                device=gate_logits.device, dtype=torch.bool
            ).view(1, -1)
            gate_logits = gate_logits.masked_fill(~available, -1e4)

        if self.training and self.branch_dropout > 0:
            keep = torch.rand_like(gate_logits).ge(self.branch_dropout)
            if available_branches is not None:
                keep = keep & available
            no_branch = ~keep.any(dim=1)
            if no_branch.any():
                fallback = gate_logits.argmax(dim=1)
                keep[no_branch, fallback[no_branch]] = True
            gate_logits = gate_logits.masked_fill(~keep, -1e4)

        gate_weights = F.softmax(gate_logits, dim=1)
        scales = self.logit_scales.clamp(-2.0, 2.0).exp().view(1, -1, 1)
        logits = (gate_weights.unsqueeze(-1) * branch_logits * scales).sum(dim=1)
        return logits, gate_weights


def branch_disagreement_feature(branch_logits: torch.Tensor) -> torch.Tensor:
    """Normalized variance of branch FAKE probabilities in the range [0, 1]."""
    fake_probabilities = F.softmax(branch_logits, dim=-1)[..., 1]
    return 4.0 * fake_probabilities.var(dim=1, unbiased=False, keepdim=True)

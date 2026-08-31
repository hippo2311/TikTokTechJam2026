from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {tuple(features.shape)}")

        features = F.normalize(features, dim=1)
        logits = features @ features.t() / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).float().to(features.device)
        diagonal = torch.eye(features.size(0), device=features.device)
        positive_mask = positive_mask - diagonal

        exp_logits = torch.exp(logits) * (1.0 - diagonal)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

        positive_counts = positive_mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts
        return -mean_log_prob_pos.mean()

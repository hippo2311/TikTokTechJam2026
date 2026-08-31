from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aigc_detector.param_budget import print_branch_params


class FixedSRM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
                [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, -2, 4, -2, 0], [0, 1, -2, 1, 0]],
            ],
            dtype=torch.float32,
        )
        kernels = kernels.unsqueeze(1) / 12.0
        self.register_buffer("kernels", kernels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1, keepdim=True)
        return F.conv2d(gray, self.kernels, padding=2)


class NoiseBranch(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        num_classes: int = 2,
        version: str = "legacy",
        truncation: float = 0.25,
    ) -> None:
        super().__init__()
        if version not in {"legacy", "improved"}:
            raise ValueError(f"Unsupported noise branch version: {version}")
        self.version = version
        self.truncation = truncation
        self.srm = FixedSRM()
        if version == "legacy":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(base_channels * 4, num_classes)
        else:
            self.improved_encoder = nn.Sequential(
                nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.improved_classifier = nn.Sequential(
                nn.LayerNorm(base_channels * 4 + 6),
                nn.Linear(base_channels * 4 + 6, 64),
                nn.SiLU(inplace=True),
                nn.Dropout(p=0.15),
                nn.Linear(64, num_classes),
            )
        print_branch_params("noise", self)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        residuals = self.srm(images)
        if self.version == "legacy":
            features = self.encoder(residuals).flatten(1)
            logits = self.classifier(features)
            return {"features": features, "logits": logits}

        truncated = residuals.clamp(-self.truncation, self.truncation)
        mean = truncated.mean(dim=(-2, -1), keepdim=True)
        std = truncated.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        normalized = (truncated - mean) / std
        encoded = self.improved_encoder(normalized).flatten(1)
        energy = torch.cat(
            [
                residuals.abs().mean(dim=(-2, -1)),
                residuals.std(dim=(-2, -1)),
            ],
            dim=1,
        )
        features = torch.cat([encoded, energy], dim=1)
        logits = self.improved_classifier(features)
        return {"features": features, "logits": logits}

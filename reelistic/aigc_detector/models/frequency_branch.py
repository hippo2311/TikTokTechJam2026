from __future__ import annotations

import torch
import torch.nn as nn

from aigc_detector.param_budget import print_branch_params


class FrequencyBranch(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
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
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 4, num_classes)
        print_branch_params("frequency", self)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self._to_log_spectrum(images)
        features = self.encoder(spectrum).flatten(1)
        logits = self.classifier(features)
        return {"features": features, "logits": logits}

    def _to_log_spectrum(self, images: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(images, norm="ortho")
        shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        return torch.log1p(torch.abs(shifted))

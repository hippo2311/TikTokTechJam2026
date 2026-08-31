from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from aigc_detector.data.handcrafted import compute_handcrafted_features, compute_quality_features
from aigc_detector.models.frequency_branch import FrequencyBranch
from aigc_detector.models.fusion_head import FusionMLP, QualityAwareFusion
from aigc_detector.models.noise_branch import NoiseBranch
from aigc_detector.models.semantic_branch import SemanticBranch
from aigc_detector.models.texture_branch import TextureBranch
from aigc_detector.param_budget import enforce_budget


class TemperatureScaler(nn.Module):
    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-3)


@dataclass
class EnsembleOutputs:
    logits: torch.Tensor
    texture_logits: torch.Tensor
    frequency_logits: torch.Tensor
    noise_logits: torch.Tensor
    semantic_logits: torch.Tensor
    texture_projections: torch.Tensor
    handcrafted_features: torch.Tensor
    quality_features: torch.Tensor | None = None
    gate_weights: torch.Tensor | None = None


class AIGCDetectionEnsemble(nn.Module):
    def __init__(
        self,
        semantic_backbone: str = "mobilenetv3_small_100.lamb_in1k",
        texture_backbone: str = "resnet18",
        semantic_pretrained: bool = True,
        texture_pretrained: bool = True,
        image_size: int = 128,
        top_k_patches: int = 1,
        quality_aware_fusion: bool = False,
        noise_version: str = "legacy",
        noise_enabled: bool = True,
        branch_dropout: float = 0.15,
        explicit_gate_disagreement: bool = False,
    ) -> None:
        super().__init__()
        patch_size = max(32, image_size // 2)
        patch_stride = max(16, patch_size // 2)
        self.texture_branch = TextureBranch(
            pretrained=texture_pretrained,
            backbone_name=texture_backbone,
            top_k_patches=top_k_patches,
            patch_size=patch_size,
            patch_stride=patch_stride,
            input_size=image_size,
        )
        self.frequency_branch = FrequencyBranch()
        self.noise_branch = NoiseBranch(version=noise_version) if noise_enabled else None
        self.semantic_branch = SemanticBranch(
            backbone_name=semantic_backbone,
            pretrained=semantic_pretrained,
        )
        self.quality_aware_fusion = quality_aware_fusion
        if quality_aware_fusion:
            self.fusion_head = QualityAwareFusion(
                branch_dropout=branch_dropout,
                explicit_disagreement=explicit_gate_disagreement,
            )
            self.register_buffer(
                "available_branches",
                torch.tensor([True, True, noise_enabled, True], dtype=torch.bool),
            )
        else:
            if not noise_enabled:
                raise ValueError("Legacy fusion requires the noise branch to be enabled.")
            self.fusion_head = FusionMLP()
        self.temperature_scaler = TemperatureScaler()

        enforce_budget(
            texture_model=self.texture_branch,
            freq_model=self.frequency_branch,
            noise_model=self.noise_branch,
            semantic_model=self.semantic_branch,
            fusion_model=self.fusion_head,
        )

    def forward(
        self,
        images: torch.Tensor,
        apply_temperature: bool = False,
        semantic_features: torch.Tensor | None = None,
    ) -> EnsembleOutputs:
        texture = self.texture_branch(images)
        frequency = self.frequency_branch(images)
        if self.noise_branch is None:
            noise = {
                "features": images.new_zeros((images.size(0), 0)),
                "logits": images.new_zeros((images.size(0), 2)),
            }
        else:
            noise = self.noise_branch(images)
        semantic = (
            self.semantic_branch(images)
            if semantic_features is None
            else self.semantic_branch.classify_features(semantic_features)
        )
        handcrafted = compute_handcrafted_features(images)
        quality = None
        gate_weights = None
        if self.quality_aware_fusion:
            quality = compute_quality_features(images)
            logits, gate_weights = self.fusion_head(
                texture_logits=texture["logits"],
                frequency_logits=frequency["logits"],
                noise_logits=noise["logits"],
                semantic_logits=semantic["logits"],
                quality_features=quality,
                available_branches=self.available_branches,
            )
        else:
            logits = self.fusion_head(
                texture_logits=texture["logits"],
                frequency_logits=frequency["logits"],
                noise_logits=noise["logits"],
                semantic_logits=semantic["logits"],
                handcrafted_features=handcrafted,
            )
        if apply_temperature:
            logits = self.temperature_scaler(logits)
        return EnsembleOutputs(
            logits=logits,
            texture_logits=texture["logits"],
            frequency_logits=frequency["logits"],
            noise_logits=noise["logits"],
            semantic_logits=semantic["logits"],
            texture_projections=texture["projections"],
            handcrafted_features=handcrafted,
            quality_features=quality,
            gate_weights=gate_weights,
        )

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(images, apply_temperature=True)
        return F.softmax(outputs.logits, dim=1)[:, 1]

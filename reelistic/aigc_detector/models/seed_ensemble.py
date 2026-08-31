"""Variance-reduction ensemble with one shared frozen semantic backbone pass."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from aigc_detector.models.ensemble import (
    AIGCDetectionEnsemble,
    EnsembleOutputs,
    TemperatureScaler,
)
from aigc_detector.param_budget import shared_seed_ensemble_breakdown


class SharedBackboneSeedEnsemble(nn.Module):
    """Average raw seed logits, then apply one ensemble-level temperature.

    The frozen semantic backbone is shared by object identity and evaluated once.
    Every seed keeps its own semantic probe, forensic branches, and fusion gate.
    Individual checkpoint temperatures are intentionally discarded.
    """

    def __init__(self, models: Sequence[AIGCDetectionEnsemble]) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("A seed ensemble requires at least two models.")
        self.models = nn.ModuleList(models)
        self._validate_and_share_backbone()
        for model in self.models:
            model.temperature_scaler = nn.Identity()
        self.temperature_scaler = TemperatureScaler()
        breakdown = shared_seed_ensemble_breakdown(
            self.models, extra_models=[self.temperature_scaler]
        )
        self.parameter_breakdown = breakdown
        print("[params] shared_seed_ensemble:", breakdown, flush=True)

    def _validate_and_share_backbone(self) -> None:
        reference = self.models[0].semantic_branch
        reference_state = reference.backbone.state_dict()
        for index, model in enumerate(self.models[1:], start=1):
            semantic = model.semantic_branch
            if semantic.backbone_name != reference.backbone_name:
                raise ValueError("All seeds must use the same semantic backbone.")
            candidate_state = semantic.backbone.state_dict()
            if candidate_state.keys() != reference_state.keys() or any(
                not torch.equal(reference_state[key], candidate_state[key])
                for key in reference_state
            ):
                raise ValueError(
                    f"Seed {index} has different frozen semantic-backbone weights."
                )
            semantic.backbone = reference.backbone

    def forward(self, images: torch.Tensor, apply_temperature: bool = False) -> EnsembleOutputs:
        semantic_features = self.models[0].semantic_branch.extract_features(images)
        outputs = [
            model(images, apply_temperature=False, semantic_features=semantic_features)
            for model in self.models
        ]

        def average(attribute: str) -> torch.Tensor:
            return torch.stack([getattr(output, attribute) for output in outputs]).mean(0)

        logits = average("logits")
        if apply_temperature:
            logits = self.temperature_scaler(logits)
        gate_weights = None
        if all(output.gate_weights is not None for output in outputs):
            gate_weights = average("gate_weights")
        quality_features = outputs[0].quality_features
        return EnsembleOutputs(
            logits=logits,
            texture_logits=average("texture_logits"),
            frequency_logits=average("frequency_logits"),
            noise_logits=average("noise_logits"),
            semantic_logits=average("semantic_logits"),
            texture_projections=average("texture_projections"),
            handcrafted_features=outputs[0].handcrafted_features,
            quality_features=quality_features,
            gate_weights=gate_weights,
        )

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(images, apply_temperature=True)
        return F.softmax(outputs.logits, dim=1)[:, 1]

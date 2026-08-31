from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aigc_detector.param_budget import print_branch_params

SEMANTIC_BACKBONE_ALIASES = {
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "clip_vit_b32": "vit_base_patch32_clip_224.openai",
    "clip_vit_l14": "vit_large_patch14_clip_224.openai",
}


class SemanticBranch(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenetv3_small_100.lamb_in1k",
        pretrained: bool = True,
        probe_hidden_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("SemanticBranch requires timm. Install requirements first.") from exc

        resolved_backbone = SEMANTIC_BACKBONE_ALIASES.get(backbone_name, backbone_name)
        self.backbone = timm.create_model(
            resolved_backbone,
            pretrained=pretrained,
            num_classes=0,
        )
        self.backbone.eval()
        pretrained_config = getattr(self.backbone, "pretrained_cfg", {}) or {}
        configured_size = pretrained_config.get("input_size", (3, 224, 224))[-1]
        is_fixed_vit = resolved_backbone.startswith("vit_")
        probe_size = configured_size if is_fixed_vit else 128
        with torch.inference_mode():
            probe_output = self.backbone(torch.zeros(1, 3, probe_size, probe_size))
        if probe_output.ndim != 2:
            raise ValueError(
                f"Expected pooled 2D features from {resolved_backbone}, "
                f"got shape {tuple(probe_output.shape)}"
            )
        feature_dim = probe_output.shape[1]

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.probe = nn.Sequential(
            nn.Linear(feature_dim, probe_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(probe_hidden_dim, num_classes),
        )
        self.backbone_name = resolved_backbone
        self.input_size = configured_size if is_fixed_vit else None
        if "clip" in resolved_backbone:
            image_mean = pretrained_config.get(
                "mean", (0.48145466, 0.4578275, 0.40821073)
            )
            image_std = pretrained_config.get(
                "std", (0.26862954, 0.26130258, 0.27577711)
            )
        else:
            image_mean = (0.485, 0.456, 0.406)
            image_std = (0.229, 0.224, 0.225)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std).view(1, 3, 1, 1),
        )
        print_branch_params("semantic", self)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run the frozen semantic backbone once for reusable CLIP features."""
        if self.input_size is not None and images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        normalized = (images - self.image_mean) / self.image_std
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(normalized)
        return features

    def classify_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply this seed's lightweight probe to shared backbone features."""
        logits = self.probe(features)
        return {"features": features, "logits": logits}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.classify_features(self.extract_features(images))

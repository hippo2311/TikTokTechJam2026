from __future__ import annotations

import math
import re
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class SRMResidualAdapter(nn.Module):
    """Fixed high-pass residual filters followed by a small trainable CNN."""

    def __init__(self, feature_dim: int, mean: tuple[float, ...], std: tuple[float, ...]):
        super().__init__()
        # Three zero-sum spatial-rich-model-inspired high-pass filters. They
        # are buffers, therefore they remain fixed and add no trainable bulk.
        kernels = torch.tensor([
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[0, 0, 1, 0, 0], [0, 1, -2, 1, 0], [1, -2, 4, -2, 1], [0, 1, -2, 1, 0], [0, 0, 1, 0, 0]],
        ], dtype=torch.float32).unsqueeze(1) / 12.0
        self.register_buffer("srm_kernels", kernels)
        self.register_buffer("normalization_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("normalization_std", torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer("laplacian_kernel", torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, feature_dim), nn.GELU(),
        )

    def forward(self, normalized_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = (normalized_image * self.normalization_std + self.normalization_mean).clamp(0, 1)
        gray = image.mean(dim=1, keepdim=True)
        residual = F.conv2d(gray, self.srm_kernels, padding=2)
        features = self.encoder(residual)
        residual_energy = residual.abs().mean(dim=(1, 2, 3), keepdim=False)
        laplacian_energy = F.conv2d(gray, self.laplacian_kernel, padding=1).square().mean(dim=(1, 2, 3), keepdim=False)
        quality = torch.log1p(torch.stack((residual_energy, laplacian_energy), dim=1))
        return features, quality


class DINOv3Forensic(nn.Module):
    """DINOv3 ViT-L/16 with three tapped multi-scale forensic branches."""
    def __init__(self, config: dict):
        super().__init__()
        model_cfg, head_cfg = config["model"], config["forensic_head"]
        self.backbone = AutoModel.from_pretrained(model_cfg["backbone"])
        if model_cfg.get("gradient_checkpointing", False):
            if not hasattr(self.backbone, "gradient_checkpointing_enable"):
                raise RuntimeError("This backbone does not support gradient checkpointing")
            self.backbone.gradient_checkpointing_enable()
        self.tapped_layers = tuple(model_cfg["tapped_layers"])
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        hidden_size, local_dim = int(self.backbone.config.hidden_size), int(head_cfg["local_feature_dim"])
        self.local_branches = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_size, local_dim, kernel_size=head_cfg["local_aggregation_kernel"], padding=head_cfg["local_aggregation_padding"]), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()) for _ in self.tapped_layers])
        self.cls_branches = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, head_cfg["classifier_hidden_dim"]), nn.GELU(), nn.Dropout(head_cfg["classifier_dropout"]), nn.Linear(head_cfg["classifier_hidden_dim"], local_dim)) for _ in self.tapped_layers])
        fused_dim = 2 * local_dim * len(self.tapped_layers)
        self.normalizer = nn.LayerNorm(fused_dim)
        self.residual_adapter_enabled = bool(head_cfg.get("residual_adapter_enabled", False))
        self.residual_feature_dim = int(head_cfg.get("residual_feature_dim", 0)) if self.residual_adapter_enabled else 0
        if self.residual_adapter_enabled:
            if self.residual_feature_dim < 1:
                raise ValueError("residual_feature_dim must be positive when residual_adapter_enabled=true")
            self.residual_adapter = SRMResidualAdapter(
                self.residual_feature_dim,
                tuple(config["data"].get("normalization_mean", (0.485, 0.456, 0.406))),
                tuple(config["data"].get("normalization_std", (0.229, 0.224, 0.225))),
            )
            gate_hidden_dim = int(head_cfg.get("residual_gate_hidden_dim", 256))
            self.quality_gate = nn.Sequential(
                nn.Linear(fused_dim + self.residual_feature_dim + 2, gate_hidden_dim), nn.GELU(),
                nn.Linear(gate_hidden_dim, 1), nn.Sigmoid(),
            )
            classifier_dim = fused_dim + self.residual_feature_dim
            self.fusion_normalizer = nn.LayerNorm(classifier_dim)
        else:
            self.residual_adapter = None
            self.quality_gate = None
            classifier_dim = fused_dim
            self.fusion_normalizer = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(classifier_dim, head_cfg["classifier_hidden_dim"]), nn.GELU(), nn.Dropout(head_cfg["classifier_dropout"]), nn.Linear(head_cfg["classifier_hidden_dim"], 1))
        self.projector = nn.Sequential(nn.Linear(classifier_dim, head_cfg["projection_hidden_dim"]), nn.GELU(), nn.Linear(head_cfg["projection_hidden_dim"], head_cfg["projection_output_dim"]))
        self.backbone_frozen = bool(model_cfg.get("freeze_backbone", False))
        self.unfrozen_backbone_blocks: tuple[int, ...] = ()
        requested_blocks = int(model_cfg.get("unfreeze_last_blocks", 0))
        if requested_blocks:
            if self.backbone_frozen:
                raise ValueError("freeze_backbone and unfreeze_last_blocks cannot both be enabled")
            self.unfrozen_backbone_blocks = self._unfreeze_last_blocks(requested_blocks)
        elif self.backbone_frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _unfreeze_last_blocks(self, count: int) -> tuple[int, ...]:
        """Fine-tune only the final transformer blocks, not the whole ViT."""
        if count < 1:
            raise ValueError("unfreeze_last_blocks must be positive")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        blocks: dict[int, list[nn.Parameter]] = {}
        pattern = re.compile(r"(?:encoder\.(?:layer|layers)|blocks)\.(\d+)\.")
        for name, parameter in self.backbone.named_parameters():
            match = pattern.search(name)
            if match:
                blocks.setdefault(int(match.group(1)), []).append(parameter)
        if not blocks:
            raise RuntimeError("Could not identify DINO transformer blocks for partial fine-tuning; refusing a silent frozen run.")
        selected = tuple(sorted(blocks)[-count:])
        for index in selected:
            for parameter in blocks[index]:
                parameter.requires_grad = True
        # Train the final normalization with the final blocks when present.
        for name, parameter in self.backbone.named_parameters():
            if name.startswith(("layernorm", "norm")):
                parameter.requires_grad = True
        return selected

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        requires_grad = any(parameter.requires_grad for parameter in self.backbone.parameters())
        with torch.set_grad_enabled(requires_grad): outputs = self.backbone(pixel_values=images, output_hidden_states=True)
        if max(self.tapped_layers) >= len(outputs.hidden_states):
            raise ValueError(
                f"Requested tapped layer {max(self.tapped_layers)}, but this backbone returned only "
                f"{len(outputs.hidden_states) - 1} transformer blocks. Update model.tapped_layers."
            )
        features = []
        for branch_index, layer_index in enumerate(self.tapped_layers):
            hidden = outputs.hidden_states[layer_index]
            cls_token, patches = hidden[:, 0], hidden[:, 1 + self.num_register_tokens:]
            side = int(math.isqrt(patches.shape[1]))
            if side * side != patches.shape[1]: raise ValueError(f"Patch count {patches.shape[1]} is not square")
            patch_map = patches.transpose(1, 2).reshape(images.shape[0], patches.shape[-1], side, side)
            features.extend((self.local_branches[branch_index](patch_map), self.cls_branches[branch_index](cls_token)))
        return self.normalizer(torch.cat(features, dim=1))

    def _fuse(self, images: torch.Tensor, dino_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.residual_adapter is None:
            return dino_features, None
        residual_features, quality = self.residual_adapter(images)
        gate = self.quality_gate(torch.cat((dino_features, residual_features, quality), dim=1))
        return self.fusion_normalizer(torch.cat((dino_features, gate * residual_features), dim=1)), gate.squeeze(1)

    def forward(self, image: torch.Tensor, augmented_image: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        clean_features, clean_gate = self._fuse(image, self._encode(image))
        result = {"logits": self.classifier(clean_features).squeeze(1), "projection": self.projector(clean_features)}
        if clean_gate is not None:
            result["residual_gate"] = clean_gate
        if augmented_image is not None:
            augmented_features, augmented_gate = self._fuse(augmented_image, self._encode(augmented_image))
            result["augmented_logits"] = self.classifier(augmented_features).squeeze(1)
            result["augmented_projection"] = self.projector(augmented_features)
            if augmented_gate is not None:
                result["augmented_residual_gate"] = augmented_gate
        return result

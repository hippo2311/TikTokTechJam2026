from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aigc_detector.param_budget import print_branch_params


class TexturePatchExtractor(nn.Module):
    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        top_k: int = 1,
        output_size: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.top_k = top_k
        self.output_size = output_size
        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1, keepdim=True)
        response = F.conv2d(gray, self.laplacian, padding=1).square()
        scores = F.avg_pool2d(response, kernel_size=self.patch_size, stride=self.stride)
        bsz, _, gh, gw = scores.shape
        top_k = min(self.top_k, gh * gw)
        _, indices = torch.topk(scores.flatten(1), k=top_k, dim=1)

        patches = []
        for batch_index in range(bsz):
            batch_patches = []
            for index in indices[batch_index]:
                row = (index // gw).item() * self.stride
                col = (index % gw).item() * self.stride
                patch = images[
                    batch_index : batch_index + 1,
                    :,
                    row : row + self.patch_size,
                    col : col + self.patch_size,
                ]
                patch = F.interpolate(
                    patch,
                    size=(self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=False,
                )
                batch_patches.append(patch)
            patches.append(torch.cat(batch_patches, dim=0))
        return torch.stack(patches, dim=0)


class TextureBranch(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        projection_dim: int = 128,
        num_classes: int = 2,
        top_k_patches: int = 1,
        patch_size: int = 64,
        patch_stride: int = 32,
        input_size: int = 128,
    ) -> None:
        super().__init__()
        self.patch_extractor = TexturePatchExtractor(
            patch_size=patch_size,
            stride=patch_stride,
            top_k=top_k_patches,
            output_size=input_size,
        )
        backbone, feature_dim = build_resnet_backbone(backbone_name, pretrained)
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, projection_dim),
        )
        print_branch_params("texture", self)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        patches = self.patch_extractor(images)
        batch_size, num_patches = patches.shape[:2]
        patch_batch = patches.reshape(batch_size * num_patches, 3, *patches.shape[-2:])
        patch_batch = (patch_batch - self.image_mean) / self.image_std
        patch_features = self.backbone(patch_batch)
        patch_features = patch_features.view(batch_size, num_patches, -1)
        features = patch_features.mean(dim=1)
        logits = self.classifier(features)
        projections = self.projection_head(features)
        return {"features": features, "logits": logits, "projections": projections}


def build_resnet_backbone(backbone_name: str, pretrained: bool):
    try:
        from torchvision.models import (
            ResNet18_Weights,
            ResNet50_Weights,
            resnet18,
            resnet50,
        )

        if backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported texture backbone: {backbone_name}")
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    except ImportError:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("TextureBranch requires torchvision or timm.") from exc
        model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        return model, model.num_features

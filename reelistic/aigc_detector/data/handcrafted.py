from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_handcrafted_features(images: torch.Tensor) -> torch.Tensor:
    compression = compression_artifact_score(images)
    edge = edge_density(images)
    return torch.stack([compression, edge], dim=1)


def compute_quality_features(images: torch.Tensor) -> torch.Tensor:
    """Return differentiable image-quality measurements for soft routing."""
    compression = compression_artifact_score(images)
    edge = edge_density(images)
    gray = images.mean(dim=1, keepdim=True)
    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
    sharpness = laplacian.var(dim=(1, 2, 3), unbiased=False)
    local_mean = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
    noise_level = (gray - local_mean).abs().mean(dim=(1, 2, 3))
    horizontal_energy = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(
        dim=(1, 2, 3)
    )
    vertical_energy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean(
        dim=(1, 2, 3)
    )
    high_frequency = 0.5 * (horizontal_energy + vertical_energy)
    brightness = gray.mean(dim=(1, 2, 3))
    contrast = gray.std(dim=(1, 2, 3), unbiased=False)
    saturation = (images.max(dim=1).values - images.min(dim=1).values).mean(
        dim=(1, 2)
    )
    return torch.stack(
        [
            compression,
            edge,
            sharpness,
            noise_level,
            high_frequency,
            brightness,
            contrast,
            saturation,
        ],
        dim=1,
    )


def compression_artifact_score(images: torch.Tensor) -> torch.Tensor:
    # Measure block boundary discontinuities as a simple JPEG artifact proxy.
    vertical = (images[:, :, :, 8::8] - images[:, :, :, 7:-1:8]).abs().mean(dim=(1, 2, 3))
    horizontal = (images[:, :, 8::8, :] - images[:, :, 7:-1:8, :]).abs().mean(dim=(1, 2, 3))
    return 0.5 * (vertical + horizontal)


def edge_density(images: torch.Tensor) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=images.device,
        dtype=images.dtype,
    ).unsqueeze(1)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=images.device,
        dtype=images.dtype,
    ).unsqueeze(1)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return (mag > mag.mean(dim=(1, 2, 3), keepdim=True)).float().mean(dim=(1, 2, 3))

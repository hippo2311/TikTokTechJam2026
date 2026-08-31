from __future__ import annotations

from functools import partial
from typing import Any


def _center_crop_fraction(image, fraction: float, **_kwargs):
    """Crop an arbitrary-sized numpy image around its centre; tensor resizing happens later."""
    height, width = image.shape[:2]
    crop_height, crop_width = max(1, int(height * fraction)), max(1, int(width * fraction))
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return image[top:top + crop_height, left:left + crop_width]


def build_evaluation_augmentation(kind: str, value: float | int | None):
    """Build one deterministic evaluation condition from the workshop specification."""
    import albumentations as A
    import cv2

    if kind == "clean":
        return None
    if kind == "jpeg":
        try: return A.Compose([A.ImageCompression(quality_range=(int(value), int(value)), p=1.0)])
        except TypeError: return A.Compose([A.ImageCompression(quality_lower=int(value), quality_upper=int(value), p=1.0)])
    if kind == "gaussian_blur":
        return A.Compose([A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(float(value), float(value)), p=1.0)])
    if kind == "resize":
        try: return A.Compose([A.Downscale(scale_range=(float(value), float(value)), interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=1.0)])
        except TypeError: return A.Compose([A.Downscale(scale_min=float(value), scale_max=float(value), interpolation=cv2.INTER_AREA, p=1.0)])
    if kind == "gaussian_noise":
        try: return A.Compose([A.GaussNoise(std_range=(float(value), float(value)), p=1.0)])
        except TypeError: return A.Compose([A.GaussNoise(var_limit=((255 * float(value)) ** 2, (255 * float(value)) ** 2), p=1.0)])
    if kind == "color_jitter":
        return A.Compose([A.ColorJitter(brightness=float(value), contrast=float(value), saturation=float(value), hue=0.0, p=1.0)])
    if kind == "center_crop":
        return A.Compose([A.Lambda(image=partial(_center_crop_fraction, fraction=float(value)), p=1.0)])
    raise ValueError(f"Unknown evaluation transform: {kind}")


def build_curriculum_augmentation(stage: dict[str, Any]):
    """Build Albumentations for one TOML curriculum stage."""
    import albumentations as A
    import cv2

    transforms = []
    qualities = stage.get("jpeg_quality", [])
    if qualities:
        low, high = min(qualities), max(qualities)
        try: transforms.append(A.ImageCompression(quality_range=(low, high), p=1.0))
        except TypeError: transforms.append(A.ImageCompression(quality_lower=low, quality_upper=high, p=1.0))
    sigmas = stage.get("gaussian_blur_sigma", [])
    if sigmas: transforms.append(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(min(sigmas), max(sigmas)), p=1.0))
    scales = stage.get("resize_scale", [])
    if scales:
        low, high = min(scales), max(scales)
        try: transforms.append(A.Downscale(scale_range=(low, high), interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=1.0))
        except TypeError: transforms.append(A.Downscale(scale_min=low, scale_max=high, interpolation=cv2.INTER_AREA, p=1.0))
    noise = stage.get("gaussian_noise_sigma", [])
    if noise:
        try: transforms.append(A.GaussNoise(std_range=(min(noise), max(noise)), p=1.0))
        except TypeError: transforms.append(A.GaussNoise(var_limit=((255 * min(noise)) ** 2, (255 * max(noise)) ** 2), p=1.0))
    jitter = max(stage.get("color_jitter_strength", [0.0]))
    if jitter: transforms.append(A.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.0, p=1.0))
    crop_fractions = stage.get("center_crop_fraction", [])
    if crop_fractions:
        transforms.append(A.Lambda(image=partial(_center_crop_fraction, fraction=min(crop_fractions)), p=1.0))
    return A.Compose(transforms)


def build_blur_noise_augmentation(stage: dict[str, Any]):
    """Sample exactly one hard view: Gaussian blur *or* Gaussian noise.

    Applying both transformations serially makes it impossible to attribute a
    robustness change to either degradation.  This variant deliberately keeps
    the paired-view objective focused on the two observed weak conditions.
    """
    import albumentations as A

    transforms = []
    sigmas = stage.get("gaussian_blur_sigma", [])
    if sigmas:
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(min(sigmas), max(sigmas)), p=1.0))
    noise = stage.get("gaussian_noise_sigma", [])
    if noise:
        try:
            transforms.append(A.GaussNoise(std_range=(min(noise), max(noise)), p=1.0))
        except TypeError:
            transforms.append(A.GaussNoise(var_limit=((255 * min(noise)) ** 2, (255 * max(noise)) ** 2), p=1.0))
    if not transforms:
        raise ValueError("blur_noise_single augmentation requires blur and/or noise settings")
    return A.Compose([A.OneOf(transforms, p=1.0)])


def build_single_transform_augmentation(stage: dict[str, Any]):
    """Sample exactly one configured real-world transform per paired view.

    The optional ``transform_weights`` mapping controls selection frequency;
    blur and noise can be oversampled because development evaluation found
    them harder, while all listed transformations remain represented.
    """
    import albumentations as A
    import cv2

    weights = stage.get("transform_weights", {})
    transforms = []

    def add(transform, name: str) -> None:
        transform.p = float(weights.get(name, 1.0))
        transforms.append(transform)

    qualities = stage.get("jpeg_quality", [])
    if qualities:
        low, high = min(qualities), max(qualities)
        try: add(A.ImageCompression(quality_range=(low, high), p=1.0), "jpeg")
        except TypeError: add(A.ImageCompression(quality_lower=low, quality_upper=high, p=1.0), "jpeg")
    sigmas = stage.get("gaussian_blur_sigma", [])
    if sigmas:
        add(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(min(sigmas), max(sigmas)), p=1.0), "gaussian_blur")
    scales = stage.get("resize_scale", [])
    if scales:
        low, high = min(scales), max(scales)
        try: add(A.Downscale(scale_range=(low, high), interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=1.0), "resize")
        except TypeError: add(A.Downscale(scale_min=low, scale_max=high, interpolation=cv2.INTER_AREA, p=1.0), "resize")
    noise = stage.get("gaussian_noise_sigma", [])
    if noise:
        try: add(A.GaussNoise(std_range=(min(noise), max(noise)), p=1.0), "gaussian_noise")
        except TypeError: add(A.GaussNoise(var_limit=((255 * min(noise)) ** 2, (255 * max(noise)) ** 2), p=1.0), "gaussian_noise")
    jitter = max(stage.get("color_jitter_strength", [0.0]))
    if jitter:
        add(A.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.0, p=1.0), "color_jitter")
    crop_fractions = stage.get("center_crop_fraction", [])
    if crop_fractions:
        add(A.Lambda(image=partial(_center_crop_fraction, fraction=min(crop_fractions)), p=1.0), "center_crop")
    if not transforms:
        raise ValueError("single_transform augmentation requires at least one configured transform")
    return A.Compose([A.OneOf(transforms, p=1.0)])

from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def basic_eval_transform(image_size: int = 224):
    """Phase 0 deterministic preprocessing only; no robustness augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RobustnessTransformPlaceholder:
    """Reserved for Phase 1 JPEG/blur/resize/noise/jitter/crop transforms."""

    def __call__(self, image):
        return image

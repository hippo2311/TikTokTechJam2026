from __future__ import annotations

from io import BytesIO
import random
from typing import Callable

from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class StochasticCompressionAugment:
    """Realistic corruption augmentation for small native-resolution images.

    Twenty percent of samples remain clean. Other samples use explicit native
    downsample sizes, real JPEG encode/decode, and noise both before and after
    the model resize so robustness is not tied to one operation order.
    """

    def __init__(
        self,
        image_size: int = 128,
        clean_probability: float = 0.2,
        robustness_focus: bool = False,
    ) -> None:
        self.image_size = image_size
        self.clean_probability = clean_probability
        self.robustness_focus = robustness_focus
        self.base = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if random.random() < self.clean_probability:
            return self.base(image)

        if random.random() < 0.55:
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-0.2, 0.2))
            image = TF.adjust_contrast(image, 1.0 + random.uniform(-0.2, 0.2))
            image = TF.adjust_saturation(image, 1.0 + random.uniform(-0.2, 0.2))
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))

        if random.random() < 0.35:
            crop_ratio = 0.8
            crop_h = int(image.height * crop_ratio)
            crop_w = int(image.width * crop_ratio)
            top = max((image.height - crop_h) // 2, 0)
            left = max((image.width - crop_w) // 2, 0)
            image = TF.crop(image, top, left, crop_h, crop_w)

        recipes = [
            "noise_resize",
            "resize_jpeg",
            "jpeg_resize_jpeg",
            "blur_mix",
            "post_noise",
        ]
        weights = [1.4, 1.4, 1.0, 1.4, 1.2] if self.robustness_focus else None
        recipe = random.choices(recipes, weights=weights, k=1)[0]
        if recipe == "noise_resize":
            image = add_native_noise(image, random.choice([0.02, 0.05, 0.10]))
            image = downsample_to_explicit_size(image)
        elif recipe == "resize_jpeg":
            image = downsample_to_explicit_size(image)
            image = real_jpeg_roundtrip(image, random.choice([30, 50, 70, 90]))
        elif recipe == "jpeg_resize_jpeg":
            image = real_jpeg_roundtrip(image, random.choice([50, 70, 90]))
            image = downsample_to_explicit_size(image)
            image = real_jpeg_roundtrip(image, random.choice([30, 50, 70]))
        elif recipe == "blur_mix":
            image = image.filter(
                ImageFilter.GaussianBlur(radius=random.choice([0.5, 1.0, 2.0]))
            )
            if random.random() < 0.5:
                image = real_jpeg_roundtrip(image, random.choice([30, 50, 70, 90]))

        tensor = self.base(image)
        if recipe == "post_noise" or random.random() < 0.15:
            noise_sigma = random.choice([0.02, 0.05, 0.10])
            tensor = (tensor + torch.randn_like(tensor) * noise_sigma).clamp(0.0, 1.0)

        return tensor


def build_eval_transform(image_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )


def simulate_jpeg_quality(tensor: torch.Tensor, quality: int) -> torch.Tensor:
    # Approximate JPEG degradation by quantizing to coarser bins for lower qualities.
    bins = {90: 255.0, 70: 128.0, 50: 64.0, 30: 32.0}[quality]
    return torch.round(tensor * bins) / bins


def real_jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def downsample_to_explicit_size(image: Image.Image) -> Image.Image:
    original_size = image.size
    candidates = [size for size in (24, 16, 8) if size < min(original_size)]
    if not candidates:
        return image
    target = random.choice(candidates)
    reduced = image.resize((target, target), resample=Image.Resampling.BILINEAR)
    return reduced.resize(original_size, resample=Image.Resampling.BICUBIC)


def add_native_noise(image: Image.Image, sigma: float) -> Image.Image:
    tensor = TF.pil_to_tensor(image).float().div(255.0)
    tensor = (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)
    return TF.to_pil_image(tensor)

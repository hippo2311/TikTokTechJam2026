from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from aigc_detector.data.augmentations import build_eval_transform
from aigc_detector.data.datasets import build_train_val_cal_test_splits
from aigc_detector.metrics import binary_classification_metrics
from aigc_detector.models.ensemble import AIGCDetectionEnsemble
from aigc_detector.utils.device import default_device


DEFAULT_CONDITIONS = [
    "clean",
    "color_shift",
    "center_crop_80",
    "downsample_50",
    "downsample_25",
    "blur_0.5",
    "blur_1.0",
    "blur_2.0",
    "jpeg_90",
    "jpeg_70",
    "jpeg_50",
    "jpeg_30",
    "noise_0.02",
    "noise_0.05",
    "noise_0.10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate corruption robustness.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="Manifest directory used for this checkpoint; required locally if its saved cluster path is unavailable.",
    )
    parser.add_argument("--split", choices=["val", "calibration", "test"], default="val")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--output-json", default="robustness_metrics.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--source",
        default=None,
        help="Evaluate only this dataset source from the combined split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    seed = args.seed if args.seed is not None else checkpoint_args.get("seed", 42)
    image_size = checkpoint_args.get("image_size", 128)
    model = build_model(checkpoint_args)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    eval_transform = build_eval_transform(image_size)
    saved_manifest_dir = checkpoint_args.get("manifest_dir")
    manifest_dir = args.manifest_dir
    if manifest_dir is None and saved_manifest_dir and Path(saved_manifest_dir).is_dir():
        manifest_dir = saved_manifest_dir
    train, val, calibration, test = build_train_val_cal_test_splits(
        args.data_dir,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        val_fraction=checkpoint_args.get("val_fraction", 0.05),
        calibration_fraction=checkpoint_args.get("calibration_fraction", 0.05),
        seed=seed,
        # Training samples are unused here. A zero cap means "do not limit" and
        # avoids violating the manifest's source/class/family coverage guard.
        max_train_samples=0,
        max_val_samples=(args.max_samples if args.source is None and args.split == "val" else 0),
        max_calibration_samples=(
            args.max_samples if args.source is None and args.split == "calibration" else 0
        ),
        max_test_samples=(args.max_samples if args.source is None and args.split == "test" else 0),
        manifest_dir=manifest_dir,
    )
    datasets = {"val": val, "calibration": calibration, "test": test}
    samples = select_samples(
        datasets[args.split].samples,
        source=args.source,
        max_samples=args.max_samples,
        seed=seed,
    )
    results = {}
    for condition in args.conditions:
        operation = build_condition(condition, seed)
        loader = DataLoader(
            CorruptionDataset(samples, operation, image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        result = evaluate_condition(model, loader, device)
        results[condition] = result
        metrics = result["fusion"]
        print(
            f"[{condition}] accuracy={metrics['accuracy']:.4f} "
            f"auc={metrics['roc_auc']:.4f} ece={metrics['ece']:.4f}",
            flush=True,
        )

    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "sample_count": len(samples),
        "sample_fingerprint": sample_fingerprint(samples),
        "source": args.source,
        "seed": seed,
        "conditions": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[robustness] wrote {output_path}")


def sample_fingerprint(samples) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        identity = (
            str(sample.path),
            str(sample.label),
            str(sample.source),
            str(sample.family),
        )
        digest.update("\0".join(identity).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_samples(samples, source: str | None, max_samples: int, seed: int):
    selected = [sample for sample in samples if source is None or sample.source == source]
    if source is not None and not selected:
        available = sorted({sample.source for sample in samples})
        raise ValueError(f"Source {source!r} is unavailable; choose from {available}")
    if max_samples <= 0 or len(selected) <= max_samples:
        return selected

    rng = random.Random(seed)
    grouped = {}
    for sample in selected:
        grouped.setdefault(sample.label, []).append(sample)
    labels = sorted(grouped)
    base, remainder = divmod(max_samples, len(labels))
    limited = []
    for index, label in enumerate(labels):
        label_samples = list(grouped[label])
        rng.shuffle(label_samples)
        limited.extend(label_samples[: base + (1 if index < remainder else 0)])
    rng.shuffle(limited)
    return limited


def build_model(checkpoint_args: dict) -> AIGCDetectionEnsemble:
    return AIGCDetectionEnsemble(
        semantic_backbone=checkpoint_args.get(
            "semantic_backbone", "mobilenetv3_small_100.lamb_in1k"
        ),
        texture_backbone=checkpoint_args.get("texture_backbone", "resnet18"),
        semantic_pretrained=False,
        texture_pretrained=False,
        image_size=checkpoint_args.get("image_size", 128),
        top_k_patches=checkpoint_args.get("top_k_patches", 1),
        quality_aware_fusion=checkpoint_args.get("quality_aware_fusion", False),
        noise_version=checkpoint_args.get("noise_version", "legacy"),
        noise_enabled=checkpoint_args.get("noise_enabled", True),
        branch_dropout=checkpoint_args.get("branch_dropout", 0.15),
        explicit_gate_disagreement=checkpoint_args.get(
            "explicit_gate_disagreement", False
        ),
    )


class CorruptionDataset(Dataset):
    def __init__(self, samples, operation, image_size: int) -> None:
        self.samples = list(samples)
        self.operation = operation
        self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = source.convert("RGB")
        return self.transform(self.operation(image, index)), sample.label


@torch.inference_mode()
def evaluate_condition(model, loader, device: torch.device) -> dict:
    labels = []
    probabilities = {key: [] for key in ["fusion", "texture", "frequency", "noise", "semantic"]}
    gate_weights = []
    for images, batch_labels in loader:
        outputs = model(images.to(device), apply_temperature=True)
        labels.extend(batch_labels.tolist())
        logits = {
            "fusion": outputs.logits,
            "texture": outputs.texture_logits,
            "frequency": outputs.frequency_logits,
            "noise": outputs.noise_logits,
            "semantic": outputs.semantic_logits,
        }
        for key, branch_logits in logits.items():
            probabilities[key].extend(torch.softmax(branch_logits, dim=1)[:, 1].cpu().tolist())
        if outputs.gate_weights is not None:
            gate_weights.append(outputs.gate_weights.cpu())
    result = {
        key: binary_classification_metrics(labels, values)
        for key, values in probabilities.items()
    }
    if gate_weights:
        mean_weights = torch.cat(gate_weights).mean(dim=0).tolist()
        result["mean_gate_weights"] = dict(
            zip(["texture", "frequency", "noise", "semantic"], mean_weights)
        )
    return result


def build_condition(name: str, seed: int):
    if name == "clean":
        return lambda image, index: image
    if name == "color_shift":
        def color(image, index):
            image = ImageEnhance.Brightness(image).enhance(1.2)
            image = ImageEnhance.Contrast(image).enhance(0.8)
            image = ImageEnhance.Color(image).enhance(1.2)
            return TF.adjust_hue(image, 0.03)
        return color
    if name == "center_crop_80":
        def crop(image, index):
            width, height = image.size
            crop_width, crop_height = round(width * 0.8), round(height * 0.8)
            left, top = (width - crop_width) // 2, (height - crop_height) // 2
            return image.crop(
                (left, top, left + crop_width, top + crop_height)
            ).resize((width, height), Image.Resampling.BICUBIC)
        return crop
    if name.startswith("downsample_"):
        ratio = float(name.split("_")[1]) / 100.0
        def downsample(image, index):
            size = (max(4, round(image.width * ratio)), max(4, round(image.height * ratio)))
            return image.resize(size, Image.Resampling.BILINEAR).resize(
                image.size, Image.Resampling.BICUBIC
            )
        return downsample
    if name.startswith("blur_"):
        radius = float(name.split("_")[1])
        return lambda image, index: image.filter(ImageFilter.GaussianBlur(radius=radius))
    if name.startswith("jpeg_"):
        quality = int(name.split("_")[1])
        def jpeg(image, index):
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB")
        return jpeg
    if name.startswith("noise_"):
        sigma = float(name.split("_")[1])
        def noise(image, index):
            array = np.asarray(image, dtype=np.float32) / 255.0
            generator = np.random.default_rng(seed + index)
            array = np.clip(array + generator.normal(0, sigma, array.shape), 0, 1)
            return Image.fromarray((array * 255).astype(np.uint8))
        return noise
    raise ValueError(f"Unknown robustness condition: {name}")


if __name__ == "__main__":
    main()

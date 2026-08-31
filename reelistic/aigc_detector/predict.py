from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from aigc_detector.data.augmentations import build_eval_transform
from aigc_detector.models.ensemble import AIGCDetectionEnsemble
from aigc_detector.utils.device import default_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble inference over a directory of images.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of images to score.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint.")
    parser.add_argument("--output-json", type=str, default="predictions.json")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})

    model = AIGCDetectionEnsemble(
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = args.image_size or checkpoint_args.get("image_size", 128)
    transform = build_eval_transform(image_size=image_size)
    image_paths = [
        path
        for path in sorted(Path(args.image_dir).rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if args.max_images is not None and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    dataset = PredictionDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    results = []
    with torch.inference_mode():
        for tensors, paths in loader:
            predictions = model.predict_proba(tensors.to(device)).cpu().tolist()
            results.extend(
                {"image_path": path, "pred": float(prediction)}
                for path, prediction in zip(paths, predictions)
            )

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"[predict] wrote {len(results)} predictions to {output_path}")


class PredictionDataset(Dataset):
    def __init__(self, image_paths, transform) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path)


if __name__ == "__main__":
    main()

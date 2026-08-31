#!/usr/bin/env python3
"""Predict AIGC probability for every image in a file or directory.

Output is a JSON list with exactly the required fields:
``image_path`` and ``pred`` (the probability that the image is AIGC-generated).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from evaluate_onnx import IMAGE_EXTENSIONS, preprocess, sigmoid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write image_path and AIGC probability for every input image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="One image file or a directory searched recursively.")
    parser.add_argument("--output", default="predictions.json", help="Destination JSON file.")
    parser.add_argument("--model", default="models/reelistic_dino/checkpoints/reelistic_dinov3.onnx")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    return parser.parse_args()


def discover_images(raw_input: str) -> list[Path]:
    source = Path(raw_input).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    if source.is_dir():
        images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No supported images found below {source}")
        return images
    raise FileNotFoundError(source)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    available = ort.get_available_providers()
    if args.provider not in available:
        raise ValueError(f"Provider {args.provider!r} unavailable; choose from {available}")
    images = discover_images(args.input)
    session = ort.InferenceSession(str(model), providers=[args.provider])
    results: list[dict] = []
    for offset in range(0, len(images), args.batch_size):
        paths = images[offset:offset + args.batch_size]
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = np.stack(tensors).astype(np.float32, copy=False)
        logits = np.asarray(session.run(["logit"], {"image": batch})[0]).reshape(-1)
        for path, probability in zip(paths, sigmoid(logits)):
            results.append({"image_path": str(path), "pred": float(probability)})
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Predicted {len(results)} image(s); wrote {output}")


if __name__ == "__main__":
    main()

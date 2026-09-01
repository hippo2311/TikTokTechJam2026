#!/usr/bin/env python3
"""Run Reelistic ONNX inference and print an authenticity verdict.

Example:
    python scripts/run_onnx_demo.py path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from evaluate_onnx import preprocess, sigmoid
from predict import discover_images


DEFAULT_THRESHOLD = 0.9648


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Reelistic ONNX model and print AI-generated/authentic verdicts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="One image file or a directory searched recursively.")
    parser.add_argument("--model", default="models/reelistic_dino/checkpoints/reelistic_dinov3.onnx")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="AI-probability cutoff selected at 5%% FPR on the clean COCO/DALL·E benchmark.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model}. Follow README.md → Quick start: run the ONNX model."
        )
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    results: list[dict[str, object]] = []
    for path in discover_images(args.input):
        with Image.open(path) as image:
            tensor = preprocess(image.convert("RGB"))[None].astype(np.float32, copy=False)
        logit = float(np.asarray(session.run(["logit"], {"image": tensor})[0]).reshape(-1)[0])
        probability = float(sigmoid(np.asarray([logit]))[0])
        verdict = "ai-generated" if probability >= args.threshold else "authentic"
        result = {
            "image_path": str(path),
            "ai_generated_probability": probability,
            "verdict": verdict,
            "threshold": args.threshold,
        }
        results.append(result)
        print(f"{path.name}: {verdict} (AI probability {probability:.4f}; threshold {args.threshold:.4f})")

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

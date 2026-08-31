#!/usr/bin/env python3
"""Run Reelistic ONNX inference on images and optional robustness conditions.

The prediction file always contains ``image_path`` and ``pred``. When labels
are available from a manifest or class-named folders, the summary also reports
classification, ranking, low-FPR, calibration, and runtime metrics.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_NAMES = {"real", "human", "human-made", "authentic", "natural", "0"}
AI_NAMES = {"fake", "ai", "aigc", "ai-generated", "generated", "synthetic", "1"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
DEFAULT_SUITE = [
    "clean",
    "jpeg:90", "jpeg:70", "jpeg:50", "jpeg:30",
    "blur:0.5", "blur:1.0", "blur:2.0",
    "resize:0.5", "resize:0.25",
    "noise:0.02", "noise:0.05", "noise:0.10",
    "color:0.10", "color:0.20",
    "crop:0.80",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Reelistic ONNX on an image, directory, or labelled CSV manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", help="Image file(s) or directories (searched recursively).")
    source.add_argument("--manifest", help="CSV with image_path/path and optional label columns.")
    parser.add_argument("--model", default="models/reelistic_dino/checkpoints/reelistic_dinov3.onnx")
    parser.add_argument("--output-dir", default="onnx_evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--transform", action="append", dest="transforms", metavar="NAME[:VALUE]",
                        help="Repeat for clean/jpeg/blur/resize/noise/color/crop. No transforms are chained.")
    parser.add_argument("--suite", action="store_true", help="Run clean plus all documented robustness conditions.")
    parser.add_argument("--seed", type=int, default=42, help="Makes noise and color jitter reproducible.")
    parser.add_argument("--provider", help="ONNX Runtime provider; defaults to the first available provider.")
    parser.add_argument("--no-folder-labels", action="store_true",
                        help="Do not infer labels from real/fake-style parent directory names.")
    return parser.parse_args()


def normalize_label(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in REAL_NAMES:
        return 0
    if text in AI_NAMES:
        return 1
    try:
        number = int(float(text))
    except ValueError as error:
        raise ValueError(f"Unsupported label {value!r}; use 0/1 or real/fake.") from error
    if number not in {0, 1}:
        raise ValueError(f"Unsupported label {value!r}; expected 0 or 1.")
    return number


def infer_folder_label(path: Path) -> int | None:
    for part in reversed(path.parent.parts):
        name = part.lower()
        if name in REAL_NAMES:
            return 0
        if name in AI_NAMES:
            return 1
    return None


def load_records(args: argparse.Namespace) -> list[dict]:
    records: list[dict] = []
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                raw_path = row.get("image_path") or row.get("path")
                if not raw_path:
                    raise ValueError(f"Missing image_path/path at {manifest}:{row_number}")
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = manifest.parent / path
                records.append({"path": path.resolve(), "label": normalize_label(row.get("label"))})
    else:
        paths: list[Path] = []
        for raw in args.input:
            candidate = Path(raw).expanduser().resolve()
            if candidate.is_file():
                paths.append(candidate)
            elif candidate.is_dir():
                paths.extend(path for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            else:
                raise FileNotFoundError(candidate)
        for path in sorted(set(paths)):
            label = None if args.no_folder_labels else infer_folder_label(path)
            records.append({"path": path, "label": label})
    if not records:
        raise ValueError("No images found.")
    missing = [str(record["path"]) for record in records if not record["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest image(s) missing; first: {missing[0]}")
    return records


def parse_condition(spec: str) -> tuple[str, float | int | None, str]:
    aliases = {"gaussian_blur": "blur", "gaussian_noise": "noise", "color_jitter": "color", "center_crop": "crop"}
    name, separator, raw_value = spec.strip().lower().partition(":")
    name = aliases.get(name, name)
    if name == "clean":
        if separator:
            raise ValueError("clean does not take a value")
        return name, None, "clean"
    defaults = {"jpeg": 90, "blur": 1.0, "resize": 0.5, "noise": 0.05, "color": 0.10, "crop": 0.80}
    if name not in defaults:
        raise ValueError(f"Unknown transform {name!r}; use clean/jpeg/blur/resize/noise/color/crop")
    value = float(raw_value) if separator else defaults[name]
    if name == "jpeg":
        value = int(value)
        if not 1 <= value <= 100:
            raise ValueError("JPEG quality must be from 1 to 100")
    elif name in {"resize", "crop"} and not 0 < value <= 1:
        raise ValueError(f"{name} fraction must be in (0, 1]")
    elif value < 0:
        raise ValueError(f"{name} value must be non-negative")
    return name, value, f"{name}:{value}"


def stable_rng(path: Path, condition: str, seed: int) -> np.random.Generator:
    digest = sha256(f"{seed}\0{path}\0{condition}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def apply_condition(image: Image.Image, path: Path, condition: tuple[str, float | int | None, str], seed: int) -> Image.Image:
    name, value, rendered = condition
    image = image.convert("RGB")
    if name == "clean":
        return image
    if name == "jpeg":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(value), subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if name == "resize":
        width, height = image.size
        down = (max(1, round(width * float(value))), max(1, round(height * float(value))))
        return image.resize(down, Image.Resampling.BOX).resize((width, height), Image.Resampling.BILINEAR)
    if name == "noise":
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        noise = stable_rng(path, rendered, seed).normal(0.0, float(value), size=pixels.shape)
        return Image.fromarray(np.uint8(np.clip(pixels + noise, 0.0, 1.0) * 255.0))
    if name == "color":
        rng = stable_rng(path, rendered, seed)
        strength = float(value)
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            image = enhancer(image).enhance(float(rng.uniform(1.0 - strength, 1.0 + strength)))
        return image
    if name == "crop":
        width, height = image.size
        crop_width, crop_height = max(1, int(width * float(value))), max(1, int(height * float(value)))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))
    raise AssertionError(name)


def preprocess(image: Image.Image, image_size: int = 224) -> np.ndarray:
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left, top = (resized.width - image_size) // 2, (resized.height - image_size) // 2
    cropped = resized.crop((left, top, left + image_size, top + image_size))
    tensor = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (tensor - MEAN) / STD


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def ranking_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(scores)), labels.size - 1]
    tp = np.cumsum(labels)[distinct].astype(float)
    fp = (distinct + 1).astype(float) - tp
    positives, negatives = labels.sum(), labels.size - labels.sum()
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    thresholds = np.r_[math.inf, scores[distinct]]
    return fpr, tpr, thresholds


def labelled_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    divide = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    npv = divide(tn, tn + fn)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics = {
        "labelled_images": int(labels.size), "threshold": threshold,
        "true_positives": tp, "true_negatives": tn, "false_positives": fp, "false_negatives": fn,
        "accuracy": divide(tp + tn, labels.size), "balanced_accuracy": None if recall is None or specificity is None else (recall + specificity) / 2,
        "precision": precision, "recall_sensitivity": recall, "specificity": specificity,
        "negative_predictive_value": npv, "f1": f1,
        "matthews_correlation": divide(tp * tn - fp * fn, denominator),
        "brier_score": float(np.mean((scores - labels) ** 2)),
    }
    bins = np.minimum((scores * 10).astype(int), 9)
    metrics["expected_calibration_error_10_bins"] = float(sum(
        np.mean(bins == index) * abs(float(np.mean(scores[bins == index])) - float(np.mean(labels[bins == index])))
        for index in range(10) if np.any(bins == index)
    ))
    if set(labels.tolist()) == {0, 1}:
        fpr, tpr, thresholds = ranking_curve(labels, scores)
        metrics["roc_auc"] = float(np.trapezoid(tpr, fpr))
        order = np.argsort(-scores, kind="mergesort")
        ordered_labels, ordered_scores = labels[order], scores[order]
        distinct = np.r_[np.flatnonzero(np.diff(ordered_scores)), labels.size - 1]
        cumulative_tp = np.cumsum(ordered_labels)[distinct].astype(float)
        precision_points = cumulative_tp / (distinct + 1)
        recall_points = cumulative_tp / np.sum(labels)
        metrics["pr_auc_average_precision"] = float(np.sum(np.diff(np.r_[0.0, recall_points]) * precision_points))
        finite_threshold = lambda value: float(value) if np.isfinite(value) else None
        for target in (0.01, 0.05):
            candidates = np.flatnonzero(fpr <= target)
            selected = candidates[np.argmax(tpr[candidates])]
            metrics[f"tpr_at_{int(target * 100)}pct_fpr"] = float(tpr[selected])
            metrics[f"threshold_at_{int(target * 100)}pct_fpr"] = finite_threshold(thresholds[selected])
        for target in (0.95, 0.99):
            candidates = np.flatnonzero(tpr >= target)
            selected = candidates[np.argmin(fpr[candidates])]
            metrics[f"fpr_at_{int(target * 100)}pct_tpr"] = float(fpr[selected])
            metrics[f"threshold_at_{int(target * 100)}pct_tpr"] = finite_threshold(thresholds[selected])
    else:
        metrics.update({"roc_auc": None, "pr_auc_average_precision": None})
    return metrics


def evaluate_condition(session: ort.InferenceSession, records: list[dict], condition, args) -> tuple[list[dict], dict]:
    predictions: list[dict] = []
    scores: list[float] = []
    labels: list[int] = []
    inference_times: list[float] = []
    started = time.perf_counter()
    for offset in range(0, len(records), args.batch_size):
        batch_records = records[offset:offset + args.batch_size]
        tensors = []
        for record in batch_records:
            with Image.open(record["path"]) as source:
                transformed = apply_condition(source, record["path"], condition, args.seed)
                tensors.append(preprocess(transformed))
        batch = np.stack(tensors).astype(np.float32, copy=False)
        before_inference = time.perf_counter()
        logits = np.asarray(session.run(["logit"], {"image": batch})[0]).reshape(-1)
        elapsed = time.perf_counter() - before_inference
        inference_times.extend([elapsed / len(batch_records)] * len(batch_records))
        probabilities = sigmoid(logits)
        for record, probability in zip(batch_records, probabilities):
            score = float(probability)
            item = {"image_path": str(record["path"]), "pred": score, "condition": condition[2]}
            if record["label"] is not None:
                item["label"] = int(record["label"])
                labels.append(int(record["label"]))
            predictions.append(item)
            scores.append(score)
    wall_seconds = time.perf_counter() - started
    score_array = np.asarray(scores)
    timing = np.asarray(inference_times) * 1000.0
    summary = {
        "condition": condition[2], "images": len(records), "labelled_images": len(labels),
        "threshold": args.threshold,
        "predicted_ai": int(np.sum(score_array >= args.threshold)),
        "predicted_real": int(np.sum(score_array < args.threshold)),
        "score_mean": float(np.mean(score_array)), "score_std": float(np.std(score_array)),
        "score_min": float(np.min(score_array)), "score_median": float(np.median(score_array)), "score_max": float(np.max(score_array)),
        "wall_seconds": wall_seconds, "throughput_images_per_second": len(records) / wall_seconds,
        "inference_ms_per_image_mean": float(np.mean(timing)),
        "inference_ms_per_image_p50": float(np.percentile(timing, 50)),
        "inference_ms_per_image_p95": float(np.percentile(timing, 95)),
    }
    if labels:
        labelled_scores = np.asarray([item["pred"] for item in predictions if "label" in item])
        summary.update(labelled_metrics(np.asarray(labels), labelled_scores, args.threshold))
    return predictions, summary


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    records = load_records(args)
    specifications = DEFAULT_SUITE if args.suite else (args.transforms or ["clean"])
    conditions = [parse_condition(spec) for spec in dict.fromkeys(specifications)]
    available = ort.get_available_providers()
    provider = args.provider or ("CPUExecutionProvider" if "CPUExecutionProvider" in available else available[0])
    if provider not in available:
        raise ValueError(f"Provider {provider!r} unavailable; choose from {available}")
    session = ort.InferenceSession(str(model), providers=[provider])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions, summaries = [], []
    for condition in conditions:
        predictions, summary = evaluate_condition(session, records, condition, args)
        all_predictions.extend(predictions)
        summaries.append(summary)
        print(f"[{condition[2]}] n={summary['images']} mean={summary['score_mean']:.4f} "
              f"throughput={summary['throughput_images_per_second']:.2f} image/s "
              f"roc_auc={summary.get('roc_auc')}")
    clean_auc = next((item.get("roc_auc") for item in summaries if item["condition"] == "clean"), None)
    if clean_auc is not None:
        for item in summaries:
            item["delta_roc_auc_vs_clean"] = None if item.get("roc_auc") is None else item["roc_auc"] - clean_auc
    prediction_path = output_dir / "predictions.json"
    summary_path = output_dir / "summary.json"
    prediction_path.write_text(json.dumps(all_predictions, indent=2, allow_nan=False), encoding="utf-8")
    report = {
        "model": str(model), "model_size_bytes": model.stat().st_size,
        "provider": provider, "available_providers": available,
        "threshold": args.threshold, "batch_size": args.batch_size, "seed": args.seed,
        "conditions": summaries,
    }
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {prediction_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

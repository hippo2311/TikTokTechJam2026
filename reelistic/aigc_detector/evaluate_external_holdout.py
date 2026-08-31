"""Evaluate a frozen checkpoint once on the isolated DALL-E/COCO holdout."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aigc_detector.data.datasets import Sample
from aigc_detector.robustness import (
    CorruptionDataset,
    build_condition,
    build_model,
    default_device,
    evaluate_condition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--conditions", nargs="+", default=["clean"])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_final_samples(data_dir: Path, manifest: Path) -> list[Sample]:
    samples = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "final_test":
                raise ValueError(f"Non-final record at {manifest}:{line_number}")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path at {manifest}:{line_number}")
            path = data_dir / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing final-test image: {path}")
            samples.append(
                Sample(
                    path=str(path),
                    label=int(record["label"]),
                    source=str(record["dataset_source"]),
                    family=str(record["family"]),
                )
            )
    if {sample.label for sample in samples} != {0, 1}:
        raise ValueError("The external holdout must contain both REAL and FAKE images.")
    return samples


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    seed = args.seed if args.seed is not None else int(checkpoint_args.get("seed", 42))
    image_size = int(checkpoint_args.get("image_size", 128))
    model = build_model(checkpoint_args)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    samples = load_final_samples(Path(args.data_dir), Path(args.manifest))
    results = {}
    for condition in args.conditions:
        loader = DataLoader(
            CorruptionDataset(samples, build_condition(condition, seed), image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        results[condition] = evaluate_condition(model, loader, device)
        metrics = results[condition]["fusion"]
        print(
            f"[{condition}] balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"auc={metrics['roc_auc']:.4f} real_recall={metrics['real_recall']:.4f} "
            f"fake_recall={metrics['fake_recall']:.4f}",
            flush=True,
        )

    payload = {
        "protocol": "frozen_external_final_holdout",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "sample_count": len(samples),
        "composition": dict(sorted(Counter(sample.family for sample in samples).items())),
        "seed": seed,
        "conditions": results,
        "interpretation_warning": (
            "REAL and FAKE classes come from different source families; report this "
            "domain-confounding limitation with all headline metrics."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(output)
    print(f"[done] wrote immutable final-holdout result to {output}", flush=True)


if __name__ == "__main__":
    main()

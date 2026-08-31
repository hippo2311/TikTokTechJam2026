"""Calibrate and evaluate a shared-CLIP raw-logit seed ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aigc_detector.calibration import fit_bounded_temperature
from aigc_detector.data.augmentations import build_eval_transform
from aigc_detector.data.datasets import build_train_val_cal_test_splits
from aigc_detector.models.seed_ensemble import SharedBackboneSeedEnsemble
from aigc_detector.robustness import (
    CorruptionDataset,
    build_condition,
    build_model,
    evaluate_condition,
    sample_fingerprint,
    select_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-calibration-samples", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_ensemble(paths: list[str]) -> tuple[SharedBackboneSeedEnsemble, dict]:
    models = []
    reference_args = None
    architecture_keys = (
        "semantic_backbone",
        "texture_backbone",
        "image_size",
        "top_k_patches",
        "quality_aware_fusion",
        "noise_version",
        "noise_enabled",
        "branch_dropout",
        "explicit_gate_disagreement",
    )
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {})
        if reference_args is None:
            reference_args = checkpoint_args
        else:
            mismatches = {
                key: (reference_args.get(key), checkpoint_args.get(key))
                for key in architecture_keys
                if reference_args.get(key) != checkpoint_args.get(key)
            }
            if mismatches:
                raise ValueError(f"Checkpoint architecture mismatch: {mismatches}")
        model = build_model(checkpoint_args)
        model.load_state_dict(checkpoint["model_state"])
        models.append(model)
    return SharedBackboneSeedEnsemble(models), reference_args or {}


@torch.inference_mode()
def calibrate(model, dataset, device, batch_size: int, num_workers: int) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits = []
    labels = []
    sources = []
    for batch in loader:
        outputs = model(batch["image"].to(device, non_blocking=True))
        logits.append(outputs.logits.cpu())
        labels.append(batch["label"].cpu())
        sources.extend(str(source) for source in batch["source"])
    if not logits:
        raise ValueError("The calibration split is empty.")
    report = fit_bounded_temperature(
        torch.cat(logits).numpy(),
        torch.cat(labels).numpy(),
        groups=sources,
    )
    model.temperature_scaler.temperature.data.fill_(float(report["temperature"]))
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model, checkpoint_args = load_ensemble(args.checkpoints)
    model.to(device).eval()

    image_size = int(checkpoint_args.get("image_size", 128))
    transform = build_eval_transform(image_size)
    _, validation, calibration, _ = build_train_val_cal_test_splits(
        args.data_dir,
        train_transform=transform,
        eval_transform=transform,
        val_fraction=checkpoint_args.get("val_fraction", 0.05),
        calibration_fraction=checkpoint_args.get("calibration_fraction", 0.05),
        seed=args.seed,
        max_train_samples=0,
        max_val_samples=0,
        max_calibration_samples=args.max_calibration_samples,
        max_test_samples=0,
        manifest_dir=args.manifest_dir,
    )
    calibration_report = calibrate(
        model, calibration, device, args.batch_size, args.num_workers
    )
    print(f"[calibration] {json.dumps(calibration_report, sort_keys=True)}", flush=True)

    for source in args.sources:
        samples = select_samples(
            validation.samples,
            source=source,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        conditions = {}
        for condition in args.conditions:
            loader = DataLoader(
                CorruptionDataset(
                    samples, build_condition(condition, args.seed), image_size
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            conditions[condition] = evaluate_condition(model, loader, device)
            auc = conditions[condition]["fusion"]["roc_auc"]
            print(f"[{source}/{condition}] auc={auc:.4f}", flush=True)
        payload = {
            "checkpoint": "raw_logit_average:" + ",".join(args.checkpoints),
            "split": "val",
            "sample_count": len(samples),
            "sample_fingerprint": sample_fingerprint(samples),
            "source": source,
            "seed": args.seed,
            "conditions": conditions,
            "ensemble": {
                "aggregation": "mean_raw_logits_then_temperature",
                "shared_semantic_forward": True,
                "temperature_calibration": calibration_report,
                "parameter_breakdown": model.parameter_breakdown,
            },
        }
        path = output_dir / f"{source}_validation.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"[report] {path}", flush=True)

    metadata = {
        "checkpoints": args.checkpoints,
        "seed": args.seed,
        "parameter_breakdown": model.parameter_breakdown,
        "temperature_calibration": calibration_report,
    }
    (output_dir / "ensemble_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[done] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()

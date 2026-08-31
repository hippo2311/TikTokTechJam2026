"""Safely recalibrate a completed ensemble checkpoint without retraining it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from aigc_detector.data.augmentations import build_eval_transform
from aigc_detector.data.datasets import build_train_val_cal_test_splits
from aigc_detector.models.ensemble import AIGCDetectionEnsemble
from cluster.train_cluster import fit_temperature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    model = AIGCDetectionEnsemble(
        semantic_backbone=saved_args.get("semantic_backbone", "mobilenetv3_small_100.lamb_in1k"),
        texture_backbone=saved_args.get("texture_backbone", "resnet18"),
        semantic_pretrained=False,
        texture_pretrained=False,
        image_size=saved_args.get("image_size", 128),
        top_k_patches=saved_args.get("top_k_patches", 1),
        quality_aware_fusion=saved_args.get("quality_aware_fusion", False),
        noise_version=saved_args.get("noise_version", "legacy"),
        noise_enabled=saved_args.get("noise_enabled", True),
        branch_dropout=saved_args.get("branch_dropout", 0.15),
        explicit_gate_disagreement=saved_args.get("explicit_gate_disagreement", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    transform = build_eval_transform(saved_args.get("image_size", 128))
    manifest_dir = args.manifest_dir
    saved_manifest_dir = saved_args.get("manifest_dir")
    if manifest_dir is None and saved_manifest_dir and Path(saved_manifest_dir).is_dir():
        manifest_dir = saved_manifest_dir
    _, _, calibration, _ = build_train_val_cal_test_splits(
        dataset_root=args.data_dir,
        train_transform=transform,
        eval_transform=transform,
        val_fraction=saved_args.get("val_fraction", 0.05),
        calibration_fraction=saved_args.get("calibration_fraction", 0.05),
        seed=saved_args.get("seed", 42),
        max_train_samples=2,
        max_val_samples=2,
        max_test_samples=2,
        manifest_dir=manifest_dir,
    )
    loader = DataLoader(
        calibration,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[recalibration] source={args.checkpoint} calibration_examples={len(calibration)}")
    fit_temperature(model, loader, device)
    result = dict(checkpoint)
    result["model_state"] = model.state_dict()
    result["recalibrated_from"] = args.checkpoint
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"[done] recalibrated_checkpoint={output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from aigc_detector.calibration import fit_bounded_temperature
from aigc_detector.data.augmentations import StochasticCompressionAugment, build_eval_transform
from aigc_detector.data.datasets import build_train_val_cal_test_splits
from aigc_detector.data.sampling import hierarchical_sample_weights
from aigc_detector.losses import SupervisedContrastiveLoss
from aigc_detector.metrics import basic_binary_classification_metrics
from aigc_detector.models.ensemble import AIGCDetectionEnsemble
from aigc_detector.utils.device import default_device
from aigc_detector.utils.progress import ProgressTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AIGC detection ensemble.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory containing local datasets.")
    parser.add_argument("--manifest-dir", type=str, default=None)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--epoch-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lambda-supcon", type=float, default=0.2)
    parser.add_argument("--semantic-backbone", type=str, default="mobilenetv3_small_100.lamb_in1k")
    parser.add_argument("--texture-backbone", type=str, default="resnet18")
    parser.add_argument("--top-k-patches", type=int, default=1)
    parser.add_argument("--legacy-fusion", action="store_true")
    parser.add_argument("--legacy-noise-branch", action="store_true")
    parser.add_argument("--disable-noise-branch", action="store_true")
    parser.add_argument("--branch-dropout", type=float, default=0.15)
    parser.add_argument("--explicit-gate-disagreement", action="store_true")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-calibration-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.quality_aware_fusion = not args.legacy_fusion
    args.noise_version = "legacy" if args.legacy_noise_branch else "improved"
    args.noise_enabled = not args.disable_noise_branch
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = StochasticCompressionAugment(image_size=args.image_size)
    eval_transform = build_eval_transform(image_size=args.image_size)
    train_dataset, val_dataset, calibration_dataset, test_dataset = build_train_val_cal_test_splits(
        dataset_root=args.data_dir,
        train_transform=train_transform,
        eval_transform=eval_transform,
        val_fraction=args.val_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_calibration_samples=args.max_calibration_samples,
        max_test_samples=args.max_test_samples,
        manifest_dir=args.manifest_dir,
    )
    print(
        f"[data] train={len(train_dataset)} val={len(val_dataset)} "
        f"calibration={len(calibration_dataset)} test={len(test_dataset)}"
    )

    train_sampler = None
    if args.balanced_sampling:
        weights, exposure = hierarchical_sample_weights(train_dataset.samples)
        epoch_samples = args.epoch_samples or len(train_dataset)
        if epoch_samples < 1:
            raise ValueError("--epoch-samples must be positive.")
        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=epoch_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[sampling] {json.dumps(exposure, sort_keys=True)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    init_args = {}
    excluded_init_prefixes = []
    if args.init_checkpoint:
        init_metadata = torch.load(args.init_checkpoint, map_location="cpu")
        init_args = init_metadata.get("args", {})
        if (
            init_args.get("semantic_backbone", "mobilenetv3_small_100.lamb_in1k")
            != args.semantic_backbone
        ):
            excluded_init_prefixes.append("semantic_branch.")
        if init_args.get("texture_backbone", "resnet18") != args.texture_backbone:
            excluded_init_prefixes.append("texture_branch.")
    semantic_needs_pretrained = not args.init_checkpoint or "semantic_branch." in excluded_init_prefixes
    texture_needs_pretrained = not args.init_checkpoint or "texture_branch." in excluded_init_prefixes

    model = AIGCDetectionEnsemble(
        semantic_backbone=args.semantic_backbone,
        texture_backbone=args.texture_backbone,
        semantic_pretrained=not args.no_pretrained and semantic_needs_pretrained,
        texture_pretrained=not args.no_pretrained and texture_needs_pretrained,
        image_size=args.image_size,
        top_k_patches=args.top_k_patches,
        quality_aware_fusion=args.quality_aware_fusion,
        noise_version=args.noise_version,
        noise_enabled=args.noise_enabled,
        branch_dropout=args.branch_dropout,
        explicit_gate_disagreement=args.explicit_gate_disagreement,
    ).to(device)
    if args.init_checkpoint:
        load_compatible_checkpoint(
            model,
            args.init_checkpoint,
            device,
            excluded_prefixes=excluded_init_prefixes,
        )

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    contrastive = SupervisedContrastiveLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    progress = ProgressTracker(total_steps=max(1, len(train_loader) * args.epochs))
    global_step = 0
    best_val_score = -1.0
    best_checkpoint_path = output_dir / "best_ensemble.pt"
    if args.init_checkpoint:
        initial_metrics = evaluate(model, val_loader, criterion, device)
        best_val_score = initial_metrics.get(
            "roc_auc", initial_metrics["balanced_accuracy"]
        )
        torch.save(
            {"model_state": model.state_dict(), "args": vars(args)},
            best_checkpoint_path,
        )
        print(f"[init-val] {format_metrics(initial_metrics)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        sampled_exposure: Counter[str] = Counter()

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if args.balanced_sampling:
                for source, label, family in zip(
                    batch["source"], batch["label"].tolist(), batch["family"]
                ):
                    sampled_exposure[f"source/{source}"] += 1
                    sampled_exposure[f"source_class/{source}/{label}"] += 1
                    sampled_exposure[
                        f"source_class_family/{source}/{label}/{family}"
                    ] += 1

            outputs = model(images)
            fusion_loss = criterion(outputs.logits, labels)
            texture_ce = criterion(outputs.texture_logits, labels)
            frequency_ce = criterion(outputs.frequency_logits, labels)
            noise_ce = (
                criterion(outputs.noise_logits, labels)
                if model.noise_branch is not None
                else outputs.logits.new_zeros(())
            )
            semantic_ce = criterion(outputs.semantic_logits, labels)
            supcon_loss = contrastive(outputs.texture_projections, labels)

            loss = (
                fusion_loss
                + texture_ce
                + 0.5 * frequency_ce
                + 0.5 * noise_ce
                + 0.2 * semantic_ce
                + args.lambda_supcon * supcon_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
            running_examples += labels.size(0)

            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader):
                message = progress.update(global_step, prefix=f"[train] epoch={epoch}")
                avg_loss = running_loss / max(running_examples, 1)
                avg_acc = running_correct / max(running_examples, 1)
                print(f"{message} loss={avg_loss:.4f} acc={avg_acc:.4f}")

        if args.balanced_sampling:
            print(
                f"[sampling-realized] epoch={epoch} "
                f"{json.dumps(dict(sorted(sampled_exposure.items())), sort_keys=True)}"
            )
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"[val] epoch={epoch} {format_metrics(val_metrics)}")
        scheduler.step()

        selection_score = val_metrics.get("roc_auc", val_metrics["balanced_accuracy"])
        if selection_score > best_val_score:
            best_val_score = selection_score
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_checkpoint_path)
            print(f"[checkpoint] saved {best_checkpoint_path}")

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    fit_temperature(model, calibration_loader, device)
    calibrated_path = output_dir / "best_ensemble_calibrated.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, calibrated_path)
    print(f"[checkpoint] saved calibrated model to {calibrated_path}")

    if not args.skip_test and len(test_dataset) > 0:
        test_metrics = evaluate(model, test_loader, criterion, device, apply_temperature=True)
        metrics_path = output_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(test_metrics, indent=2))
        print(f"[test] {format_metrics(test_metrics)}")
        print(f"[test] wrote metrics to {metrics_path}")
    elif args.skip_test:
        print("[test] skipped; use validation robustness during development")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_compatible_checkpoint(
    model,
    checkpoint_path: str,
    device: torch.device,
    excluded_prefixes=(),
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state
        and target_state[key].shape == value.shape
        and not any(key.startswith(prefix) for prefix in excluded_prefixes)
    }
    model.load_state_dict(compatible, strict=False)
    print(
        f"[init] loaded {len(compatible)}/{len(target_state)} compatible tensors "
        f"from {checkpoint_path}"
    )


@torch.inference_mode()
def evaluate(model, loader, criterion, device: torch.device, apply_temperature: bool = False):
    model.eval()
    total_loss = 0.0
    probabilities = []
    labels_all = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images, apply_temperature=apply_temperature)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item() * labels.size(0)
        probabilities.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    if not labels_all:
        raise ValueError("Evaluation loader is empty.")
    summary = basic_binary_classification_metrics(labels_all, probabilities)
    keys = ("accuracy", "balanced_accuracy", "f1", "average_precision", "roc_auc")
    metrics = {key: summary[key] for key in keys}
    metrics["loss"] = total_loss / len(labels_all)
    return {key: float(value) for key, value in metrics.items()}


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def fit_temperature(model, loader, device: torch.device) -> None:
    if len(loader.dataset) == 0:
        print("[calibration] skipped because the calibration split is empty")
        return
    model.eval()
    logits_list = []
    labels_list = []
    sources_list = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            logits_list.append(outputs.logits)
            labels_list.append(labels)
            sources_list.extend(str(source) for source in batch["source"])

    logits = torch.cat(logits_list, dim=0).float().cpu().numpy()
    labels = torch.cat(labels_list, dim=0).cpu().numpy()
    report = fit_bounded_temperature(logits, labels, groups=sources_list)
    model.temperature_scaler.temperature.data.fill_(float(report["temperature"]))
    print(f"[calibration] {json.dumps(report, sort_keys=True)}")


if __name__ == "__main__":
    main()

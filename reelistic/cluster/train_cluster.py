"""Slurm/CUDA training entry point kept separate from the working Mac trainer.

This module intentionally avoids scikit-learn, supports CUDA mixed precision,
and writes a resumable checkpoint after every epoch. Final best-model files use
the same ``model_state``/``args`` format as ``aigc_detector.predict``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from aigc_detector.data.augmentations import StochasticCompressionAugment, build_eval_transform
from aigc_detector.data.datasets import build_train_val_cal_test_splits
from aigc_detector.data.sampling import hierarchical_sample_weights
from aigc_detector.calibration import fit_bounded_temperature
from aigc_detector.losses import SupervisedContrastiveLoss
from aigc_detector.metrics import (
    average_precision as _average_precision,
    basic_binary_classification_metrics,
    roc_auc,
)
from aigc_detector.models.ensemble import AIGCDetectionEnsemble
from aigc_detector.selection import checkpoint_selection_components, checkpoint_selection_score
from aigc_detector.utils.progress import ProgressTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AIGC ensemble on Slurm/CUDA.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="Load fixed train/validation/calibration/test JSONL manifests from this directory.",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Balance expected exposure by source, then class, then family.",
    )
    parser.add_argument(
        "--epoch-samples",
        type=int,
        default=None,
        help="Number of replacement samples per balanced epoch; defaults to manifest train size.",
    )
    parser.add_argument("--output-dir", default="cluster_outputs/training_run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients to raise effective batch size without extra GPU memory.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lambda-supcon", type=float, default=0.2)
    parser.add_argument(
        "--fusion-reliability-weight",
        type=float,
        default=0.0,
        help="Penalize fusion when its per-image CE exceeds the semantic branch; zero preserves prior behavior.",
    )
    parser.add_argument("--clean-probability", type=float, default=0.2)
    parser.add_argument(
        "--robustness-focus",
        action="store_true",
        help="Increase exposure to severe downsample, blur, and noise recipes.",
    )
    parser.add_argument("--semantic-backbone", default="mobilenetv3_small_100.lamb_in1k")
    parser.add_argument("--texture-backbone", default="resnet18")
    parser.add_argument("--top-k-patches", type=int, default=1)
    parser.add_argument("--legacy-fusion", action="store_true")
    parser.add_argument("--legacy-noise-branch", action="store_true")
    parser.add_argument("--disable-noise-branch", action="store_true")
    parser.add_argument("--branch-dropout", type=float, default=0.15)
    parser.add_argument(
        "--explicit-gate-disagreement",
        action="store_true",
        help="Add a zero-initialized branch-disagreement correction to gate logits.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-calibration-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Resume from latest_training.pt.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights from a compatible Mac/cluster checkpoint.",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "mean_source_auc", "robust_source_auc"),
        default="mean_source_auc",
        help="Choose checkpoints by aggregate, mean-source, or weakest-source/fusion-aware AUC.",
    )
    parser.add_argument(
        "--gate-warmup",
        action="store_true",
        help="Freeze texture/frequency/semantic branches; train the repaired noise branch and fusion gate.",
    )
    parser.add_argument(
        "--fusion-only-warmup",
        action="store_true",
        help="Freeze all four evidence branches and train only the fusion gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1.")
    if args.fusion_reliability_weight < 0:
        raise ValueError("--fusion-reliability-weight must be non-negative.")
    if not 0.0 <= args.clean_probability <= 1.0:
        raise ValueError("--clean-probability must be between zero and one.")
    args.quality_aware_fusion = not args.legacy_fusion
    args.noise_version = "legacy" if args.legacy_noise_branch else "improved"
    args.noise_enabled = not args.disable_noise_branch
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable inside the Slurm container.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_ensemble.pt"
    latest_path = output_dir / "latest_training.pt"
    print_runtime(device)

    train_transform = StochasticCompressionAugment(
        image_size=args.image_size,
        clean_probability=args.clean_probability,
        robustness_focus=args.robustness_focus,
    )
    eval_transform = build_eval_transform(image_size=args.image_size)
    train_dataset, val_dataset, calibration_dataset, test_dataset = (
        build_train_val_cal_test_splits(
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
    )
    print(
        f"[data] train={len(train_dataset)} val={len(val_dataset)} "
        f"calibration={len(calibration_dataset)} test={len(test_dataset)}"
    )
    print(
        f"[batch] physical={args.batch_size} accumulation={args.grad_accum_steps} "
        f"effective={args.batch_size * args.grad_accum_steps}",
        flush=True,
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_sampler = None
    if args.balanced_sampling:
        weights, exposure = hierarchical_sample_weights(train_dataset.samples)
        epoch_samples = args.epoch_samples or len(train_dataset)
        if epoch_samples < 1:
            raise ValueError("--epoch-samples must be positive.")
        generator = torch.Generator().manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=epoch_samples,
            replacement=True,
            generator=generator,
        )
        print(f"[sampling] {json.dumps(exposure, sort_keys=True)}", flush=True)
        print(f"[sampling] epoch_samples={epoch_samples}", flush=True)
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    load_path = args.resume or args.init_checkpoint
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
    semantic_needs_pretrained = (
        load_path is None or "semantic_branch." in excluded_init_prefixes
    )
    texture_needs_pretrained = (
        load_path is None or "texture_branch." in excluded_init_prefixes
    )
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
    if args.gate_warmup and args.fusion_only_warmup:
        raise ValueError("Choose only one warmup mode.")
    if args.gate_warmup:
        frozen_modules = [
            model.texture_branch,
            model.frequency_branch,
            model.semantic_branch,
        ]
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(
            f"[stage] gate_warmup=true trainable_parameters={trainable}",
            flush=True,
        )
    elif args.fusion_only_warmup:
        frozen_modules = [
            model.texture_branch,
            model.frequency_branch,
            model.noise_branch,
            model.semantic_branch,
        ]
        for module in frozen_modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        for parameter in model.temperature_scaler.parameters():
            parameter.requires_grad = False
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"[stage] fusion_only_warmup=true trainable_parameters={trainable}",
            flush=True,
        )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    start_epoch = 1
    best_val_score = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_score = float(checkpoint["best_val_score"])
        print(f"[resume] continuing at epoch={start_epoch} from {args.resume}")
    elif args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        load_compatible_state(
            model,
            checkpoint["model_state"],
            args.init_checkpoint,
            excluded_prefixes=excluded_init_prefixes,
        )

    criterion = nn.CrossEntropyLoss()
    contrastive = SupervisedContrastiveLoss()
    amp_enabled = device.type == "cuda" and not args.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if args.init_checkpoint:
        initial_metrics = evaluate(model, val_loader, criterion, device)
        best_val_score = checkpoint_selection_score(initial_metrics, args.selection_metric)
        save_model_checkpoint(best_path, model, args)
        print(f"[init-val] {format_metrics(initial_metrics)}", flush=True)
        print(f"[checkpoint] preserved initialization at {best_path}", flush=True)
    remaining_epochs = max(args.epochs - start_epoch + 1, 0)
    progress = ProgressTracker(total_steps=max(1, len(train_loader) * remaining_epochs))
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        sampled_exposure: Counter[str] = Counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if args.balanced_sampling:
                for source, label, family in zip(
                    batch["source"], batch["label"].tolist(), batch["family"]
                ):
                    sampled_exposure[f"source/{source}"] += 1
                    sampled_exposure[f"source_class/{source}/{label}"] += 1
                    sampled_exposure[
                        f"source_class_family/{source}/{label}/{family}"
                    ] += 1
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(images)
                loss = combined_loss(
                    outputs,
                    labels,
                    criterion,
                    contrastive,
                    args.lambda_supcon,
                    noise_enabled=model.noise_branch is not None,
                    fusion_reliability_weight=args.fusion_reliability_weight,
                )
            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = (
                batch_index % args.grad_accum_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
            running_examples += labels.size(0)
            if batch_index % args.log_interval == 0 or batch_index == len(train_loader):
                status = progress.update(global_step, prefix=f"[train] epoch={epoch}")
                print(
                    f"{status} loss={running_loss / running_examples:.4f} "
                    f"acc={running_correct / running_examples:.4f}",
                    flush=True,
                )

        if args.balanced_sampling:
            print(
                f"[sampling-realized] epoch={epoch} "
                f"{json.dumps(dict(sorted(sampled_exposure.items())), sort_keys=True)}",
                flush=True,
            )
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"[val] epoch={epoch} {format_metrics(val_metrics)}", flush=True)
        scheduler.step()
        selection_score = checkpoint_selection_score(val_metrics, args.selection_metric)
        selection_components = checkpoint_selection_components(val_metrics)
        print(
            f"[selection] mode={args.selection_metric} score={selection_score:.6f} "
            f"{json.dumps(selection_components, sort_keys=True)}",
            flush=True,
        )
        if selection_score > best_val_score:
            best_val_score = selection_score
            save_model_checkpoint(best_path, model, args)
            print(f"[checkpoint] saved best model to {best_path}", flush=True)
        save_training_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_score,
            args,
        )
        print(f"[checkpoint] saved resumable state to {latest_path}", flush=True)

    if not best_path.exists():
        raise FileNotFoundError(
            f"No best checkpoint exists at {best_path}; check --epochs and --resume values."
        )
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    fit_temperature(model, calibration_loader, device)
    calibrated_path = output_dir / "best_ensemble_calibrated.pt"
    save_model_checkpoint(calibrated_path, model, args)

    if args.skip_test:
        print(f"[done] checkpoint={calibrated_path} test=skipped", flush=True)
    else:
        test_metrics = evaluate(model, test_loader, criterion, device, apply_temperature=True)
        metrics_path = output_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(test_metrics, indent=2))
        print(f"[test] {format_metrics(test_metrics)}", flush=True)
        print(f"[done] checkpoint={calibrated_path} metrics={metrics_path}", flush=True)


def combined_loss(
    outputs,
    labels,
    criterion,
    contrastive,
    lambda_supcon: float,
    noise_enabled: bool = True,
    fusion_reliability_weight: float = 0.0,
):
    noise_loss = (
        criterion(outputs.noise_logits, labels)
        if noise_enabled
        else outputs.logits.new_zeros(())
    )
    fusion_loss = criterion(outputs.logits, labels)
    reliability_loss = fusion_reliability_penalty(
        outputs.logits, outputs.semantic_logits, labels
    )
    return (
        fusion_loss
        + criterion(outputs.texture_logits, labels)
        + 0.5 * criterion(outputs.frequency_logits, labels)
        + 0.5 * noise_loss
        + 0.2 * criterion(outputs.semantic_logits, labels)
        + lambda_supcon * contrastive(outputs.texture_projections, labels)
        + fusion_reliability_weight * reliability_loss
    )


def fusion_reliability_penalty(
    fusion_logits: torch.Tensor,
    semantic_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Only penalize examples where fusion is worse than frozen semantic evidence."""
    fusion_ce = F.cross_entropy(fusion_logits, labels, reduction="none")
    semantic_ce = F.cross_entropy(
        semantic_logits.detach(), labels, reduction="none"
    )
    return F.relu(fusion_ce - semantic_ce).mean()


@torch.inference_mode()
def evaluate(model, loader, criterion, device: torch.device, apply_temperature: bool = False):
    model.eval()
    total_loss = 0.0
    labels_all = []
    probabilities = []
    semantic_probabilities = []
    sources_all = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images, apply_temperature=apply_temperature)
        total_loss += criterion(outputs.logits, labels).item() * labels.size(0)
        probabilities.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist())
        semantic_probabilities.extend(
            torch.softmax(outputs.semantic_logits, dim=1)[:, 1].cpu().tolist()
        )
        labels_all.extend(labels.cpu().tolist())
        sources_all.extend(str(source) for source in batch["source"])
    if not labels_all:
        raise ValueError("Evaluation loader is empty.")
    metrics = binary_metrics(np.asarray(labels_all), np.asarray(probabilities))
    labels_array = np.asarray(labels_all)
    probabilities_array = np.asarray(probabilities)
    semantic_probabilities_array = np.asarray(semantic_probabilities)
    sources_array = np.asarray(sources_all)
    for source in sorted(set(sources_all)):
        source_mask = sources_array == source
        source_labels = labels_array[source_mask]
        if len(np.unique(source_labels)) < 2:
            continue
        safe_source = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
        metrics[f"source_auc_{safe_source}"] = roc_auc(
            source_labels, probabilities_array[source_mask]
        )
        semantic_auc = roc_auc(
            source_labels, semantic_probabilities_array[source_mask]
        )
        metrics[f"source_semantic_auc_{safe_source}"] = semantic_auc
        metrics[f"fusion_minus_semantic_auc_{safe_source}"] = (
            metrics[f"source_auc_{safe_source}"] - semantic_auc
        )
    metrics["semantic_roc_auc"] = roc_auc(
        labels_array, semantic_probabilities_array
    )
    metrics["loss"] = total_loss / len(labels_all)
    return {key: float(value) for key, value in metrics.items()}


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = basic_binary_classification_metrics(labels, probabilities)
    keys = ("accuracy", "balanced_accuracy", "f1", "average_precision", "roc_auc")
    return {key: float(metrics[key]) for key in keys}


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compatibility wrapper for older tests and analysis scripts."""
    return _average_precision(labels, scores)


def fit_temperature(model, loader, device: torch.device) -> None:
    model.eval()
    logits_all = []
    labels_all = []
    sources_all = []
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["image"].to(device, non_blocking=True))
            logits_all.append(outputs.logits)
            labels_all.append(batch["label"].to(device, non_blocking=True))
            sources_all.extend(str(source) for source in batch["source"])
    if not logits_all:
        print("[calibration] skipped because the calibration split is empty")
        return
    logits = torch.cat(logits_all).float().cpu().numpy()
    labels = torch.cat(labels_all).cpu().numpy()
    report = fit_bounded_temperature(logits, labels, groups=sources_all)
    temperature = float(report["temperature"])
    model.temperature_scaler.temperature.data.fill_(temperature)
    print(f"[calibration] {json.dumps(report, sort_keys=True)}", flush=True)


def save_model_checkpoint(path: Path, model, args: argparse.Namespace) -> None:
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, path)


def load_compatible_state(
    model, source_state, source_name: str, excluded_prefixes=()
) -> None:
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
        f"from {source_name}"
    )


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, best_val_score, args):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_score": best_val_score,
            "args": vars(args),
        },
        path,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_runtime(device: torch.device) -> None:
    print(f"[runtime] torch={torch.__version__} device={device}")
    if device.type == "cuda":
        print(f"[runtime] gpu={torch.cuda.get_device_name(device)} cuda={torch.version.cuda}")


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


if __name__ == "__main__":
    main()

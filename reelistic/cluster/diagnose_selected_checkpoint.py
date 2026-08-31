"""Gate-2/3 operating-point and focused family diagnosis for one checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from aigc_detector.data.augmentations import build_eval_transform
from aigc_detector.data.datasets import build_train_val_cal_test_splits
from aigc_detector.metrics import (
    binary_classification_metrics,
    operating_point_metrics,
    roc_auc,
)
from aigc_detector.robustness import (
    CorruptionDataset,
    build_condition,
    build_model,
    sample_fingerprint,
    select_samples,
)


BRANCH_NAMES = ("texture", "frequency", "noise", "semantic")
SEVERE_CONDITIONS = ("downsample_25", "blur_2.0", "jpeg_30", "noise_0.10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--sources", nargs="+", default=["birdy654", "external_pilot", "WildFake"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-samples", type=int, default=5000)
    parser.add_argument("--family-samples-per-class", type=int, default=1000)
    parser.add_argument("--min-family-samples", type=int, default=100)
    parser.add_argument("--weak-family-count", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def infer(
    model,
    samples,
    condition: str,
    image_size: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    semantic_enabled: bool = True,
) -> dict[str, object]:
    if not semantic_enabled and not getattr(model, "quality_aware_fusion", False):
        raise ValueError("Semantic-mask ablation requires quality-aware fusion.")
    original_available = None
    if not semantic_enabled:
        original_available = model.available_branches.clone()
        model.available_branches[3] = False
    try:
        loader = DataLoader(
            CorruptionDataset(samples, build_condition(condition, seed), image_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        labels = []
        probabilities = {"fusion": [], **{name: [] for name in BRANCH_NAMES}}
        gates = []
        for images, batch_labels in loader:
            outputs = model(images.to(device, non_blocking=True), apply_temperature=True)
            labels.extend(batch_labels.tolist())
            logits = {
                "fusion": outputs.logits,
                "texture": outputs.texture_logits,
                "frequency": outputs.frequency_logits,
                "noise": outputs.noise_logits,
                "semantic": outputs.semantic_logits,
            }
            for name, values in logits.items():
                probabilities[name].extend(
                    torch.softmax(values, dim=1)[:, 1].cpu().tolist()
                )
            if outputs.gate_weights is not None:
                gates.append(outputs.gate_weights.cpu().numpy())
    finally:
        if original_available is not None:
            model.available_branches.copy_(original_available)

    label_array = np.asarray(labels, dtype=np.int64)
    probability_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in probabilities.items()
    }
    gate_array = np.concatenate(gates) if gates else None
    return {
        "labels": label_array,
        "probabilities": probability_arrays,
        "gate_weights": gate_array,
    }


def report_inference(raw: dict, *, include_fusion_curves: bool = False) -> dict:
    labels = raw["labels"]
    probabilities = raw["probabilities"]
    result = {
        name: binary_classification_metrics(
            labels,
            values,
            include_curves=(include_fusion_curves and name == "fusion"),
        )
        for name, values in probabilities.items()
    }
    gates = raw["gate_weights"]
    if gates is not None:
        result["mean_gate_weights"] = dict(
            zip(BRANCH_NAMES, gates.mean(axis=0).astype(float).tolist())
        )
    return result


def bootstrap_intervals(labels, scores, repetitions: int, seed: int) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(group) == 0 for group in groups):
        return {}
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(repetitions):
        selected = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        sampled_labels = labels[selected]
        sampled_scores = scores[selected]
        operating = operating_point_metrics(sampled_labels, sampled_scores)
        values["roc_auc"].append(roc_auc(sampled_labels, sampled_scores))
        values["tpr_at_1pct_fpr"].append(operating["tpr_at_1pct_fpr"])
        values["fpr_at_99pct_tpr"].append(operating["fpr_at_99pct_tpr"])
    return {
        name: {
            "lower_95": float(np.nanpercentile(metric_values, 2.5)),
            "upper_95": float(np.nanpercentile(metric_values, 97.5)),
            "repetitions": repetitions,
        }
        for name, metric_values in values.items()
    }


def disagreement_report(raw: dict) -> dict:
    labels = raw["labels"]
    probabilities = raw["probabilities"]
    branch_matrix = np.column_stack([probabilities[name] for name in BRANCH_NAMES])
    disagreement = branch_matrix.var(axis=1)
    predictions = (probabilities["fusion"] >= 0.5).astype(np.int64)
    errors = predictions != labels
    gates = raw["gate_weights"]
    result = {
        "mean_branch_probability_variance": float(disagreement.mean()),
        "mean_variance_correct": float(disagreement[~errors].mean()) if (~errors).any() else None,
        "mean_variance_incorrect": float(disagreement[errors].mean()) if errors.any() else None,
        "error_count": int(errors.sum()),
    }
    if disagreement.std() > 0 and errors.std() > 0:
        result["correlation_disagreement_with_error"] = float(
            np.corrcoef(disagreement, errors.astype(float))[0, 1]
        )
    if gates is not None:
        result["mean_gate_correct"] = dict(
            zip(BRANCH_NAMES, gates[~errors].mean(axis=0).astype(float).tolist())
        ) if (~errors).any() else None
        result["mean_gate_incorrect"] = dict(
            zip(BRANCH_NAMES, gates[errors].mean(axis=0).astype(float).tolist())
        ) if errors.any() else None
        if disagreement.std() > 0 and gates[:, 3].std() > 0:
            result["correlation_disagreement_with_semantic_gate"] = float(
                np.corrcoef(disagreement, gates[:, 3])[0, 1]
            )
    return result


def deterministic_subset(samples, limit: int, seed: int, token: str):
    samples = list(samples)
    digest = hashlib.sha256(f"{seed}:{token}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(samples)
    return samples[:limit]


def family_contrasts(
    samples,
    max_per_class: int,
    minimum: int,
    seed: int,
) -> list[dict]:
    by_family = defaultdict(list)
    by_label = defaultdict(list)
    for sample in samples:
        by_family[(sample.label, sample.family)].append(sample)
        by_label[sample.label].append(sample)
    contrasts = []
    for (label, family), family_samples in sorted(by_family.items()):
        opposite = by_label[1 - label]
        count = min(len(family_samples), len(opposite), max_per_class)
        if count < minimum:
            continue
        target = deterministic_subset(family_samples, count, seed, f"target:{label}:{family}")
        reference = deterministic_subset(opposite, count, seed, f"reference:{1-label}")
        combined = deterministic_subset(target + reference, count * 2, seed, f"mix:{family}")
        contrasts.append(
            {
                "family": family,
                "target_label": int(label),
                "samples_per_class": count,
                "samples": combined,
                "sample_fingerprint": sample_fingerprint(combined),
            }
        )
    return contrasts


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = build_model(checkpoint_args)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()
    image_size = int(checkpoint_args.get("image_size", 128))
    transform = build_eval_transform(image_size)
    _, validation, _, _ = build_train_val_cal_test_splits(
        args.data_dir,
        train_transform=transform,
        eval_transform=transform,
        val_fraction=checkpoint_args.get("val_fraction", 0.05),
        calibration_fraction=checkpoint_args.get("calibration_fraction", 0.05),
        seed=args.seed,
        max_train_samples=0,
        max_val_samples=0,
        max_calibration_samples=0,
        max_test_samples=0,
        manifest_dir=args.manifest_dir,
    )

    source_results = {}
    source_raw = []
    for source_index, source in enumerate(args.sources):
        samples = select_samples(
            validation.samples, source, args.source_samples, args.seed
        )
        raw = infer(
            model, samples, "clean", image_size, args.seed, device,
            args.batch_size, args.num_workers,
        )
        report = report_inference(raw, include_fusion_curves=True)
        report["sample_fingerprint"] = sample_fingerprint(samples)
        report["bootstrap_95_ci"] = bootstrap_intervals(
            raw["labels"], raw["probabilities"]["fusion"],
            args.bootstrap_repetitions, args.seed + source_index,
        )
        report["disagreement"] = disagreement_report(raw)
        source_results[source] = report
        source_raw.append(raw)
        fusion = report["fusion"]
        print(
            f"[operating] source={source} n={len(samples)} "
            f"auc={fusion['roc_auc']:.4f} "
            f"tpr@1%fpr={fusion['tpr_at_1pct_fpr']:.4f} "
            f"fpr@99%tpr={fusion['fpr_at_99pct_tpr']:.4f}",
            flush=True,
        )

    combined_raw = {
        "labels": np.concatenate([raw["labels"] for raw in source_raw]),
        "probabilities": {
            name: np.concatenate([raw["probabilities"][name] for raw in source_raw])
            for name in ("fusion", *BRANCH_NAMES)
        },
        "gate_weights": np.concatenate([raw["gate_weights"] for raw in source_raw]),
    }
    overall = report_inference(combined_raw, include_fusion_curves=True)
    overall["bootstrap_95_ci"] = bootstrap_intervals(
        combined_raw["labels"], combined_raw["probabilities"]["fusion"],
        args.bootstrap_repetitions, args.seed + 100,
    )
    overall["disagreement"] = disagreement_report(combined_raw)

    wildfake_samples = [
        sample for sample in validation.samples if sample.source == "WildFake"
    ]
    contrasts = family_contrasts(
        wildfake_samples,
        args.family_samples_per_class,
        args.min_family_samples,
        args.seed,
    )
    clean_families = {}
    contrast_lookup = {}
    for contrast in contrasts:
        raw = infer(
            model, contrast["samples"], "clean", image_size, args.seed, device,
            args.batch_size, args.num_workers,
        )
        report = report_inference(raw)
        report.update(
            {
                "target_label": contrast["target_label"],
                "samples_per_class": contrast["samples_per_class"],
                "sample_fingerprint": contrast["sample_fingerprint"],
                "disagreement": disagreement_report(raw),
            }
        )
        key = f"label{contrast['target_label']}:{contrast['family']}"
        clean_families[key] = report
        contrast_lookup[key] = contrast
        print(
            f"[family-clean] family={key} n={2 * contrast['samples_per_class']} "
            f"fusion_auc={report['fusion']['roc_auc']:.4f}",
            flush=True,
        )

    fake_families = [
        key for key, report in clean_families.items() if report["target_label"] == 1
    ]
    weakest = sorted(
        fake_families, key=lambda key: clean_families[key]["fusion"]["roc_auc"]
    )[: args.weak_family_count]
    focused = {}
    for key in weakest:
        contrast = contrast_lookup[key]
        focused[key] = {}
        for condition in ("clean", *SEVERE_CONDITIONS):
            full_raw = infer(
                model, contrast["samples"], condition, image_size, args.seed, device,
                args.batch_size, args.num_workers,
            )
            masked_raw = infer(
                model, contrast["samples"], condition, image_size, args.seed, device,
                args.batch_size, args.num_workers, semantic_enabled=False,
            )
            full = report_inference(full_raw)
            masked = report_inference(masked_raw)
            focused[key][condition] = {
                "full": full,
                "semantic_mask": masked,
                "fusion_auc_delta_mask_minus_full": float(
                    masked["fusion"]["roc_auc"] - full["fusion"]["roc_auc"]
                ),
            }
            print(
                f"[focused] family={key} condition={condition} "
                f"full_auc={full['fusion']['roc_auc']:.4f} "
                f"mask_auc={masked['fusion']['roc_auc']:.4f}",
                flush=True,
            )

    payload = {
        "protocol": "selected_checkpoint_gate2_gate3_development_diagnosis",
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "operating_point_source_samples": args.source_samples,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "source_clean": source_results,
        "overall_clean": overall,
        "wildfake_family_protocol": (
            "Each target family is contrasted against an equal-sized deterministic "
            "pool from the opposite class; family AUC is not computed from a "
            "single-class family alone."
        ),
        "wildfake_clean_by_family": clean_families,
        "weakest_fake_families": weakest,
        "focused_family_conditions": focused,
        "final_external_holdout_used": False,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True))
    temporary.replace(output_path)
    print(f"[done] diagnosis={output_path}", flush=True)


if __name__ == "__main__":
    main()

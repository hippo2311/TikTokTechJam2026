"""Checkpoint-selection policies kept independent of the training runtime."""

from __future__ import annotations

from statistics import fmean


def checkpoint_selection_components(metrics: dict[str, float]) -> dict[str, float]:
    source_aucs = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("source_auc_")
    ]
    if not source_aucs:
        fallback = float(metrics.get("roc_auc", metrics["balanced_accuracy"]))
        return {
            "mean_source_auc": fallback,
            "weakest_source_auc": fallback,
            "weakest_fusion_gap": 0.0,
            "fusion_gap_penalty": 0.0,
        }
    fusion_gaps = [
        float(value)
        for key, value in metrics.items()
        if key.startswith("fusion_minus_semantic_auc_")
    ]
    weakest_gap = min(fusion_gaps) if fusion_gaps else 0.0
    return {
        "mean_source_auc": fmean(source_aucs),
        "weakest_source_auc": min(source_aucs),
        "weakest_fusion_gap": weakest_gap,
        "fusion_gap_penalty": max(0.0, -weakest_gap),
    }


def checkpoint_selection_score(metrics: dict[str, float], mode: str) -> float:
    if mode == "roc_auc":
        return float(metrics.get("roc_auc", metrics["balanced_accuracy"]))
    components = checkpoint_selection_components(metrics)
    if mode == "mean_source_auc":
        return components["mean_source_auc"]
    if mode == "robust_source_auc":
        return (
            0.5 * components["mean_source_auc"]
            + 0.5 * components["weakest_source_auc"]
            - 0.25 * components["fusion_gap_penalty"]
        )
    raise ValueError(f"Unknown checkpoint selection mode: {mode}")

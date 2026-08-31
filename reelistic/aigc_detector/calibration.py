"""Deterministic, bounded temperature calibration utilities."""

from __future__ import annotations

import numpy as np

from aigc_detector.metrics import binary_classification_metrics


def fit_bounded_temperature(
    logits,
    labels,
    *,
    groups=None,
    minimum: float = 0.05,
    maximum: float = 10.0,
    grid_size: int = 401,
) -> dict[str, object]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) != len(labels):
        raise ValueError("Expected logits with shape [N, 2] and one label per row.")
    if minimum <= 0 or maximum <= minimum or grid_size < 3:
        raise ValueError("Temperature bounds and grid size are invalid.")

    selected = (
        balanced_group_indices(labels, groups)
        if groups is not None
        else balanced_indices(labels)
    )
    if not len(selected):
        raise ValueError("Calibration requires both binary classes.")
    balanced_logits = logits[selected]
    balanced_labels = labels[selected]
    temperatures = np.geomspace(minimum, maximum, grid_size)
    losses = np.asarray(
        [negative_log_likelihood(balanced_logits, balanced_labels, value) for value in temperatures]
    )
    best_index = int(np.nanargmin(losses))
    candidate = float(temperatures[best_index])
    before = calibration_report(balanced_logits, balanced_labels, 1.0)

    status = "accepted"
    if best_index in {0, len(temperatures) - 1}:
        status = "boundary_rejected"
        candidate = 1.0
    elif not np.isfinite(losses[best_index]) or losses[best_index] >= before["log_loss"]:
        status = "non_improving_rejected"
        candidate = 1.0

    after = calibration_report(balanced_logits, balanced_labels, candidate)
    result = {
        "status": status,
        "temperature": candidate,
        "balanced_examples": int(len(selected)),
        "examples_per_class": int(len(selected) // 2),
        "before": before,
        "after": after,
    }
    if groups is not None:
        unique_groups = np.unique(np.asarray(groups, dtype=str))
        result["balanced_sources"] = unique_groups.tolist()
        result["examples_per_source_class"] = int(len(selected) // (2 * len(unique_groups)))
    return result


def balanced_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    count = min(len(negative), len(positive))
    if count == 0:
        return np.asarray([], dtype=np.int64)
    selected = np.concatenate([negative[:count], positive[:count]])
    return np.sort(selected)


def balanced_group_indices(labels: np.ndarray, groups) -> np.ndarray:
    """Select equal counts for every source/class pair used in calibration."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    if len(labels) != len(groups):
        raise ValueError("Calibration groups must have one value per label.")
    group_names = np.unique(groups)
    buckets = [
        np.flatnonzero((groups == group) & (labels == label))
        for group in group_names
        for label in (0, 1)
    ]
    count = min((len(bucket) for bucket in buckets), default=0)
    if count == 0:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.concatenate([bucket[:count] for bucket in buckets]))


def calibration_report(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict[str, float]:
    probabilities = probabilities_from_logits(logits, temperature)
    metrics = binary_classification_metrics(labels, probabilities)
    return {
        "log_loss": metrics["log_loss"],
        "ece": metrics["ece"],
        "brier": metrics["brier"],
    }


def probabilities_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials[:, 1] / exponentials.sum(axis=1)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probabilities = probabilities_from_logits(logits, temperature)
    epsilon = 1e-12
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))

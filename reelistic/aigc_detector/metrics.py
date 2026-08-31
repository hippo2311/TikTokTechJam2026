"""Dependency-light binary classification metrics used by every entry point."""

from __future__ import annotations

import numpy as np


def basic_binary_classification_metrics(labels, probabilities) -> dict[str, object]:
    """Return threshold and ranking metrics without calibration/curve overhead."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    return {
        "sample_count": int(len(labels)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "fake_recall": float(true_positive / max(positive_count, 1)),
        "real_recall": float(true_negative / max(negative_count, 1)),
        "accuracy": float((true_positive + true_negative) / max(len(labels), 1)),
        "balanced_accuracy": float(
            0.5
            * (
                true_positive / max(positive_count, 1)
                + true_negative / max(negative_count, 1)
            )
        ),
        "f1": float(
            2
            * true_positive
            / max(2 * true_positive + false_positive + false_negative, 1)
        ),
        "average_precision": average_precision(labels, probabilities),
        "roc_auc": roc_auc(labels, probabilities),
    }


def binary_classification_metrics(
    labels, probabilities, calibration_bins: int = 15, include_curves: bool = False
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    epsilon = 1e-7
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    operating = operating_point_metrics(labels, probabilities)
    result = {
        **basic_binary_classification_metrics(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities, calibration_bins),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        ),
        **operating,
    }
    if include_curves:
        result["roc_curve"] = json_safe_curve(
            roc_curve_points(labels, probabilities)
        )
        result["precision_recall_curve"] = json_safe_curve(
            precision_recall_curve_points(labels, probabilities)
        )
    return result


def json_safe_curve(curve: dict[str, list[float]]) -> dict[str, list[float | None]]:
    return {
        name: [float(value) if np.isfinite(value) else None for value in values]
        for name, values in curve.items()
    }


def operating_point_metrics(labels, scores) -> dict[str, float]:
    """Return deployment-oriented operating points using tied-score thresholds."""
    curve = roc_curve_points(labels, scores)
    fpr = np.asarray(curve["fpr"], dtype=np.float64)
    tpr = np.asarray(curve["tpr"], dtype=np.float64)
    thresholds = np.asarray(curve["thresholds"], dtype=np.float64)
    if not len(fpr) or np.isnan(fpr).all() or np.isnan(tpr).all():
        return {
            "tpr_at_1pct_fpr": float("nan"),
            "threshold_at_1pct_fpr": float("nan"),
            "actual_fpr_at_1pct_target": float("nan"),
            "fpr_at_99pct_tpr": float("nan"),
            "threshold_at_99pct_tpr": float("nan"),
            "actual_tpr_at_99pct_target": float("nan"),
        }

    allowed = np.flatnonzero(fpr <= 0.01 + 1e-12)
    low_fpr_index = int(allowed[np.argmax(tpr[allowed])])
    sufficient = np.flatnonzero(tpr >= 0.99 - 1e-12)
    high_tpr_index = int(sufficient[np.argmin(fpr[sufficient])])
    return {
        "tpr_at_1pct_fpr": float(tpr[low_fpr_index]),
        "threshold_at_1pct_fpr": float(thresholds[low_fpr_index]),
        "actual_fpr_at_1pct_target": float(fpr[low_fpr_index]),
        "fpr_at_99pct_tpr": float(fpr[high_tpr_index]),
        "threshold_at_99pct_tpr": float(thresholds[high_tpr_index]),
        "actual_tpr_at_99pct_target": float(tpr[high_tpr_index]),
    }


def roc_curve_points(labels, scores) -> dict[str, list[float]]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return {"fpr": [], "tpr": [], "thresholds": []}
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores)) + 1, len(scores)]
    cumulative_positive = np.cumsum(ordered_labels == 1)
    cumulative_negative = np.cumsum(ordered_labels == 0)
    indices = group_ends - 1
    return {
        "fpr": [0.0, *(cumulative_negative[indices] / negative_count).astype(float).tolist()],
        "tpr": [0.0, *(cumulative_positive[indices] / positive_count).astype(float).tolist()],
        "thresholds": [float("inf"), *ordered_scores[indices].astype(float).tolist()],
    }


def precision_recall_curve_points(labels, scores) -> dict[str, list[float]]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        return {"precision": [], "recall": [], "thresholds": []}
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores)) + 1, len(scores)]
    cumulative_positive = np.cumsum(ordered_labels == 1)
    indices = group_ends - 1
    predicted_positive = group_ends.astype(np.float64)
    return {
        "precision": [1.0, *(cumulative_positive[indices] / predicted_positive).astype(float).tolist()],
        "recall": [0.0, *(cumulative_positive[indices] / positive_count).astype(float).tolist()],
        "thresholds": [float("inf"), *ordered_scores[indices].astype(float).tolist()],
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels)
    precision = cumulative_positive / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels == 1].sum() / positive_count)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    predictions = (probabilities >= 0.5).astype(np.int64)
    confidences = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (predictions == labels).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            error += mask.mean() * abs(confidences[mask].mean() - correctness[mask].mean())
    return float(error)

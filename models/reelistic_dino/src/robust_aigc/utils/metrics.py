"""Small, dependency-light per-epoch metrics writer."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def binary_operating_point_metrics(labels, scores) -> dict[str, float]:
    """Compute binary validation metrics for positive (AI-generated) scores.

    Scores must increase with confidence that the image is AI-generated (label 1).
    `tpr_at_1_fpr` is higher-is-better; `fpr_at_99_tpr` is lower-is-better.
    Values are fractions rather than percent strings so CSV/JSONL stay numerical.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1 or len(labels_array) != len(scores_array):
        raise ValueError("labels and scores must be one-dimensional arrays with equal length")
    if len(labels_array) == 0 or set(np.unique(labels_array)) != {0, 1}:
        raise ValueError("operating-point metrics require at least one real (0) and one AI (1) example")
    fpr, tpr, thresholds = roc_curve(labels_array, scores_array, pos_label=1)

    def tpr_at_fpr(target_fpr: float) -> tuple[float, float]:
        candidates = np.flatnonzero(fpr <= target_fpr)
        selected = candidates[np.argmax(tpr[candidates])]
        return float(tpr[selected]), float(thresholds[selected])

    def fpr_at_tpr(target_tpr: float) -> tuple[float, float]:
        candidates = np.flatnonzero(tpr >= target_tpr)
        selected = candidates[np.argmin(fpr[candidates])]
        return float(fpr[selected]), float(thresholds[selected])

    tpr_at_1_fpr, threshold_at_1_fpr = tpr_at_fpr(0.01)
    tpr_at_5_fpr, threshold_at_5_fpr = tpr_at_fpr(0.05)
    fpr_at_99_tpr, threshold_at_99_tpr = fpr_at_tpr(0.99)
    fpr_at_95_tpr, threshold_at_95_tpr = fpr_at_tpr(0.95)
    return {
        "roc_auc": float(roc_auc_score(labels_array, scores_array)),
        "tpr_at_1_fpr": tpr_at_1_fpr,
        "tpr_at_5_fpr": tpr_at_5_fpr,
        "fpr_at_99_tpr": fpr_at_99_tpr,
        "fpr_at_95_tpr": fpr_at_95_tpr,
        "threshold_at_1_fpr": threshold_at_1_fpr,
        "threshold_at_5_fpr": threshold_at_5_fpr,
        "threshold_at_99_tpr": threshold_at_99_tpr,
        "threshold_at_95_tpr": threshold_at_95_tpr,
    }


def binary_classification_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float]:
    """Metrics used for every clean or transformed evaluation condition."""
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    metrics = binary_operating_point_metrics(labels_array, scores_array)
    predictions = (scores_array >= threshold).astype(np.int64)
    metrics.update({
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
    })
    return metrics


class EpochMetricsWriter:
    """Append one machine-readable metrics record per epoch in JSONL and CSV."""

    def __init__(self, output_root: str | Path, run_name: str):
        self.run_dir = Path(output_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"

    def write(self, epoch: int, metrics: Mapping[str, float | int]) -> dict:
        record = {"epoch": epoch, **{key: float(value) for key, value in metrics.items()}}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if new_file:
                writer.writeheader()
            writer.writerow(record)
        return record


class TensorBoardMetricsWriter:
    """Write per-epoch scalar metrics to TensorBoard event files."""

    def __init__(self, output_root: str | Path, run_name: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("TensorBoard is not installed. Run the project setup script.") from error
        self.log_dir = Path(output_root) / run_name / "tensorboard"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def write(self, epoch: int, metrics: Mapping[str, float | int]) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(f"epoch/{name}", float(value), global_step=epoch)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class EpochReporter:
    """One end-of-epoch reporting path: console, file log, JSONL, CSV, TensorBoard."""

    def __init__(self, output_root: str | Path, run_name: str, tensorboard: bool = True):
        self.run_dir = Path(output_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = EpochMetricsWriter(output_root, run_name)
        self.tensorboard = TensorBoardMetricsWriter(output_root, run_name) if tensorboard else None
        self.logger = logging.getLogger(f"robust_aigc.{run_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            stream = logging.StreamHandler(); stream.setFormatter(formatter)
            file_handler = logging.FileHandler(self.run_dir / "training.log", encoding="utf-8"); file_handler.setFormatter(formatter)
            self.logger.addHandler(stream); self.logger.addHandler(file_handler)

    def report(self, epoch: int, metrics: Mapping[str, float | int]) -> dict:
        record = self.metrics.write(epoch, metrics)
        if self.tensorboard:
            self.tensorboard.write(epoch, metrics)
        rendered = " | ".join(f"{name}={value:.6f}" for name, value in record.items() if name != "epoch")
        self.logger.info("epoch=%d | %s", epoch, rendered)
        return record

    def close(self) -> None:
        if self.tensorboard:
            self.tensorboard.close()

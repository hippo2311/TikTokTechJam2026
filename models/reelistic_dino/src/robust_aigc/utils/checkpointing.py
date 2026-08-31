"""Resumable, atomic checkpoint handling for future training scripts."""
from __future__ import annotations

import os
import re
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch


Mode = Literal["max", "min"]


@dataclass
class CheckpointManager:
    """Keep a resumable `last.pt` and a validation-selected `best.pt` per run."""

    root: Path
    run_name: str
    monitor: str = "internal_val_accuracy"
    mode: Mode = "max"
    best_value: float | None = None
    tracked_metrics: dict[str, Mode] = field(default_factory=lambda: {
        "tpr_at_1_fpr": "max",
        "fpr_at_99_tpr": "min",
    })
    best_values: dict[str, float] = field(default_factory=dict)
    save_last: bool = True
    save_primary_best: bool = True
    include_optimizer_state: bool = True
    include_scheduler_state: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        if any(mode not in {"max", "min"} for mode in self.tracked_metrics.values()):
            raise ValueError("tracked metric modes must be 'max' or 'min'")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.run_dir / "checkpoint_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.best_value = state.get("best_value")
            self.best_values = {name: float(value) for name, value in state.get("best_values", {}).items()}

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_name

    @staticmethod
    def _is_better(value: float, best_value: float | None, mode: Mode) -> bool:
        if best_value is None:
            return True
        return value > best_value if mode == "max" else value < best_value

    @staticmethod
    def _safe_metric_name(metric: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)

    def _atomic_save(self, payload: dict[str, Any], destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def save_epoch(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        metrics: dict[str, float],
        training_args: dict[str, Any] | None = None,
        ema_state_dict: dict[str, Any] | None = None,
    ) -> dict[str, Path | bool]:
        """Save resumable state each epoch and best state only on improvement."""
        if self.monitor not in metrics:
            raise KeyError(f"Checkpoint monitor '{self.monitor}' missing from metrics: {sorted(metrics)}")
        monitor_value = float(metrics[self.monitor])
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer and self.include_optimizer_state else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler and self.include_scheduler_state else None,
            "metrics": metrics,
            "monitor": self.monitor,
            "monitor_value": monitor_value,
            "training_args": training_args or {},
            "ema_state_dict": ema_state_dict,
        }
        last_path = self.run_dir / "last.pt"
        if self.save_last:
            self._atomic_save(payload, last_path)
        is_best = self._is_better(monitor_value, self.best_value, self.mode)
        if is_best and self.save_primary_best:
            self.best_value = monitor_value
            payload["best_value"] = self.best_value
            self._atomic_save(payload, self.run_dir / "best.pt")
        updated_metrics: list[str] = []
        for metric, metric_mode in self.tracked_metrics.items():
            if metric not in metrics:
                continue
            value = float(metrics[metric])
            if self._is_better(value, self.best_values.get(metric), metric_mode):
                self.best_values[metric] = value
                metric_payload = {**payload, "best_metric": metric, "best_value": value}
                self._atomic_save(metric_payload, self.run_dir / f"best_{self._safe_metric_name(metric)}.pt")
                updated_metrics.append(metric)
        self._write_tracking_state()
        return {"last": last_path if self.save_last else False,
                "best": self.run_dir / "best.pt" if self.save_primary_best else False, "is_best": is_best,
                "updated_best_metrics": updated_metrics}

    def _write_tracking_state(self) -> Path:
        path = self.run_dir / "checkpoint_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"best_value": self.best_value, "best_values": self.best_values}, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path

    def write_metadata(self) -> Path:
        path = self.run_dir / "checkpoint_config.json"
        path.write_text(json.dumps(asdict(self), default=str, indent=2) + "\n", encoding="utf-8")
        return path

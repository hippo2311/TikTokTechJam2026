from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with a clear error for absent/malformed config files."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    return value


def load_toml(path: str | Path, *, validate_experiment: bool = True) -> dict[str, Any]:
    """Load TOML, optionally enforcing the full training-experiment schema."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")
    with config_path.open("rb") as handle:
        value = tomllib.load(handle)
    if validate_experiment:
        validate_experiment_config(value)
    return value


def validate_experiment_config(config: dict[str, Any]) -> None:
    required = {"run", "data", "model", "forensic_head", "loss", "optimizer", "scheduler", "training", "metrics", "logging", "curriculum"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Experiment configuration is missing sections: {', '.join(sorted(missing))}")
    data = config["data"]
    if data.get("allow_organizer_demo_for_training"):
        raise ValueError("organizer_demo must remain evaluation-only and cannot be enabled for training")
    if "organizer_demo" in data.get("train_splits", []):
        raise ValueError("organizer_demo must never be listed in train_splits")
    if config["model"].get("parameter_budget", 0) > 2_000_000_000:
        raise ValueError("Model parameter budget cannot exceed 2B")

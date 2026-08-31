from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import load_yaml


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    cache_root: Path
    checkpoint_root: Path
    output_root: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _as_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def resolve_paths(config_path: str | Path | None = None, create: bool = False) -> ProjectPaths:
    """Resolve storage paths: environment variables > local YAML > repo defaults."""
    root = _repo_root()
    candidate = Path(config_path) if config_path else root / "configs" / "paths.yaml"
    values = load_yaml(candidate) if candidate.exists() else {}
    defaults = {
        "data_root": root / "data",
        "cache_root": root / "data" / "cache",
        "checkpoint_root": root / "checkpoints",
        "output_root": root / "outputs",
    }
    resolved = {}
    for key, default in defaults.items():
        env_key = f"AIGC_{key.upper()}"
        resolved[key] = _as_path(os.environ.get(env_key, values.get(key, default)), root)
    paths = ProjectPaths(**resolved)
    if create:
        for path in paths.__dict__.values():
            path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_caches(paths: ProjectPaths) -> None:
    """Redirect model/dataset caches away from the default home-directory cache."""
    hf_cache = paths.cache_root / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
    os.environ.setdefault("TORCH_HOME", str(paths.cache_root / "torch"))

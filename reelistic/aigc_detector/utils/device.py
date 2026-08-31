"""Runtime-device selection shared by training, evaluation, and prediction."""

from __future__ import annotations

import torch


def default_device() -> str:
    """Prefer CUDA, then Apple MPS, while retaining a CPU fallback."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

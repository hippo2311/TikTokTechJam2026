"""AIGC image detection ensemble package.

The model import is intentionally lazy so data-audit utilities can run in a
small CPU environment without importing PyTorch and every model dependency.
"""

__all__ = ["AIGCDetectionEnsemble"]


def __getattr__(name: str):
    if name == "AIGCDetectionEnsemble":
        from .models.ensemble import AIGCDetectionEnsemble

        return AIGCDetectionEnsemble
    raise AttributeError(name)

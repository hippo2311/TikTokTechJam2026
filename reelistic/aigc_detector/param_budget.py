from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch.nn as nn

PARAM_BUDGET = 2_000_000_000


def count_params(model: Optional[nn.Module]) -> int:
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())


def count_unique_params(models: Iterable[nn.Module]) -> int:
    """Count shared Parameter objects once across a deployment graph."""
    unique = {}
    for model in models:
        for parameter in model.parameters():
            unique[id(parameter)] = parameter
    return sum(parameter.numel() for parameter in unique.values())


def shared_seed_ensemble_breakdown(
    models: Iterable[nn.Module], extra_models: Iterable[nn.Module] = ()
) -> Dict[str, int]:
    """Report the real deployment count for models sharing a frozen backbone."""
    models = list(models)
    extra_models = list(extra_models)
    if not models:
        raise ValueError("At least one seed model is required.")
    backbones = [model.semantic_branch.backbone for model in models]
    shared_backbone = count_unique_params(backbones)
    seed_specific_total = count_unique_params(models) - shared_backbone
    extra_total = count_unique_params(extra_models)
    unique_total = count_unique_params([*models, *extra_models])
    breakdown = {
        "shared_semantic_backbone": shared_backbone,
        "all_seed_specific_parameters": seed_specific_total,
        "shared_ensemble_parameters": extra_total,
        "total_unique_parameters": unique_total,
        "naive_unshared_parameters": (
            sum(count_params(model) for model in models) + extra_total
        ),
    }
    assert unique_total < PARAM_BUDGET, f"Exceeds 2B budget: {unique_total}"
    return breakdown


def print_branch_params(name: str, model: nn.Module) -> int:
    params = count_params(model)
    print(f"[params] {name}: {params:,}")
    return params


def enforce_budget(
    texture_model: nn.Module,
    freq_model: nn.Module,
    noise_model: nn.Module,
    semantic_model: nn.Module,
    fusion_model: Optional[nn.Module] = None,
) -> Dict[str, int]:
    breakdown = {
        "texture": count_params(texture_model),
        "frequency": count_params(freq_model),
        "noise": count_params(noise_model),
        "semantic": count_params(semantic_model),
        "fusion": count_params(fusion_model),
    }
    total = sum(breakdown.values())
    print("[params] breakdown:", breakdown, "TOTAL:", total)
    assert total < PARAM_BUDGET, f"Exceeds 2B budget: {total}"
    return breakdown

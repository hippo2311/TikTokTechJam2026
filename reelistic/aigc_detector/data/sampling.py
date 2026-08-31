"""Hierarchical source/class/family-balanced sampling utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence


def hierarchical_sample_weights(samples: Sequence[object]) -> tuple[list[float], dict]:
    """Give equal expected exposure to sources, classes, and families.

    Within each source, present classes equally. Within each source/class pair,
    present families equally. Images within a family share its probability.
    Empty combinations are ignored rather than fabricated.
    """
    if not samples:
        raise ValueError("Cannot balance an empty sample collection.")

    grouped: dict[str, dict[int, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for index, sample in enumerate(samples):
        source = str(getattr(sample, "source"))
        label = int(getattr(sample, "label"))
        family = str(getattr(sample, "family", "unknown")) or "unknown"
        grouped[source][label][family].append(index)

    weights = [0.0] * len(samples)
    source_probability = 1.0 / len(grouped)
    expected = Counter()
    for source, labels in grouped.items():
        class_probability = source_probability / len(labels)
        for label, families in labels.items():
            family_probability = class_probability / len(families)
            for family, indices in families.items():
                image_probability = family_probability / len(indices)
                for index in indices:
                    weights[index] = image_probability
                expected[f"source/{source}"] += family_probability
                expected[f"source_class/{source}/{label}"] += family_probability
                expected[f"source_class_family/{source}/{label}/{family}"] += family_probability

    total = sum(weights)
    weights = [weight / total for weight in weights]
    report = {
        "strategy": "source_class_family_balanced",
        "sample_count": len(samples),
        "source_count": len(grouped),
        "expected_fraction": dict(sorted(expected.items())),
    }
    return weights, report

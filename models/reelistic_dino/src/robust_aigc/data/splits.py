from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .registry import write_manifest

SEED = 42


def assign_deterministic_splits(records: list[dict], seed: int = SEED) -> list[dict]:
    """Assign 80/10/10 splits stratified by label and generator metadata."""
    groups: dict[tuple[int, str | None], list[dict]] = defaultdict(list)
    for record in records:
        groups[(int(record["label"]), record.get("generator"))].append(dict(record))
    output: list[dict] = []
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
        train_end, val_end = round(len(group) * 0.8), round(len(group) * 0.9)
        for index, record in enumerate(group):
            record["split"] = "train" if index < train_end else "internal_val" if index < val_end else "test"
            output.append(record)
    return output


def assign_train_internal_validation(records: list[dict], seed: int = SEED) -> list[dict]:
    """Make a 90/10 train/internal-validation split without creating a test set."""
    groups: dict[tuple[int, str | None], list[dict]] = defaultdict(list)
    for record in records:
        groups[(int(record["label"]), record.get("generator"))].append(dict(record))
    output: list[dict] = []
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
        train_end = round(len(group) * 0.9)
        for index, record in enumerate(group):
            record["split"] = "train" if index < train_end else "internal_val"
            output.append(record)
    return output


def preserve_directory_splits(records: list[dict], root: str | Path, seed: int = SEED) -> list[dict]:
    """Respect conventional train/validation/test folders when a dataset has them.

    Official test folders remain held out. Training-folder records get a fresh
    internal validation split for model selection. An unstructured dataset
    falls back to the standard deterministic 80/10/10 split.
    """
    root_path = Path(root).expanduser().resolve()
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        relative_parts = [part.lower() for part in Path(record["path"]).resolve().relative_to(root_path).parts]
        if any(part in {"test", "testing"} for part in relative_parts):
            buckets["test"].append(dict(record))
        elif any(part in {"val", "valid", "validation", "dev"} for part in relative_parts):
            buckets["internal_val"].append(dict(record))
        elif any(part in {"train", "training"} for part in relative_parts):
            buckets["train_source"].append(dict(record))
        else:
            buckets["unstructured"].append(dict(record))
    if not (buckets["test"] or buckets["internal_val"] or buckets["train_source"]):
        return assign_deterministic_splits(records, seed=seed)
    output = assign_train_internal_validation(buckets["train_source"], seed=seed)
    for split in ("internal_val", "test"):
        for record in buckets[split]:
            record["split"] = split
            output.append(record)
    # Unknown-layout records can safely supplement training but do not enter
    # an official held-out test set.
    output.extend(assign_train_internal_validation(buckets["unstructured"], seed=seed))
    return output


def persist_split_manifests(records: list[dict], manifest_dir: str | Path, prefix: str) -> dict[str, Path]:
    directory = Path(manifest_dir)
    result = {}
    for split in ("train", "internal_val", "test"):
        result[split] = write_manifest([record for record in records if record["split"] == split], directory / f"{prefix}_{split}.csv")
    return result


def validate_split_isolation(records: list[dict]) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        path = record["path"]
        split = record["split"]
        if path in by_split[split]:
            raise ValueError(f"Duplicate path within {split}: {path}")
        by_split[split].add(path)
    for left, right in (("train", "internal_val"), ("train", "test"), ("internal_val", "test")):
        overlap = by_split[left] & by_split[right]
        if overlap:
            raise ValueError(f"Leakage: {left} and {right} share {len(overlap)} path(s), e.g. {next(iter(overlap))}")
    if by_split["train"] & by_split["organizer_demo"]:
        raise ValueError("Leakage: organizer_demo paths must never appear in training.")

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_NAMES = {"real", "authentic", "natural", "0"}
AI_NAMES = {"fake", "ai", "aigc", "generated", "synthetic", "diffusion_based", "gan_based", "other_based", "1"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    identifier: str


DATASETS = {
    "sid": DatasetSpec("sid", "huggingface", "saberzl/SID_Set"),
    "cifake": DatasetSpec("cifake", "kaggle", "birdy654/cifake-real-and-ai-generated-synthetic-images"),
    "wildfake": DatasetSpec("wildfake", "modelscope", "hy2628982280/WildFake"),
}


def sid_binary_label(raw_label: int) -> int:
    """Map SID's three source categories to the detector's binary target.

    The raw value is used only while preparing images and is not emitted to the
    final manifest, so it cannot accidentally become a model feature.
    """
    if raw_label == 0:
        return 0
    if raw_label in {1, 2}:
        return 1
    raise ValueError(f"Unexpected SID label: {raw_label}; expected 0, 1, or 2")


def _label_for_path(path: Path) -> tuple[int, str | None] | None:
    parts = [part.lower() for part in path.parts]
    for index, part in enumerate(parts):
        if part in REAL_NAMES:
            return 0, None
        if part in AI_NAMES:
            # The nearest folder above a generic class label is useful later as
            # a generator hint, but remains None if the layout has no such hint.
            # WildFake stores an AI family followed by a generator, e.g.
            # ``Diffusion_based/ADM``.  Preserve that useful provenance
            # without exposing it to the model as an input feature.
            if part in {"diffusion_based", "gan_based", "other_based"} and index + 1 < len(parts):
                hint = parts[index + 1]
            else:
                hint = parts[index - 1] if index > 0 and parts[index - 1] not in {"train", "val", "test"} else None
            return 1, hint
    return None


def build_records_from_directory(root: str | Path, source_dataset: str, split: str = "train") -> list[dict]:
    """Scan common real/fake folder layouts into the canonical record schema."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset not found at {root_path}. Set AIGC_DATA_ROOT or run the download script.")
    records: list[dict] = []
    skipped = 0
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        labelled = _label_for_path(path.relative_to(root_path))
        if labelled is None:
            skipped += 1
            continue
        label, generator = labelled
        records.append({"path": str(path), "label": label, "source_dataset": source_dataset,
                        "generator": generator, "split": split})
    if not records:
        raise ValueError(
            f"No labelled images found below {root_path}. Expected directories named real/authentic and fake/ai/generated."
        )
    if skipped:
        print(f"[WARNING] Skipped {skipped} image(s) whose path did not reveal a real/AI class.")
    return records


def write_manifest(records: Iterable[dict], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "source_dataset", "generator", "split"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
    return destination


def load_manifest(path: str | Path) -> list[dict]:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    required = {"path", "label", "source_dataset", "generator", "split"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"Manifest must include columns: {', '.join(sorted(required))}")
    for record in records:
        record["label"] = int(record["label"])
        record["generator"] = record["generator"] or None
    return records

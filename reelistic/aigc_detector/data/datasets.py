from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, List, Sequence

from PIL import Image
try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Allow manifest/audit tests in a lightweight CPU environment.
    class Dataset:  # type: ignore[no-redef]
        pass


ALLOWED_DATASETS = {"sid_set", "cifake", "wildfake"}
RESERVED_WILDFAKE_TOKENS = ("demo", "validation", "val2017", "dall", "advanced")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    source: str
    family: str = "unknown"


class ImageClassificationDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as source_image:
            image = source_image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "path": sample.path,
            "source": sample.source,
            "family": sample.family,
        }


def build_local_splits(
    dataset_root: str,
    train_transform,
    eval_transform,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
):
    train, val, _, _ = build_train_val_cal_test_splits(
        dataset_root=dataset_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        val_fraction=val_fraction,
        calibration_fraction=0.0,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    return train, val


def build_train_val_cal_test_splits(
    dataset_root: str,
    train_transform,
    eval_transform,
    val_fraction: float = 0.05,
    calibration_fraction: float = 0.05,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_calibration_samples: int | None = None,
    max_test_samples: int | None = None,
    manifest_dir: str | None = None,
):
    """Build leakage-free, stratified train/validation/calibration/test splits.

    A directory named ``test`` is never used for model selection or calibration.
    If no explicit validation directory exists, validation and calibration samples
    are carved out of the explicit training directory per class.
    """
    if val_fraction < 0 or calibration_fraction < 0 or val_fraction + calibration_fraction >= 1:
        raise ValueError("Split fractions must be non-negative and sum to less than one.")

    root = Path(dataset_root)
    rng = random.Random(seed)
    if manifest_dir is not None:
        split_samples = load_manifest_splits(root, Path(manifest_dir))
        train_samples = split_samples["train"]
        val_samples = split_samples["validation"]
        calibration_samples = split_samples["calibration"]
        test_samples = split_samples["test"]
        train_samples = _limit_samples_hierarchical(train_samples, max_train_samples, rng)
        val_samples = _limit_samples_hierarchical(val_samples, max_val_samples, rng)
        calibration_samples = _limit_samples_hierarchical(
            calibration_samples, max_calibration_samples, rng
        )
        test_samples = _limit_samples_stratified(test_samples, max_test_samples, rng)
        return (
            ImageClassificationDataset(train_samples, transform=train_transform),
            ImageClassificationDataset(val_samples, transform=eval_transform),
            ImageClassificationDataset(calibration_samples, transform=eval_transform),
            ImageClassificationDataset(test_samples, transform=eval_transform),
        )

    dataset_units = _find_dataset_units(root)

    if dataset_units:
        train_samples: List[Sample] = []
        val_samples: List[Sample] = []
        calibration_samples: List[Sample] = []
        test_samples: List[Sample] = []
        for unit in dataset_units:
            unit_train, unit_val, unit_calibration, unit_test = _split_dataset_unit(
                unit,
                val_fraction=val_fraction,
                calibration_fraction=calibration_fraction,
                rng=rng,
            )
            train_samples.extend(unit_train)
            val_samples.extend(unit_val)
            calibration_samples.extend(unit_calibration)
            test_samples.extend(unit_test)
    else:
        samples = discover_samples(root)
        if not samples:
            raise FileNotFoundError(f"No labeled images found under {root}")
        train_samples, val_samples, calibration_samples = _stratified_train_val_cal_split(
            samples,
            val_fraction=val_fraction,
            calibration_fraction=calibration_fraction,
            rng=rng,
        )

    train_samples = _limit_samples_stratified(train_samples, max_train_samples, rng)
    val_samples = _limit_samples_stratified(val_samples, max_val_samples, rng)
    calibration_samples = _limit_samples_stratified(
        calibration_samples, max_calibration_samples, rng
    )
    test_samples = _limit_samples_stratified(test_samples, max_test_samples, rng)
    return (
        ImageClassificationDataset(train_samples, transform=train_transform),
        ImageClassificationDataset(val_samples, transform=eval_transform),
        ImageClassificationDataset(calibration_samples, transform=eval_transform),
        ImageClassificationDataset(test_samples, transform=eval_transform),
    )


def _find_dataset_units(root: Path) -> List[Path]:
    """Return independently split dataset roots below a combined data directory.

    A combined root may contain one dataset with an explicit validation folder
    and another with only train/test folders. Treating those folders globally
    leaks the latter dataset's entire training set into model fitting. Each
    top-level dataset therefore gets its own train/validation/calibration split.
    """
    split_names = {"train", "val", "valid", "validation", "test"}
    direct_directory_names = {
        child.name.lower() for child in root.iterdir() if child.is_dir()
    }
    if direct_directory_names & split_names:
        return [root]

    units = []
    for child in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
        if any(
            path.is_dir() and path.name.lower() in split_names
            for path in child.rglob("*")
        ):
            units.append(child)
    return units


def _split_dataset_unit(
    root: Path,
    val_fraction: float,
    calibration_fraction: float,
    rng: random.Random,
) -> tuple[List[Sample], List[Sample], List[Sample], List[Sample]]:
    train_samples = _with_source(_collect_named_splits(root, {"train"}), root.name)
    val_samples = _with_source(
        _collect_named_splits(root, {"val", "valid", "validation"}), root.name
    )
    test_samples = _with_source(_collect_named_splits(root, {"test"}), root.name)

    if val_samples:
        val_samples, calibration_samples = _stratified_partition(
            val_samples,
            first_fraction=0.5 if calibration_fraction > 0 else 1.0,
            rng=rng,
        )
    else:
        train_samples, val_samples, calibration_samples = _stratified_train_val_cal_split(
            train_samples,
            val_fraction=val_fraction,
            calibration_fraction=calibration_fraction,
            rng=rng,
        )
    return train_samples, val_samples, calibration_samples, test_samples


def _with_source(samples: Sequence[Sample], source: str) -> List[Sample]:
    return [
        Sample(path=sample.path, label=sample.label, source=source, family=sample.family)
        for sample in samples
    ]


def load_manifest_splits(root: Path, manifest_dir: Path) -> dict[str, List[Sample]]:
    """Load immutable split manifests without rediscovering dataset folders."""
    output: dict[str, List[Sample]] = {}
    for split in ("train", "validation", "calibration", "test"):
        manifest_path = manifest_dir / f"{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Required manifest is missing: {manifest_path}")
        samples: List[Sample] = []
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != split:
                    raise ValueError(
                        f"Manifest split mismatch at {manifest_path}:{line_number}"
                    )
                relative = Path(record["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"Unsafe manifest path at {manifest_path}:{line_number}: {relative}"
                    )
                image_path = root / relative
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest image is missing at {manifest_path}:{line_number}: {image_path}"
                    )
                samples.append(
                    Sample(
                        path=str(image_path),
                        label=int(record["label"]),
                        source=str(record["dataset_source"]),
                        family=str(record.get("family", "unknown")),
                    )
                )
        output[split] = samples
    return output


def _collect_named_splits(root: Path, names: set[str]) -> List[Sample]:
    directories = [
        path for path in root.rglob("*") if path.is_dir() and path.name.lower() in names
    ]
    samples: List[Sample] = []
    for directory in directories:
        samples.extend(_collect_generic_image_folder(directory))
    return samples


def _stratified_train_val_cal_split(
    samples: Sequence[Sample],
    val_fraction: float,
    calibration_fraction: float,
    rng: random.Random,
) -> tuple[List[Sample], List[Sample], List[Sample]]:
    grouped = _group_by_label(samples)
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    calibration_samples: List[Sample] = []
    for label_samples in grouped.values():
        label_samples = list(label_samples)
        rng.shuffle(label_samples)
        val_count = int(len(label_samples) * val_fraction)
        cal_count = int(len(label_samples) * calibration_fraction)
        val_samples.extend(label_samples[:val_count])
        calibration_samples.extend(label_samples[val_count : val_count + cal_count])
        train_samples.extend(label_samples[val_count + cal_count :])
    return train_samples, val_samples, calibration_samples


def _stratified_partition(
    samples: Sequence[Sample], first_fraction: float, rng: random.Random
) -> tuple[List[Sample], List[Sample]]:
    first: List[Sample] = []
    second: List[Sample] = []
    for label_samples in _group_by_label(samples).values():
        label_samples = list(label_samples)
        rng.shuffle(label_samples)
        split_index = int(len(label_samples) * first_fraction)
        first.extend(label_samples[:split_index])
        second.extend(label_samples[split_index:])
    return first, second


def _group_by_label(samples: Sequence[Sample]) -> dict[int, List[Sample]]:
    grouped: dict[int, List[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)
    return grouped


def discover_explicit_splits(root: Path) -> tuple[List[Sample], List[Sample]]:
    train_dirs = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == "train"]
    val_dirs = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() in {"val", "valid", "validation"}
    ]

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    for directory in train_dirs:
        train_samples.extend(_collect_generic_image_folder(directory))
    for directory in val_dirs:
        val_samples.extend(_collect_generic_image_folder(directory))
    return train_samples, val_samples


def discover_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []

    # Enforce the requested public benchmark rules when those datasets are present.
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name.lower()
        if dataset_name in ALLOWED_DATASETS:
            samples.extend(_collect_named_dataset(dataset_dir))
        elif _looks_like_image_folder(dataset_dir):
            samples.extend(_collect_generic_image_folder(dataset_dir))

    if not samples and _looks_like_image_folder(root):
        samples.extend(_collect_generic_image_folder(root))

    return samples


def _collect_named_dataset(dataset_dir: Path) -> List[Sample]:
    lowered = dataset_dir.name.lower()
    if lowered == "wildfake" and any(token in str(dataset_dir).lower() for token in RESERVED_WILDFAKE_TOKENS):
        return []

    samples: List[Sample] = []
    for class_dir in dataset_dir.rglob("*"):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        if class_name not in {"real", "fake"}:
            continue
        if lowered == "wildfake" and any(token in str(class_dir).lower() for token in RESERVED_WILDFAKE_TOKENS):
            continue
        label = 1 if class_name == "fake" else 0
        samples.extend(_files_to_samples(class_dir.glob("*"), label, dataset_dir.name))
    return samples


def _collect_generic_image_folder(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for class_dir in root.rglob("*"):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        if class_name not in {"real", "fake"}:
            continue
        label = 1 if class_name == "fake" else 0
        samples.extend(_files_to_samples(class_dir.glob("*"), label, root.name))
    return samples


def _files_to_samples(paths: Iterable[Path], label: int, source: str) -> List[Sample]:
    return [
        Sample(path=str(path), label=label, source=source, family=source)
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _looks_like_image_folder(path: Path) -> bool:
    names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    return {"real", "fake"}.issubset(names) or "train" in names or "test" in names


def _limit_samples(samples: Sequence[Sample], max_samples: int | None, rng: random.Random) -> List[Sample]:
    samples = list(samples)
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng.shuffle(samples)
    return samples[:max_samples]


def _limit_samples_stratified(
    samples: Sequence[Sample], max_samples: int | None, rng: random.Random
) -> List[Sample]:
    samples = list(samples)
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    grouped = _group_by_label(samples)
    labels = sorted(grouped)
    base, remainder = divmod(max_samples, len(labels))
    limited: List[Sample] = []
    for index, label in enumerate(labels):
        label_samples = list(grouped[label])
        rng.shuffle(label_samples)
        count = base + (1 if index < remainder else 0)
        limited.extend(label_samples[:count])
    rng.shuffle(limited)
    return limited


def _limit_samples_hierarchical(
    samples: Sequence[Sample], max_samples: int | None, rng: random.Random
) -> List[Sample]:
    """Bound a manifest smoke run without dropping small sources or families."""
    samples = list(samples)
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    groups: dict[tuple[str, int, str], List[Sample]] = {}
    for sample in samples:
        groups.setdefault((sample.source, sample.label, sample.family), []).append(sample)
    keys = sorted(groups)
    if max_samples < len(keys):
        raise ValueError(
            f"max_samples={max_samples} cannot represent all {len(keys)} "
            "source/class/family groups. Raise the limit."
        )
    base, remainder = divmod(max_samples, len(keys))
    limited: List[Sample] = []
    for index, key in enumerate(keys):
        group = list(groups[key])
        rng.shuffle(group)
        limited.extend(group[: base + (1 if index < remainder else 0)])
    # If small groups could not fill their quota, top up deterministically from
    # the remaining pool while retaining the guaranteed group coverage.
    if len(limited) < max_samples:
        selected = {sample.path for sample in limited}
        remainder_pool = [sample for sample in samples if sample.path not in selected]
        rng.shuffle(remainder_pool)
        limited.extend(remainder_pool[: max_samples - len(limited)])
    rng.shuffle(limited)
    return limited

"""Build audited, deterministic manifests for birdy654, SID, and WildFake."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.external_data import stable_score
from cluster.wildfake_archives import ARCHIVES_BY_KEY


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
SPLIT_PRIORITY = {"test": 0, "calibration": 1, "validation": 2, "train": 3}


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    label: int
    dataset_source: str
    family: str
    split: str
    content_hash: str = ""
    bytes: int = 0
    archive_origin: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--no-verify-images", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Image verification/hash workers. Workers recycle to bound PIL memory.",
    )
    parser.add_argument(
        "--allow-decode-failures",
        action="store_true",
        help="Exclude unreadable files instead of failing the manifest build.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def enrich_record_task(
    task: tuple[str, ManifestRecord, bool],
) -> tuple[ManifestRecord | None, str | None]:
    root_string, record, verify_images = task
    path = Path(root_string) / record.path
    try:
        if verify_images:
            verify_image(path)
        return (
            replace(
                record,
                content_hash=sha256_file(path),
                bytes=path.stat().st_size,
            ),
            None,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def class_label(parts: tuple[str, ...]) -> int | None:
    names = {part.upper() for part in parts}
    if "REAL" in names and "FAKE" in names:
        raise ValueError(f"Ambiguous REAL/FAKE path: {'/'.join(parts)}")
    if "FAKE" in names:
        return 1
    if "REAL" in names:
        return 0
    return None


def normalize_split(name: str) -> str:
    lowered = name.lower()
    if lowered in {"val", "valid", "validation"}:
        return "explicit_validation"
    return lowered


def infer_standard_family(source: str, path: Path) -> str:
    if source == "external_pilot":
        return path.stem.split("_", 1)[0] or "unknown"
    return source


def scan_standard_sources(root: Path) -> list[ManifestRecord]:
    records = []
    for source_root in sorted(path for path in root.iterdir() if path.is_dir()):
        if source_root.name.lower() == "wildfake":
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(source_root)
            split_name = next(
                (part for part in relative.parts if part.lower() in SPLIT_NAMES), None
            )
            if split_name is None:
                continue
            label = class_label(relative.parts)
            if label is None:
                continue
            records.append(
                ManifestRecord(
                    path=str(path.relative_to(root)),
                    label=label,
                    dataset_source=source_root.name,
                    family=infer_standard_family(source_root.name, path),
                    split=normalize_split(split_name),
                )
            )
    return records


def infer_wildfake_family(archive_key: str, path: Path, archive_root: Path) -> str:
    archive = ARCHIVES_BY_KEY[archive_key]
    directories = [
        part
        for part in path.relative_to(archive_root).parts[:-1]
        if part.lower()
        not in {
            "images",
            "diffusion_based",
            "real",
            "fake",
            archive.key.lower(),
            archive.family.lower(),
        }
    ]
    suffix = directories[0] if directories else ""
    return f"{archive.family}:{suffix}" if suffix else archive.family


def scan_wildfake(root: Path) -> list[ManifestRecord]:
    raw_root = root / "WildFake" / "raw"
    if not raw_root.exists():
        return []
    records = []
    for archive_root in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        archive_key = archive_root.name.lower()
        if archive_key not in ARCHIVES_BY_KEY:
            raise ValueError(
                f"Unapproved WildFake archive directory: {archive_root}. "
                f"Approved keys are {sorted(ARCHIVES_BY_KEY)}"
            )
        marker = archive_root / ".complete.json"
        if not marker.exists():
            raise FileNotFoundError(f"WildFake archive is incomplete: {archive_root}")
        archive = ARCHIVES_BY_KEY[archive_key]
        marker_data = json.loads(marker.read_text())
        if marker_data.get("remote_path") != archive.remote_path:
            raise ValueError(f"Archive marker mismatch: {marker}")
        for path in sorted(archive_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            records.append(
                ManifestRecord(
                    path=str(path.relative_to(root)),
                    label=archive.label,
                    dataset_source="WildFake",
                    family=infer_wildfake_family(archive_key, path, archive_root),
                    split="family_pending",
                    archive_origin=archive.remote_path,
                )
            )
    return records


def deterministic_partition(
    records: list[ManifestRecord],
    val_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> list[ManifestRecord]:
    ordered = sorted(records, key=lambda item: stable_score(seed, item.path))
    val_count = round(len(ordered) * val_fraction)
    cal_count = round(len(ordered) * calibration_fraction)
    if len(ordered) >= 20:
        val_count = max(val_count, 1)
        cal_count = max(cal_count, 1)
    output = []
    for index, record in enumerate(ordered):
        if index < val_count:
            split = "validation"
        elif index < val_count + cal_count:
            split = "calibration"
        else:
            split = "train"
        output.append(replace(record, split=split))
    return output


def split_standard_records(
    records: list[ManifestRecord],
    val_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> list[ManifestRecord]:
    output = [record for record in records if record.split == "test"]
    grouped: dict[tuple[str, int], list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.split != "test":
            grouped[(record.dataset_source, record.label)].append(record)
    for (source, _), group in sorted(grouped.items()):
        explicit = [record for record in group if record.split == "explicit_validation"]
        training = [record for record in group if record.split == "train"]
        if explicit:
            ordered = sorted(explicit, key=lambda item: stable_score(seed, item.path))
            boundary = len(ordered) // 2 if calibration_fraction > 0 else len(ordered)
            output.extend(replace(record, split="validation") for record in ordered[:boundary])
            output.extend(replace(record, split="calibration") for record in ordered[boundary:])
            output.extend(training)
        else:
            output.extend(
                deterministic_partition(
                    training,
                    val_fraction=val_fraction,
                    calibration_fraction=calibration_fraction,
                    seed=stable_score(seed, source),
                )
            )
    return output


def family_split_names(
    families: list[str], val_fraction: float, calibration_fraction: float, seed: int
) -> dict[str, str]:
    families = sorted(set(families), key=lambda name: stable_score(seed, name))
    if len(families) < 3:
        raise ValueError(
            "WildFake family-aware splitting requires at least three families per class."
        )
    val_count = max(1, round(len(families) * val_fraction))
    cal_count = max(1, round(len(families) * calibration_fraction))
    while val_count + cal_count >= len(families):
        if val_count >= cal_count and val_count > 1:
            val_count -= 1
        elif cal_count > 1:
            cal_count -= 1
        else:
            raise ValueError("Not enough WildFake families to preserve a training family.")
    assignments = {}
    for index, family in enumerate(families):
        if index < val_count:
            assignments[family] = "validation"
        elif index < val_count + cal_count:
            assignments[family] = "calibration"
        else:
            assignments[family] = "train"
    return assignments


def split_wildfake_records(
    records: list[ManifestRecord],
    val_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> list[ManifestRecord]:
    assignments: dict[tuple[int, str], str] = {}
    for label in (0, 1):
        families = [record.family for record in records if record.label == label]
        label_assignments = family_split_names(
            families,
            val_fraction=val_fraction,
            calibration_fraction=calibration_fraction,
            seed=stable_score(seed, "wildfake", label),
        )
        assignments.update(
            {(label, family): split for family, split in label_assignments.items()}
        )
    return [
        replace(record, split=assignments[(record.label, record.family)])
        for record in records
    ]


def enrich_and_deduplicate(
    root: Path,
    records: Iterable[ManifestRecord],
    verify_images: bool,
    workers: int = 1,
) -> tuple[list[ManifestRecord], list[dict], Counter[str]]:
    enriched = []
    failures: Counter[str] = Counter()
    if workers < 1:
        raise ValueError("workers must be at least 1")

    tasks = ((str(root), record, verify_images) for record in records)
    if workers == 1:
        results = map(enrich_record_task, tasks)
        pool = None
    else:
        pool = mp.Pool(processes=workers, maxtasksperchild=500)
        results = pool.imap(enrich_record_task, tasks, chunksize=32)

    try:
        for index, (record, failure) in enumerate(results, start=1):
            if record is not None:
                enriched.append(record)
            elif failure is not None:
                failures[failure] += 1
            if index % 10000 == 0:
                print(f"[manifest] scanned={index}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    canonical = {}
    duplicates = []
    unique = []
    for record in sorted(
        enriched,
        key=lambda item: (SPLIT_PRIORITY[item.split], item.path),
    ):
        previous = canonical.get(record.content_hash)
        if previous is None:
            canonical[record.content_hash] = record
            unique.append(record)
        else:
            duplicates.append(
                {
                    "excluded_path": record.path,
                    "excluded_split": record.split,
                    "canonical_path": previous.path,
                    "canonical_split": previous.split,
                    "content_hash": record.content_hash,
                }
            )
    return unique, duplicates, failures


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def build_manifests(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    val_fraction: float,
    calibration_fraction: float,
    verify_images: bool = True,
    allow_decode_failures: bool = False,
    workers: int = 1,
) -> dict:
    if val_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("Validation and calibration fractions must both be positive.")
    if val_fraction + calibration_fraction >= 1:
        raise ValueError("Validation and calibration fractions must sum to less than one.")

    standard = split_standard_records(
        scan_standard_sources(data_dir), val_fraction, calibration_fraction, seed
    )
    wildfake_pending = scan_wildfake(data_dir)
    wildfake = (
        split_wildfake_records(
            wildfake_pending, val_fraction, calibration_fraction, seed
        )
        if wildfake_pending
        else []
    )
    unique, duplicates, failures = enrich_and_deduplicate(
        data_dir, standard + wildfake, verify_images, workers=workers
    )
    if failures and not allow_decode_failures:
        examples = list(failures.items())[:5]
        raise ValueError(
            f"Manifest build rejected {sum(failures.values())} unreadable files; "
            f"examples={examples}. Use --allow-decode-failures only after review."
        )

    counts: Counter[str] = Counter()
    families: dict[str, set[str]] = defaultdict(set)
    for record in unique:
        counts[f"{record.split}/{record.dataset_source}/{record.label}"] += 1
        families[f"{record.split}/{record.dataset_source}/{record.label}"].add(record.family)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "calibration", "test"):
        rows = [asdict(record) for record in unique if record.split == split]
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "duplicates.jsonl", duplicates)

    audit = {
        "status": "ok" if not failures else "decode_failures_excluded",
        "data_dir": str(data_dir.resolve()),
        "seed": seed,
        "val_fraction": val_fraction,
        "calibration_fraction": calibration_fraction,
        "record_count": len(unique),
        "duplicate_count": len(duplicates),
        "decode_failure_count": sum(failures.values()),
        "decode_failures": dict(failures),
        "counts": dict(sorted(counts.items())),
        "family_counts": {key: len(value) for key, value in sorted(families.items())},
    }
    temporary = output_dir / "audit.json.tmp"
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True))
    temporary.replace(output_dir / "audit.json")
    print(f"[done] {json.dumps(audit, sort_keys=True)}", flush=True)
    return audit


def main() -> None:
    args = parse_args()
    build_manifests(
        data_dir=Path(args.data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        seed=args.seed,
        val_fraction=args.val_fraction,
        calibration_fraction=args.calibration_fraction,
        verify_images=not args.no_verify_images,
        allow_decode_failures=args.allow_decode_failures,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

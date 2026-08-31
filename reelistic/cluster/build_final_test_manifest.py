"""Build an audited manifest for the isolated DALL-E/COCO final holdout."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.wildfake_archives import FINAL_TEST_ARCHIVES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_EXPECTED_COUNTS = {
    "dalle_advanced": 8843,
    "coco_val2017": 4998,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--training-manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-verify-images", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def training_hashes(manifest_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "validation", "calibration", "test"):
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Required training manifest is missing: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                content_hash = str(record.get("content_hash", ""))
                if len(content_hash) != 64:
                    raise ValueError(
                        f"Missing SHA-256 at {path}:{line_number}; rebuild manifests first."
                    )
                hashes.add(content_hash)
    return hashes


def selected_images(archive_key: str, archive_root: Path) -> list[Path]:
    paths = [
        path
        for path in sorted(archive_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if archive_key == "dalle_advanced":
        advanced = [
            path
            for path in paths
            if "advanced" in "/".join(path.relative_to(archive_root).parts).lower()
        ]
        if not advanced:
            raise ValueError(f"DALL-E Advanced directory not found below {archive_root}")
        return advanced
    if archive_key == "coco_val2017":
        validation = [
            path
            for path in paths
            if "val2017" in {
                part.lower() for part in path.relative_to(archive_root).parts
            }
        ]
        if not validation:
            raise ValueError(f"COCO val2017 directory not found below {archive_root}")
        return validation
    return paths


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def build_final_test_manifest(
    data_dir: Path,
    training_manifest_dir: Path,
    output_dir: Path,
    *,
    verify_images: bool = True,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    expected_counts = expected_counts or DEFAULT_EXPECTED_COUNTS
    existing_hashes = training_hashes(training_manifest_dir)
    final_hashes: dict[str, dict[str, object]] = {}
    internal_duplicates = []
    rows = []
    unique_rows = []
    counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()

    raw_root = data_dir / "WildFake" / "final_test" / "raw"
    unknown = (
        {path.name for path in raw_root.iterdir() if path.is_dir()}
        - {archive.key for archive in FINAL_TEST_ARCHIVES}
        if raw_root.is_dir()
        else set()
    )
    if unknown:
        raise ValueError(f"Unknown final-test archive directories: {sorted(unknown)}")

    for archive in FINAL_TEST_ARCHIVES:
        archive_root = raw_root / archive.key
        marker = archive_root / ".complete.json"
        if not marker.is_file():
            raise FileNotFoundError(f"Final-test archive is incomplete: {archive_root}")
        marker_data = json.loads(marker.read_text())
        if marker_data.get("remote_path") != archive.remote_path:
            raise ValueError(f"Archive marker mismatch: {marker}")
        if marker_data.get("archive_set") not in {None, "final-test"}:
            raise ValueError(f"Archive marker is not final-test data: {marker}")

        images = selected_images(archive.key, archive_root)
        expected = expected_counts.get(archive.key)
        if expected is not None and len(images) != expected:
            raise ValueError(
                f"Expected {expected} images for {archive.key}, found {len(images)}."
            )
        for index, path in enumerate(images, start=1):
            if verify_images:
                with Image.open(path) as image:
                    image.verify()
            content_hash = sha256_file(path)
            if content_hash in existing_hashes:
                raise ValueError(f"Final-test image overlaps a model-development split: {path}")
            previous = final_hashes.get(content_hash)
            if previous is not None:
                if int(previous["label"]) != archive.label:
                    raise ValueError(
                        "Conflicting REAL/FAKE labels for duplicate final-test content: "
                        f"{path} and {previous['path']}"
                    )
                internal_duplicates.append(
                    {
                        "duplicate_path": str(path.relative_to(data_dir)),
                        "canonical_path": str(previous["path"]),
                        "content_hash": content_hash,
                        "label": archive.label,
                        "family": archive.family,
                    }
                )
            else:
                final_hashes[content_hash] = {
                    "path": str(path.relative_to(data_dir)),
                    "label": archive.label,
                }
            row = {
                "path": str(path.relative_to(data_dir)),
                "label": archive.label,
                "dataset_source": "WildFake_external_final",
                "family": archive.family,
                "split": "final_test",
                "content_hash": content_hash,
                "bytes": path.stat().st_size,
                "archive_origin": archive.remote_path,
            }
            rows.append(row)
            if previous is None:
                unique_rows.append(row)
                unique_counts[f"label/{archive.label}"] += 1
                unique_counts[f"family/{archive.family}"] += 1
            counts[f"label/{archive.label}"] += 1
            counts[f"family/{archive.family}"] += 1
            if index % 1000 == 0:
                print(f"[final-manifest] archive={archive.key} scanned={index}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "final_test.jsonl", rows)
    write_jsonl(output_dir / "final_test_unique.jsonl", unique_rows)
    write_jsonl(output_dir / "internal_duplicates.jsonl", internal_duplicates)
    audit = {
        "status": "ok_with_internal_duplicates" if internal_duplicates else "ok",
        "record_count": len(rows),
        "unique_content_count": len(final_hashes),
        "counts": dict(sorted(counts.items())),
        "unique_counts": dict(sorted(unique_counts.items())),
        "training_hash_count": len(existing_hashes),
        "cross_split_duplicate_count": 0,
        "within_final_duplicate_count": len(internal_duplicates),
        "expected_counts": expected_counts,
    }
    temporary = output_dir / "audit.json.tmp"
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True))
    temporary.replace(output_dir / "audit.json")
    print(f"[done] {json.dumps(audit, sort_keys=True)}", flush=True)
    return audit


def main() -> None:
    args = parse_args()
    build_final_test_manifest(
        data_dir=Path(args.data_dir).resolve(),
        training_manifest_dir=Path(args.training_manifest_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        verify_images=not args.no_verify_images,
    )


if __name__ == "__main__":
    main()

"""Download, safely extract, and persist one approved WildFake archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.wildfake_archives import (
    APPROVED_ARCHIVES,
    FINAL_TEST_ARCHIVES,
    archive_for_index,
    archive_for_key,
    final_test_archive_for_index,
    final_test_archive_for_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=False)
    selection.add_argument("--archive-key")
    selection.add_argument("--archive-index", type=int)
    parser.add_argument(
        "--archive-set",
        choices=("training", "final-test"),
        default="training",
        help="Keep final-test archives physically separate from training data.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--scratch-multiplier", type=float, default=3.0)
    parser.add_argument("--persistent-multiplier", type=float, default=2.0)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def require_free_space(path: Path, required_gb: float, purpose: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    print(
        f"[storage] purpose={purpose} path={path} free_gb={free_gb:.2f} "
        f"required_gb={required_gb:.2f}",
        flush=True,
    )
    if free_gb < required_gb:
        raise OSError(
            f"Insufficient {purpose} space at {path}: {free_gb:.2f} GB free, "
            f"{required_gb:.2f} GB required."
        )


def safe_extract(archive_path: Path, destination: Path) -> tuple[int, int]:
    destination = destination.resolve()
    file_count = 0
    uncompressed_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename!r}")
            if member.is_dir():
                continue
            file_count += 1
            uncompressed_bytes += member.file_size
        archive.extractall(destination)
    return file_count, uncompressed_bytes


def locate_download(root: Path, remote_path: str) -> Path:
    expected_name = Path(remote_path).name
    matches = [path for path in root.rglob(expected_name) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one downloaded {expected_name!r} below {root}, "
            f"found {len(matches)}."
        )
    return matches[0]


def download_command(remote_path: str, download_root: Path) -> list[str]:
    """Request one exact repository file instead of relying on glob matching."""
    return [
        "modelscope",
        "download",
        "--dataset",
        "hy2628982280/WildFake",
        remote_path,
        "--local_dir",
        str(download_root),
    ]


def main() -> None:
    args = parse_args()
    inventory = (
        APPROVED_ARCHIVES
        if args.archive_set == "training"
        else FINAL_TEST_ARCHIVES
    )
    if args.list:
        for index, archive in enumerate(inventory):
            print(
                f"{index:02d} {archive.key:10} label={archive.label} "
                f"compressed_gb={archive.compressed_gb:.3f} {archive.remote_path}"
            )
        return
    if args.archive_key is None and args.archive_index is None:
        raise ValueError("Choose --archive-key or --archive-index, or use --list.")
    if args.archive_set == "training":
        archive = (
            archive_for_key(args.archive_key)
            if args.archive_key is not None
            else archive_for_index(args.archive_index)
        )
        relative_destination = Path("raw") / archive.key
    else:
        archive = (
            final_test_archive_for_key(args.archive_key)
            if args.archive_key is not None
            else final_test_archive_for_index(args.archive_index)
        )
        relative_destination = Path("final_test") / "raw" / archive.key
    if args.scratch_multiplier < 1 or args.persistent_multiplier < 1:
        raise ValueError("Storage multipliers must be at least 1.")

    output_root = Path(args.output_dir).resolve()
    destination = output_root / relative_destination
    completion_marker = destination / ".complete.json"
    if completion_marker.exists():
        report = json.loads(completion_marker.read_text())
        if report.get("remote_path") != archive.remote_path:
            raise ValueError(f"Completion marker at {completion_marker} does not match request.")
        print(f"[skip] archive={archive.key} already complete at {destination}")
        return
    if destination.exists():
        raise FileExistsError(
            f"Partial destination exists without a completion marker: {destination}. "
            "Inspect it before retrying."
        )

    scratch_root = Path(args.scratch_dir).resolve() / archive.key
    download_root = scratch_root / "download"
    extracted_root = scratch_root / "extracted"
    required_scratch = archive.compressed_gb * args.scratch_multiplier
    required_persistent = archive.compressed_gb * args.persistent_multiplier
    require_free_space(scratch_root.parent, required_scratch, "scratch")
    require_free_space(output_root, required_persistent, "persistent")
    download_root.mkdir(parents=True, exist_ok=False)
    extracted_root.mkdir(parents=True, exist_ok=False)

    command = download_command(archive.remote_path, download_root)
    print(f"[download] archive={archive.key} remote={archive.remote_path}", flush=True)
    subprocess.run(command, check=True)
    downloaded = locate_download(download_root, archive.remote_path)
    file_count, uncompressed_bytes = safe_extract(downloaded, extracted_root)
    if file_count == 0:
        raise ValueError(f"Archive {downloaded} contained no files.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(extracted_root, destination)
    report = {
        "archive_set": args.archive_set,
        "archive_key": archive.key,
        "remote_path": archive.remote_path,
        "label": archive.label,
        "family": archive.family,
        "compressed_bytes": downloaded.stat().st_size,
        "uncompressed_bytes": uncompressed_bytes,
        "file_count": file_count,
    }
    completion_marker.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[done] {json.dumps(report, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()

"""Create a small immutable manifest for the selected development package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile


SOURCE_DIRECTORIES = ("aigc_detector", "cluster", "slurm", "tests")
ROOT_SOURCE_FILES = (
    "README.md",
    "RECOMMENDATIONS.md",
    "SOC_CLUSTER_TRAINING_GUIDE.md",
    "PROJECT_WORKFLOW.svg",
    "requirements.txt",
    ".gitignore",
)
SOURCE_SUFFIXES = {".py", ".sh", ".sbatch", ".md", ".svg", ".txt", ".json"}
EXCLUDED_PARTS = {"__pycache__", ".git", ".cache", ".envs", "logs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--uncalibrated-checkpoint", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--parameter-count", type=int, required=True)
    parser.add_argument("--selection-name", required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Report must use NAME=PATH syntax: {value!r}")
    name, raw_path = value.split("=", 1)
    if not name or "/" in name or ".." in name:
        raise ValueError(f"Unsafe report name: {name!r}")
    return name, Path(raw_path)


def source_files(project_dir: Path) -> list[Path]:
    files = []
    for relative in ROOT_SOURCE_FILES:
        path = project_dir / relative
        if path.is_file():
            files.append(path)
    for directory in SOURCE_DIRECTORIES:
        root = project_dir / directory
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(project_dir)
            if (
                path.is_file()
                and path.suffix.lower() in SOURCE_SUFFIXES
                and not EXCLUDED_PARTS.intersection(relative.parts)
            ):
                files.append(path)
    return sorted(set(files), key=lambda path: str(path.relative_to(project_dir)))


def source_snapshot(project_dir: Path, output: Path) -> dict[str, object]:
    files = source_files(project_dir)
    records = []
    combined = hashlib.sha256()
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            relative = path.relative_to(project_dir)
            digest = sha256_file(path)
            records.append(
                {
                    "path": str(relative),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
            combined.update(str(relative).encode("utf-8"))
            combined.update(b"\0")
            combined.update(digest.encode("ascii"))
            combined.update(b"\n")
            archive.add(path, arcname=str(relative), recursive=False)
    return {
        "file_count": len(records),
        "combined_sha256": combined.hexdigest(),
        "archive": file_record(output),
        "files": records,
    }


def git_state(project_dir: Path) -> dict[str, object] | None:
    if not (project_dir / ".git").exists():
        return None

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(project_dir), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(run("diff", "--binary").encode()).hexdigest(),
    }


def runtime_state() -> dict[str, object]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
    except ImportError:
        result["torch"] = None
        return result
    result.update(
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
    return result


def checkpoint_args(path: Path) -> dict[str, object]:
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return dict(checkpoint.get("args", {}))


def build_package(args: argparse.Namespace) -> Path:
    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen package: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent)
    )
    reports_dir = temporary / "reports"
    reports_dir.mkdir()

    calibrated = Path(args.checkpoint)
    uncalibrated = Path(args.uncalibrated_checkpoint)
    manifest_dir = Path(args.manifest_dir)
    required_manifests = (
        "audit.json",
        "train.jsonl",
        "validation.jsonl",
        "calibration.jsonl",
    )
    manifests = {
        name: file_record(manifest_dir / name) for name in required_manifests
    }
    optional_test = manifest_dir / "test.jsonl"
    if optional_test.is_file():
        manifests[optional_test.name] = file_record(optional_test)

    reports = {}
    for raw_report in args.report:
        name, path = parse_named_path(raw_report)
        record = file_record(path)
        destination = reports_dir / f"{name}{path.suffix}"
        shutil.copy2(path, destination)
        record["frozen_copy"] = str(destination.relative_to(temporary))
        reports[name] = record

    snapshot = source_snapshot(project_dir, temporary / "source_snapshot.tar.gz")
    payload = {
        "protocol": "frozen_development_package_before_external_holdout",
        "selection_name": args.selection_name,
        "global_probability_threshold": args.threshold,
        "parameter_count": args.parameter_count,
        "parameter_limit": 2_000_000_000,
        "calibrated_checkpoint": file_record(calibrated),
        "uncalibrated_checkpoint": file_record(uncalibrated),
        "checkpoint_args": checkpoint_args(calibrated),
        "development_manifests": manifests,
        "development_reports": reports,
        "source_snapshot": snapshot,
        "git": git_state(project_dir),
        "runtime": runtime_state(),
        "external_holdout_used": False,
        "known_limitations": [
            "WildFake TPR at 1% FPR remains low under cross-domain shift.",
            "Semantic masking still improves severe downsampling and blur on VQDM.",
            "The disagreement-gate gain was seed-dependent across confirmation runs.",
            "The final COCO/DALL-E holdout couples source with class.",
        ],
    }
    manifest_path = temporary / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    verification = {
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "source_archive_sha256": sha256_file(temporary / "source_snapshot.tar.gz"),
    }
    (temporary / "VERIFY.json").write_text(json.dumps(verification, indent=2))
    os.replace(temporary, output_dir)
    return output_dir


def main() -> None:
    output = build_package(parse_args())
    print(f"[freeze] package={output}", flush=True)
    print(f"[freeze] manifest={output / 'freeze_manifest.json'}", flush=True)
    print("[done] development package frozen; external holdout untouched", flush=True)


if __name__ == "__main__":
    main()

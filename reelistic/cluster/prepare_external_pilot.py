"""Stream and stage bounded, deterministic SID-Set/WildFake pilot subsets.

The script writes only selected images plus JSONL provenance manifests. It never
downloads either complete dataset and applies the WildFake demo-set denylist
before a record can enter a train/validation directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import heapq
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

from PIL import Image

from cluster.external_data import (
    SID_LABEL_NAMES,
    safe_component,
    sid_binary_label,
    stable_score,
    wildfake_exclusion_reason,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sid-train-per-type", type=int, default=1000)
    parser.add_argument("--sid-val-per-type", type=int, default=200)
    parser.add_argument("--wildfake-fake", type=int, default=2000)
    parser.add_argument("--wildfake-real", type=int, default=1000)
    parser.add_argument("--wildfake-max-per-family", type=int, default=400)
    parser.add_argument("--wildfake-max-scan", type=int, default=100000)
    parser.add_argument("--skip-sid", action="store_true")
    parser.add_argument("--skip-wildfake", action="store_true")
    return parser.parse_args()


def image_extension(blob: bytes) -> str:
    with Image.open(io.BytesIO(blob)) as image:
        fmt = (image.format or "png").lower()
        image.verify()
    return ".jpg" if fmt in {"jpeg", "jpg"} else f".{fmt}"


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_sid(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import Image as HFImage
    from datasets import load_dataset

    manifest: list[dict[str, Any]] = []
    token = os.environ.get("HF_TOKEN") or None
    requested = {"train": args.sid_train_per_type, "validation": args.sid_val_per_type}
    for remote_split, per_type in requested.items():
        if per_type <= 0:
            continue
        local_split = "train" if remote_split == "train" else "val"
        stream = load_dataset(
            "saberzl/SID_Set",
            split=remote_split,
            streaming=True,
            token=token,
        ).cast_column("image", HFImage(decode=False))
        counts: Counter[int] = Counter()
        for row in stream:
            label = int(row["label"])
            if label not in SID_LABEL_NAMES or counts[label] >= per_type:
                continue
            blob = row["image"]["bytes"]
            if not blob:
                continue
            sample_id = safe_component(row.get("img_id"), f"sid_{counts[label]}")
            binary_label = sid_binary_label(label)
            class_name = "FAKE" if binary_label else "REAL"
            destination = (
                Path(args.output_dir) / local_split / "sid_set" / class_name
                / f"{SID_LABEL_NAMES[label]}_{sample_id}{image_extension(blob)}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(blob)
            manifest.append(
                {
                    "dataset": "sid_set",
                    "remote_split": remote_split,
                    "local_split": local_split,
                    "sample_id": str(row.get("img_id")),
                    "original_label": label,
                    "subtype": SID_LABEL_NAMES[label],
                    "binary_label": binary_label,
                    "path": str(destination),
                }
            )
            counts[label] += 1
            if all(counts[value] >= per_type for value in SID_LABEL_NAMES):
                break
        print(f"[sid] split={remote_split} counts={dict(counts)}", flush=True)
    return manifest


def select_wildfake_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    from modelscope.msdatasets import MsDataset

    dataset = MsDataset.load(
        "hy2628982280/WildFake",
        subset_name="default",
        split="train",
        use_streaming=True,
    )
    targets = {0: args.wildfake_real, 1: args.wildfake_fake}
    heaps: dict[int, list[tuple[int, int, dict[str, Any]]]] = {0: [], 1: []}
    family_counts: dict[tuple[int, str], int] = defaultdict(int)
    excluded: Counter[str] = Counter()
    for index, row in enumerate(dataset):
        if index >= args.wildfake_max_scan:
            break
        reason = wildfake_exclusion_reason(row)
        if reason:
            excluded[reason] += 1
            continue
        label = 1 if int(row.get("IsFake", 0)) else 0
        if targets[label] <= 0:
            continue
        family = safe_component(
            f"{row.get('Generator', 'unknown')}__{row.get('Architecture', 'unknown')}",
            "unknown",
        )
        if family_counts[(label, family)] >= args.wildfake_max_per_family:
            continue
        score = stable_score(args.seed, row.get("Image_path"), row.get("Num"))
        heap = heaps[label]
        item = (-score, index, dict(row))
        if len(heap) < targets[label]:
            heapq.heappush(heap, item)
            family_counts[(label, family)] += 1
        elif score < -heap[0][0]:
            removed = heapq.heapreplace(heap, item)[2]
            removed_family = safe_component(
                f"{removed.get('Generator', 'unknown')}__{removed.get('Architecture', 'unknown')}",
                "unknown",
            )
            family_counts[(label, removed_family)] -= 1
            family_counts[(label, family)] += 1
    selected = [item[2] for label in sorted(heaps) for item in sorted(heaps[label])]
    print(
        f"[wildfake] scanned<= {args.wildfake_max_scan} selected_real={len(heaps[0])} "
        f"selected_fake={len(heaps[1])} exclusions={dict(excluded)}",
        flush=True,
    )
    return selected, excluded


def prepare_wildfake(args: argparse.Namespace) -> list[dict[str, Any]]:
    from modelscope.hub.file_download import dataset_file_download

    rows, excluded = select_wildfake_rows(args)
    manifest: list[dict[str, Any]] = []
    cache_dir = Path(args.output_dir).parent / ".wildfake_download_cache"
    for index, row in enumerate(rows, start=1):
        remote_path = str(row["Image_path"]).removeprefix("./")
        downloaded = Path(
            dataset_file_download(
                dataset_id="hy2628982280/WildFake",
                file_path=remote_path,
                cache_dir=str(cache_dir),
            )
        )
        label = 1 if int(row.get("IsFake", 0)) else 0
        class_name = "FAKE" if label else "REAL"
        family = safe_component(
            f"{row.get('Generator', 'unknown')}__{row.get('Architecture', 'unknown')}",
            "unknown",
        )
        destination = (
            Path(args.output_dir) / "train" / "wildfake" / class_name / family
            / safe_component(Path(remote_path).name)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
        record = {key: row.get(key) for key in (
            "Generator", "Architecture", "Weight", "Category", "IsAdvanced", "IsFake", "Num"
        )}
        record.update(
            {
                "dataset": "wildfake",
                "remote_path": remote_path,
                "local_split": "train",
                "binary_label": label,
                "family": family,
                "path": str(destination),
            }
        )
        manifest.append(record)
        if index % 100 == 0:
            print(f"[wildfake] downloaded={index}/{len(rows)}", flush=True)
    exclusion_rows = [{"reason": reason, "count": count} for reason, count in excluded.items()]
    write_manifest(Path(args.manifest_dir) / "wildfake_exclusions.jsonl", exclusion_rows)
    return manifest


def main() -> None:
    args = parse_args()
    if args.wildfake_max_per_family < 1 or args.wildfake_max_scan < 1:
        raise ValueError("WildFake family and scan limits must be positive.")
    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_sid:
        sid_rows = prepare_sid(args)
        write_manifest(manifest_dir / "sid_set.jsonl", sid_rows)
    if not args.skip_wildfake:
        wildfake_rows = prepare_wildfake(args)
        write_manifest(manifest_dir / "wildfake.jsonl", wildfake_rows)
    print(f"[done] data={output_dir} manifests={manifest_dir}", flush=True)


if __name__ == "__main__":
    main()
    # Some streaming-provider builds abort while tearing down background C
    # threads after all output is safely closed. Exit directly after flushing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from cluster.build_dataset_manifests import scan_wildfake
from cluster.build_final_test_manifest import build_final_test_manifest
from cluster.wildfake_archives import FINAL_TEST_ARCHIVES


class FinalTestManifestTests(unittest.TestCase):
    def test_holdout_is_counted_hashed_and_kept_separate(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "Dataset"
            manifest_dir = data_dir / "manifests"
            manifest_dir.mkdir(parents=True)
            for split in ("train", "validation", "calibration", "test"):
                (manifest_dir / f"{split}.jsonl").write_text(
                    json.dumps({"content_hash": str(split).encode().hex().ljust(64, "0")})
                    + "\n"
                )

            expected_counts = {"dalle_advanced": 2, "coco_val2017": 1}
            for archive in FINAL_TEST_ARCHIVES:
                archive_root = (
                    data_dir / "WildFake" / "final_test" / "raw" / archive.key
                )
                image_root = (
                    archive_root / "Advanced"
                    if archive.key == "dalle_advanced"
                    else archive_root / "coco" / "coco2017" / "val2017"
                )
                image_root.mkdir(parents=True)
                for index in range(expected_counts[archive.key]):
                    Image.new(
                        "RGB",
                        (8, 8),
                        color=(
                            archive.label * 100,
                            0 if archive.key == "dalle_advanced" else index * 50,
                            20,
                        ),
                    ).save(image_root / f"{index}.png")
                if archive.key == "coco_val2017":
                    train_root = archive_root / "coco" / "coco2017" / "train2017"
                    train_root.mkdir(parents=True)
                    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(
                        train_root / "must_not_enter_final_manifest.png"
                    )
                (archive_root / ".complete.json").write_text(
                    json.dumps(
                        {
                            "archive_set": "final-test",
                            "remote_path": archive.remote_path,
                        }
                    )
                )

            output_dir = data_dir / "WildFake" / "final_test" / "manifests"
            audit = build_final_test_manifest(
                data_dir,
                manifest_dir,
                output_dir,
                expected_counts=expected_counts,
            )
            self.assertEqual(audit["status"], "ok_with_internal_duplicates")
            self.assertEqual(audit["record_count"], 3)
            self.assertEqual(audit["unique_content_count"], 2)
            self.assertEqual(audit["within_final_duplicate_count"], 1)
            rows = [
                json.loads(line)
                for line in (output_dir / "final_test.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["split"] for row in rows}, {"final_test"})
            self.assertEqual({row["label"] for row in rows}, {0, 1})
            unique_rows = [
                json.loads(line)
                for line in (output_dir / "final_test_unique.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(unique_rows), 2)
            self.assertEqual(len({row["content_hash"] for row in unique_rows}), 2)
            self.assertTrue(
                all("WildFake/final_test/raw" in row["path"] for row in rows)
            )
            duplicates = [
                json.loads(line)
                for line in (output_dir / "internal_duplicates.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(duplicates[0]["label"], 1)
            self.assertEqual(scan_wildfake(data_dir), [])


if __name__ == "__main__":
    unittest.main()

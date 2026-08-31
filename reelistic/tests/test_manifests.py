from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from aigc_detector.data.datasets import load_manifest_splits
from cluster.build_dataset_manifests import build_manifests, scan_wildfake
from cluster.wildfake_archives import ARCHIVES_BY_KEY


def save_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value // 2, 255 - value)).save(path)


class ManifestTests(unittest.TestCase):
    def test_build_is_deduplicated_family_aware_and_loadable(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Dataset"
            output = root / "manifests"
            value = 1
            for split, count in (("train", 22), ("test", 2)):
                for class_name in ("REAL", "FAKE"):
                    for index in range(count):
                        save_image(root / "birdy654" / split / class_name / f"{index}.png", value)
                        value += 1

            # An exact train/test duplicate must remain only in the protected test split.
            source = root / "birdy654" / "test" / "REAL" / "0.png"
            duplicate = root / "birdy654" / "train" / "REAL" / "duplicate.png"
            duplicate.write_bytes(source.read_bytes())

            for archive_key in ("afhq", "ffhq", "imagenet", "adm", "ddim", "ddpm"):
                archive = ARCHIVES_BY_KEY[archive_key]
                archive_root = root / "WildFake" / "raw" / archive_key
                save_image(archive_root / "Images" / f"{archive_key}.png", value)
                value += 1
                (archive_root / ".complete.json").write_text(
                    json.dumps({"remote_path": archive.remote_path})
                )

            audit = build_manifests(root, output, 42, 0.05, 0.05)
            self.assertEqual(audit["status"], "ok")
            self.assertGreaterEqual(audit["duplicate_count"], 1)

            manifests = load_manifest_splits(root, output)
            hashes_by_split = {}
            for split, samples in manifests.items():
                hashes_by_split[split] = {
                    hashlib.sha256(Path(sample.path).read_bytes()).hexdigest()
                    for sample in samples
                }
            split_names = list(hashes_by_split)
            for index, first in enumerate(split_names):
                for second in split_names[index + 1 :]:
                    self.assertTrue(hashes_by_split[first].isdisjoint(hashes_by_split[second]))

            family_splits = {}
            for split, samples in manifests.items():
                for sample in samples:
                    if sample.source == "WildFake":
                        previous = family_splits.setdefault((sample.label, sample.family), split)
                        self.assertEqual(previous, split)
            self.assertEqual(len(family_splits), 6)

    def test_unapproved_wildfake_directory_is_rejected(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Dataset"
            save_image(root / "WildFake" / "raw" / "dalle" / "x.png", 1)
            with self.assertRaises(ValueError):
                build_manifests(root, root / "manifests", 42, 0.05, 0.05)

    def test_approved_archive_may_contain_internal_coco_named_subset(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Dataset"
            archive_root = root / "WildFake" / "raw" / "gan"
            image_path = (
                archive_root
                / "GAN_based"
                / "Advanced"
                / "DF-GAN"
                / "samples"
                / "coco"
                / "000.png"
            )
            save_image(image_path, 1)
            archive = ARCHIVES_BY_KEY["gan"]
            (archive_root / ".complete.json").write_text(
                json.dumps({"remote_path": archive.remote_path})
            )

            records = scan_wildfake(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].archive_origin, archive.remote_path)

    def test_parallel_enrichment_matches_single_worker(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Dataset"
            for split in ("train", "test"):
                for label_name in ("REAL", "FAKE"):
                    for index in range(2):
                        save_image(
                            root / "birdy654" / split / label_name / f"{index}.png",
                            index + (20 if label_name == "FAKE" else 1),
                        )

            single = build_manifests(
                root, root / "single", 42, 0.2, 0.2, workers=1
            )
            parallel = build_manifests(
                root, root / "parallel", 42, 0.2, 0.2, workers=2
            )

            self.assertEqual(single["counts"], parallel["counts"])
            self.assertEqual(single["duplicate_count"], parallel["duplicate_count"])


if __name__ == "__main__":
    unittest.main()

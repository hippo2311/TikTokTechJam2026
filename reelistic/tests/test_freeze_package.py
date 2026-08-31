from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cluster.freeze_development_package import build_package, sha256_file


class FreezePackageTests(unittest.TestCase):
    def test_freeze_records_hashes_and_refuses_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aigc_detector").mkdir()
            (root / "aigc_detector" / "model.py").write_text("VALUE = 1\n")
            manifests = root / "manifests"
            manifests.mkdir()
            for name in (
                "audit.json",
                "train.jsonl",
                "validation.jsonl",
                "calibration.jsonl",
            ):
                (manifests / name).write_text(name)
            calibrated = root / "calibrated.pt"
            uncalibrated = root / "best.pt"
            calibrated.write_bytes(b"calibrated")
            uncalibrated.write_bytes(b"uncalibrated")
            report = root / "report.json"
            report.write_text("{}")
            output = root / "frozen"
            args = Namespace(
                project_dir=str(root),
                checkpoint=str(calibrated),
                uncalibrated_checkpoint=str(uncalibrated),
                manifest_dir=str(manifests),
                output_dir=str(output),
                report=[f"diagnosis={report}"],
                threshold=0.5,
                parameter_count=10,
                selection_name="test",
            )
            with patch(
                "cluster.freeze_development_package.checkpoint_args",
                return_value={"seed": 1},
            ), patch(
                "cluster.freeze_development_package.runtime_state",
                return_value={"python": "test"},
            ):
                result = build_package(args)
                payload = json.loads((result / "freeze_manifest.json").read_text())
                self.assertEqual(
                    payload["calibrated_checkpoint"]["sha256"],
                    sha256_file(calibrated),
                )
                self.assertFalse(payload["external_holdout_used"])
                self.assertTrue((result / "source_snapshot.tar.gz").is_file())
                with self.assertRaises(FileExistsError):
                    build_package(args)


if __name__ == "__main__":
    unittest.main()

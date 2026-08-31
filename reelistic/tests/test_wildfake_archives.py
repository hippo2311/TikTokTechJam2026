from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from cluster.download_wildfake_archive import download_command, safe_extract
from cluster.wildfake_archives import (
    APPROVED_ARCHIVES,
    EXCLUDED_REMOTE_PATHS,
    FINAL_TEST_ARCHIVES,
    archive_for_index,
    archive_for_key,
    final_test_archive_for_index,
    final_test_archive_for_key,
)


class WildFakeArchiveTests(unittest.TestCase):
    def test_download_uses_exact_repository_path(self):
        command = download_command("Images/Real/Coco.zip", Path("/tmp/download"))
        self.assertIn("Images/Real/Coco.zip", command)
        self.assertNotIn("--include", command)

    def test_inventory_is_allowlist_only(self):
        paths = {archive.remote_path for archive in APPROVED_ARCHIVES}
        self.assertTrue(paths.isdisjoint(EXCLUDED_REMOTE_PATHS))
        self.assertEqual(len(paths), len(APPROVED_ARCHIVES))
        self.assertEqual(archive_for_index(0), APPROVED_ARCHIVES[0])
        self.assertEqual(archive_for_key("ADM").key, "adm")
        with self.assertRaises(KeyError):
            archive_for_key("dalle")

    def test_final_test_inventory_is_separate_and_explicit(self):
        training_paths = {archive.remote_path for archive in APPROVED_ARCHIVES}
        final_paths = {archive.remote_path for archive in FINAL_TEST_ARCHIVES}
        self.assertEqual(final_paths, EXCLUDED_REMOTE_PATHS)
        self.assertTrue(training_paths.isdisjoint(final_paths))
        self.assertEqual(final_test_archive_for_index(0).label, 1)
        self.assertEqual(final_test_archive_for_key("COCO_VAL2017").label, 0)
        self.assertEqual(
            final_test_archive_for_key("COCO_VAL2017").remote_path,
            "Images/Real/coco.zip",
        )
        with self.assertRaises(KeyError):
            final_test_archive_for_key("adm")

    def test_zip_path_traversal_is_rejected(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "blocked")
            with self.assertRaises(ValueError):
                safe_extract(archive, root / "extract")
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()

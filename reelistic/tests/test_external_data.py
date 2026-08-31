import unittest

from cluster.external_data import (
    safe_component,
    sid_binary_label,
    stable_score,
    wildfake_exclusion_reason,
)


class ExternalDataSafetyTests(unittest.TestCase):
    def test_sid_binary_mapping(self):
        self.assertEqual(sid_binary_label(0), 0)
        self.assertEqual(sid_binary_label(1), 1)
        self.assertEqual(sid_binary_label(2), 1)
        with self.assertRaises(ValueError):
            sid_binary_label(3)

    def test_wildfake_demo_real_is_excluded(self):
        row = {"IsFake": 0, "Category": "COCO val2017", "Image_path": "real/a.jpg"}
        self.assertEqual(wildfake_exclusion_reason(row), "reserved_demo_coco_val2017")

    def test_wildfake_demo_fake_is_excluded(self):
        row = {
            "IsFake": 1,
            "IsAdvanced": 1,
            "Generator": "DALL-E",
            "Image_path": "fake/a.png",
        }
        self.assertEqual(wildfake_exclusion_reason(row), "reserved_demo_dalle_advanced")

    def test_non_demo_record_is_retained(self):
        row = {"IsFake": 1, "IsAdvanced": 0, "Architecture": "ADM"}
        self.assertIsNone(wildfake_exclusion_reason(row))

    def test_selection_helpers_are_deterministic_and_path_safe(self):
        self.assertEqual(stable_score(42, "a"), stable_score(42, "a"))
        self.assertNotEqual(stable_score(42, "a"), stable_score(42, "b"))
        self.assertEqual(safe_component("../DALL E/x"), "DALL_E_x")


if __name__ == "__main__":
    unittest.main()

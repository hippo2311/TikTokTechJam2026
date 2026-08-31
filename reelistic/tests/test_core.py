from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch

from aigc_detector.data.datasets import build_train_val_cal_test_splits
from aigc_detector.models.ensemble import AIGCDetectionEnsemble
from aigc_detector.param_budget import PARAM_BUDGET, count_params
from aigc_detector.robustness import select_samples


class DataSplitTests(unittest.TestCase):
    def test_test_directory_is_held_out_and_limits_are_balanced(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split, count in (("train", 20), ("test", 6)):
                for class_name in ("REAL", "FAKE"):
                    class_directory = root / split / class_name
                    class_directory.mkdir(parents=True)
                    for index in range(count):
                        Image.new("RGB", (8, 8)).save(class_directory / f"{index}.png")

            train, val, calibration, test = build_train_val_cal_test_splits(
                str(root),
                train_transform=None,
                eval_transform=None,
                val_fraction=0.2,
                calibration_fraction=0.1,
                seed=7,
                max_train_samples=10,
                max_test_samples=4,
            )

            self.assertEqual(len(train), 10)
            self.assertEqual(len(val), 8)
            self.assertEqual(len(calibration), 4)
            self.assertEqual(len(test), 4)
            self.assertEqual([sample.label for sample in train.samples].count(0), 5)
            self.assertTrue(all("/test/" in sample.path for sample in test.samples))
            training_paths = {sample.path for sample in train.samples + val.samples + calibration.samples}
            self.assertTrue(training_paths.isdisjoint(sample.path for sample in test.samples))

    def test_combined_datasets_are_split_independently_without_overlap(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for dataset_name in ("birdy654", "sid"):
                for split, count in (("train", 20), ("test", 4)):
                    for class_name in ("REAL", "FAKE"):
                        class_directory = root / dataset_name / split / class_name
                        class_directory.mkdir(parents=True)
                        for index in range(count):
                            Image.new("RGB", (8, 8)).save(class_directory / f"{index}.png")

            # SID has an explicit validation split while birdy654 does not.
            for class_name in ("REAL", "FAKE"):
                class_directory = root / "sid" / "val" / class_name
                class_directory.mkdir(parents=True)
                for index in range(6):
                    Image.new("RGB", (8, 8)).save(class_directory / f"{index}.png")

            train, val, calibration, test = build_train_val_cal_test_splits(
                str(root),
                train_transform=None,
                eval_transform=None,
                val_fraction=0.2,
                calibration_fraction=0.1,
                seed=7,
            )

            train_paths = {sample.path for sample in train.samples}
            held_out_paths = {sample.path for sample in val.samples + calibration.samples}
            self.assertTrue(train_paths.isdisjoint(held_out_paths))
            self.assertTrue(any(sample.source == "birdy654" for sample in val.samples))
            self.assertTrue(any(sample.source == "sid" for sample in val.samples))
            self.assertEqual(len(test), 16)

    def test_robustness_source_filter_is_balanced_and_deterministic(self):
        from aigc_detector.data.datasets import Sample

        samples = [
            Sample(path=f"birdy-{label}-{index}", label=label, source="birdy654")
            for label in (0, 1)
            for index in range(10)
        ] + [Sample(path="sid-0", label=0, source="external_pilot")]
        first = select_samples(samples, "birdy654", max_samples=8, seed=42)
        second = select_samples(samples, "birdy654", max_samples=8, seed=42)
        self.assertEqual([sample.path for sample in first], [sample.path for sample in second])
        self.assertEqual([sample.label for sample in first].count(0), 4)
        self.assertEqual([sample.label for sample in first].count(1), 4)


class ModelTests(unittest.TestCase):
    def test_lite_model_forward_and_parameter_budget(self):
        model = AIGCDetectionEnsemble(
            semantic_pretrained=False,
            texture_pretrained=False,
            image_size=128,
        ).eval()
        self.assertLess(count_params(model), PARAM_BUDGET)
        with torch.inference_mode():
            outputs = model(torch.rand(2, 3, 128, 128))
        self.assertEqual(tuple(outputs.logits.shape), (2, 2))
        self.assertEqual(tuple(model.predict_proba(torch.rand(2, 3, 128, 128)).shape), (2,))


if __name__ == "__main__":
    unittest.main()

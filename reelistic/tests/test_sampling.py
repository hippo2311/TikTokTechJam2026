from __future__ import annotations

from collections import defaultdict
import random
import unittest

from aigc_detector.data.datasets import Sample, _limit_samples_hierarchical
from aigc_detector.data.sampling import hierarchical_sample_weights


class HierarchicalSamplingTests(unittest.TestCase):
    def test_sources_classes_and_families_receive_equal_expected_exposure(self):
        samples = []
        for index in range(20):
            samples.append(Sample(f"bird-real-{index}", 0, "birdy654", "birdy654"))
        for index in range(5):
            samples.append(Sample(f"bird-fake-{index}", 1, "birdy654", "birdy654"))
        for family, count in (("ADM", 40), ("GAN", 2)):
            for index in range(count):
                samples.append(Sample(f"wild-{family}-{index}", 1, "WildFake", family))
        for family in ("afhq", "ffhq"):
            samples.append(Sample(f"wild-{family}", 0, "WildFake", family))

        weights, report = hierarchical_sample_weights(samples)
        totals = defaultdict(float)
        family_totals = defaultdict(float)
        for sample, weight in zip(samples, weights):
            totals[(sample.source, sample.label)] += weight
            family_totals[(sample.source, sample.label, sample.family)] += weight

        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(totals[("birdy654", 0)], 0.25)
        self.assertAlmostEqual(totals[("birdy654", 1)], 0.25)
        self.assertAlmostEqual(totals[("WildFake", 0)], 0.25)
        self.assertAlmostEqual(totals[("WildFake", 1)], 0.25)
        self.assertAlmostEqual(family_totals[("WildFake", 1, "ADM")], 0.125)
        self.assertAlmostEqual(family_totals[("WildFake", 1, "GAN")], 0.125)
        self.assertEqual(report["source_count"], 2)

    def test_empty_collection_is_rejected(self):
        with self.assertRaises(ValueError):
            hierarchical_sample_weights([])

    def test_hierarchical_limit_preserves_every_source_class_family(self):
        samples = []
        for source, label, family, count in (
            ("birdy654", 0, "birdy654", 20),
            ("birdy654", 1, "birdy654", 20),
            ("WildFake", 0, "afhq", 20),
            ("WildFake", 1, "ADM", 20),
            ("external_pilot", 0, "real", 2),
            ("external_pilot", 1, "full", 2),
        ):
            for index in range(count):
                samples.append(Sample(f"{source}-{label}-{family}-{index}", label, source, family))

        limited = _limit_samples_hierarchical(samples, 18, random.Random(42))
        observed = {(sample.source, sample.label, sample.family) for sample in limited}

        self.assertEqual(len(limited), 18)
        self.assertEqual(len(observed), 6)


if __name__ == "__main__":
    unittest.main()

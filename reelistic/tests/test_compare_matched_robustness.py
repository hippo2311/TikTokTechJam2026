import unittest

from cluster.compare_matched_robustness import compare, validate_matching


def report(source, clean_auc, corrupt_auc, semantic_delta=0.01):
    return {
        "source": source,
        "seed": 42,
        "sample_count": 100,
        "sample_fingerprint": f"{source}-matched-fingerprint",
        "conditions": {
            condition: {
                "fusion": {
                    "roc_auc": clean_auc if condition == "clean" else corrupt_auc
                },
                "semantic": {
                    "roc_auc": (
                        clean_auc if condition == "clean" else corrupt_auc
                    )
                    - semantic_delta
                },
            }
            for condition in [
                "clean",
                "blur_2.0",
                "downsample_25",
                "jpeg_30",
                "noise_0.10",
            ]
        },
    }


class MatchedRobustnessComparisonTests(unittest.TestCase):
    def test_weakest_source_ranks_before_higher_mean(self):
        sources = ["birdy654", "external_pilot", "WildFake"]
        reports = {
            "seed42": {source: report(source, 0.90, 0.80) for source in sources},
            "seed43": {
                "birdy654": report("birdy654", 0.91, 0.90),
                "external_pilot": report("external_pilot", 0.91, 0.90),
                "WildFake": report("WildFake", 0.90, 0.82),
            },
            "seed44": {
                "birdy654": report("birdy654", 0.95, 0.94),
                "external_pilot": report("external_pilot", 0.95, 0.94),
                "WildFake": report("WildFake", 0.89, 0.79),
            },
        }
        validate_matching(reports, sources, expected_seed=42)
        result = compare(reports, sources, "seed42", 0.01)
        self.assertEqual(result["provisional_winner"], "seed43")

    def test_mismatched_seed_is_rejected(self):
        reports = {
            "seed42": {"birdy654": report("birdy654", 0.9, 0.8)},
            "seed43": {"birdy654": report("birdy654", 0.9, 0.8)},
        }
        reports["seed43"]["birdy654"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "expected 42"):
            validate_matching(reports, ["birdy654"], expected_seed=42)

    def test_mismatched_sample_fingerprint_is_rejected(self):
        reports = {
            "seed42": {"birdy654": report("birdy654", 0.9, 0.8)},
            "seed43": {"birdy654": report("birdy654", 0.9, 0.8)},
        }
        reports["seed43"]["birdy654"]["sample_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_matching(reports, ["birdy654"], expected_seed=42)


if __name__ == "__main__":
    unittest.main()

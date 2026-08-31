"""Validate and summarize matched robustness reports across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


DEFAULT_SOURCES = ("birdy654", "external_pilot", "WildFake")
SEVERE_CONDITIONS = ("downsample_25", "blur_2.0", "jpeg_30", "noise_0.10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--reference", default="seed42")
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--clean-regression-tolerance", type=float, default=0.01)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_reports(root: Path, candidates: list[str], sources: list[str]) -> dict:
    reports = {}
    for candidate in candidates:
        reports[candidate] = {}
        for source in sources:
            path = root / candidate / f"{source}_validation.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing matched report: {path}")
            reports[candidate][source] = json.loads(path.read_text())
    return reports


def validate_matching(reports: dict, sources: list[str], expected_seed: int) -> None:
    candidates = list(reports)
    reference = candidates[0]
    for source in sources:
        reference_report = reports[reference][source]
        reference_conditions = set(reference_report["conditions"])
        reference_count = int(reference_report["sample_count"])
        reference_fingerprint = reference_report.get("sample_fingerprint")
        if not reference_fingerprint:
            raise ValueError(f"{reference}/{source} is missing sample_fingerprint")
        if int(reference_report["seed"]) != expected_seed:
            raise ValueError(
                f"{reference}/{source} used seed={reference_report['seed']}, "
                f"expected {expected_seed}"
            )
        for candidate in candidates[1:]:
            report = reports[candidate][source]
            if int(report["seed"]) != expected_seed:
                raise ValueError(
                    f"{candidate}/{source} used seed={report['seed']}, "
                    f"expected {expected_seed}"
                )
            if int(report["sample_count"]) != reference_count:
                raise ValueError(
                    f"Sample-count mismatch for {source}: "
                    f"{reference}={reference_count}, "
                    f"{candidate}={report['sample_count']}"
                )
            if report.get("sample_fingerprint") != reference_fingerprint:
                raise ValueError(
                    f"Sample-fingerprint mismatch for {candidate}/{source}"
                )
            if set(report["conditions"]) != reference_conditions:
                raise ValueError(f"Condition mismatch for {candidate}/{source}")


def summarize_candidate(source_reports: dict, sources: list[str]) -> dict:
    pair_aucs = []
    semantic_aucs = []
    severe_aucs = []
    source_mean_auc = {}
    clean_auc = {}
    source_sample_count = {}

    for source in sources:
        report = source_reports[source]
        source_sample_count[source] = int(report["sample_count"])
        condition_aucs = []
        for condition, values in report["conditions"].items():
            fusion_auc = float(values["fusion"]["roc_auc"])
            semantic_auc = float(values["semantic"]["roc_auc"])
            condition_aucs.append(fusion_auc)
            pair_aucs.append(fusion_auc)
            semantic_aucs.append(semantic_auc)
            if condition in SEVERE_CONDITIONS:
                severe_aucs.append(fusion_auc)
            if condition == "clean":
                clean_auc[source] = fusion_auc
        source_mean_auc[source] = mean(condition_aucs)

    fusion_gaps = [
        fusion_auc - semantic_auc
        for fusion_auc, semantic_auc in zip(pair_aucs, semantic_aucs)
    ]
    return {
        "sample_count_by_source": source_sample_count,
        "clean_auc_by_source": clean_auc,
        "clean_mean_auc": mean(clean_auc.values()),
        "source_mean_auc": source_mean_auc,
        "weakest_source_mean_auc": min(source_mean_auc.values()),
        "mean_condition_source_auc": mean(pair_aucs),
        "severe_mean_auc": mean(severe_aucs),
        "mean_fusion_minus_semantic_auc": mean(fusion_gaps),
        "worst_fusion_minus_semantic_auc": min(fusion_gaps),
    }


def compare(
    reports: dict,
    sources: list[str],
    reference: str,
    clean_regression_tolerance: float,
) -> dict:
    if reference not in reports:
        raise ValueError(f"Reference candidate {reference!r} is unavailable")
    summaries = {
        candidate: summarize_candidate(source_reports, sources)
        for candidate, source_reports in reports.items()
    }
    reference_clean = summaries[reference]["clean_auc_by_source"]
    for summary in summaries.values():
        regressions = {
            source: summary["clean_auc_by_source"][source] - reference_clean[source]
            for source in sources
        }
        summary["clean_auc_delta_vs_reference"] = regressions
        summary["passes_clean_regression_gate"] = all(
            delta >= -clean_regression_tolerance for delta in regressions.values()
        )

    eligible = [
        candidate
        for candidate, summary in summaries.items()
        if summary["passes_clean_regression_gate"]
    ]
    ranking = sorted(
        eligible,
        key=lambda candidate: (
            summaries[candidate]["weakest_source_mean_auc"],
            summaries[candidate]["mean_condition_source_auc"],
            summaries[candidate]["clean_mean_auc"],
        ),
        reverse=True,
    )
    return {
        "reference": reference,
        "clean_regression_tolerance": clean_regression_tolerance,
        "ranking_rule": [
            "passes_clean_regression_gate",
            "weakest_source_mean_auc",
            "mean_condition_source_auc",
            "clean_mean_auc",
        ],
        "summaries": summaries,
        "eligible_ranking": ranking,
        "provisional_winner": ranking[0] if ranking else None,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    reports = load_reports(root, args.candidates, args.sources)
    validate_matching(reports, args.sources, args.expected_seed)
    result = compare(
        reports,
        args.sources,
        args.reference,
        args.clean_regression_tolerance,
    )
    result.update(
        {
            "root": str(root),
            "expected_seed": args.expected_seed,
            "sources": args.sources,
            "candidates": args.candidates,
        }
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"[comparison] provisional_winner={result['provisional_winner']} "
        f"ranking={result['eligible_ranking']}",
        flush=True,
    )
    for candidate in result["eligible_ranking"]:
        summary = result["summaries"][candidate]
        print(
            f"[candidate] name={candidate} "
            f"weakest_source_mean_auc={summary['weakest_source_mean_auc']:.4f} "
            f"mean_auc={summary['mean_condition_source_auc']:.4f} "
            f"clean_mean_auc={summary['clean_mean_auc']:.4f} "
            f"severe_mean_auc={summary['severe_mean_auc']:.4f} "
            f"mean_fusion_gap={summary['mean_fusion_minus_semantic_auc']:.4f}",
            flush=True,
        )
    print(f"[done] comparison={output_path}", flush=True)


if __name__ == "__main__":
    main()

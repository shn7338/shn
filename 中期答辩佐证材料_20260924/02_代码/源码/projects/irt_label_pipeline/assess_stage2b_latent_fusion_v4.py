#!/usr/bin/env python3
"""Apply the predeclared V4 acceptance gate to full-test evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs_inversion_dataset import load_json
from run_sionna_dataset import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    baseline = load_json(args.baseline_report.resolve())
    candidate = load_json(args.candidate_report.resolve())
    acceptance = config["acceptance"]
    baseline_rmse = float(baseline["final_rmse_db"])
    candidate_rmse = float(candidate["final_rmse_db"])
    improvement = baseline_rmse - candidate_rmse
    baseline_center = baseline["position_bins"]["by_center_distance"]
    candidate_center = candidate["position_bins"]["by_center_distance"]
    near_baseline = float(baseline_center["000-064m"]["final_rmse_db"])
    near_candidate = float(candidate_center["000-064m"]["final_rmse_db"])
    far_candidate = float(candidate_center["192m-plus"]["final_rmse_db"])
    checks = {
        "overall_rmse": {
            "passed": candidate_rmse <= float(acceptance["maximum_test_rmse_db"]),
            "actual_db": candidate_rmse,
            "maximum_db": float(acceptance["maximum_test_rmse_db"]),
        },
        "improvement": {
            "passed": improvement >= float(acceptance["minimum_improvement_db"]),
            "actual_db": improvement,
            "minimum_db": float(acceptance["minimum_improvement_db"]),
        },
        "far_center_bin": {
            "passed": far_candidate
            <= float(acceptance["maximum_far_center_bin_rmse_db"]),
            "actual_db": far_candidate,
            "maximum_db": float(acceptance["maximum_far_center_bin_rmse_db"]),
        },
        "near_center_degradation": {
            "passed": near_candidate - near_baseline
            <= float(acceptance["maximum_near_center_bin_degradation_db"]),
            "actual_db": near_candidate - near_baseline,
            "maximum_db": float(
                acceptance["maximum_near_center_bin_degradation_db"]
            ),
        },
    }
    accepted = all(bool(row["passed"]) for row in checks.values())
    report = {
        "version": 1,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "baseline_report": str(args.baseline_report.resolve()),
        "candidate_report": str(args.candidate_report.resolve()),
        "baseline_final_rmse_db": baseline_rmse,
        "candidate_final_rmse_db": candidate_rmse,
        "improvement_db": improvement,
        "checks": checks,
        "recommendation": (
            "promote V4 as the recommended Stage2B checkpoint"
            if accepted
            else "retain V3 as the recommended Stage2B checkpoint"
        ),
    }
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

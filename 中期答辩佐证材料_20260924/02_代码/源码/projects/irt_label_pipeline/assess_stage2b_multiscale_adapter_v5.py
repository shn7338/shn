#!/usr/bin/env python3
"""Apply the predeclared V5 gate against both V3 and V4 reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs_inversion_dataset import load_json
from run_sionna_dataset import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v3-report", type=Path, required=True)
    parser.add_argument("--v4-report", type=Path, required=True)
    parser.add_argument("--v5-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    v3 = load_json(args.v3_report.resolve())
    v4 = load_json(args.v4_report.resolve())
    v5 = load_json(args.v5_report.resolve())
    gate = config["acceptance"]
    v3_rmse = float(v3["final_rmse_db"])
    v4_rmse = float(v4["final_rmse_db"])
    v5_rmse = float(v5["final_rmse_db"])
    v4_bins = v4["position_bins"]["by_center_distance"]
    v5_bins = v5["position_bins"]["by_center_distance"]
    near_change = (
        float(v5_bins["000-064m"]["final_rmse_db"])
        - float(v4_bins["000-064m"]["final_rmse_db"])
    )
    checks = {
        "overall_rmse": {
            "passed": v5_rmse <= float(gate["maximum_test_rmse_db"]),
            "actual_db": v5_rmse,
            "maximum_db": float(gate["maximum_test_rmse_db"]),
        },
        "improvement_vs_v3": {
            "passed": v3_rmse - v5_rmse
            >= float(gate["minimum_improvement_vs_v3_db"]),
            "actual_db": v3_rmse - v5_rmse,
            "minimum_db": float(gate["minimum_improvement_vs_v3_db"]),
        },
        "improvement_vs_v4": {
            "passed": v4_rmse - v5_rmse
            >= float(gate["minimum_improvement_vs_v4_db"]),
            "actual_db": v4_rmse - v5_rmse,
            "minimum_db": float(gate["minimum_improvement_vs_v4_db"]),
        },
        "far_center_bin": {
            "passed": float(v5_bins["192m-plus"]["final_rmse_db"])
            <= float(gate["maximum_far_center_bin_rmse_db"]),
            "actual_db": float(v5_bins["192m-plus"]["final_rmse_db"]),
            "maximum_db": float(gate["maximum_far_center_bin_rmse_db"]),
        },
        "near_center_degradation_vs_v4": {
            "passed": near_change
            <= float(gate["maximum_near_center_bin_degradation_vs_v4_db"]),
            "actual_db": near_change,
            "maximum_db": float(
                gate["maximum_near_center_bin_degradation_vs_v4_db"]
            ),
        },
    }
    accepted = all(bool(row["passed"]) for row in checks.values())
    report = {
        "version": 1,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "v3_final_rmse_db": v3_rmse,
        "v4_final_rmse_db": v4_rmse,
        "v5_final_rmse_db": v5_rmse,
        "v5_improvement_vs_v3_db": v3_rmse - v5_rmse,
        "v5_improvement_vs_v4_db": v4_rmse - v5_rmse,
        "checks": checks,
        "recommendation": (
            "promote V5 as the recommended Stage2B checkpoint"
            if accepted
            else "do not promote V5 under the predeclared gate"
        ),
    }
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

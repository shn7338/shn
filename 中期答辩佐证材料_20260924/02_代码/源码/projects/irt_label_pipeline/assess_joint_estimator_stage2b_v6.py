#!/usr/bin/env python3
"""Apply the predeclared V6 map and estimator preservation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs_inversion_dataset import load_json
from run_sionna_dataset import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v5-report", type=Path, required=True)
    parser.add_argument("--v6-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    v5 = load_json(args.v5_report.resolve())
    v6 = load_json(args.v6_report.resolve())
    gate = config["acceptance"]
    v5_rmse = float(v5["final_rmse_db"])
    v6_rmse = float(v6["final_rmse_db"])
    checks = {
        "overall_rmse": {
            "passed": v6_rmse <= float(gate["maximum_test_rmse_db"]),
            "actual": v6_rmse,
            "maximum": float(gate["maximum_test_rmse_db"]),
        },
        "improvement_vs_v5": {
            "passed": v5_rmse - v6_rmse
            >= float(gate["minimum_improvement_vs_v5_db"]),
            "actual": v5_rmse - v6_rmse,
            "minimum": float(gate["minimum_improvement_vs_v5_db"]),
        },
        "location_preserved": {
            "passed": float(v6["estimated_location_error_px_mean"])
            <= float(gate["maximum_test_location_mean_error_px"]),
            "actual": float(v6["estimated_location_error_px_mean"]),
            "maximum": float(gate["maximum_test_location_mean_error_px"]),
        },
        "power_preserved": {
            "passed": float(v6["estimated_power_abs_error_db_mean"])
            <= float(gate["maximum_test_power_mae_db"]),
            "actual": float(v6["estimated_power_abs_error_db_mean"]),
            "maximum": float(gate["maximum_test_power_mae_db"]),
        },
        "direction_preserved": {
            "passed": float(v6["estimated_direction_abs_error_deg_mean"])
            <= float(gate["maximum_test_direction_mae_deg"]),
            "actual": float(v6["estimated_direction_abs_error_deg_mean"]),
            "maximum": float(gate["maximum_test_direction_mae_deg"]),
        },
        "far_center_bin": {
            "passed": float(
                v6["position_bins"]["by_center_distance"]["192m-plus"][
                    "final_rmse_db"
                ]
            )
            <= float(gate["maximum_far_center_bin_rmse_db"]),
            "actual": float(
                v6["position_bins"]["by_center_distance"]["192m-plus"][
                    "final_rmse_db"
                ]
            ),
            "maximum": float(gate["maximum_far_center_bin_rmse_db"]),
        },
    }
    accepted = all(bool(row["passed"]) for row in checks.values())
    report = {
        "version": 1,
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "v5_final_rmse_db": v5_rmse,
        "v6_final_rmse_db": v6_rmse,
        "v6_improvement_vs_v5_db": v5_rmse - v6_rmse,
        "checks": checks,
        "recommendation": (
            "promote the V6 Stage2B/estimator checkpoint pair"
            if accepted
            else "retain V5 unless raw V6 accuracy is preferred experimentally"
        ),
    }
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate Sionna paper-pilot arrays and directional behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paper = config["paper_parameters"]
    samples = int(args.samples or paper["samples_per_tx"])
    output_root = Path(config["output_root"])
    names = ["iso"] + [
        f"az{int(value):03d}" for value in paper["directional_azimuths_deg"]
    ]
    arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    resolved_downtilts_deg: dict[str, float] = {}
    failures: list[str] = []
    expected_shape = (int(paper["rows"]), int(paper["cols"]))

    for name in names:
        path = output_root / f"{name}_samples_{samples}.npz"
        if not path.exists():
            failures.append(f"missing {path}")
            continue
        with np.load(path) as data:
            array = data["path_gain_db"].astype(np.float32)
            valid = data["valid_mask"].astype(bool)
            outdoor = data["outdoor_mask"].astype(bool)
            if "directional_downtilt_deg" in data.files:
                resolved_downtilts_deg[name] = float(
                    data["directional_downtilt_deg"]
                )
            if array.shape != expected_shape:
                failures.append(f"{name}: shape {array.shape} != {expected_shape}")
            if valid.shape != expected_shape:
                failures.append(f"{name}: invalid valid-mask shape {valid.shape}")
            if np.any(valid & ~outdoor):
                failures.append(f"{name}: indoor cells marked valid")
            if np.any(~np.isfinite(array[valid])):
                failures.append(f"{name}: non-finite values inside valid mask")
            if np.any(np.isfinite(array[~valid])):
                failures.append(f"{name}: finite values outside valid mask")
            arrays[name] = array
            masks[name] = valid

    if resolved_downtilts_deg:
        values = np.asarray(list(resolved_downtilts_deg.values()))
        if not np.all(np.isfinite(values)):
            failures.append("non-finite resolved directional downtilt")
        elif not np.allclose(values, values[0], atol=1e-5):
            failures.append("resolved directional downtilt differs by variant")

    pairwise: dict[str, float] = {}
    orientation_checks: dict[str, dict[str, float | int]] = {}
    if len(arrays) == len(names):
        for name in names[1:]:
            common = masks["iso"] & masks[name]
            if not np.any(common):
                failures.append(f"{name}: no common valid cells with iso")
                continue
            difference = arrays[name][common] - arrays["iso"][common]
            pairwise[f"{name}_vs_iso_mae_db"] = float(
                np.mean(np.abs(difference))
            )
            if np.allclose(difference, 0.0, atol=1e-5):
                failures.append(f"{name}: directional map equals isotropic map")
        directional_names = names[1:]
        for index, left in enumerate(directional_names):
            for right in directional_names[index + 1 :]:
                common = masks[left] & masks[right]
                if not np.any(common):
                    continue
                mae = float(
                    np.mean(np.abs(arrays[left][common] - arrays[right][common]))
                )
                pairwise[f"{left}_vs_{right}_mae_db"] = mae
                if mae < 1e-4:
                    failures.append(f"{left} and {right}: maps are identical")

        rows, cols = expected_shape
        resolution_x = float(paper["cell_size_m"][0])
        resolution_y = float(paper["cell_size_m"][1])
        x_centers = (np.arange(cols) + 0.5) * resolution_x
        y_centers = (
            float(paper["area_size_m"][1])
            - (np.arange(rows) + 0.5) * resolution_y
        )
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        tx_x, tx_y = (float(value) for value in paper["transmitter_xy"])
        delta_x = x_grid - tx_x
        delta_y = y_grid - tx_y
        radius = np.hypot(delta_x, delta_y)
        bearing = np.degrees(np.arctan2(delta_y, delta_x)) % 360.0
        for azimuth in (int(value) for value in paper["directional_azimuths_deg"]):
            name = f"az{azimuth:03d}"
            common = masks["iso"] & masks[name]
            angular_difference = np.abs(
                (bearing - float(azimuth) + 180.0) % 360.0 - 180.0
            )
            annulus = common & (radius >= 140.0) & (radius <= 330.0)
            front = annulus & (angular_difference <= 15.0)
            back = annulus & (angular_difference >= 165.0)
            if not np.any(front) or not np.any(back):
                failures.append(f"{name}: insufficient front/back cells")
                continue
            directional_delta = arrays[name] - arrays["iso"]
            front_median = float(np.nanmedian(directional_delta[front]))
            back_median = float(np.nanmedian(directional_delta[back]))
            front_advantage = front_median - back_median
            orientation_checks[name] = {
                "front_cells": int(front.sum()),
                "back_cells": int(back.sum()),
                "front_median_delta_db": front_median,
                "back_median_delta_db": back_median,
                "front_minus_back_db": front_advantage,
            }
            if front_advantage <= 1.0:
                failures.append(
                    f"{name}: front/back advantage {front_advantage:.3f} dB "
                    "does not confirm the requested azimuth"
                )

    report = {
        "status": "ok" if not failures else "failed",
        "samples_per_tx": samples,
        "expected_shape": list(expected_shape),
        "variants_found": sorted(arrays),
        "resolved_directional_downtilts_deg": resolved_downtilts_deg,
        "pairwise": pairwise,
        "orientation_checks": orientation_checks,
        "failures": failures,
    }
    report_path = output_root / f"validation_samples_{samples}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

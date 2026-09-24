#!/usr/bin/env python3
"""Validate tile_000001 direct WinProp isotropic and 0/90-degree maps."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from compact_irt_result import find_power_text, parse_winprop_ascii


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iso-result-dir", type=Path, required=True)
    parser.add_argument("--az000-result-dir", type=Path, required=True)
    parser.add_argument("--az090-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def delta_metrics(
    directional: np.ndarray,
    isotropic: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta = directional.astype(np.float64) - isotropic.astype(np.float64)
    finite = np.isfinite(delta)
    values = delta[finite]
    if not values.size:
        raise ValueError("directional/isotropic comparison has no finite pixels")
    return delta, {
        "common_valid_pixels": int(values.size),
        "minimum_db": float(np.min(values)),
        "maximum_db": float(np.max(values)),
        "mean_db": float(np.mean(values)),
        "percentiles_db": {
            str(percentile): float(np.percentile(values, percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
    }


def region_mean(delta: np.ndarray, region: np.ndarray) -> float:
    values = delta[region & np.isfinite(delta)]
    if not values.size:
        raise ValueError("orientation validation region has no finite pixels")
    return float(np.mean(values))


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    tile = "tile_000001"
    metadata_path = Path(config["prepared_root"]) / tile / "metadata.json"
    grid = load_json(metadata_path)["grid"]
    result_dirs = {
        "iso": args.iso_result_dir,
        "az000": args.az000_result_dir,
        "az090": args.az090_result_dir,
    }
    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, Any] = {}
    for name, result_dir in result_dirs.items():
        power_path = find_power_text(result_dir)
        arrays[name], sources[name] = parse_winprop_ascii(power_path, grid)

    delta_000, metrics_000 = delta_metrics(arrays["az000"], arrays["iso"])
    delta_090, metrics_090 = delta_metrics(arrays["az090"], arrays["iso"])
    rows, cols = np.indices(arrays["iso"].shape)
    north = rows < arrays["iso"].shape[0] // 2
    south = ~north
    west = cols < arrays["iso"].shape[1] // 2
    east = ~west
    region_means = {
        "az000": {
            "north_db": region_mean(delta_000, north),
            "south_db": region_mean(delta_000, south),
            "east_db": region_mean(delta_000, east),
            "west_db": region_mean(delta_000, west),
        },
        "az090": {
            "north_db": region_mean(delta_090, north),
            "south_db": region_mean(delta_090, south),
            "east_db": region_mean(delta_090, east),
            "west_db": region_mean(delta_090, west),
        },
    }
    north_south_contrast = (
        region_means["az000"]["north_db"] - region_means["az000"]["south_db"]
    )
    east_west_contrast = (
        region_means["az090"]["east_db"] - region_means["az090"]["west_db"]
    )
    iso_mask = np.isfinite(arrays["iso"])
    mask_mismatch = {
        name: int(np.count_nonzero(np.isfinite(arrays[name]) != iso_mask))
        for name in ("az000", "az090")
    }
    checks = {
        "grid_is_128_by_128": arrays["iso"].shape == (128, 128),
        "az000_points_north": north_south_contrast >= 10.0,
        "az090_points_east": east_west_contrast >= 10.0,
        "az000_gain_does_not_exceed_6p3_dbi": metrics_000["maximum_db"] <= 6.4,
        "az090_gain_does_not_exceed_6p3_dbi": metrics_090["maximum_db"] <= 6.4,
        "directional_mask_mismatch_at_most_one_pixel": max(mask_mismatch.values()) <= 1,
    }
    passed = all(checks.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "tile_000001_direct_validation.png"
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        ("Isotropic IRT power", arrays["iso"], "turbo", None, None),
        ("0° direct power", arrays["az000"], "turbo", None, None),
        ("90° direct power", arrays["az090"], "turbo", None, None),
        ("0° minus isotropic", delta_000, "coolwarm", -24, 6.3),
        ("90° minus isotropic", delta_090, "coolwarm", -24, 6.3),
        (
            "0° minus 90°",
            arrays["az000"] - arrays["az090"],
            "coolwarm",
            -20,
            20,
        ),
    ]
    for axis, (title, array, cmap, vmin, vmax) in zip(axes.flat, panels):
        image = axis.imshow(array, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("column (west → east)")
        axis.set_ylabel("row (north → south)")
        figure.colorbar(image, ax=axis, shrink=0.82, label="dB / dBm")
    figure.suptitle(
        "tile_000001 direct WinProp IRT validation "
        f"(north–south={north_south_contrast:.2f} dB, "
        f"east–west={east_west_contrast:.2f} dB)"
    )
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    report = {
        "version": 1,
        "tile": tile,
        "method": "direct WinProp maps; no offline directional reweighting",
        "grid": grid,
        "array_convention": "row 0 north; column 0 west",
        "sources": sources,
        "valid_pixels": {
            name: int(np.count_nonzero(np.isfinite(array)))
            for name, array in arrays.items()
        },
        "mask_mismatch_vs_iso_pixels": mask_mismatch,
        "directional_minus_isotropic": {
            "az000": metrics_000,
            "az090": metrics_090,
        },
        "region_mean_directional_gain_db": region_means,
        "orientation_contrast_db": {
            "az000_north_minus_south": north_south_contrast,
            "az090_east_minus_west": east_west_contrast,
        },
        "checks": checks,
        "passed": passed,
        "figure": str(figure_path),
    }
    report_path = args.output_dir / "tile_000001_direct_validation.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

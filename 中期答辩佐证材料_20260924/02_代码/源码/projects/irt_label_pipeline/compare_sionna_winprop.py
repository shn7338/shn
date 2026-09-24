#!/usr/bin/env python3
"""Compare aligned Sionna path-gain maps with compact WinProp labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sionna-root", type=Path, required=True)
    parser.add_argument("--winprop-compact", type=Path, required=True)
    parser.add_argument("--reference-sionna-root", type=Path)
    parser.add_argument("--samples", type=int, default=7_000_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_sionna(root: Path, name: str, samples: int) -> np.ndarray:
    path = root / f"{name}_samples_{samples}.npz"
    with np.load(path) as data:
        return data["path_gain_db"].astype(np.float32)


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    common = np.isfinite(left) & np.isfinite(right)
    if not np.any(common):
        raise ValueError("maps have no common finite pixels")
    left_values = left[common].astype(np.float64)
    right_values = right[common].astype(np.float64)
    difference = left_values - right_values
    bias = float(np.mean(difference))
    centered = difference - bias
    correlation = float(np.corrcoef(left_values, right_values)[0, 1])
    absolute = np.abs(difference)
    return {
        "common_pixels": int(common.sum()),
        "left_finite_pixels": int(np.isfinite(left).sum()),
        "right_finite_pixels": int(np.isfinite(right).sum()),
        "left_mean_db": float(np.mean(left_values)),
        "right_mean_db": float(np.mean(right_values)),
        "left_minus_right_mean_db": bias,
        "left_minus_right_median_db": float(np.median(difference)),
        "mae_db": float(np.mean(absolute)),
        "rmse_db": float(np.sqrt(np.mean(np.square(difference)))),
        "bias_removed_rmse_db": float(np.sqrt(np.mean(np.square(centered)))),
        "absolute_error_p50_db": float(np.percentile(absolute, 50.0)),
        "absolute_error_p90_db": float(np.percentile(absolute, 90.0)),
        "absolute_error_p95_db": float(np.percentile(absolute, 95.0)),
        "pearson_correlation": correlation,
    }


def save_iso_comparison(
    path: Path,
    sionna: np.ndarray,
    winprop: np.ndarray,
    sionna_title: str,
) -> None:
    common = np.isfinite(sionna) & np.isfinite(winprop)
    combined = np.concatenate(
        [sionna[np.isfinite(sionna)], winprop[np.isfinite(winprop)]]
    )
    vmin, vmax = np.percentile(combined, [2.0, 98.0])
    difference = np.full_like(sionna, np.nan)
    difference[common] = sionna[common] - winprop[common]
    difference_limit = max(
        1.0, float(np.percentile(np.abs(difference[common]), 98.0))
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(15.5, 5.0), constrained_layout=True
    )
    first = axes[0].imshow(sionna, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(sionna_title)
    second = axes[1].imshow(winprop, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("WinProp 3.5 GHz, IRT 2/2/2")
    third = axes[2].imshow(
        difference,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title("Sionna - WinProp (dB)")
    for axis in axes:
        axis.set_xlabel("column (west to east)")
        axis.set_ylabel("row (north to south)")
    figure.colorbar(first, ax=axes[:2], label="Path gain (dB)", shrink=0.88)
    figure.colorbar(third, ax=axes[2], label="Difference (dB)", shrink=0.88)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    variants = ["iso", "az022", "az073", "az092", "az271"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.winprop_compact) as data:
        winprop = {
            name: data[name].astype(np.float32)
            for name in variants
        }
    sionna = {
        name: load_sionna(args.sionna_root, name, args.samples)
        for name in variants
    }
    sionna_summary = json.loads(
        (
            args.sionna_root / f"summary_samples_{args.samples}.json"
        ).read_text(encoding="utf-8")
    )
    parameters = sionna_summary["parameters"]
    frequency_ghz = float(parameters["frequency_hz"]) / 1e9
    max_depth = int(parameters["max_depth"])
    tx_polarization = str(parameters["tx_polarization"])
    rx_polarization = str(parameters["rx_polarization"])
    downtilt_deg = float(parameters["directional_downtilt_deg"])

    report: dict[str, Any] = {
        "version": 1,
        "samples_per_tx": args.samples,
        "map_convention": "128x128 north-up; row 0 north; col 0 west",
        "quantity": "path_gain_db",
        "sionna_root": str(args.sionna_root),
        "winprop_compact": str(args.winprop_compact),
        "sionna_parameters": {
            "frequency_ghz": frequency_ghz,
            "max_depth": max_depth,
            "tx_polarization": tx_polarization,
            "rx_polarization": rx_polarization,
            "directional_downtilt_deg": downtilt_deg,
        },
        "comparison_warning": (
            f"Sionna uses maximum depth {max_depth}, {tx_polarization}/"
            f"{rx_polarization} TX/RX polarization and {downtilt_deg:g} "
            "degree downtilt. The available WinProp baseline uses 2 "
            "reflections, 2 diffractions, 2 combined interactions, vertical "
            "polarization and a 10 degree directional-pattern downtilt. "
            "Any remaining implementation and material differences still "
            "prevent numerical equivalence."
        ),
        "sionna_35ghz_minus_winprop_35ghz": {
            name: metrics(sionna[name], winprop[name])
            for name in variants
        },
    }
    if args.reference_sionna_root is not None:
        reference = {
            name: load_sionna(
                args.reference_sionna_root, name, args.samples
            )
            for name in variants
        }
        report["sionna_35ghz_minus_reference_sionna"] = {
            name: metrics(sionna[name], reference[name])
            for name in variants
        }
        report["reference_sionna_root"] = str(args.reference_sionna_root)

    report_path = args.output_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_iso_comparison(
        args.output_dir / "iso_sionna_vs_winprop.png",
        sionna["iso"],
        winprop["iso"],
        f"Sionna {frequency_ghz:g} GHz, depth {max_depth}",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

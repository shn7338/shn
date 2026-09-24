"""Create the three normalized model inputs for one new 4 m building tile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-height", type=Path, required=True, help="Raw building-height NPY in metres")
    parser.add_argument("--grid-metadata", type=Path, required=True, help="JSON containing a WinProp-style grid object")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tx-x", type=float, required=True)
    parser.add_argument("--tx-y", type=float, required=True)
    parser.add_argument("--tx-height-m", type=float, required=True)
    parser.add_argument("--frequency-mhz", type=float, default=3500.0)
    parser.add_argument("--receiver-height-m", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalization_path = args.model_dir / "normalization.json"
    tx_normalization_path = args.model_dir / "tx_feature_normalization.json"
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    tx_normalization = json.loads(tx_normalization_path.read_text(encoding="utf-8"))
    source_metadata = json.loads(args.grid_metadata.read_text(encoding="utf-8"))
    grid = source_metadata["grid"]

    rows, cols = int(grid["rows"]), int(grid["cols"])
    xmin, ymin = float(grid["xmin"]), float(grid["ymin"])
    xmax, ymax = float(grid["xmax"]), float(grid["ymax"])
    resolution = float(grid["resolution"])
    if rows != 128 or cols != 128 or not math.isclose(resolution, 4.0):
        raise ValueError(
            f"This checkpoint requires a 128x128 grid at 4 m resolution; got {rows}x{cols} at {resolution} m"
        )
    if not math.isclose(xmax - xmin, cols * resolution, rel_tol=0, abs_tol=1e-4):
        raise ValueError("Grid x extent is inconsistent with columns and resolution")
    if not math.isclose(ymax - ymin, rows * resolution, rel_tol=0, abs_tol=1e-4):
        raise ValueError("Grid y extent is inconsistent with rows and resolution")

    expected_frequencies = [float(value) for value in tx_normalization["frequency_mhz_values"]]
    expected_rx_heights = [float(value) for value in tx_normalization["receiver_height_m_values"]]
    if not any(math.isclose(args.frequency_mhz, value, abs_tol=1e-6) for value in expected_frequencies):
        raise ValueError(
            f"Checkpoint was trained for frequency {expected_frequencies} MHz, not {args.frequency_mhz} MHz"
        )
    if not any(math.isclose(args.receiver_height_m, value, abs_tol=1e-6) for value in expected_rx_heights):
        raise ValueError(
            f"Checkpoint was trained for receiver height {expected_rx_heights} m, not {args.receiver_height_m} m"
        )

    height = np.load(args.building_height, allow_pickle=False).astype(np.float32, copy=False)
    if height.shape != (rows, cols):
        raise ValueError(f"Building-height shape mismatch: {height.shape} != {(rows, cols)}")
    if not np.isfinite(height).all() or float(height.min()) < 0:
        raise ValueError("Building height must contain finite, non-negative values in metres")
    height_cap = float(normalization["statistics"]["height"]["cap_m"])
    height_norm = np.clip(height, 0, height_cap) / height_cap

    col = int(math.floor((args.tx_x - xmin) / resolution))
    row = int(math.floor((ymax - args.tx_y) / resolution))
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"TX coordinate lies outside the grid: row={row}, col={col}")
    if not math.isfinite(args.tx_height_m) or args.tx_height_m <= 0:
        raise ValueError("TX height must be a positive finite value in metres")

    tx_height_cap = float(tx_normalization["transmitter_height"]["cap_m"])
    tx_map = np.zeros((rows, cols), dtype=np.float32)
    tx_map[row, col] = min(args.tx_height_m, tx_height_cap) / tx_height_cap

    x_coords = xmin + (np.arange(cols, dtype=np.float32) + 0.5) * resolution
    y_coords = ymax - (np.arange(rows, dtype=np.float32) + 0.5) * resolution
    distance_m = np.hypot(y_coords[:, None] - args.tx_y, x_coords[None, :] - args.tx_x)
    distance_max = float(tx_normalization["distance"]["max_m"])
    distance_norm = np.clip(
        np.log1p(distance_m) / math.log1p(distance_max), 0, 1
    ).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "building_height_norm.npy": height_norm.astype(np.float32),
        "tx_position_height_norm.npy": tx_map,
        "tx_distance_norm.npy": distance_norm,
    }
    for filename, array in outputs.items():
        np.save(args.output_dir / filename, array, allow_pickle=False)

    metadata = {
        "array_convention": "array[row, col]; row=0 is north and col=0 is west",
        "grid": grid,
        "source_building_height": str(args.building_height.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "normalization": str(normalization_path.resolve()),
        "tx_feature_normalization": str(tx_normalization_path.resolve()),
        "tx": {
            "x": args.tx_x,
            "y": args.tx_y,
            "height_m": args.tx_height_m,
            "row": row,
            "col": col,
            "height_was_clipped": args.tx_height_m > tx_height_cap,
        },
        "frequency_mhz": args.frequency_mhz,
        "receiver_height_m": args.receiver_height_m,
        "input_ranges": {
            filename: {"min": float(array.min()), "max": float(array.max())}
            for filename, array in outputs.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"tx_row={row} tx_col={col}")
    for filename in (*outputs, "metadata.json"):
        print(args.output_dir / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

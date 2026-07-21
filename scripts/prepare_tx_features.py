#!/usr/bin/env python3
"""Create transmitter-height marker and distance-map features for each tile.

The feature files are added to an existing normalized dataset directory:

* tx_position_height_norm.npy: all zero except the TX pixel, which stores
  normalized transmitter height.
* tx_distance_norm.npy: log-normalized 2D distance from every prediction pixel
  centre to the transmitter.

Feature normalization limits are derived from training tiles in splits.csv only.
Raw arrays and existing normalized arrays are never altered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


TX_MAP = "tx_position_height_norm.npy"
DISTANCE_MAP = "tx_distance_norm.npy"


@dataclass(frozen=True)
class TxTile:
    name: str
    raw_dir: str
    split: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    resolution: float
    rows: int
    cols: int
    tx_x: float
    tx_y: float
    tx_height_m: float
    frequency_mhz: float
    receiver_height_m: float


class TxFeatureError(RuntimeError):
    pass


def _atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_winprop_header(path: Path) -> tuple[float, float, float, float, float]:
    tx: tuple[float, float, float] | None = None
    frequency: float | None = None
    receiver_height: float | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN_DATA":
                break
            tokens = line.replace("\t", " ").split()
            if line.startswith("ANTENNA 1\tLOCATION"):
                tx = tuple(map(float, tokens[-3:]))  # type: ignore[assignment]
            elif line.startswith("ANTENNA 1\tFREQUENCY"):
                frequency = float(tokens[-1])
            elif line.startswith("HEIGHT"):
                receiver_height = float(tokens[-1])
    if tx is None or frequency is None or receiver_height is None:
        raise TxFeatureError(f"Incomplete WinProp header: {path}")
    return tx[0], tx[1], tx[2], frequency, receiver_height


def load_tiles(raw_root: Path, normalized_root: Path) -> list[TxTile]:
    split_path = normalized_root / "splits.csv"
    if not split_path.is_file():
        raise TxFeatureError(f"Missing {split_path}; run normalize_winprop_dataset.py first")
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    tiles: list[TxTile] = []
    for row in split_rows:
        name, split = row["tile"], row["split"]
        raw_dir = raw_root / name
        metadata_path = raw_dir / "metadata.json"
        if not metadata_path.is_file():
            raise TxFeatureError(f"Missing raw metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        grid = metadata["grid"]
        pathloss = Path(metadata["input_winprop_path_loss"])
        tx_x, tx_y, tx_height, frequency, receiver_height = _read_winprop_header(pathloss)
        tiles.append(
            TxTile(
                name=name,
                raw_dir=str(raw_dir),
                split=split,
                xmin=float(grid["xmin"]),
                ymin=float(grid["ymin"]),
                xmax=float(grid["xmax"]),
                ymax=float(grid["ymax"]),
                resolution=float(grid["resolution"]),
                rows=int(grid["rows"]),
                cols=int(grid["cols"]),
                tx_x=tx_x,
                tx_y=tx_y,
                tx_height_m=tx_height,
                frequency_mhz=frequency,
                receiver_height_m=receiver_height,
            )
        )
    return tiles


def tile_max_distance(tile: TxTile) -> float:
    xs = (tile.xmin + 0.5 * tile.resolution, tile.xmax - 0.5 * tile.resolution)
    ys = (tile.ymin + 0.5 * tile.resolution, tile.ymax - 0.5 * tile.resolution)
    return max(math.hypot(x - tile.tx_x, y - tile.tx_y) for x in xs for y in ys)


def build_tx_feature_config(tiles: list[TxTile]) -> dict[str, Any]:
    train = [tile for tile in tiles if tile.split == "train"]
    if not train:
        raise TxFeatureError("No training tiles in splits.csv")
    tx_heights = np.array([tile.tx_height_m for tile in train], dtype=np.float64)
    distance_max = max(tile_max_distance(tile) for tile in train)
    frequencies = sorted({tile.frequency_mhz for tile in tiles})
    receiver_heights = sorted({tile.receiver_height_m for tile in tiles})
    return {
        "version": 1,
        "source": "WinProp Path Loss file headers",
        "normalization_fit_split": "train",
        "transmitter_height": {
            "cap_m": float(tx_heights.max()),
            "p99_m": float(np.percentile(tx_heights, 99)),
            "min_m": float(tx_heights.min()),
            "normalization": "min(tx_height_m, cap_m) / cap_m at the transmitter pixel; zero elsewhere",
        },
        "distance": {
            "max_m": float(distance_max),
            "normalization": "log1p(distance_m) / log1p(max_m)",
        },
        "frequency_mhz_values": frequencies,
        "receiver_height_m_values": receiver_heights,
        "tile_count": len(tiles),
        "training_tile_count": len(train),
    }


def generate_one(tile_dict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    tile = TxTile(**tile_dict)
    started = time.perf_counter()
    output_dir = Path(config["output_root"]) / tile.name
    tx_path, distance_path = output_dir / TX_MAP, output_dir / DISTANCE_MAP
    if not config["overwrite"] and tx_path.is_file() and distance_path.is_file():
        return {"tile": tile.name, "split": tile.split, "status": "skipped", "seconds": round(time.perf_counter() - started, 3)}
    try:
        # Coordinates are grid-cell centres, with row 0 at the northern edge.
        col = int(math.floor((tile.tx_x - tile.xmin) / tile.resolution))
        row = int(math.floor((tile.ymax - tile.tx_y) / tile.resolution))
        tx_inside_grid = 0 <= row < tile.rows and 0 <= col < tile.cols
        tx_map = np.zeros((tile.rows, tile.cols), dtype=np.float32)
        height_cap = float(config["tx_height_cap_m"])
        if tx_inside_grid:
            tx_map[row, col] = min(tile.tx_height_m, height_cap) / height_cap

        x_coords = tile.xmin + (np.arange(tile.cols, dtype=np.float32) + 0.5) * tile.resolution
        y_coords = tile.ymax - (np.arange(tile.rows, dtype=np.float32) + 0.5) * tile.resolution
        distance_m = np.hypot(y_coords[:, None] - tile.tx_y, x_coords[None, :] - tile.tx_x)
        distance_norm = np.log1p(distance_m) / math.log1p(float(config["distance_max_m"]))
        distance_norm = np.clip(distance_norm, 0, 1).astype(np.float32)

        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_save(tx_path, tx_map)
        _atomic_save(distance_path, distance_norm)
        return {
            "tile": tile.name,
            "split": tile.split,
            "status": "ok",
            "tx_row": row,
            "tx_col": col,
            "tx_inside_grid": tx_inside_grid,
            "tx_height_m": tile.tx_height_m,
            "seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        return {"tile": tile.name, "split": tile.split, "status": "error", "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - started, 3)}


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    path = root / "tx_feature_summary.csv"
    fields = ["tile", "split", "status", "tx_row", "tx_col", "tx_inside_grid", "tx_height_m", "seconds", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="prepared_4m directory")
    parser.add_argument("--normalized-root", type=Path, required=True, help="normalized_4m_v1 directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel generation workers; 2-4 recommended")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Only generate first N sorted tiles; useful for a pilot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.raw_root.is_dir() or not args.normalized_root.is_dir():
        print("--raw-root and --normalized-root must exist", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2
    tiles = load_tiles(args.raw_root, args.normalized_root)
    config_info = build_tx_feature_config(tiles)
    _atomic_json(args.normalized_root / "tx_feature_normalization.json", config_info)
    print(json.dumps(config_info, ensure_ascii=False, indent=2))
    if args.limit is not None:
        tiles = tiles[: args.limit]
    config = {
        "output_root": str(args.normalized_root),
        "tx_height_cap_m": config_info["transmitter_height"]["cap_m"],
        "distance_max_m": config_info["distance"]["max_m"],
        "overwrite": args.overwrite,
    }
    tile_dicts = [asdict(tile) for tile in tiles]
    results: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = (generate_one(tile, config) for tile in tile_dicts)
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index == 1 or index % 500 == 0 or index == len(tile_dicts):
                print(f"TX features [{index:,}/{len(tile_dicts):,}] {result['tile']}: {result['status']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(generate_one, tile, config) for tile in tile_dicts]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if index == 1 or index % 500 == 0 or index == len(tile_dicts):
                    print(f"TX features [{index:,}/{len(tile_dicts):,}] {result['tile']}: {result['status']}")
    results.sort(key=lambda item: item["tile"])
    write_summary(args.normalized_root, results)
    errors = sum(row["status"] == "error" for row in results)
    outside = sum(row.get("tx_inside_grid") is False for row in results if row["status"] == "ok")
    print(f"TX feature generation complete: ok={sum(row['status'] == 'ok' for row in results):,}, skipped={sum(row['status'] == 'skipped' for row in results):,}, error={errors:,}, tx_outside_grid={outside:,}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

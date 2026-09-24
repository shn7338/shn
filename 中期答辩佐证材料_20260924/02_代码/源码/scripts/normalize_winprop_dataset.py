#!/usr/bin/env python3
"""Create leakage-safe splits, normalized arrays, and valid-label masks.

This script never modifies raw WinProp preparation outputs. It assigns tiles to
spatial blocks, fits normalization only on training tiles, then writes a
separate normalized dataset:

  <output-root>/normalization.json
  <output-root>/splits.csv
  <output-root>/train_tiles.txt, val_tiles.txt, test_tiles.txt
  <output-root>/tile_xxxxxx/building_height_norm.npy
  <output-root>/tile_xxxxxx/path_gain_norm.npy
  <output-root>/tile_xxxxxx/path_gain_valid_mask.npy

``path_gain_valid_mask.npy`` is True only where the original WinProp output is
finite. Invalid N.C. pixels are filled with zero in the normalized array solely
to make tensor storage convenient; training loss must use this mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


RAW_HEIGHT = "building_height_4m.npy"
RAW_GAIN = "path_gain_4m.npy"
RAW_METADATA = "metadata.json"
NORM_HEIGHT = "building_height_norm.npy"
NORM_GAIN = "path_gain_norm.npy"
VALID_MASK = "path_gain_valid_mask.npy"


@dataclass(frozen=True)
class TileInfo:
    name: str
    source_dir: str
    center_x: float
    center_y: float
    block_x: int
    block_y: int
    split: str


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, values: np.ndarray) -> None:
        """Merge a batch with this running mean/variance accumulator."""
        if values.size == 0:
            return
        data = values.astype(np.float64, copy=False)
        batch_count = int(data.size)
        batch_mean = float(data.mean(dtype=np.float64))
        batch_m2 = float(np.square(data - batch_mean, dtype=np.float64).sum(dtype=np.float64))
        if self.count == 0:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count < 2:
            return float("nan")
        return math.sqrt(self.m2 / self.count)


class DatasetError(RuntimeError):
    pass


def stable_block_score(seed: int, block_x: int, block_y: int) -> float:
    raw = f"{seed}:{block_x}:{block_y}".encode("ascii")
    number = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), byteorder="big")
    return number / 2**64


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, array)
    os.replace(temp, path)


def atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def read_raw_tiles(input_root: Path, block_size_m: float, seed: int, train_fraction: float, val_fraction: float) -> list[TileInfo]:
    tiles: list[TileInfo] = []
    missing: list[str] = []
    for tile_dir in sorted(path for path in input_root.glob("tile_*") if path.is_dir()):
        required = [tile_dir / RAW_HEIGHT, tile_dir / RAW_GAIN, tile_dir / RAW_METADATA]
        if not all(path.is_file() for path in required):
            missing.append(tile_dir.name)
            continue
        try:
            metadata = json.loads((tile_dir / RAW_METADATA).read_text(encoding="utf-8"))
            grid = metadata["grid"]
            center_x = (float(grid["xmin"]) + float(grid["xmax"])) / 2
            center_y = (float(grid["ymin"]) + float(grid["ymax"])) / 2
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Cannot read grid metadata in {tile_dir}: {exc}") from exc
        block_x = math.floor(center_x / block_size_m)
        block_y = math.floor(center_y / block_size_m)
        score = stable_block_score(seed, block_x, block_y)
        if score < train_fraction:
            split = "train"
        elif score < train_fraction + val_fraction:
            split = "val"
        else:
            split = "test"
        tiles.append(TileInfo(tile_dir.name, str(tile_dir), center_x, center_y, block_x, block_y, split))
    if missing:
        preview = ", ".join(missing[:10])
        raise DatasetError(f"{len(missing)} incomplete input tiles, e.g. {preview}")
    if not tiles:
        raise DatasetError(f"No complete tile_* directories found in {input_root}")
    return tiles


def write_split_files(output_root: Path, tiles: list[TileInfo]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    split_csv = output_root / "splits.csv"
    with split_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["tile", "split", "block_x", "block_y", "center_x", "center_y", "source_dir"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tile in tiles:
            writer.writerow(
                {
                    "tile": tile.name,
                    "split": tile.split,
                    "block_x": tile.block_x,
                    "block_y": tile.block_y,
                    "center_x": f"{tile.center_x:.3f}",
                    "center_y": f"{tile.center_y:.3f}",
                    "source_dir": tile.source_dir,
                }
            )
    for split in ("train", "val", "test"):
        names = [tile.name for tile in tiles if tile.split == split]
        (output_root / f"{split}_tiles.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def _load_arrays(tile: TileInfo) -> tuple[np.ndarray, np.ndarray]:
    source_dir = Path(tile.source_dir)
    height = np.load(source_dir / RAW_HEIGHT, mmap_mode="r")
    gain = np.load(source_dir / RAW_GAIN, mmap_mode="r")
    if height.shape != gain.shape or height.ndim != 2:
        raise DatasetError(f"Shape mismatch in {tile.name}: height={height.shape}, gain={gain.shape}")
    return height, gain


def compute_statistics(tiles: list[TileInfo], height_percentile: float, histogram_bins: int = 4096) -> dict[str, Any]:
    train_tiles = [tile for tile in tiles if tile.split == "train"]
    if not train_tiles:
        raise DatasetError("Training split has no tiles")

    moments = RunningMoments()
    height_positive_count = 0
    max_height = 0.0
    valid_gain_count = 0
    invalid_gain_count = 0
    expected_shape: tuple[int, ...] | None = None

    for index, tile in enumerate(train_tiles, start=1):
        height, gain = _load_arrays(tile)
        if expected_shape is None:
            expected_shape = height.shape
        elif height.shape != expected_shape:
            raise DatasetError(f"Mixed grid shapes: {tile.name} is {height.shape}, expected {expected_shape}")
        positive = height[np.isfinite(height) & (height > 0)]
        if positive.size:
            height_positive_count += int(positive.size)
            max_height = max(max_height, float(positive.max()))
        valid = np.isfinite(gain)
        values = gain[valid]
        moments.add(values)
        valid_gain_count += int(values.size)
        invalid_gain_count += int((~valid).sum())
        if index == 1 or index % 1000 == 0 or index == len(train_tiles):
            print(f"stats pass 1 [{index:,}/{len(train_tiles):,}]")

    if height_positive_count == 0 or max_height <= 0:
        raise DatasetError("No positive building heights in training split")
    if moments.count == 0 or not np.isfinite(moments.std) or moments.std <= 0:
        raise DatasetError("Path-gain training statistics are invalid")

    # A fixed-bin streaming histogram avoids loading hundreds of millions of
    # height pixels at once. With 4096 bins, the P99 error is at most max_height/4096.
    histogram = np.zeros(histogram_bins, dtype=np.int64)
    for index, tile in enumerate(train_tiles, start=1):
        height, _ = _load_arrays(tile)
        positive = height[np.isfinite(height) & (height > 0)]
        if positive.size:
            bin_index = np.minimum((positive / max_height * histogram_bins).astype(np.int64), histogram_bins - 1)
            histogram += np.bincount(bin_index, minlength=histogram_bins)
        if index == 1 or index % 1000 == 0 or index == len(train_tiles):
            print(f"stats pass 2 [{index:,}/{len(train_tiles):,}]")
    rank = math.ceil(height_percentile / 100 * height_positive_count) - 1
    bin_index = int(np.searchsorted(np.cumsum(histogram), rank + 1, side="left"))
    height_cap = (bin_index + 0.5) / histogram_bins * max_height

    return {
        "raw_array_shape": list(expected_shape or ()),
        "height": {
            "source": "training positive building pixels only",
            "percentile": height_percentile,
            "histogram_bins": histogram_bins,
            "positive_pixel_count": height_positive_count,
            "max_observed_m": max_height,
            "cap_m": height_cap,
            "normalization": "clip(height_m, 0, cap_m) / cap_m",
        },
        "path_gain": {
            "source": "training finite pixels only",
            "valid_pixel_count": valid_gain_count,
            "invalid_pixel_count": invalid_gain_count,
            "mean_db": moments.mean,
            "std_db": moments.std,
            "normalization": "(path_gain_db - mean_db) / std_db",
        },
        "invalid_target_handling": "NaN raw pixels become 0 in path_gain_norm.npy and False in path_gain_valid_mask.npy",
    }


def normalize_one(tile_dict: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    tile = TileInfo(**tile_dict)
    started = time.perf_counter()
    output_dir = Path(config["output_root"]) / tile.name
    target_files = [output_dir / NORM_HEIGHT, output_dir / NORM_GAIN, output_dir / VALID_MASK]
    if not config["overwrite"] and all(path.is_file() for path in target_files):
        return {"tile": tile.name, "split": tile.split, "status": "skipped", "seconds": round(time.perf_counter() - started, 3)}
    try:
        height, gain = _load_arrays(tile)
        cap = float(config["height_cap"])
        mean = float(config["gain_mean"])
        std = float(config["gain_std"])
        if cap <= 0 or std <= 0:
            raise DatasetError("Invalid normalization parameters")
        height_norm = (np.clip(np.asarray(height, dtype=np.float32), 0, cap) / cap).astype(np.float32, copy=False)
        valid_mask = np.isfinite(gain)
        gain_norm = np.zeros(gain.shape, dtype=np.float32)
        gain_norm[valid_mask] = (np.asarray(gain[valid_mask], dtype=np.float32) - mean) / std
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(output_dir / NORM_HEIGHT, height_norm)
        atomic_save_npy(output_dir / NORM_GAIN, gain_norm)
        atomic_save_npy(output_dir / VALID_MASK, valid_mask)
        return {
            "tile": tile.name,
            "split": tile.split,
            "status": "ok",
            "valid_gain_pixels": int(valid_mask.sum()),
            "seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        return {
            "tile": tile.name,
            "split": tile.split,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - started, 3),
        }


def write_normalization_summary(output_root: Path, results: list[dict[str, Any]]) -> None:
    path = output_root / "normalization_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["tile", "split", "status", "valid_gain_pixels", "seconds", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Raw prepared_4m directory")
    parser.add_argument("--output-root", type=Path, required=True, help="New directory for normalized arrays")
    parser.add_argument("--seed", type=int, default=20260710, help="Deterministic spatial-split seed")
    parser.add_argument("--block-size-m", type=float, default=2048.0, help="Spatial split block edge length in metres")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--height-percentile", type=float, default=99.0, help="Training-only positive-height percentile used as clip cap")
    parser.add_argument("--workers", type=int, default=1, help="Parallel normalization workers; 2-4 recommended for SSD")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing normalized tile outputs")
    parser.add_argument("--dry-run", action="store_true", help="Create only split files and report counts; do not compute stats or arrays")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_root.is_dir():
        print(f"Input root does not exist: {args.input_root}", file=sys.stderr)
        return 2
    if args.workers < 1 or args.block_size_m <= 0:
        print("--workers and --block-size-m must be positive", file=sys.stderr)
        return 2
    if not (0 < args.train_fraction < 1 and 0 < args.val_fraction < 1 and args.train_fraction + args.val_fraction < 1):
        print("Split fractions must be positive and sum to less than 1", file=sys.stderr)
        return 2
    if not (0 < args.height_percentile <= 100):
        print("--height-percentile must be in (0, 100]", file=sys.stderr)
        return 2

    tiles = read_raw_tiles(args.input_root, args.block_size_m, args.seed, args.train_fraction, args.val_fraction)
    split_counts = {split: sum(tile.split == split for tile in tiles) for split in ("train", "val", "test")}
    block_count = len({(tile.block_x, tile.block_y) for tile in tiles})
    print(f"Tiles={len(tiles):,}; spatial blocks={block_count:,}; splits={split_counts}")
    write_split_files(args.output_root, tiles)
    if args.dry_run:
        return 0

    statistics = compute_statistics(tiles, args.height_percentile)
    normalization = {
        "version": 1,
        "created_at_unix": time.time(),
        "split_strategy": {
            "name": "spatial_block_hash",
            "seed": args.seed,
            "block_size_m": args.block_size_m,
            "fractions_requested": {"train": args.train_fraction, "val": args.val_fraction, "test": 1 - args.train_fraction - args.val_fraction},
            "tile_counts": split_counts,
            "spatial_block_count": block_count,
        },
        "statistics": statistics,
    }
    atomic_write_json(args.output_root / "normalization.json", normalization)
    print(json.dumps(normalization["statistics"], ensure_ascii=False, indent=2))

    config = {
        "output_root": str(args.output_root),
        "height_cap": statistics["height"]["cap_m"],
        "gain_mean": statistics["path_gain"]["mean_db"],
        "gain_std": statistics["path_gain"]["std_db"],
        "overwrite": args.overwrite,
    }
    tile_dicts = [asdict(tile) for tile in tiles]
    results: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = (normalize_one(tile, config) for tile in tile_dicts)
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index == 1 or index % 500 == 0 or index == len(tile_dicts):
                print(f"normalize [{index:,}/{len(tile_dicts):,}] {result['tile']}: {result['status']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(normalize_one, tile, config) for tile in tile_dicts]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if index == 1 or index % 500 == 0 or index == len(tile_dicts):
                    print(f"normalize [{index:,}/{len(tile_dicts):,}] {result['tile']}: {result['status']}")
    results.sort(key=lambda item: item["tile"])
    write_normalization_summary(args.output_root, results)
    errors = sum(result["status"] == "error" for result in results)
    print(f"Normalization complete: ok={sum(r['status'] == 'ok' for r in results):,}, skipped={sum(r['status'] == 'skipped' for r in results):,}, error={errors:,}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

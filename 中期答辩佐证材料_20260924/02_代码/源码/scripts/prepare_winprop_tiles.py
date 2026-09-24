#!/usr/bin/env python3
"""Prepare aligned building-height and WinProp path-gain arrays for tile datasets.

For every ``tile_*`` directory, the script:
1. reads the grid extent and resolution from a WinProp ``* Path Loss.txt`` file;
2. rasterizes the tile shapefile's building-height field to that exact grid; and
3. writes the raw WinProp values into the same north-up array orientation.

The output arrays use ``array[row, col]`` convention: row 0 is the northernmost
row and column 0 is the westernmost column.  Values from the WinProp Path Loss
file are preserved exactly; they are *not* automatically relabelled as RSRP.

Requires: numpy, geopandas and rasterio.  Install instructions are in
``scripts/prepare_winprop_tiles_requirements.txt``.
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


@dataclass(frozen=True)
class GridSpec:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    resolution: float
    rows: int
    cols: int


class TileInputError(RuntimeError):
    """A tile cannot be safely processed with the available input files."""


def _parse_header_pair(line: str, key: str) -> tuple[float, float] | None:
    parts = line.replace("\t", " ").split()
    if not parts or parts[0] != key or len(parts) < 3:
        return None
    return float(parts[1]), float(parts[2])


def read_winprop_path_gain(path: Path) -> tuple[GridSpec, np.ndarray, dict[str, str]]:
    """Read a WinProp ASCII Path Loss export into a north-up floating array."""
    lower_left: tuple[float, float] | None = None
    upper_right: tuple[float, float] | None = None
    resolution: float | None = None
    headers: dict[str, str] = {}
    records: list[tuple[float, float, float]] = []
    non_computable_records = 0
    in_data = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "BEGIN_DATA":
                in_data = True
                continue
            if line == "END_DATA":
                break
            if not in_data:
                if line.startswith("LOWER_LEFT"):
                    lower_left = _parse_header_pair(line, "LOWER_LEFT")
                elif line.startswith("UPPER_RIGHT"):
                    upper_right = _parse_header_pair(line, "UPPER_RIGHT")
                elif line.startswith("RESOLUTION"):
                    resolution = float(line.replace("\t", " ").split()[-1])
                else:
                    parts = line.replace("\t", " ").split(maxsplit=1)
                    if len(parts) == 2:
                        headers[parts[0]] = parts[1]
                continue

            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError as exc:
                raise TileInputError(f"Malformed data row in {path}: {line!r}") from exc
            # WinProp uses N.C. (not computable) for points outside its valid
            # prediction domain. Preserve these cells as NaN rather than turning
            # them into a physically meaningful value such as zero.
            if parts[2].strip().upper() in {"N.C.", "N.C", "NC", "NAN"}:
                value = float("nan")
                non_computable_records += 1
            else:
                try:
                    value = float(parts[2])
                except ValueError as exc:
                    raise TileInputError(f"Malformed data row in {path}: {line!r}") from exc
            records.append((x, y, value))

    if not lower_left or not upper_right or resolution is None:
        raise TileInputError(f"Missing LOWER_LEFT, UPPER_RIGHT, or RESOLUTION in {path}")
    if resolution <= 0:
        raise TileInputError(f"Invalid resolution {resolution} in {path}")

    xmin, ymin = lower_left
    xmax, ymax = upper_right
    raw_cols = (xmax - xmin) / resolution
    raw_rows = (ymax - ymin) / resolution
    cols, rows = round(raw_cols), round(raw_rows)
    if (
        cols <= 0
        or rows <= 0
        or not math.isclose(raw_cols, cols, abs_tol=1e-6)
        or not math.isclose(raw_rows, rows, abs_tol=1e-6)
    ):
        raise TileInputError(
            f"Grid extent is not an integral number of cells in {path}: "
            f"{raw_cols} columns x {raw_rows} rows"
        )

    grid = GridSpec(xmin, ymin, xmax, ymax, resolution, rows, cols)
    array = np.full((rows, cols), np.nan, dtype=np.float32)
    duplicate_cells = 0
    out_of_bounds = 0

    for x, y, value in records:
        # WinProp records are coordinates of cell centres. This formula also
        # works for any ordering in the text file, not only its usual SW->N order.
        col = int(math.floor((x - xmin) / resolution))
        row = int(math.floor((ymax - y) / resolution))
        if not (0 <= row < rows and 0 <= col < cols):
            out_of_bounds += 1
            continue
        if np.isfinite(array[row, col]):
            duplicate_cells += 1
        array[row, col] = value

    headers["records_read"] = str(len(records))
    headers["duplicate_cells"] = str(duplicate_cells)
    headers["out_of_bounds_records"] = str(out_of_bounds)
    headers["non_computable_records"] = str(non_computable_records)
    headers["missing_cells"] = str(int(np.isnan(array).sum()))
    return grid, array, headers


def rasterize_building_heights(
    shapefile: Path,
    grid: GridSpec,
    height_field: str,
    all_touched: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize buildings using the exact WinProp grid and north-up transform."""
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise TileInputError(
            "Missing GIS dependencies. Install geopandas and rasterio; see "
            "scripts/prepare_winprop_tiles_requirements.txt"
        ) from exc

    buildings = gpd.read_file(shapefile)
    if buildings.empty:
        raise TileInputError(f"No features in {shapefile}")
    if buildings.crs is None:
        raise TileInputError(f"No CRS found in {shapefile}; cannot verify coordinate alignment")
    if height_field not in buildings.columns:
        raise TileInputError(
            f"Height field {height_field!r} not found in {shapefile}; "
            f"available fields: {', '.join(map(str, buildings.columns))}"
        )

    geom_bounds = buildings.total_bounds  # xmin, ymin, xmax, ymax
    if (
        geom_bounds[2] <= grid.xmin
        or geom_bounds[0] >= grid.xmax
        or geom_bounds[3] <= grid.ymin
        or geom_bounds[1] >= grid.ymax
    ):
        raise TileInputError(
            f"Shapefile bounds {geom_bounds.tolist()} do not overlap the WinProp grid. "
            "They are probably in different CRSs."
        )

    shapes: list[tuple[Any, float]] = []
    invalid_height_count = 0
    for geometry, value in zip(buildings.geometry, buildings[height_field]):
        if geometry is None or geometry.is_empty:
            continue
        try:
            height = float(value)
        except (TypeError, ValueError):
            invalid_height_count += 1
            continue
        if not np.isfinite(height) or height < 0:
            invalid_height_count += 1
            continue
        shapes.append((geometry, height))

    if not shapes:
        raise TileInputError(f"No valid non-negative {height_field} values in {shapefile}")

    # Rasterio applies later geometries last. Sorting ascending therefore makes
    # the highest building win if overlapping polygons occur.
    shapes.sort(key=lambda item: item[1])
    transform = from_origin(grid.xmin, grid.ymax, grid.resolution, grid.resolution)
    height_map = rasterize(
        shapes,
        out_shape=(grid.rows, grid.cols),
        transform=transform,
        fill=0.0,
        all_touched=all_touched,
        dtype="float32",
    )

    info = {
        "shapefile_crs": str(buildings.crs),
        "source_feature_count": int(len(buildings)),
        "valid_height_feature_count": len(shapes),
        "invalid_height_feature_count": invalid_height_count,
        "source_bounds": [float(v) for v in geom_bounds],
        "height_covered_cells": int(np.count_nonzero(height_map)),
        "height_min_m": float(height_map[height_map > 0].min()) if np.any(height_map > 0) else None,
        "height_max_m": float(height_map.max()),
    }
    return height_map, info


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, array)
    os.replace(temp, path)


def _atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def _find_single_input(tile_dir: Path, suffix: str, description: str) -> Path:
    matches = sorted(tile_dir.glob(suffix))
    if len(matches) != 1:
        raise TileInputError(f"Expected exactly one {description} in {tile_dir}; found {len(matches)}")
    return matches[0]


def _find_single_pathloss_file(tile_dir: Path, pattern: str, result_dir_suffix: str) -> Path:
    """Find a prediction export without accidentally selecting test-result folders."""
    search_root = tile_dir
    if result_dir_suffix:
        search_root = tile_dir / f"{tile_dir.name}{result_dir_suffix}"
        if not search_root.is_dir():
            raise TileInputError(
                f"Expected WinProp result directory {search_root.name!r} in {tile_dir}; "
                "set --result-dir-suffix '' only if a different layout is intentional"
            )
    matches = sorted(search_root.rglob(pattern))
    if len(matches) != 1:
        raise TileInputError(
            f"Expected exactly one WinProp path-loss file matching {pattern!r} in {search_root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def process_tile(tile_dir_string: str, config: dict[str, Any]) -> dict[str, Any]:
    """Process one tile. Kept module-level so it works with Windows multiprocessing."""
    started = time.perf_counter()
    tile_dir = Path(tile_dir_string)
    tile_name = tile_dir.name
    output_dir = Path(config["output_root"]) / tile_name
    resolution_tag = config["resolution_tag"]
    height_path = output_dir / f"building_height_{resolution_tag}.npy"
    gain_path = output_dir / f"path_gain_{resolution_tag}.npy"
    metadata_path = output_dir / "metadata.json"

    if not config["overwrite"] and all(p.exists() for p in (height_path, gain_path, metadata_path)):
        return {"tile": tile_name, "status": "skipped", "seconds": round(time.perf_counter() - started, 3)}

    try:
        shapefile = _find_single_input(tile_dir, "*.shp", "shapefile")
        pathloss_file = _find_single_pathloss_file(
            tile_dir, config["pathloss_pattern"], config["result_dir_suffix"]
        )
        grid, path_gain, winprop_info = read_winprop_path_gain(pathloss_file)
        height_map, building_info = rasterize_building_heights(
            shapefile, grid, config["height_field"], config["all_touched"]
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_save_npy(height_path, height_map)
        _atomic_save_npy(gain_path, path_gain)
        metadata = {
            "tile": tile_name,
            "array_convention": "array[row, col]; row=0 is north and col=0 is west",
            "grid": asdict(grid),
            "input_shapefile": str(shapefile),
            "input_winprop_path_loss": str(pathloss_file),
            "height_field": config["height_field"],
            "all_touched": config["all_touched"],
            "path_gain_semantics": (
                "Raw dB values from the WinProp 'Path Loss' ASCII export. "
                "They are not automatically converted to RSRP."
            ),
            "winprop": winprop_info,
            "buildings": building_info,
        }
        _atomic_write_json(metadata_path, metadata)
        return {
            "tile": tile_name,
            "status": "ok",
            "rows": grid.rows,
            "cols": grid.cols,
            "missing_path_gain_cells": int(np.isnan(path_gain).sum()),
            "height_covered_cells": building_info["height_covered_cells"],
            "seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # Keep a batch running when one tile is malformed.
        return {
            "tile": tile_name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - started, 3),
        }


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tile", "status", "rows", "cols", "missing_path_gain_cells", "height_covered_cells", "seconds", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing tile_* directories")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory in which aligned arrays will be written")
    parser.add_argument("--height-field", default="HEIGHT_M", help="Building-height attribute in the shapefile")
    parser.add_argument("--pathloss-pattern", default="* Path Loss.txt", help="Glob used below each tile to find WinProp output")
    parser.add_argument(
        "--result-dir-suffix",
        default="_result1",
        help="Use <tile name><suffix> as the WinProp result folder (default: _result1); use '' to search all subfolders",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes; start with 1, then use 4-8")
    parser.add_argument("--limit", type=int, help="Only process the first N sorted tiles; useful for a pilot run")
    parser.add_argument("--overwrite", action="store_true", help="Recreate outputs even if the three output files already exist")
    parser.add_argument("--dry-run", action="store_true", help="Only count/list matching tiles; do not read or write data")
    parser.add_argument("--pixel-center-only", action="store_true", help="Do not burn pixels merely touched by a building polygon")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2
    if not args.input_root.is_dir():
        print(f"Input root does not exist: {args.input_root}", file=sys.stderr)
        return 2

    tiles = sorted(path for path in args.input_root.glob("tile_*") if path.is_dir())
    if args.limit is not None:
        tiles = tiles[: args.limit]
    print(f"Found {len(tiles):,} tile directories under {args.input_root}")
    if args.dry_run:
        for tile in tiles[:20]:
            print(tile.name)
        if len(tiles) > 20:
            print(f"... plus {len(tiles) - 20:,} more")
        return 0
    if not tiles:
        return 0

    # The tag is determined from the source header per tile, but this project is
    # expected to be 4 m. Keep the requested names stable and validate shape in metadata.
    config = {
        "output_root": str(args.output_root),
        "height_field": args.height_field,
        "pathloss_pattern": args.pathloss_pattern,
        "result_dir_suffix": args.result_dir_suffix,
        "all_touched": not args.pixel_center_only,
        "overwrite": args.overwrite,
        "resolution_tag": "4m",
    }

    results: list[dict[str, Any]] = []
    total = len(tiles)
    if args.workers == 1:
        iterator = (process_tile(str(tile), config) for tile in tiles)
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index == 1 or index % 100 == 0 or index == total:
                print(f"[{index:,}/{total:,}] {result['tile']}: {result['status']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_tile, str(tile), config) for tile in tiles]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if index == 1 or index % 100 == 0 or index == total:
                    print(f"[{index:,}/{total:,}] {result['tile']}: {result['status']}")

    results.sort(key=lambda item: item["tile"])
    summary_path = args.output_root / "batch_summary.csv"
    write_summary(summary_path, results)
    counts = {status: sum(result["status"] == status for result in results) for status in ("ok", "skipped", "error")}
    print(f"Complete. ok={counts['ok']:,}, skipped={counts['skipped']:,}, error={counts['error']:,}")
    print(f"Summary: {summary_path}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parse WinProp ASCII maps and write a compact, aligned tile NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument(
        "--variants", nargs="+", default=["iso", "az000", "az090"]
    )
    parser.add_argument(
        "--variant-source",
        action="append",
        default=[],
        metavar="VARIANT=RESULT_DIR",
        help=(
            "Override a variant's raw result directory. May be repeated; "
            "useful for non-overwriting validation across diagnostic roots."
        ),
    )
    parser.add_argument(
        "--quantity",
        choices=("path_loss", "power"),
        default="path_loss",
        help=(
            "WinProp ASCII quantity to compact. With 0 dBm transmitter power, "
            "the exported path-gain values in Path Loss.txt and Power.txt are "
            "numerically identical; path_loss is the formal label source."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_variant_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(
                f"invalid --variant-source {value!r}; expected VARIANT=RESULT_DIR"
            )
        if name in sources:
            raise ValueError(f"duplicate --variant-source for {name!r}")
        sources[name] = Path(raw_path)
    return sources


def find_power_text(result_dir: Path) -> Path:
    candidates = sorted(result_dir.rglob("*Power.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one Power.txt below {result_dir}, found {len(candidates)}"
        )
    return candidates[0]


def find_path_loss_text(result_dir: Path) -> Path:
    candidates = sorted(result_dir.rglob("*Path Loss.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one Path Loss.txt below {result_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def parse_winprop_ascii(
    path: Path,
    grid: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    header: dict[str, str] = {}
    records: list[tuple[float, float, float]] = []
    in_data = False
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "BEGIN_DATA":
            in_data = True
            continue
        if not in_data:
            parts = line.split("\t")
            if len(parts) >= 2:
                header[parts[0]] = " ".join(parts[1:])
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        x, y = float(parts[0]), float(parts[1])
        value = float(parts[2]) if parts[2].upper() not in {"N.C.", "NC"} else np.nan
        records.append((x, y, value))

    if not records:
        raise ValueError(f"{path} contains no result records")
    rows, cols = int(grid["rows"]), int(grid["cols"])
    xmin, ymin = float(grid["xmin"]), float(grid["ymin"])
    resolution = float(grid["resolution"])
    source_x = np.asarray(sorted({record[0] for record in records}), dtype=np.float64)
    source_y = np.asarray(sorted({record[1] for record in records}), dtype=np.float64)
    if source_x.size < 2 or source_y.size < 2:
        raise ValueError(f"{path} does not contain a 2-D result grid")
    source_dx = float(np.median(np.diff(source_x)))
    source_dy = float(np.median(np.diff(source_y)))
    if (
        not np.allclose(np.diff(source_x), source_dx, atol=1e-6, rtol=0.0)
        or not np.allclose(np.diff(source_y), source_dy, atol=1e-6, rtol=0.0)
        or not np.isclose(source_dx, resolution, atol=1e-6, rtol=0.0)
        or not np.isclose(source_dy, resolution, atol=1e-6, rtol=0.0)
    ):
        raise ValueError(
            f"{path} source grid is irregular or has an unexpected resolution: "
            f"dx={source_dx}, dy={source_dy}, expected={resolution}"
        )
    if len(records) != source_x.size * source_y.size:
        raise ValueError(
            f"{path} source grid is incomplete: records={len(records)}, "
            f"expected={source_x.size * source_y.size}"
        )
    source = np.full((source_y.size, source_x.size), np.nan, dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    source_x0, source_y0 = float(source_x[0]), float(source_y[0])
    for x, y, value in records:
        col = int(round((x - source_x0) / resolution))
        south_row = int(round((y - source_y0) / resolution))
        key = (south_row, col)
        if key in seen:
            raise ValueError(f"duplicate source grid cell {key} in {path}")
        seen.add(key)
        source[south_row, col] = value
    if len(seen) != len(records):
        raise ValueError(f"{path} source grid contains duplicate coordinates")

    target_x = xmin + (np.arange(cols, dtype=np.float64) + 0.5) * resolution
    target_y = ymin + (np.arange(rows, dtype=np.float64) + 0.5) * resolution
    exact_grid = (
        source_x.size == cols
        and source_y.size == rows
        and np.allclose(source_x, target_x, atol=1e-6, rtol=0.0)
        and np.allclose(source_y, target_y, atol=1e-6, rtol=0.0)
    )
    if exact_grid:
        target_south = source
        resampling = {
            "applied": False,
            "method": "none",
            "reason": "WinProp output centers exactly match target centers",
        }
    else:
        target_south = np.full((rows, cols), np.nan, dtype=np.float64)
        for south_row, y in enumerate(target_y):
            y_position = (y - source_y0) / resolution
            y_low = int(np.floor(y_position + 1e-12))
            y_fraction = float(y_position - y_low)
            if abs(y_fraction) < 1e-10:
                y_fraction = 0.0
            y_high = y_low if y_fraction == 0.0 else y_low + 1
            if y_low < 0 or y_high >= source_y.size:
                raise ValueError(
                    f"{path} does not cover target y center {y:.9f}"
                )
            for col, x in enumerate(target_x):
                x_position = (x - source_x0) / resolution
                x_low = int(np.floor(x_position + 1e-12))
                x_fraction = float(x_position - x_low)
                if abs(x_fraction) < 1e-10:
                    x_fraction = 0.0
                x_high = x_low if x_fraction == 0.0 else x_low + 1
                if x_low < 0 or x_high >= source_x.size:
                    raise ValueError(
                        f"{path} does not cover target x center {x:.9f}"
                    )
                neighbors = (
                    (source[y_low, x_low], (1.0 - y_fraction) * (1.0 - x_fraction)),
                    (source[y_low, x_high], (1.0 - y_fraction) * x_fraction),
                    (source[y_high, x_low], y_fraction * (1.0 - x_fraction)),
                    (source[y_high, x_high], y_fraction * x_fraction),
                )
                active = [
                    (value, weight)
                    for value, weight in neighbors
                    if weight > 1e-12
                ]
                if not active or any(not np.isfinite(value) for value, _ in active):
                    continue
                linear_power = sum(
                    weight * np.power(10.0, value / 10.0)
                    for value, weight in active
                )
                target_south[south_row, col] = 10.0 * np.log10(linear_power)
        resampling = {
            "applied": True,
            "method": (
                "bilinear interpolation in linear-power space; target is NaN "
                "when any non-zero-weight source neighbor is N.C."
            ),
            "source_shape_south_up": [
                int(source_y.size),
                int(source_x.size),
            ],
            "source_first_center": [source_x0, source_y0],
            "source_last_center": [
                float(source_x[-1]),
                float(source_y[-1]),
            ],
            "target_first_center": [
                float(target_x[0]),
                float(target_y[0]),
            ],
            "target_last_center": [
                float(target_x[-1]),
                float(target_y[-1]),
            ],
            "target_minus_source_first_center_m": [
                float(target_x[0] - source_x0),
                float(target_y[0] - source_y0),
            ],
        }
    array = np.flip(target_south, axis=0).astype(np.float32, copy=False)
    finite = array[np.isfinite(array)]
    if not finite.size:
        raise ValueError(f"{path} has no finite target-grid values")
    metadata = {
        "source": str(path),
        "header": header,
        "records": len(records),
        "valid_pixels": int(np.isfinite(array).sum()),
        "minimum_db": float(np.min(finite)),
        "maximum_db": float(np.max(finite)),
        "resampling": resampling,
    }
    return array, metadata


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    metadata_path = Path(config["prepared_root"]) / args.tile / "metadata.json"
    tile_metadata = load_json(metadata_path)
    grid = tile_metadata["grid"]
    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, Any] = {}
    variant_sources = parse_variant_sources(args.variant_source)
    unknown_sources = sorted(set(variant_sources) - set(args.variants))
    if unknown_sources:
        raise ValueError(
            "variant sources were supplied for variants not being compacted: "
            + ", ".join(unknown_sources)
        )
    for variant in args.variants:
        result_dir = variant_sources.get(
            variant,
            Path(config["output_root"]) / "raw_results" / args.tile / variant,
        )
        result_path = (
            find_path_loss_text(result_dir)
            if args.quantity == "path_loss"
            else find_power_text(result_dir)
        )
        array, source_metadata = parse_winprop_ascii(result_path, grid)
        arrays[variant] = array.astype(np.float16)
        sources[variant] = {
            **source_metadata,
            "sha256": sha256_file(result_path),
        }

    output = args.output or (
        Path(config["output_root"]) / "compact_tiles" / f"{args.tile}.npz"
    )
    if output.exists():
        raise FileExistsError(f"refusing to replace compact tile: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output)
    metadata = {
        "version": config["version"],
        "tile": args.tile,
        "array_convention": "array[row, col]; row=0 north, col=0 west",
        "grid": grid,
        "dtype": "float16",
        "quantity": args.quantity,
        "variants": args.variants,
        "sources": sources,
        "valid_pixels": {
            name: int(np.count_nonzero(np.isfinite(array)))
            for name, array in arrays.items()
        },
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
    }
    metadata_path = output.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary, metadata_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive directional WinProp maps from one isotropic per-ray ASCII result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from compact_irt_result import find_power_text, parse_winprop_ascii


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument(
        "--azimuths",
        type=float,
        nargs="*",
        help="North-origin clockwise degrees; defaults to four deterministic values.",
    )
    parser.add_argument(
        "--validate-direct",
        action="store_true",
        help="Compare derived azimuths with matching direct WinProp result directories.",
    )
    parser.add_argument(
        "--direct-result-root",
        type=Path,
        help=(
            "Optional root containing raw_results/<tile>/<variant> for direct "
            "validation; defaults to the configured output_root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional compact-output root; defaults to the configured "
            "output_root. This is useful for non-overwriting diagnostics."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def deterministic_azimuths(config: dict[str, Any], tile: str) -> list[float]:
    seed = int(config["directional_antenna"]["azimuth_seed"])
    tile_number = int(tile.split("_")[1])
    generator = np.random.default_rng(np.random.SeedSequence([seed, tile_number]))
    values = generator.choice(360, size=4, replace=False)
    return [float(value) for value in values]


def variant_name(azimuth_deg: float) -> str:
    rounded = int(round(azimuth_deg)) % 360
    if not math.isclose(azimuth_deg % 360.0, float(rounded), abs_tol=1e-8):
        raise ValueError("WinProp label azimuths must be whole degrees")
    return f"az{rounded:03d}"


def load_apa(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    records: list[tuple[float, float, float]] = []
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
        source_encoding = "utf-8"
    except UnicodeDecodeError:
        # AMan 2020 exports its header with a Windows-1252 copyright byte
        # even on systems whose active Windows code page is GBK. The numeric
        # pattern records are ASCII-compatible.
        text = raw_bytes.decode("cp1252", errors="strict")
        source_encoding = "cp1252"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line[0] in {"#", "*"}:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"invalid APA line: {raw_line!r}")
        records.append(tuple(float(value) for value in parts))
    theta_values = sorted({record[0] for record in records})
    phi_values = sorted({record[1] for record in records})
    if theta_values[0] != 0.0 or theta_values[-1] != 180.0:
        raise ValueError("APA theta coverage must be [0, 180] degrees")
    theta_step = float(np.median(np.diff(theta_values)))
    phi_step = float(np.median(np.diff(phi_values)))
    if (
        not np.allclose(np.diff(theta_values), theta_step)
        or not np.allclose(np.diff(phi_values), phi_step)
        or not math.isclose(theta_step, phi_step)
    ):
        raise ValueError("APA must use one regular theta/phi grid")
    if phi_values[0] != 0.0 or not (
        math.isclose(phi_values[-1], 360.0)
        or math.isclose(phi_values[-1] + phi_step, 360.0)
    ):
        raise ValueError(
            "APA phi coverage must be [0, 360] or one circular grid "
            "without the duplicate 360-degree seam"
        )
    source_phi_samples = len(phi_values)
    gains = np.full((len(theta_values), len(phi_values)), np.nan, dtype=np.float64)
    for theta, phi, gain in records:
        theta_index = int(round(theta / theta_step))
        phi_index = int(round(phi / phi_step))
        if np.isfinite(gains[theta_index, phi_index]):
            raise ValueError(f"duplicate APA sample at theta={theta}, phi={phi}")
        gains[theta_index, phi_index] = gain
    if not np.all(np.isfinite(gains)):
        raise ValueError("APA grid is incomplete")
    if math.isclose(phi_values[-1], 360.0):
        if not np.allclose(gains[:, 0], gains[:, -1], atol=1e-8):
            raise ValueError("APA phi=0 and phi=360 seams do not match")
    else:
        gains = np.concatenate((gains, gains[:, :1]), axis=1)
    return gains, {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(records),
        "theta_samples": len(theta_values),
        "phi_samples": int(gains.shape[1]),
        "source_phi_samples": source_phi_samples,
        "resolution_deg": theta_step,
        "minimum_gain_dbi": float(np.min(gains)),
        "maximum_gain_dbi": float(np.max(gains)),
        "source_encoding": source_encoding,
    }


def interpolate_pattern(
    gains: np.ndarray,
    resolution_deg: float,
    theta_deg: np.ndarray,
    relative_phi_deg: np.ndarray,
) -> np.ndarray:
    theta = np.clip(np.asarray(theta_deg, dtype=np.float64), 0.0, 180.0)
    phi = np.mod(np.asarray(relative_phi_deg, dtype=np.float64), 360.0)
    theta_position = theta / resolution_deg
    phi_position = phi / resolution_deg
    theta_low = np.floor(theta_position).astype(np.int64)
    phi_low = np.floor(phi_position).astype(np.int64)
    theta_high = np.minimum(theta_low + 1, gains.shape[0] - 1)
    phi_high = phi_low + 1
    theta_fraction = theta_position - theta_low
    phi_fraction = phi_position - phi_low
    low_phi_gain = (
        gains[theta_low, phi_low] * (1.0 - theta_fraction)
        + gains[theta_high, phi_low] * theta_fraction
    )
    high_phi_gain = (
        gains[theta_low, phi_high] * (1.0 - theta_fraction)
        + gains[theta_high, phi_high] * theta_fraction
    )
    return low_phi_gain * (1.0 - phi_fraction) + high_phi_gain * phi_fraction


def find_ray_text(result_dir: Path) -> Path:
    candidates = sorted(result_dir.rglob("*Rays.str"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one Rays.str below {result_dir}, found {len(candidates)}"
        )
    return candidates[0]


def parse_ray_header(path: Path) -> dict[str, Any]:
    header: dict[str, Any] = {}
    with path.open("r", encoding="mbcs", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("POINT "):
                break
            if not line or line.startswith("*"):
                continue
            if line.startswith("LL_EDGE "):
                header["lower_left"] = [
                    float(value) for value in line.split(maxsplit=1)[1].split(",")
                ]
            elif line.startswith("HEIGHT "):
                header["height_m"] = float(line.split()[1])
            elif line.startswith("RESOLUTION "):
                header["resolution_m"] = float(line.split()[1])
            elif line.startswith("COLUMNS "):
                header["columns"] = int(line.split()[1])
            elif line.startswith("LINES "):
                header["lines"] = int(line.split()[1])
            elif line.startswith("ANTENNA "):
                header["transmitter_xyz_m"] = [
                    float(value) for value in line.split()[1:4]
                ]
            elif line.startswith("FREQUENCY "):
                header["frequency_mhz"] = float(line.split()[1])
            elif line.startswith("PATHS "):
                header["paths"] = line.split()[1]
            elif line.startswith("ANGLES "):
                header["angles"] = line.split()[1]
    return header


def validate_ray_header(
    header: dict[str, Any],
    grid: dict[str, Any],
    config: dict[str, Any],
) -> None:
    expected = {
        "lower_left": [float(grid["xmin"]), float(grid["ymin"])],
        "height_m": float(config["grid"]["receiver_height_m"]),
        "resolution_m": float(grid["resolution"]),
        "columns": int(grid["cols"]),
        "lines": int(grid["rows"]),
        "frequency_mhz": float(config["physics"]["frequency_mhz"]),
        "paths": "YES",
        "angles": "YES",
    }
    for key, expected_value in expected.items():
        actual = header.get(key)
        if isinstance(expected_value, list):
            matches = np.allclose(actual, expected_value, atol=1e-6, rtol=0.0)
        elif isinstance(expected_value, float):
            matches = math.isclose(
                float(actual), expected_value, abs_tol=1e-6, rel_tol=0.0
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"ray header mismatch for {key}: actual={actual}, "
                f"expected={expected_value}"
            )


def ray_reweight_offsets(
    ray_path: Path,
    grid: dict[str, Any],
    azimuths: list[float],
    pattern: np.ndarray,
    resolution_deg: float,
    expected_point_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows, cols = int(grid["rows"]), int(grid["cols"])
    if expected_point_mask.shape != (rows, cols):
        raise ValueError(
            f"unexpected ray POINT mask shape: {expected_point_mask.shape}"
        )
    xmin, ymin = float(grid["xmin"]), float(grid["ymin"])
    cell_resolution = float(grid["resolution"])
    names = [variant_name(value) for value in azimuths]
    offsets = {
        name: np.full((rows, cols), np.nan, dtype=np.float32) for name in names
    }
    current_cell: tuple[int, int] | None = None
    field_strengths: list[float] = []
    dod_azimuths_deg: list[float] = []
    dod_elevations_deg: list[float] = []
    points = 0
    paths = 0
    maximum_paths_per_point = 0
    out_of_bounds = 0
    seen: set[tuple[int, int]] = set()

    def finish_point() -> None:
        nonlocal maximum_paths_per_point
        if current_cell is None:
            return
        maximum_paths_per_point = max(maximum_paths_per_point, len(field_strengths))
        if not field_strengths:
            return
        field_db = np.asarray(field_strengths, dtype=np.float64)
        field_power = np.power(10.0, field_db / 10.0)
        denominator = float(np.sum(field_power))
        theta = np.asarray(dod_elevations_deg, dtype=np.float64)
        world_phi = np.asarray(dod_azimuths_deg, dtype=np.float64)
        row, col = current_cell
        for name, north_clockwise_azimuth in zip(names, azimuths):
            world_boresight = (90.0 - north_clockwise_azimuth) % 360.0
            relative_phi = np.mod(world_phi - world_boresight, 360.0)
            gain_db = interpolate_pattern(
                pattern,
                resolution_deg,
                theta,
                relative_phi,
            )
            numerator = float(np.sum(field_power * np.power(10.0, gain_db / 10.0)))
            offsets[name][row, col] = 10.0 * math.log10(numerator / denominator)

    with ray_path.open("r", encoding="mbcs", errors="strict") as handle:
        for raw_line in handle:
            if raw_line.startswith("POINT "):
                finish_point()
                parts = raw_line.split()
                x, y = float(parts[1]), float(parts[2])
                col = int(round((x - xmin) / cell_resolution - 0.5))
                south_row = int(round((y - ymin) / cell_resolution - 0.5))
                row = rows - 1 - south_row
                current_cell = (row, col)
                field_strengths = []
                dod_azimuths_deg = []
                dod_elevations_deg = []
                points += 1
                if not (0 <= row < rows and 0 <= col < cols):
                    out_of_bounds += 1
                    current_cell = None
                elif (row, col) in seen:
                    raise ValueError(f"duplicate ray POINT at grid cell {(row, col)}")
                else:
                    seen.add((row, col))
            elif raw_line.startswith("PATH "):
                if current_cell is None:
                    continue
                parts = raw_line.split()
                if len(parts) < 10:
                    raise ValueError(f"invalid ray PATH line: {raw_line!r}")
                field_strengths.append(float(parts[2]))
                dod_azimuths_deg.append(math.degrees(float(parts[6])) % 360.0)
                dod_elevations_deg.append(math.degrees(float(parts[7])))
                paths += 1
    finish_point()
    expected_cells = {
        (int(row), int(col)) for row, col in np.argwhere(expected_point_mask)
    }
    missing_cells = expected_cells - seen
    unexpected_cells = seen - expected_cells
    if (
        points != len(expected_cells)
        or seen != expected_cells
        or out_of_bounds
    ):
        raise ValueError(
            f"incomplete ray grid: points={points}, unique={len(seen)}, "
            f"out_of_bounds={out_of_bounds}, expected={len(expected_cells)}, "
            f"missing={len(missing_cells)}, unexpected={len(unexpected_cells)}"
        )
    return offsets, {
        "points": points,
        "paths": paths,
        "maximum_paths_per_point": maximum_paths_per_point,
        "mean_paths_per_point": paths / points,
        "out_of_bounds": out_of_bounds,
        "indoor_points_omitted_by_winprop": rows * cols - points,
    }


def comparison_metrics(
    derived: np.ndarray,
    direct: np.ndarray,
) -> dict[str, float | int | bool]:
    finite = np.isfinite(derived) & np.isfinite(direct)
    difference = derived[finite].astype(np.float64) - direct[finite].astype(np.float64)
    if difference.size == 0:
        raise ValueError("direct comparison has no finite pixels")
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    p95 = float(np.percentile(absolute, 95))
    return {
        "pixels": int(difference.size),
        "bias_db": float(np.mean(difference)),
        "mae_db": float(np.mean(absolute)),
        "rmse_db": rmse,
        "p95_absolute_error_db": p95,
        "maximum_absolute_error_db": float(np.max(absolute)),
        "passed": rmse <= 0.35 and p95 <= 0.75,
    }


def main() -> int:
    args = parse_args()
    if re.fullmatch(r"tile_\d{6}", args.tile) is None:
        raise ValueError(f"invalid tile name: {args.tile!r}")
    config = load_json(args.config)
    metadata_path = Path(config["prepared_root"]) / args.tile / "metadata.json"
    tile_metadata = load_json(metadata_path)
    grid = tile_metadata["grid"]
    result_root = Path(config["output_root"]) / "raw_results" / args.tile
    iso_result_dir = result_root / "iso"
    ray_path = find_ray_text(iso_result_dir)
    power_path = find_power_text(iso_result_dir)
    ray_header = parse_ray_header(ray_path)
    validate_ray_header(ray_header, grid, config)

    antenna = config["directional_antenna"]
    pattern_path = Path(antenna["pattern_base"]).with_suffix(".apa")
    pattern, pattern_metadata = load_apa(pattern_path)
    if not math.isclose(
        pattern_metadata["maximum_gain_dbi"],
        float(antenna["boresight_gain_dbi"]),
        abs_tol=1e-6,
    ):
        raise ValueError("APA maximum gain does not match the configuration")
    azimuths = (
        [float(value) % 360.0 for value in args.azimuths]
        if args.azimuths
        else deterministic_azimuths(config, args.tile)
    )
    names = [variant_name(value) for value in azimuths]
    if len(names) != len(set(names)):
        raise ValueError("azimuth list contains duplicate whole-degree variants")

    started = time.perf_counter()
    iso_power, iso_source = parse_winprop_ascii(power_path, grid)
    building_height = np.load(
        Path(config["prepared_root"]) / args.tile / "building_height_4m.npy",
        allow_pickle=False,
    )
    building_mask = building_height > 0
    winprop_valid_mask = np.isfinite(iso_power)
    offsets, ray_stats = ray_reweight_offsets(
        ray_path,
        grid,
        azimuths,
        pattern,
        float(pattern_metadata["resolution_deg"]),
        winprop_valid_mask,
    )
    arrays: dict[str, np.ndarray] = {"iso": iso_power}
    for name in names:
        derived = iso_power + offsets[name]
        arrays[name] = derived.astype(np.float32)
    elapsed_seconds = time.perf_counter() - started

    comparisons: dict[str, Any] = {}
    if args.validate_direct:
        direct_result_root = (
            args.direct_result_root
            if args.direct_result_root is not None
            else Path(config["output_root"])
        )
        direct_tile_root = direct_result_root / "raw_results" / args.tile
        for name in names:
            direct_path = find_power_text(direct_tile_root / name)
            direct, _ = parse_winprop_ascii(direct_path, grid)
            comparisons[name] = {
                "source": str(direct_path),
                **comparison_metrics(arrays[name], direct),
            }

    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(config["output_root"])
    )
    output_dir = output_root / "compact_tiles"
    output_path = output_dir / f"{args.tile}.npz"
    if output_path.exists():
        raise FileExistsError(f"refusing to replace compact tile: {output_path}")
    compact_arrays = {
        name: array.astype(np.float16) for name, array in arrays.items()
    }
    atomic_save_npz(output_path, **compact_arrays)
    raw_path: Path | None = None
    if args.tile == "tile_000001":
        raw_path = output_dir / f"{args.tile}_validation_f32.npz"
        if raw_path.exists():
            raise FileExistsError(f"refusing to replace validation tile: {raw_path}")
        atomic_save_npz(raw_path, **arrays)

    metadata: dict[str, Any] = {
        "version": config["version"],
        "tile": args.tile,
        "array_convention": "array[row, col]; row=0 north, col=0 west",
        "grid": grid,
        "azimuth_convention": "degrees clockwise from north",
        "azimuths_deg": azimuths,
        "variants": ["iso", *names],
        "dtype": "float16",
        "superposition": config["physics"]["superposition"],
        "method": (
            "WinProp isotropic IRT per-ray power reweighting using the ray DoD "
            "angles and the configured AMan antenna gain pattern"
        ),
        "ray_header": ray_header,
        "ray_statistics": ray_stats,
        "validity_mask_audit": {
            "winprop_finite_cells": int(np.count_nonzero(winprop_valid_mask)),
            "winprop_missing_cells": int(np.count_nonzero(~winprop_valid_mask)),
            "prepared_building_cells": int(np.count_nonzero(building_mask)),
            "finite_cells_inside_prepared_building_mask": int(
                np.count_nonzero(winprop_valid_mask & building_mask)
            ),
            "missing_cells_outside_prepared_building_mask": int(
                np.count_nonzero(~winprop_valid_mask & ~building_mask)
            ),
            "authority": "WinProp Power.txt N.C. status",
        },
        "pattern": pattern_metadata,
        "iso_source": iso_source,
        "direct_validation": comparisons,
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "compact": str(output_path),
            "validation_float32": str(raw_path) if raw_path else None,
        },
        "hashes_sha256": {
            "ray_str": sha256_file(ray_path),
            "iso_power_txt": sha256_file(power_path),
            "compact": sha256_file(output_path),
            "validation_float32": sha256_file(raw_path) if raw_path else None,
        },
    }
    metadata["direct_validation_passed"] = (
        all(item["passed"] for item in comparisons.values())
        if comparisons
        else None
    )
    metadata_output = output_path.with_suffix(".json")
    atomic_write_json(metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if comparisons and not metadata["direct_validation_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

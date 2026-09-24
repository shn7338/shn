#!/usr/bin/env python3
"""Generate the random-base-station inversion Pilot dataset.

One resumable work item is one building tile at one transmitter location. Each
accepted item contains one isotropic path-gain map, four directional signal
strength maps, sparse measurements, and all latent base-station ground truth.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from run_sionna_dataset import (
    atomic_write_json,
    load_json,
    read_selection,
    run_logged,
    sha256_file,
    sha256_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SITE_RUNNER = SCRIPT_DIR / "run_sionna_paper_pilot.py"
SITE_VALIDATOR = SCRIPT_DIR / "validate_sionna_paper_pilot.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument("--limit-sites", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def selected_buildings(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts = config["pilot_split_counts"]
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        count = int(counts[split])
        if not 0 < count <= len(split_rows):
            raise ValueError(f"invalid Pilot building count for {split}: {count}")
        selected.extend(split_rows[:count])
    return selected


def sample_tile_locations(
    config: dict[str, Any], tile: str
) -> list[tuple[str, float, float]]:
    location = config["location_sampling"]
    margin = float(location["margin_m"])
    half = float(config["radio_map"]["area_size_m"][0]) / 2.0
    maximum = 2.0 * half - margin
    minimum_separation = float(location["minimum_pairwise_separation_m"])
    quadrants = [
        ("southwest", (margin, half), (margin, half)),
        ("southeast", (half, maximum), (margin, half)),
        ("northwest", (margin, half), (half, maximum)),
        ("northeast", (half, maximum), (half, maximum)),
    ]
    expected_sites = int(config["sites_per_building"])
    rng = np.random.default_rng(
        stable_seed(config["randomness"]["location_seed"], tile)
    )
    strategy = str(location.get("strategy", "quadrants")).lower()
    if strategy == "radial_bins":
        radius_bins = [
            (float(bounds[0]), float(bounds[1]))
            for bounds in location["center_radius_bins_m"]
        ]
        if expected_sites != len(radius_bins):
            raise ValueError("sites_per_building must equal the number of radial bins")
        center = half
        for _ in range(10000):
            points: list[tuple[str, float, float]] = []
            valid = True
            for lower, upper in radius_bins:
                accepted: tuple[str, float, float] | None = None
                for _attempt in range(1000):
                    radius = math.sqrt(rng.uniform(lower * lower, upper * upper))
                    angle = rng.uniform(0.0, 2.0 * math.pi)
                    x_m = center + radius * math.cos(angle)
                    y_m = center + radius * math.sin(angle)
                    if margin <= x_m <= maximum and margin <= y_m <= maximum:
                        accepted = (
                            f"radius_{int(lower):03d}_{int(upper):03d}m",
                            float(x_m),
                            float(y_m),
                        )
                        break
                if accepted is None:
                    valid = False
                    break
                points.append(accepted)
            if not valid:
                continue
            distances = [
                math.hypot(left[1] - right[1], left[2] - right[2])
                for index, left in enumerate(points)
                for right in points[index + 1 :]
            ]
            if min(distances) >= minimum_separation:
                return points
        raise RuntimeError(f"failed to sample separated radial-bin sites for {tile}")
    if strategy != "quadrants":
        raise ValueError(f"unsupported location sampling strategy: {strategy}")
    if expected_sites != len(quadrants):
        raise ValueError("quadrant sampling requires exactly four sites")
    for _ in range(10000):
        points = [
            (name, float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))
            for name, x_range, y_range in quadrants
        ]
        distances = [
            math.hypot(left[1] - right[1], left[2] - right[2])
            for index, left in enumerate(points)
            for right in points[index + 1 :]
        ]
        if min(distances) >= minimum_separation:
            return points
    raise RuntimeError(f"failed to sample separated sites for {tile}")


def build_entries(
    config: dict[str, Any], buildings: list[dict[str, str]]
) -> list[dict[str, Any]]:
    prepared_root = Path(config["prepared_root"])
    width = float(config["radio_map"]["area_size_m"][0])
    height = float(config["radio_map"]["area_size_m"][1])
    cell_x = float(config["radio_map"]["cell_size_m"][0])
    cell_y = float(config["radio_map"]["cell_size_m"][1])
    rows = int(config["radio_map"]["rows"])
    cols = int(config["radio_map"]["cols"])
    entries: list[dict[str, Any]] = []
    for building in buildings:
        tile = building["tile"]
        tile_number = int(tile.split("_")[1])
        height_map_path = prepared_root / tile / "building_height_4m.npy"
        height_map = np.load(height_map_path, allow_pickle=False)
        if height_map.shape != (rows, cols):
            raise ValueError(f"unexpected building map shape for {tile}: {height_map.shape}")
        transmitter_height = float(np.max(height_map)) + 5.0
        for site_index, (quadrant, x_m, y_m) in enumerate(
            sample_tile_locations(config, tile)
        ):
            site_id = f"{tile}_site{site_index:02d}"
            azimuth_rng = np.random.default_rng(
                stable_seed(config["randomness"]["azimuth_seed"], site_id)
            )
            azimuths = sorted(
                int(value)
                for value in azimuth_rng.choice(
                    360,
                    size=int(config["radio_map"]["directional_labels_per_site"]),
                    replace=False,
                ).tolist()
            )
            power_rng = np.random.default_rng(
                stable_seed(config["randomness"]["effective_power_seed"], site_id)
            )
            power_range = config["effective_power"]["range_db"]
            effective_power_db = float(
                power_rng.uniform(float(power_range[0]), float(power_range[1]))
            )
            col_px = x_m / cell_x - 0.5
            row_px = (height - y_m) / cell_y - 0.5
            entries.append(
                {
                    "site_id": site_id,
                    "tile": tile,
                    "tile_number": tile_number,
                    "split": building["split"],
                    "split_index": int(building["split_index"]),
                    "site_index": site_index,
                    "quadrant": quadrant,
                    "tx_xy_m": [x_m, y_m],
                    "tx_row_col_px": [row_px, col_px],
                    "tx_xy_norm": [x_m / width, y_m / height],
                    "tx_row_col_norm": [row_px / (rows - 1), col_px / (cols - 1)],
                    "tx_height_m": transmitter_height,
                    "effective_power_db": effective_power_db,
                    "azimuths_deg": azimuths,
                    "variants": ["iso"] + [f"az{value:03d}" for value in azimuths],
                    "ray_seed": stable_seed(
                        config["randomness"]["ray_seed"], site_id
                    ),
                    "sparse_seeds": [
                        stable_seed(
                            config["randomness"]["sparse_seed"], site_id, direction
                        )
                        for direction in range(len(azimuths))
                    ],
                }
            )
    return entries


def build_manifest(
    config: dict[str, Any], config_path: Path, entries: list[dict[str, Any]]
) -> dict[str, Any]:
    signature_payload = {
        "selection_sha256": sha256_file(Path(config["selection_csv"])),
        "source_root": config["source_root"],
        "prepared_root": config["prepared_root"],
        "pilot_split_counts": config["pilot_split_counts"],
        "sites_per_building": config["sites_per_building"],
        "location_sampling": config["location_sampling"],
        "effective_power": config["effective_power"],
        "sparse_measurements": config["sparse_measurements"],
        "radio_map": config["radio_map"],
        "scene": config["scene"],
        "randomness": config["randomness"],
        "entries": entries,
    }
    return {
        "version": int(config["version"]),
        "name": config["name"],
        "method": config["storage"]["dataset_method"],
        "config": str(config_path),
        "selection_csv": config["selection_csv"],
        "selection_sha256": signature_payload["selection_sha256"],
        "dataset_signature_sha256": sha256_json(signature_payload),
        "building_count": sum(int(v) for v in config["pilot_split_counts"].values()),
        "site_count": len(entries),
        "propagation_map_count": len(entries)
        * (1 + int(config["radio_map"]["directional_labels_per_site"])),
        "directional_training_sample_count": len(entries)
        * int(config["radio_map"]["directional_labels_per_site"]),
        "entries": entries,
    }


def ensure_manifest(output_root: Path, expected: dict[str, Any]) -> Path:
    path = output_root / "run_manifest.json"
    if path.exists():
        existing = load_json(path)
        if existing.get("dataset_signature_sha256") != expected["dataset_signature_sha256"]:
            raise RuntimeError(
                "the existing run manifest has different inputs or parameters; "
                "use a new output_root"
            )
    else:
        atomic_write_json(path, expected)
    return path


def write_indexes(
    output_root: Path,
    buildings: list[dict[str, str]],
    entries: list[dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    building_path = output_root / "selection_buildings.csv"
    with building_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "tile",
            "split",
            "split_index",
            "block_x",
            "block_y",
            "center_x",
            "center_y",
            "source_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(buildings)
    site_path = output_root / "site_index.csv"
    with site_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "site_id",
            "tile",
            "tile_number",
            "split",
            "split_index",
            "site_index",
            "quadrant",
            "tx_x_m",
            "tx_y_m",
            "tx_row_px",
            "tx_col_px",
            "tx_height_m",
            "effective_power_db",
            "azimuth_0_deg",
            "azimuth_1_deg",
            "azimuth_2_deg",
            "azimuth_3_deg",
            "sample_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "site_id": entry["site_id"],
                    "tile": entry["tile"],
                    "tile_number": entry["tile_number"],
                    "split": entry["split"],
                    "split_index": entry["split_index"],
                    "site_index": entry["site_index"],
                    "quadrant": entry["quadrant"],
                    "tx_x_m": f"{entry['tx_xy_m'][0]:.9f}",
                    "tx_y_m": f"{entry['tx_xy_m'][1]:.9f}",
                    "tx_row_px": f"{entry['tx_row_col_px'][0]:.9f}",
                    "tx_col_px": f"{entry['tx_row_col_px'][1]:.9f}",
                    "tx_height_m": f"{entry['tx_height_m']:.6f}",
                    "effective_power_db": f"{entry['effective_power_db']:.9f}",
                    **{
                        f"azimuth_{index}_deg": azimuth
                        for index, azimuth in enumerate(entry["azimuths_deg"])
                    },
                    "sample_path": str(
                        output_root
                        / "samples"
                        / entry["split"]
                        / f"{entry['site_id']}.npz"
                    ),
                }
            )


def site_paths(
    output_root: Path, entry: dict[str, Any]
) -> tuple[Path, Path, Path]:
    sample = output_root / "samples" / entry["split"] / f"{entry['site_id']}.npz"
    metadata = sample.with_suffix(".json")
    completion = output_root / "completion" / f"{entry['site_id']}.json"
    return sample, metadata, completion


def site_work_root(
    config: dict[str, Any], output_root: Path, entry: dict[str, Any]
) -> Path:
    base = Path(config.get("work_root", output_root / "work_sites"))
    return base / entry["site_id"]


def is_complete(output_root: Path, entry: dict[str, Any], verify_hash: bool = False) -> bool:
    sample, metadata_path, completion_path = site_paths(output_root, entry)
    if not (sample.is_file() and metadata_path.is_file() and completion_path.is_file()):
        return False
    try:
        metadata = load_json(metadata_path)
        completion = load_json(completion_path)
        expected_hash = metadata["output_sha256"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    basic = (
        completion.get("status") == "ok"
        and completion.get("compact_sha256") == expected_hash
        and sample.stat().st_size == int(metadata.get("output_bytes", -1))
    )
    return basic and (not verify_hash or sha256_file(sample) == expected_hash)


def site_config(
    config: dict[str, Any], entry: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    tile = entry["tile"]
    paper = {
        key: value
        for key, value in config["radio_map"].items()
        if key != "directional_labels_per_site"
    }
    paper["transmitter_xy"] = entry["tx_xy_m"]
    paper["radio_map_center_xy"] = [256.0, 256.0]
    paper["directional_azimuths_deg"] = entry["azimuths_deg"]
    paper["seed"] = entry["ray_seed"]
    return {
        "version": int(config["version"]),
        "name": entry["site_id"],
        "tile": tile,
        "site_id": entry["site_id"],
        "source_shapefile": str(Path(config["source_root"]) / tile / f"{tile}.shp"),
        "prepared_metadata": str(Path(config["prepared_root"]) / tile / "metadata.json"),
        "building_height_map": str(
            Path(config["prepared_root"]) / tile / "building_height_4m.npy"
        ),
        # Mitsuba's Windows PLY loader cannot reliably open non-ASCII paths.
        # Keep ray-tracing work on the configured ASCII-only E: path while the
        # accepted compact samples remain under the user-facing D: dataset.
        "output_root": str(site_work_root(config, output_root, entry)),
        "environment_python": config["environment_python"],
        "paper_parameters": paper,
        "scene": config["scene"],
        "output": {
            "array_convention": "array[row,col]; row 0 north, col 0 west",
            "quantity": "path_gain_db",
            "indoor_pixels": "NaN",
            "unhit_outdoor_pixels": "NaN",
            "save_linear_path_gain": True,
            "save_png": bool(config["execution"]["save_png"]),
        },
        "provenance": {
            "dataset": config["name"],
            "dataset_signature_sha256": config["dataset_signature_sha256"],
            "runtime_sionna_version": config["runtime_sionna_version"],
        },
    }


def transmitter_heatmap(
    entry: dict[str, Any], rows: int, cols: int, sigma_px: float
) -> np.ndarray:
    row, col = (float(value) for value in entry["tx_row_col_px"])
    row_grid, col_grid = np.meshgrid(
        np.arange(rows, dtype=np.float32),
        np.arange(cols, dtype=np.float32),
        indexing="ij",
    )
    heatmap = np.exp(
        -((row_grid - row) ** 2 + (col_grid - col) ** 2) / (2.0 * sigma_px**2)
    )
    return heatmap.astype(np.float16)


def validate_and_compact(
    config: dict[str, Any], entry: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    samples = int(config["radio_map"]["samples_per_tx"])
    rows = int(config["radio_map"]["rows"])
    cols = int(config["radio_map"]["cols"])
    expected_shape = (rows, cols)
    work_root = site_work_root(config, output_root, entry)
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    building_height: np.ndarray | None = None
    outdoor_mask: np.ndarray | None = None
    directional_downtilt_deg = float("nan")
    for variant_index, variant in enumerate(entry["variants"]):
        path = work_root / f"{variant}_samples_{samples}.npz"
        with np.load(path, allow_pickle=False) as data:
            path_gain = data["path_gain_db"].astype(np.float32)
            valid = data["valid_mask"].astype(bool)
            outdoor = data["outdoor_mask"].astype(bool)
            saved_building = data["building_height_m"].astype(np.float32)
            transmitter_xyz = data["transmitter_xyz_m"].astype(np.float32)
            if "directional_downtilt_deg" in data.files:
                directional_downtilt_deg = float(data["directional_downtilt_deg"])
        if path_gain.shape != expected_shape or valid.shape != expected_shape:
            raise ValueError(f"{entry['site_id']} {variant}: invalid array shape")
        if not np.array_equal(np.isfinite(path_gain), valid):
            raise ValueError(f"{entry['site_id']} {variant}: finite-mask mismatch")
        if np.any(valid & ~outdoor) or int(valid.sum()) < int(
            config["sparse_measurements"]["point_count"]
        ):
            raise ValueError(f"{entry['site_id']} {variant}: insufficient valid outdoor cells")
        expected_xyz = np.asarray(
            [*entry["tx_xy_m"], entry["tx_height_m"]], dtype=np.float32
        )
        if not np.allclose(transmitter_xyz, expected_xyz, atol=1e-4):
            raise ValueError(
                f"{entry['site_id']} {variant}: saved transmitter coordinates differ"
            )
        if building_height is None:
            building_height = saved_building
            outdoor_mask = outdoor
        elif not np.array_equal(saved_building, building_height):
            raise ValueError(f"{entry['site_id']} {variant}: building map differs")
        maps.append(path_gain)
        masks.append(valid)
    assert building_height is not None and outdoor_mask is not None
    iso_path_gain = maps[0]
    directional_path_gain = np.stack(maps[1:], axis=0)
    directional_signal = directional_path_gain + float(entry["effective_power_db"])
    directional_valid = np.stack(masks[1:], axis=0)
    point_count = int(config["sparse_measurements"]["point_count"])
    sparse_indices = np.empty((len(entry["azimuths_deg"]), point_count), dtype=np.uint16)
    sparse_signal_db = np.empty_like(sparse_indices, dtype=np.float16)
    reference = config["normalization_reference"]["signal_strength"]
    sparse_signal_norm = np.empty_like(sparse_indices, dtype=np.float16)
    for direction, sparse_seed in enumerate(entry["sparse_seeds"]):
        valid_indices = np.flatnonzero(directional_valid[direction])
        rng = np.random.default_rng(int(sparse_seed))
        selected = np.sort(
            rng.choice(valid_indices, size=point_count, replace=False).astype(np.uint16)
        )
        values = directional_signal[direction].flat[selected]
        sparse_indices[direction] = selected
        sparse_signal_db[direction] = values.astype(np.float16)
        sparse_signal_norm[direction] = (
            (values - float(reference["mean_db"])) / float(reference["std_db"])
        ).astype(np.float16)
    azimuths = np.asarray(entry["azimuths_deg"], dtype=np.float32)
    azimuth_rad = np.deg2rad(azimuths)
    power = config["effective_power"]
    power_mean = float(power["normalization_mean_db"])
    power_std = float(power["normalization_std_db"])
    building_cap = float(config["normalization_reference"]["building_height_cap_m"])
    tx_height_cap = float(config["normalization_reference"]["tx_height_cap_m"])
    arrays = {
        "building_height_m": building_height.astype(np.float16),
        "building_height_norm": (
            np.clip(building_height, 0.0, building_cap) / building_cap
        ).astype(np.float16),
        "outdoor_mask": outdoor_mask,
        "isotropic_path_gain_db": iso_path_gain.astype(np.float16),
        "isotropic_valid_mask": masks[0],
        "directional_signal_strength_db": directional_signal.astype(np.float16),
        "directional_valid_mask": directional_valid,
        "sparse_flat_indices": sparse_indices,
        "sparse_signal_strength_db": sparse_signal_db,
        "sparse_signal_strength_norm_reference": sparse_signal_norm,
        "tx_location_heatmap": transmitter_heatmap(
            entry,
            rows,
            cols,
            float(config["location_sampling"]["heatmap_sigma_px"]),
        ),
        "tx_xy_m": np.asarray(entry["tx_xy_m"], dtype=np.float32),
        "tx_row_col_px": np.asarray(entry["tx_row_col_px"], dtype=np.float32),
        "tx_xy_norm": np.asarray(entry["tx_xy_norm"], dtype=np.float32),
        "tx_row_col_norm": np.asarray(entry["tx_row_col_norm"], dtype=np.float32),
        "tx_height_m": np.asarray(entry["tx_height_m"], dtype=np.float32),
        "tx_height_norm": np.asarray(
            min(float(entry["tx_height_m"]), tx_height_cap) / tx_height_cap,
            dtype=np.float32,
        ),
        "effective_power_db": np.asarray(entry["effective_power_db"], dtype=np.float32),
        "effective_power_norm": np.asarray(
            (float(entry["effective_power_db"]) - power_mean) / power_std,
            dtype=np.float32,
        ),
        "azimuth_deg": azimuths.astype(np.uint16),
        "azimuth_sin_cos": np.stack(
            [np.sin(azimuth_rad), np.cos(azimuth_rad)], axis=-1
        ).astype(np.float32),
        "directional_downtilt_deg": np.asarray(
            directional_downtilt_deg, dtype=np.float32
        ),
        "tile_number": np.asarray(entry["tile_number"], dtype=np.uint32),
        "site_index": np.asarray(entry["site_index"], dtype=np.uint8),
    }
    sample_path, metadata_path, _ = site_paths(output_root, entry)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sample_path.with_suffix(".npz.tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, sample_path)
    metadata = {
        "version": int(config["version"]),
        "site_id": entry["site_id"],
        "tile": entry["tile"],
        "split": entry["split"],
        "site_index": entry["site_index"],
        "quadrant": entry["quadrant"],
        "array_convention": "row 0 north, col 0 west; flat index = row*128+col",
        "coordinate_convention": "local x east, local y north; azimuth 0 east and 90 north",
        "tx_xy_m": entry["tx_xy_m"],
        "tx_row_col_px": entry["tx_row_col_px"],
        "tx_height_m": entry["tx_height_m"],
        "effective_power_db": entry["effective_power_db"],
        "signal_formula": "directional_signal_strength_db = directional_path_gain_db + effective_power_db",
        "azimuths_deg": entry["azimuths_deg"],
        "azimuth_target": "[sin(azimuth), cos(azimuth)]",
        "resolved_directional_downtilt_deg": directional_downtilt_deg,
        "sparse_point_count_per_direction": point_count,
        "sparse_noise_db": 0.0,
        "location_heatmap_sigma_px": float(
            config["location_sampling"]["heatmap_sigma_px"]
        ),
        "normalization_reference": config["normalization_reference"],
        "schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "output": str(sample_path),
        "output_bytes": sample_path.stat().st_size,
        "output_sha256": sha256_file(sample_path),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def cleanup_work(config: dict[str, Any], entry: dict[str, Any], output_root: Path) -> int:
    if bool(config["execution"]["retain_work_files"]):
        return 0
    work_root = site_work_root(config, output_root, entry).resolve()
    expected_parent = Path(config.get("work_root", output_root / "work_sites")).resolve()
    if work_root.parent != expected_parent or not work_root.is_dir():
        return 0
    removed = sum(1 for path in work_root.rglob("*") if path.is_file())
    shutil.rmtree(work_root)
    return removed


def finalize_site(
    config: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
    started: float,
    attempt: int,
    reused_raw: bool,
) -> dict[str, Any]:
    samples = int(config["radio_map"]["samples_per_tx"])
    validation_path = (
        site_work_root(config, output_root, entry)
        / f"validation_samples_{samples}.json"
    )
    validation = load_json(validation_path)
    metadata = validate_and_compact(config, entry, output_root)
    removed = cleanup_work(config, entry, output_root)
    completion = {
        "version": int(config["version"]),
        "status": "ok",
        "site_id": entry["site_id"],
        "tile": entry["tile"],
        "split": entry["split"],
        "site_index": entry["site_index"],
        "attempt": attempt,
        "reused_existing_raw": reused_raw,
        "validation_status": validation.get("status"),
        "validation_warnings": validation.get("warnings", []),
        "elapsed_seconds": time.perf_counter() - started,
        "compact": metadata["output"],
        "compact_sha256": metadata["output_sha256"],
        "removed_generated_work_files": removed,
    }
    _, _, completion_path = site_paths(output_root, entry)
    atomic_write_json(completion_path, completion)
    failure_path = output_root / "failures" / f"{entry['site_id']}.json"
    if failure_path.exists():
        failure_path.unlink()
    return completion


def process_site(
    config: dict[str, Any], entry: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    site_id = entry["site_id"]
    if is_complete(output_root, entry):
        return {"site_id": site_id, "status": "skipped_complete", "elapsed_seconds": 0.0}
    cfg = site_config(config, entry, output_root)
    config_path = output_root / "site_configs" / f"{site_id}.json"
    atomic_write_json(config_path, cfg)
    python = str(config["environment_python"])
    samples = int(config["radio_map"]["samples_per_tx"])
    timeout_seconds = int(config["execution"]["site_timeout_seconds"])
    max_attempts = int(config["execution"]["max_attempts"])
    logs = output_root / "logs" / site_id
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["MPLCONFIGDIR"] = str(output_root / "mplconfig" / "_shared")
    work_root = site_work_root(config, output_root, entry)
    validation_command = [
        python,
        str(SITE_VALIDATOR),
        "--config",
        str(config_path),
        "--samples",
        str(samples),
    ]
    errors: list[str] = []
    raw_outputs_exist = all(
        (work_root / f"{variant}_samples_{samples}.npz").is_file()
        for variant in entry["variants"]
    )
    if raw_outputs_exist:
        code, timeout_error = run_logged(
            validation_command,
            logs / "reuse.validation.stdout.log",
            logs / "reuse.validation.stderr.log",
            timeout_seconds,
            env,
        )
        if timeout_error is None and code == 0:
            try:
                return finalize_site(config, entry, output_root, started, 0, True)
            except Exception as exc:
                errors.append(f"existing raw postprocess failed: {type(exc).__name__}: {exc}")
        else:
            errors.append(timeout_error or f"existing raw validator returned {code}")
    for attempt in range(1, max_attempts + 1):
        command = [
            python,
            str(SITE_RUNNER),
            "--config",
            str(config_path),
            "--samples",
            str(samples),
            "--variants",
            "all",
            "--overwrite",
        ]
        code, timeout_error = run_logged(
            command,
            logs / f"attempt_{attempt}.stdout.log",
            logs / f"attempt_{attempt}.stderr.log",
            timeout_seconds,
            env,
        )
        if timeout_error is not None or code != 0:
            errors.append(timeout_error or f"generator returned {code}")
            continue
        validation_code, validation_timeout = run_logged(
            validation_command,
            logs / f"attempt_{attempt}.validation.stdout.log",
            logs / f"attempt_{attempt}.validation.stderr.log",
            timeout_seconds,
            env,
        )
        if validation_timeout is not None or validation_code != 0:
            errors.append(validation_timeout or f"validator returned {validation_code}")
            continue
        try:
            return finalize_site(config, entry, output_root, started, attempt, False)
        except Exception as exc:
            errors.append(f"postprocess failed: {type(exc).__name__}: {exc}")
    failure = {
        "version": int(config["version"]),
        "status": "failed",
        "site_id": site_id,
        "tile": entry["tile"],
        "split": entry["split"],
        "attempts": max_attempts,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_root / "failures" / f"{site_id}.json", failure)
    return failure


def status_report(
    output_root: Path, entries: list[dict[str, Any]], verify_hash: bool = False
) -> dict[str, Any]:
    complete = {entry["site_id"] for entry in entries if is_complete(output_root, entry, verify_hash)}
    failed = {
        entry["site_id"]
        for entry in entries
        if (output_root / "failures" / f"{entry['site_id']}.json").is_file()
        and entry["site_id"] not in complete
    }
    split_total = {
        split: sum(entry["split"] == split for entry in entries)
        for split in ("train", "val", "test")
    }
    split_complete = {
        split: sum(
            entry["split"] == split and entry["site_id"] in complete
            for entry in entries
        )
        for split in ("train", "val", "test")
    }
    return {
        "status": "complete" if len(complete) == len(entries) else "incomplete",
        "building_count": len({entry["tile"] for entry in entries}),
        "site_count": len(entries),
        "sites_complete": len(complete),
        "sites_remaining": len(entries) - len(complete),
        "sites_failed": len(failed),
        "split_sites_total": split_total,
        "split_sites_complete": split_complete,
        "propagation_maps_complete": len(complete) * 5,
        "directional_training_samples_complete": len(complete) * 4,
        "failed_first": sorted(failed)[:20],
        "stop_requested": (output_root / "STOP").exists(),
    }


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray, positive_only: bool = False) -> None:
        array = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(array)
        if positive_only:
            valid &= array > 0.0
        array = array[valid]
        if not array.size:
            return
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_square += float(np.square(array).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def report(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError("cannot report empty moments")
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def finalize_dataset(
    config: dict[str, Any], entries: list[dict[str, Any]], output_root: Path
) -> dict[str, Any]:
    report = status_report(output_root, entries, verify_hash=True)
    if report["sites_complete"] != report["site_count"]:
        raise RuntimeError("cannot finalize before every Pilot site is complete")
    iso = Moments()
    directional_gain = Moments()
    signal = Moments()
    building = Moments()
    power = Moments()
    for entry in entries:
        if entry["split"] != "train":
            continue
        sample_path, _, _ = site_paths(output_root, entry)
        with np.load(sample_path, allow_pickle=False) as data:
            iso.update(data["isotropic_path_gain_db"])
            signal_values = data["directional_signal_strength_db"].astype(np.float32)
            signal.update(signal_values)
            directional_gain.update(
                signal_values - float(data["effective_power_db"].item())
            )
            building.update(data["building_height_m"], positive_only=True)
            power.update(np.asarray([float(data["effective_power_db"])]))
    normalization = {
        "version": int(config["version"]),
        "source": "completed training split of the random-BS Pilot",
        "building_height_positive_m": building.report(),
        "isotropic_path_gain_db": iso.report(),
        "directional_path_gain_db": directional_gain.report(),
        "directional_signal_strength_db": signal.report(),
        "effective_power_db": power.report(),
        "training_formulas": {
            "building_height_norm": f"clip(height_m,0,{config['normalization_reference']['building_height_cap_m']}) / {config['normalization_reference']['building_height_cap_m']}",
            "signal_strength_norm": "(signal_strength_db - directional_signal_strength_db.mean) / directional_signal_strength_db.std",
            "effective_power_norm": f"(effective_power_db - {config['effective_power']['normalization_mean_db']}) / {config['effective_power']['normalization_std_db']}",
        },
        "reference_used_for_saved_sparse_norm": config["normalization_reference"]["signal_strength"],
    }
    atomic_write_json(output_root / "normalization.json", normalization)
    readme = f"""# 随机基站反演 Pilot v1

状态：完整生成并通过 SHA-256 验证。

- 建筑：384（train/val/test = 256/64/64），按原空间块划分，建筑不会跨集合泄漏。
- 基站站点：1536（每个建筑四象限各 1 个位置）。
- 传播图：7680（每站点 1 张各向同性路径增益 + 4 张方向图）。
- 方向训练样本：6144（train/val/test = 4096/1024/1024）。
- 稀疏点：每张方向图 100 个有效室外点，无噪声、无异常点。

## 目录

- `samples/<split>/<site_id>.npz`：模型所需数组。
- `samples/<split>/<site_id>.json`：字段、单位、参数和文件哈希。
- `completion/`：逐站点完成记录，支持断点续跑。
- `run_manifest.json`：全部随机位置、功率、方向和随机种子。
- `site_index.csv`：便于表格查看的站点索引。
- `normalization.json`：仅由训练集计算的 Pilot 归一化统计。
- `VALIDATION_REPORT.json`：独立全量样本验证结果。

## 单个 NPZ 的主要字段

- `building_height_m`, `building_height_norm`：`[128,128]` 建筑输入。
- `isotropic_path_gain_db`：`[128,128]`，供现有 Stage2-A/Stage2-B 接口复用。
- `directional_signal_strength_db`：`[4,128,128]` 完整信号图真值。
- `sparse_flat_indices`, `sparse_signal_strength_db`：`[4,100]` 稀疏测点；在线用 `row=index//128, col=index%128` 还原掩码。
- `tx_location_heatmap`：`[128,128]`，高斯标准差 1.5 像素。
- `tx_xy_m`, `tx_row_col_px`, `tx_height_m`：真实位置和高度。
- `effective_power_db`：可辨识的合并链路预算偏移。
- `azimuth_deg`, `azimuth_sin_cos`：4 个真实方向标签，后者顺序为 `[sin, cos]`。

方向信号图满足：

`directional_signal_strength_db = directional_path_gain_db + effective_power_db`

因此未重复存储方向路径增益，需要时直接相减恢复。

## 归一化选择

重新训练 Pilot 模型时，从原始 dB 字段按 `normalization.json` 处理：

`signal_norm = (signal_db - {normalization['directional_signal_strength_db']['mean']:.12f}) / {normalization['directional_signal_strength_db']['std']:.12f}`

`sparse_signal_strength_norm_reference` 使用旧 Stage2-B 的归一化（mean=-57.86771677215448 dB，std=19.549090075885392 dB），只用于直接兼容现有 Stage2-B。若重新训练，应从 `sparse_signal_strength_db` 按 Pilot 统计在线归一化。

坐标约定：局部 x 向东、y 向北；数组 row=0 为北、col=0 为西；方位角 0° 向东、90° 向北。
"""
    readme_path = output_root / "DATASET_README.md"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=output_root
    ) as handle:
        handle.write(readme)
        temporary_readme = Path(handle.name)
    os.replace(temporary_readme, readme_path)
    final = {
        **report,
        "status": "complete_and_verified",
        "normalization": str(output_root / "normalization.json"),
        "dataset_readme": str(readme_path),
        "completed_at_unix": time.time(),
    }
    atomic_write_json(output_root / "FINAL_REPORT.json", final)
    return final


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output_root = Path(config["output_root"])
    rows = read_selection(Path(config["selection_csv"]))
    buildings = selected_buildings(rows, config)
    entries = build_entries(config, buildings)
    manifest = build_manifest(config, config_path, entries)
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_manifest(output_root, manifest)
    config = {**config, "dataset_signature_sha256": manifest["dataset_signature_sha256"]}
    write_indexes(output_root, buildings, entries)
    by_id = {entry["site_id"]: entry for entry in entries}
    if args.site:
        missing = [site for site in args.site if site not in by_id]
        if missing:
            raise ValueError(f"unknown site(s): {missing}")
        selected = [by_id[site] for site in args.site]
    else:
        selected = entries
    if args.limit_sites is not None:
        if args.limit_sites < 1:
            raise ValueError("--limit-sites must be positive")
        selected = selected[: args.limit_sites]
    if args.dry_run:
        payload = {
            "status": "dry_run",
            "output_root": str(output_root),
            "building_count": len(buildings),
            "site_count": len(entries),
            "selected_sites": len(selected),
            "propagation_maps": len(entries) * 5,
            "directional_training_samples": len(entries) * 4,
            "first_sites": [entry["site_id"] for entry in selected[:8]],
            "dataset_signature_sha256": manifest["dataset_signature_sha256"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.status:
        print(json.dumps(status_report(output_root, entries), ensure_ascii=False, indent=2))
        return 0
    if args.finalize_only:
        print(json.dumps(finalize_dataset(config, entries, output_root), ensure_ascii=False, indent=2))
        return 0
    stop_path = output_root / "STOP"
    if stop_path.exists():
        raise RuntimeError(f"stop flag exists; remove it before resuming: {stop_path}")
    workers = int(args.workers or config["execution"]["max_parallel_sites"])
    hard_limit = int(config["execution"]["max_parallel_sites_hard_limit"])
    if not 1 <= workers <= hard_limit:
        raise ValueError(f"workers must be in [1,{hard_limit}]")
    pending = deque(entry for entry in selected if not is_complete(output_root, entry))
    initial = status_report(output_root, entries)
    print(json.dumps({**initial, "selected_pending": len(pending)}, ensure_ascii=False), flush=True)
    active: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        while pending or active:
            while pending and len(active) < workers and not stop_path.exists():
                entry = pending.popleft()
                future = executor.submit(process_site, config, entry, output_root)
                active[future] = entry
                print(f"START {entry['site_id']} split={entry['split']}", flush=True)
            if not active:
                break
            done, _ = concurrent.futures.wait(
                active, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                entry = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "site_id": entry["site_id"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                print(
                    f"DONE {entry['site_id']} status={result['status']} "
                    f"seconds={float(result.get('elapsed_seconds', 0.0)):.2f}",
                    flush=True,
                )
                progress = status_report(output_root, entries)
                progress["last_site"] = entry["site_id"]
                progress["last_result"] = result["status"]
                atomic_write_json(output_root / "progress.json", progress)
    except KeyboardInterrupt:
        atomic_write_json(
            output_root / "interrupt.json",
            {
                "status": "interrupted",
                "time_unix": time.time(),
                "message": "completed sites are safe; rerun to resume",
            },
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    final = status_report(output_root, entries)
    if final["sites_complete"] == final["site_count"]:
        final = finalize_dataset(config, entries, output_root)
    elif stop_path.exists():
        final["status"] = "stopped"
    elif all(is_complete(output_root, entry) for entry in selected):
        final["status"] = "selected_sites_complete"
        final["selected_site_count"] = len(selected)
    else:
        final["status"] = "completed_with_failures_or_partial_selection"
    atomic_write_json(output_root / "progress.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0 if final["status"] in {
        "complete_and_verified",
        "selected_sites_complete",
        "stopped",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())

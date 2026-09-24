#!/usr/bin/env python3
"""Independently validate every compact random-BS dataset sample."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_sionna_dataset import atomic_write_json, load_json, sha256_file


EXPECTED_FIELDS: dict[str, tuple[tuple[int, ...], str]] = {
    "building_height_m": ((128, 128), "float16"),
    "building_height_norm": ((128, 128), "float16"),
    "outdoor_mask": ((128, 128), "bool"),
    "isotropic_path_gain_db": ((128, 128), "float16"),
    "isotropic_valid_mask": ((128, 128), "bool"),
    "directional_signal_strength_db": ((4, 128, 128), "float16"),
    "directional_valid_mask": ((4, 128, 128), "bool"),
    "sparse_flat_indices": ((4, 100), "uint16"),
    "sparse_signal_strength_db": ((4, 100), "float16"),
    "sparse_signal_strength_norm_reference": ((4, 100), "float16"),
    "tx_location_heatmap": ((128, 128), "float16"),
    "tx_xy_m": ((2,), "float32"),
    "tx_row_col_px": ((2,), "float32"),
    "tx_xy_norm": ((2,), "float32"),
    "tx_row_col_norm": ((2,), "float32"),
    "tx_height_m": ((), "float32"),
    "tx_height_norm": ((), "float32"),
    "effective_power_db": ((), "float32"),
    "effective_power_norm": ((), "float32"),
    "azimuth_deg": ((4,), "uint16"),
    "azimuth_sin_cos": ((4, 2), "float32"),
    "directional_downtilt_deg": ((), "float32"),
    "tile_number": ((), "uint32"),
    "site_index": ((), "uint8"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def validate_manifest(
    entries: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    split_building_counts = {
        str(split): int(count)
        for split, count in config["pilot_split_counts"].items()
    }
    sites_per_building = int(config["sites_per_building"])
    expected_site_count = sum(split_building_counts.values()) * sites_per_building
    if (
        len(entries) != expected_site_count
        or len({entry["site_id"] for entry in entries}) != expected_site_count
    ):
        raise ValueError(
            f"manifest must contain {expected_site_count} unique sites"
        )
    split_counts = Counter(entry["split"] for entry in entries)
    expected_splits = Counter(
        {
            split: building_count * sites_per_building
            for split, building_count in split_building_counts.items()
        }
    )
    if split_counts != expected_splits:
        raise ValueError(f"unexpected split counts: {split_counts}")
    sampling = config["location_sampling"]
    margin_m = float(sampling["margin_m"])
    map_width_m, map_height_m = map(float, config["radio_map"]["area_size_m"])
    power_min_db, power_max_db = map(float, config["effective_power"]["range_db"])
    direction_count = int(config["radio_map"]["directional_labels_per_site"])
    strategy = str(sampling.get("strategy", "quadrants"))
    radial_bins = [tuple(map(float, bounds)) for bounds in sampling.get("center_radius_bins_m", [])]
    if strategy == "radial_bins":
        if len(radial_bins) != sites_per_building:
            raise ValueError("radial bin count must equal sites_per_building")
        required_allocations = {
            f"radius_{lower:03.0f}_{upper:03.0f}m"
            for lower, upper in radial_bins
        }
    else:
        required_allocations = {"southwest", "southeast", "northwest", "northeast"}
        if sites_per_building != len(required_allocations):
            raise ValueError("quadrant sampling requires four sites per building")
    by_tile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_tile[entry["tile"]].append(entry)
        x_m, y_m = map(float, entry["tx_xy_m"])
        if not (margin_m <= x_m <= map_width_m - margin_m):
            raise ValueError(f"{entry['site_id']}: transmitter outside x margin")
        if not (margin_m <= y_m <= map_height_m - margin_m):
            raise ValueError(f"{entry['site_id']}: transmitter outside the margin")
        if not power_min_db <= float(entry["effective_power_db"]) <= power_max_db:
            raise ValueError(f"{entry['site_id']}: effective power outside range")
        if len(entry["azimuths_deg"]) != direction_count:
            raise ValueError(f"{entry['site_id']}: unexpected direction count")
        if len(set(entry["azimuths_deg"])) != direction_count:
            raise ValueError(f"{entry['site_id']}: azimuths are not distinct")
        if strategy == "radial_bins":
            site_index = int(entry["site_index"])
            if not 0 <= site_index < len(radial_bins):
                raise ValueError(f"{entry['site_id']}: invalid radial bin index")
            lower, upper = radial_bins[site_index]
            radius_m = math.dist((x_m, y_m), (map_width_m / 2.0, map_height_m / 2.0))
            if not lower <= radius_m <= upper:
                raise ValueError(
                    f"{entry['site_id']}: radius {radius_m} outside [{lower}, {upper}]"
                )
    minimum_separation = math.inf
    for tile, group in by_tile.items():
        if len(group) != sites_per_building:
            raise ValueError(f"{tile}: invalid site count")
        if {int(entry["site_index"]) for entry in group} != set(range(sites_per_building)):
            raise ValueError(f"{tile}: invalid site-index allocation")
        if {entry["quadrant"] for entry in group} != required_allocations:
            raise ValueError(f"{tile}: invalid location allocation")
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                minimum_separation = min(
                    minimum_separation,
                    math.dist(left["tx_xy_m"], right["tx_xy_m"]),
                )
    required_separation = float(sampling["minimum_pairwise_separation_m"])
    if minimum_separation < required_separation:
        raise ValueError(f"minimum site separation is only {minimum_separation} m")
    return {
        "building_count": len(by_tile),
        "site_count": len(entries),
        "split_site_counts": dict(split_counts),
        "location_sampling_strategy": strategy,
        "location_allocations": sorted(required_allocations),
        "minimum_pairwise_site_separation_m": minimum_separation,
    }


def validate_sample(
    root: Path,
    entry: dict[str, Any],
    config: dict[str, Any],
    verify_hash: bool,
) -> int:
    site_id = entry["site_id"]
    sample = root / "samples" / entry["split"] / f"{site_id}.npz"
    metadata_path = sample.with_suffix(".json")
    completion_path = root / "completion" / f"{site_id}.json"
    metadata = load_json(metadata_path)
    completion = load_json(completion_path)
    if completion.get("status") != "ok":
        raise ValueError(f"{site_id}: completion status is not ok")
    if sample.stat().st_size != int(metadata["output_bytes"]):
        raise ValueError(f"{site_id}: output size differs from metadata")
    if verify_hash:
        digest = sha256_file(sample)
        if digest != metadata["output_sha256"] or digest != completion["compact_sha256"]:
            raise ValueError(f"{site_id}: SHA-256 mismatch")
    with np.load(sample, allow_pickle=False) as data:
        if set(data.files) != set(EXPECTED_FIELDS):
            raise ValueError(f"{site_id}: unexpected NPZ fields")
        for field, (shape, dtype) in EXPECTED_FIELDS.items():
            array = data[field]
            if array.shape != shape or str(array.dtype) != dtype:
                raise ValueError(
                    f"{site_id} {field}: got {array.shape}/{array.dtype}, "
                    f"expected {shape}/{dtype}"
                )
        building = data["building_height_m"].astype(np.float32)
        building_cap = float(config["normalization_reference"]["building_height_cap_m"])
        expected_building_norm = np.clip(building, 0.0, building_cap) / building_cap
        if not np.allclose(
            data["building_height_norm"], expected_building_norm, atol=5e-4
        ):
            raise ValueError(f"{site_id}: building normalization mismatch")
        outdoor = data["outdoor_mask"]
        if not np.array_equal(outdoor, building <= 0.0):
            raise ValueError(f"{site_id}: outdoor mask mismatch")
        iso = data["isotropic_path_gain_db"]
        iso_valid = data["isotropic_valid_mask"]
        directional = data["directional_signal_strength_db"]
        directional_valid = data["directional_valid_mask"]
        if not np.array_equal(np.isfinite(iso), iso_valid):
            raise ValueError(f"{site_id}: isotropic finite mask mismatch")
        if not np.array_equal(np.isfinite(directional), directional_valid):
            raise ValueError(f"{site_id}: directional finite mask mismatch")
        if np.any(iso_valid & ~outdoor) or np.any(directional_valid & ~outdoor[None]):
            raise ValueError(f"{site_id}: indoor pixels marked valid")
        indices = data["sparse_flat_indices"]
        sparse = data["sparse_signal_strength_db"]
        reference = config["normalization_reference"]["signal_strength"]
        for direction in range(4):
            if len(np.unique(indices[direction])) != 100:
                raise ValueError(f"{site_id}: duplicate sparse indices")
            if not np.all(directional_valid[direction].flat[indices[direction]]):
                raise ValueError(f"{site_id}: sparse point outside valid mask")
            if not np.array_equal(
                directional[direction].flat[indices[direction]], sparse[direction]
            ):
                raise ValueError(f"{site_id}: sparse dB value mismatch")
        expected_sparse_norm = (
            (sparse.astype(np.float32) - float(reference["mean_db"]))
            / float(reference["std_db"])
        )
        if not np.allclose(
            data["sparse_signal_strength_norm_reference"],
            expected_sparse_norm,
            # Both the saved dB samples and the saved normalized samples are
            # float16, and normalization was computed before either value was
            # quantized. The combined worst-case round-off is below 0.006.
            atol=6e-3,
        ):
            raise ValueError(f"{site_id}: sparse reference normalization mismatch")
        if not np.allclose(data["tx_xy_m"], entry["tx_xy_m"], atol=1e-4):
            raise ValueError(f"{site_id}: transmitter coordinates mismatch")
        if not np.allclose(
            data["tx_row_col_px"], entry["tx_row_col_px"], atol=1e-4
        ):
            raise ValueError(f"{site_id}: transmitter pixel coordinates mismatch")
        row, col = data["tx_row_col_px"].astype(float)
        heatmap = data["tx_location_heatmap"]
        nearest_row = int(np.rint(row))
        nearest_col = int(np.rint(col))
        # A transmitter extremely close to a half-pixel boundary can create a
        # two-cell maximum plateau after float16 quantization. The nearest grid
        # cell must be one of the maxima; np.argmax alone would pick the first.
        if heatmap[nearest_row, nearest_col] != np.max(heatmap):
            raise ValueError(f"{site_id}: heatmap peak does not match transmitter")
        if not math.isclose(
            float(data["tx_height_m"]), float(entry["tx_height_m"]), abs_tol=1e-4
        ):
            raise ValueError(f"{site_id}: transmitter height mismatch")
        if not math.isclose(
            float(data["effective_power_db"]),
            float(entry["effective_power_db"]),
            abs_tol=1e-4,
        ):
            raise ValueError(f"{site_id}: effective power mismatch")
        if not np.array_equal(data["azimuth_deg"], entry["azimuths_deg"]):
            raise ValueError(f"{site_id}: azimuth mismatch")
        radians = np.deg2rad(data["azimuth_deg"].astype(np.float32))
        expected_sin_cos = np.stack([np.sin(radians), np.cos(radians)], axis=-1)
        if not np.allclose(data["azimuth_sin_cos"], expected_sin_cos, atol=1e-6):
            raise ValueError(f"{site_id}: azimuth sin/cos mismatch")
    return sample.stat().st_size


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    root = Path(config["output_root"])
    manifest = load_json(root / "run_manifest.json")
    entries = manifest["entries"]
    report: dict[str, Any] = {
        "version": 1,
        "status": "running",
        "manifest": validate_manifest(entries, config),
        "verify_hashes": not args.skip_hashes,
        "samples_checked": 0,
        "sample_bytes": 0,
        "failures": [],
    }
    started = time.perf_counter()
    for entry in entries:
        try:
            report["sample_bytes"] += validate_sample(
                root, entry, config, not args.skip_hashes
            )
            report["samples_checked"] += 1
        except Exception as exc:
            report["failures"].append(
                f"{entry['site_id']}: {type(exc).__name__}: {exc}"
            )
            if len(report["failures"]) >= 20:
                break
    report["status"] = "ok" if not report["failures"] else "failed"
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_write_json(root / "VALIDATION_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

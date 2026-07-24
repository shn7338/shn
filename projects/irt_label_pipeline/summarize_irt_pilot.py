#!/usr/bin/env python3
"""Audit a completed direct-WinProp pilot and write its quality report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    root = Path(config["output_root"])
    pilot_manifest = load_json(root / "pilot_manifest.json")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    elapsed: list[float] = []
    compact_bytes = 0
    all_values_min: list[float] = []
    all_values_max: list[float] = []
    resampled_variants = 0
    maximum_resampling_offset_m = 0.0
    maximum_directional_gain_db = -float("inf")
    maximum_directional_gain_99pct_db = -float("inf")
    directional_gain_outlier_pixels = 0
    directional_gain_common_pixels = 0
    for entry in pilot_manifest["tiles"]:
        tile = entry["tile"]
        variants = entry["variants"]
        completion_path = root / "completion" / f"{tile}.json"
        compact_path = root / "compact_tiles" / f"{tile}.npz"
        compact_metadata_path = compact_path.with_suffix(".json")
        if not (
            completion_path.is_file()
            and compact_path.is_file()
            and compact_metadata_path.is_file()
        ):
            failures.append(f"{tile}: missing completion or compact artifact")
            continue
        completion = load_json(completion_path)
        metadata = load_json(compact_metadata_path)
        if completion.get("status") != "ok":
            failures.append(f"{tile}: completion status is not ok")
            continue
        if metadata.get("variants") != variants:
            failures.append(f"{tile}: compact variant order mismatch")
            continue
        with np.load(compact_path, allow_pickle=False) as compact:
            if set(compact.files) != set(variants):
                failures.append(f"{tile}: compact keys mismatch")
                continue
            arrays = {
                name: np.asarray(compact[name], dtype=np.float32)
                for name in variants
            }
        if any(array.shape != (128, 128) for array in arrays.values()):
            failures.append(f"{tile}: array shape mismatch")
            continue
        iso_mask = np.isfinite(arrays["iso"])
        finite_counts = {
            name: int(np.count_nonzero(np.isfinite(array)))
            for name, array in arrays.items()
        }
        mismatches = {
            name: int(np.count_nonzero(np.isfinite(arrays[name]) != iso_mask))
            for name in variants[1:]
        }
        directional_gain_maxima: dict[str, float] = {}
        directional_gain_99pct: dict[str, float] = {}
        for name in variants[1:]:
            common_mask = np.isfinite(arrays[name]) & iso_mask
            if not np.any(common_mask):
                failures.append(f"{tile}: {name} has no common valid iso pixels")
                directional_gain_maxima = {}
                break
            differences = (
                arrays[name][common_mask] - arrays["iso"][common_mask]
            )
            directional_gain_maxima[name] = float(np.max(differences))
            directional_gain_99pct[name] = float(
                np.percentile(differences, 99.0)
            )
            directional_gain_outlier_pixels += int(
                np.count_nonzero(differences > 6.5)
            )
            directional_gain_common_pixels += int(differences.size)
        if not directional_gain_maxima:
            continue
        tile_resampled = 0
        tile_maximum_offset = 0.0
        for source in metadata["sources"].values():
            resampling = source.get("resampling")
            if resampling and resampling.get("applied"):
                tile_resampled += 1
                offset = resampling[
                    "target_minus_source_first_center_m"
                ]
                tile_maximum_offset = max(
                    tile_maximum_offset,
                    max(abs(float(value)) for value in offset),
                )
        resampled_variants += tile_resampled
        maximum_resampling_offset_m = max(
            maximum_resampling_offset_m,
            tile_maximum_offset,
        )
        maximum_directional_gain_db = max(
            maximum_directional_gain_db,
            max(directional_gain_maxima.values()),
        )
        maximum_directional_gain_99pct_db = max(
            maximum_directional_gain_99pct_db,
            max(directional_gain_99pct.values()),
        )
        if tile_maximum_offset >= float(config["grid"]["resolution_m"]):
            failures.append(
                f"{tile}: coordinate resampling offset is at least one cell"
            )
            continue
        finite_values = np.concatenate(
            [array[np.isfinite(array)] for array in arrays.values()]
        )
        if not finite_values.size:
            failures.append(f"{tile}: no finite label values")
            continue
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        if minimum < -300.0 or maximum > 50.0:
            failures.append(
                f"{tile}: implausible label range [{minimum}, {maximum}]"
            )
            continue
        seconds = float(completion["elapsed_seconds"])
        elapsed.append(seconds)
        compact_bytes += compact_path.stat().st_size
        all_values_min.append(minimum)
        all_values_max.append(maximum)
        rows.append(
            {
                "tile": tile,
                "variants": " ".join(variants),
                "elapsed_seconds": f"{seconds:.3f}",
                "iso_valid_pixels": finite_counts["iso"],
                "directional_valid_pixels_min": min(
                    finite_counts[name] for name in variants[1:]
                ),
                "directional_valid_pixels_max": max(
                    finite_counts[name] for name in variants[1:]
                ),
                "mask_mismatch_vs_iso_max": max(mismatches.values()),
                "directional_gain_max_db": (
                    f"{max(directional_gain_maxima.values()):.6f}"
                ),
                "directional_gain_99pct_max_db": (
                    f"{max(directional_gain_99pct.values()):.6f}"
                ),
                "resampled_variants": tile_resampled,
                "maximum_resampling_offset_m": (
                    f"{tile_maximum_offset:.6f}"
                ),
                "minimum_db": f"{minimum:.6f}",
                "maximum_db": f"{maximum:.6f}",
                "compact_bytes": compact_path.stat().st_size,
            }
        )

    expected = len(pilot_manifest["tiles"])
    checks = {
        "all_tiles_completed": len(rows) == expected and not failures,
        "five_variants_per_tile": all(
            len(entry["variants"]) == 5
            and entry["variants"][0] == "iso"
            and len(set(entry["variants"])) == 5
            for entry in pilot_manifest["tiles"]
        ),
        "compact_float16_raw_db": all(
            load_json(
                root / "compact_tiles" / f"{entry['tile']}.json"
            ).get("dtype")
            == "float16"
            for entry in pilot_manifest["tiles"]
            if (
                root / "compact_tiles" / f"{entry['tile']}.json"
            ).is_file()
        ),
        "formal_quantity_is_path_loss": all(
            load_json(
                root / "compact_tiles" / f"{entry['tile']}.json"
            ).get("quantity")
            == "path_loss"
            for entry in pilot_manifest["tiles"]
            if (
                root / "compact_tiles" / f"{entry['tile']}.json"
            ).is_file()
        ),
        "directional_gain_99pct_not_above_6p5_db": (
            maximum_directional_gain_99pct_db <= 6.5
        ),
        "directional_gain_outlier_fraction_below_0p5_percent": (
            directional_gain_common_pixels > 0
            and directional_gain_outlier_pixels
            / directional_gain_common_pixels
            < 0.005
        ),
        "resampling_offset_below_one_cell": (
            maximum_resampling_offset_m
            < float(config["grid"]["resolution_m"])
        ),
    }
    passed = all(checks.values())
    report = {
        "version": config["version"],
        "name": config["name"],
        "method": "direct WinProp; iso plus four directional runs per tile",
        "physics": config["physics"],
        "directional_antenna": config["directional_antenna"],
        "tiles_expected": expected,
        "tiles_completed": len(rows),
        "failures": failures,
        "timing_seconds": {
            "total": float(np.sum(elapsed)) if elapsed else None,
            "mean": float(np.mean(elapsed)) if elapsed else None,
            "median": float(np.median(elapsed)) if elapsed else None,
            "minimum": float(np.min(elapsed)) if elapsed else None,
            "maximum": float(np.max(elapsed)) if elapsed else None,
        },
        "storage_bytes": {
            "compact_npz": compact_bytes,
            "oib_keep": directory_bytes(root / "oib_keep"),
            "raw_results": directory_bytes(root / "raw_results"),
            "projects": directory_bytes(root / "projects"),
        },
        "label_range_db": {
            "minimum": min(all_values_min) if all_values_min else None,
            "maximum": max(all_values_max) if all_values_max else None,
        },
        "coordinate_resampling": {
            "variants_resampled": resampled_variants,
            "maximum_first_center_offset_m": maximum_resampling_offset_m,
            "maximum_allowed_offset_m": float(
                config["grid"]["resolution_m"]
            ),
        },
        "maximum_directional_gain_vs_isotropic_db": (
            maximum_directional_gain_db
            if maximum_directional_gain_db != -float("inf")
            else None
        ),
        "maximum_directional_gain_99pct_vs_isotropic_db": (
            maximum_directional_gain_99pct_db
            if maximum_directional_gain_99pct_db != -float("inf")
            else None
        ),
        "directional_gain_above_6p5_db": {
            "pixels": directional_gain_outlier_pixels,
            "common_pixels": directional_gain_common_pixels,
            "fraction": (
                directional_gain_outlier_pixels
                / directional_gain_common_pixels
                if directional_gain_common_pixels
                else None
            ),
            "interpretation": (
                "Rare direct-WinProp nonlinear path-selection outliers; "
                "tile_000007 persisted with RAYS_MAX_NUMBER disabled"
            ),
        },
        "checks": checks,
        "passed": passed,
        "tile_report_csv": str(root / "pilot_quality_tiles.csv"),
    }
    csv_path = root / "pilot_quality_tiles.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    if rows:
        with csv_temporary.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(csv_temporary, csv_path)
    atomic_write_json(root / "pilot_quality_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

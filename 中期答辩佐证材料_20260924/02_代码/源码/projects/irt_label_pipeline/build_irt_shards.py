#!/usr/bin/env python3
"""Pack completed path-gain tiles into memory-mappable NPY shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every tile in the run manifest to be complete.",
    )
    parser.add_argument(
        "--shards-root",
        type=Path,
        help="Override the default <output_root>/shards directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Override the default <output_root>/shard_manifest.csv.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Override the default <output_root>/shard_metadata.json.",
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_azimuth(variant: str) -> int:
    match = re.fullmatch(r"az(\d{3})", variant)
    if match is None:
        raise ValueError(f"invalid directional variant: {variant!r}")
    value = int(match.group(1))
    if value >= 360:
        raise ValueError(f"azimuth outside [0, 359]: {variant!r}")
    return value


def atomic_memmap(
    path: Path,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int, ...],
) -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace shard: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary shard already exists: {temporary}")
    array = open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    return array, temporary


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    output_root = Path(config["output_root"])
    pilot_manifest_path = output_root / config.get(
        "run_manifest_filename",
        config.get("storage", {}).get(
            "run_manifest_filename", "pilot_manifest.json"
        ),
    )
    if not pilot_manifest_path.is_file():
        raise FileNotFoundError(pilot_manifest_path)
    pilot_manifest = load_json(pilot_manifest_path)
    entries = pilot_manifest["tiles"]
    shard_size = args.shard_size or int(config["storage"]["shard_size_tiles"])
    if shard_size < 1:
        raise ValueError("shard size must be positive")

    ready: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in entries:
        tile = entry["tile"]
        completion_path = output_root / "completion" / f"{tile}.json"
        compact_path = output_root / "compact_tiles" / f"{tile}.npz"
        compact_metadata_path = compact_path.with_suffix(".json")
        completion = (
            load_json(completion_path) if completion_path.is_file() else {}
        )
        if not (
            completion_path.is_file()
            and compact_path.is_file()
            and compact_metadata_path.is_file()
            and completion.get("status") == "ok"
        ):
            missing.append(tile)
            continue
        compact_metadata = load_json(compact_metadata_path)
        if compact_metadata.get("variants") != entry["variants"]:
            raise RuntimeError(
                f"compact variant order differs from pilot manifest for {tile}"
            )
        ready.append(
            {
                **entry,
                "compact_path": compact_path,
                "compact_metadata": compact_metadata,
                "completion": completion,
            }
        )
    if missing and args.require_complete:
        raise RuntimeError(
            f"{len(missing)} tile(s) are incomplete; first missing: {missing[:8]}"
        )
    if not ready:
        raise RuntimeError("no completed tiles are available for sharding")

    shards_root = args.shards_root or (output_root / "shards")
    manifest_path = args.manifest or (
        output_root / "shard_manifest.csv"
    )
    metadata_path = args.metadata or (
        output_root / "shard_metadata.json"
    )
    if manifest_path.exists() or metadata_path.exists() or shards_root.exists():
        raise FileExistsError(
            "shard outputs already exist; refusing to replace the current dataset"
        )
    shards_root.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    shard_metadata: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(ready), shard_size)):
        shard_entries = ready[start : start + shard_size]
        count = len(shard_entries)
        stem = f"{shard_index:04d}"
        paths = {
            "p_iso": shards_root / f"p_iso_{stem}.npy",
            "p_dir": shards_root / f"p_dir_{stem}.npy",
            "azimuth_deg": shards_root / f"azimuth_deg_{stem}.npy",
            "tile_number": shards_root / f"tile_number_{stem}.npy",
        }
        p_iso, temp_iso = atomic_memmap(
            paths["p_iso"], np.float16, (count, 128, 128)
        )
        p_dir, temp_dir = atomic_memmap(
            paths["p_dir"], np.float16, (count, 4, 128, 128)
        )
        azimuth_deg, temp_azimuth = atomic_memmap(
            paths["azimuth_deg"], np.uint16, (count, 4)
        )
        tile_number, temp_tile = atomic_memmap(
            paths["tile_number"], np.uint32, (count,)
        )
        valid_counts: dict[str, list[int]] = {"p_iso": [], "p_dir": []}
        minima: list[float] = []
        maxima: list[float] = []
        for offset, entry in enumerate(shard_entries):
            tile = entry["tile"]
            variants = entry["variants"]
            if variants[0] != "iso" or len(variants) != 5:
                raise ValueError(
                    f"{tile} must have iso plus four directional variants"
                )
            with np.load(entry["compact_path"], allow_pickle=False) as compact:
                if set(compact.files) != set(variants):
                    raise ValueError(
                        f"{tile} compact keys differ from planned variants"
                    )
                iso = np.asarray(compact["iso"], dtype=np.float16)
                directions = np.stack(
                    [
                        np.asarray(compact[variant], dtype=np.float16)
                        for variant in variants[1:]
                    ],
                    axis=0,
                )
            if iso.shape != (128, 128) or directions.shape != (4, 128, 128):
                raise ValueError(
                    f"unexpected compact shape for {tile}: "
                    f"iso={iso.shape}, directions={directions.shape}"
                )
            p_iso[offset] = iso
            p_dir[offset] = directions
            azimuths = [parse_azimuth(value) for value in variants[1:]]
            azimuth_deg[offset] = azimuths
            number = int(tile.split("_")[1])
            tile_number[offset] = number
            iso_valid = int(np.count_nonzero(np.isfinite(iso)))
            direction_valid = int(np.count_nonzero(np.isfinite(directions)))
            valid_counts["p_iso"].append(iso_valid)
            valid_counts["p_dir"].append(direction_valid)
            finite_values = np.concatenate(
                (iso[np.isfinite(iso)], directions[np.isfinite(directions)])
            )
            minima.append(float(np.min(finite_values)))
            maxima.append(float(np.max(finite_values)))
            manifest_rows.append(
                {
                    "tile": tile,
                    "tile_number": number,
                    "shard": shard_index,
                    "offset": offset,
                    "azimuth_0_deg": azimuths[0],
                    "azimuth_1_deg": azimuths[1],
                    "azimuth_2_deg": azimuths[2],
                    "azimuth_3_deg": azimuths[3],
                    "directional_downtilt_deg": entry[
                        "compact_metadata"
                    ].get("resolved_directional_downtilt_deg"),
                    "validation_status": entry["completion"].get(
                        "validation_status", "ok_legacy_strict"
                    ),
                    "validation_warning_count": len(
                        entry["completion"].get("validation_warnings", [])
                    ),
                    "reused_existing_raw": bool(
                        entry["completion"].get("reused_existing_raw", False)
                    ),
                    "iso_valid_pixels": iso_valid,
                    "directional_valid_pixels": direction_valid,
                    "source_npz_sha256": entry["compact_metadata"][
                        "output_sha256"
                    ],
                }
            )
        for array in (p_iso, p_dir, azimuth_deg, tile_number):
            array.flush()
        # On Windows the loop variable retains the final memmap and keeps its
        # file handle open, which prevents the following atomic rename.
        del array
        del p_iso, p_dir, azimuth_deg, tile_number
        temporary_paths = {
            "p_iso": temp_iso,
            "p_dir": temp_dir,
            "azimuth_deg": temp_azimuth,
            "tile_number": temp_tile,
        }
        for name, temporary in temporary_paths.items():
            os.replace(temporary, paths[name])
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        shard_metadata.append(
            {
                "shard": shard_index,
                "tiles": count,
                "first_tile": shard_entries[0]["tile"],
                "last_tile": shard_entries[-1]["tile"],
                "paths": {name: str(path) for name, path in paths.items()},
                "hashes_sha256": hashes,
                "valid_counts": {
                    "p_iso_min": min(valid_counts["p_iso"]),
                    "p_iso_max": max(valid_counts["p_iso"]),
                    "p_dir_min": min(valid_counts["p_dir"]),
                    "p_dir_max": max(valid_counts["p_dir"]),
                },
                "minimum_db": min(minima),
                "maximum_db": max(maxima),
            }
        )
        print(
            f"SHARD {shard_index + 1}: {count} tiles "
            f"({shard_entries[0]['tile']}..{shard_entries[-1]['tile']})",
            flush=True,
        )

    manifest_temporary = manifest_path.with_suffix(".csv.tmp")
    with manifest_temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    os.replace(manifest_temporary, manifest_path)
    storage = config.get("storage", {})
    metadata = {
        "version": config["version"],
        "method": storage.get("dataset_method", "direct WinProp IRT"),
        "quantity": storage.get(
            "quantity_description",
            (
                "negative-dB path gain values exported by WinProp in "
                "Antenna Path Loss.txt; raw float16 with NaN invalids"
            ),
        ),
        "array_convention": "row 0 north; column 0 west",
        "tiles_planned": len(entries),
        "tiles_packed": len(ready),
        "tiles_missing": missing,
        "shard_size": shard_size,
        "shards": shard_metadata,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

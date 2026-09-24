#!/usr/bin/env python3
"""Verify hashes, shapes, dtypes and manifest indices of IRT NPY shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED = {
    "p_iso": (np.dtype(np.float16), lambda count: (count, 128, 128)),
    "p_dir": (np.dtype(np.float16), lambda count: (count, 4, 128, 128)),
    "azimuth_deg": (np.dtype(np.uint16), lambda count: (count, 4)),
    "tile_number": (np.dtype(np.uint32), lambda count: (count,)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    manifest_path = Path(metadata["manifest"])
    if sha256_file(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("shard manifest hash mismatch")
    with manifest_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    rows_by_shard: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_shard.setdefault(int(row["shard"]), []).append(row)

    checked_files = 0
    checked_tiles = 0
    for shard in metadata["shards"]:
        index = int(shard["shard"])
        count = int(shard["tiles"])
        shard_rows = rows_by_shard.get(index, [])
        if len(shard_rows) != count:
            raise ValueError(
                f"manifest count mismatch for shard {index}: "
                f"{len(shard_rows)} != {count}"
            )
        if sorted(int(row["offset"]) for row in shard_rows) != list(
            range(count)
        ):
            raise ValueError(f"non-contiguous offsets in shard {index}")
        arrays: dict[str, np.ndarray] = {}
        for name, (dtype, shape_factory) in EXPECTED.items():
            path = Path(shard["paths"][name])
            actual_hash = sha256_file(path)
            if actual_hash != shard["hashes_sha256"][name]:
                raise ValueError(f"hash mismatch: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = shape_factory(count)
            if array.dtype != dtype or array.shape != expected_shape:
                raise ValueError(
                    f"invalid {name} shard {index}: "
                    f"{array.dtype} {array.shape}, expected "
                    f"{dtype} {expected_shape}"
                )
            arrays[name] = array
            checked_files += 1
        if np.any(arrays["azimuth_deg"] >= 360):
            raise ValueError(f"invalid azimuth in shard {index}")
        for row in shard_rows:
            offset = int(row["offset"])
            if int(arrays["tile_number"][offset]) != int(row["tile_number"]):
                raise ValueError(
                    f"tile number mismatch for {row['tile']} in shard {index}"
                )
            iso_valid = int(
                np.count_nonzero(np.isfinite(arrays["p_iso"][offset]))
            )
            dir_valid = int(
                np.count_nonzero(np.isfinite(arrays["p_dir"][offset]))
            )
            if iso_valid != int(row["iso_valid_pixels"]):
                raise ValueError(f"isotropic valid count mismatch: {row['tile']}")
            if dir_valid != int(row["directional_valid_pixels"]):
                raise ValueError(
                    f"directional valid count mismatch: {row['tile']}"
                )
            checked_tiles += 1

    result: dict[str, Any] = {
        "status": "ok",
        "metadata": str(args.metadata),
        "tiles_verified": checked_tiles,
        "files_verified": checked_files,
        "manifest_sha256": metadata["manifest_sha256"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Select deterministic spatially dispersed train/val/test IRT tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train", type=int, default=2560)
    parser.add_argument("--val", type=int, default=320)
    parser.add_argument("--test", type=int, default=320)
    parser.add_argument("--block-size-m", type=float, default=2048.0)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spatially_dispersed_sample(
    tiles: list[str],
    count: int,
    prepared_root: Path,
    block_size_m: float,
    seed: int,
) -> list[dict[str, Any]]:
    if count > len(tiles):
        raise ValueError(f"requested {count} tiles from only {len(tiles)} candidates")
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for tile in tiles:
        metadata_path = prepared_root / tile / "metadata.json"
        metadata = load_json(metadata_path)
        grid = metadata["grid"]
        center_x = (float(grid["xmin"]) + float(grid["xmax"])) / 2.0
        center_y = (float(grid["ymin"]) + float(grid["ymax"])) / 2.0
        block = (
            math.floor(center_x / block_size_m),
            math.floor(center_y / block_size_m),
        )
        groups[block].append(tile)
    generator = np.random.default_rng(seed)
    blocks = list(groups)
    generator.shuffle(blocks)
    for block in blocks:
        generator.shuffle(groups[block])
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        added = 0
        for block in blocks:
            candidates = groups[block]
            if round_index >= len(candidates):
                continue
            tile = candidates[round_index]
            selected.append(
                {
                    "tile": tile,
                    "block_x": block[0],
                    "block_y": block[1],
                    "block_round": round_index,
                }
            )
            added += 1
            if len(selected) == count:
                break
        if not added:
            raise RuntimeError("spatial round-robin selection exhausted unexpectedly")
        round_index += 1
    return selected


def main() -> int:
    args = parse_args()
    if min(args.train, args.val, args.test) < 1:
        raise ValueError("split counts must be positive")
    if args.block_size_m <= 0:
        raise ValueError("block size must be positive")
    with args.ready_manifest.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        ready_rows = list(csv.DictReader(handle))
    ready = {
        row["Tile"]
        for row in ready_rows
        if row.get("Status", "").strip().upper() == "READY"
    }
    counts = {"train": args.train, "val": args.val, "test": args.test}
    selected: list[dict[str, Any]] = []
    candidates_by_split: dict[str, int] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        split_path = args.normalized_root / f"{split}_tiles.txt"
        split_tiles = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.strip() in ready
        ]
        candidates_by_split[split] = len(split_tiles)
        split_selected = spatially_dispersed_sample(
            split_tiles,
            counts[split],
            args.prepared_root,
            args.block_size_m,
            int(
                np.random.SeedSequence([args.seed, split_index]).generate_state(
                    1, dtype=np.uint32
                )[0]
            ),
        )
        for item in split_selected:
            item["split"] = split
            item["split_index"] = sum(
                previous["split"] == split for previous in selected
            )
            selected.append(item)
    tiles = [item["tile"] for item in selected]
    if len(tiles) != len(set(tiles)):
        raise RuntimeError("selected train/val/test tiles overlap")

    args.output_root.mkdir(parents=True, exist_ok=True)
    text_path = args.output_root / "selection_tiles.txt"
    csv_path = args.output_root / "selection_tiles.csv"
    json_path = args.output_root / "selection_metadata.json"
    for path in (text_path, csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace selection artifact: {path}")
    text_temporary = text_path.with_suffix(".txt.tmp")
    text_temporary.write_text("\n".join(tiles) + "\n", encoding="utf-8")
    os.replace(text_temporary, text_path)
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "tile",
                "split",
                "split_index",
                "block_x",
                "block_y",
                "block_round",
            ),
        )
        writer.writeheader()
        writer.writerows(selected)
    os.replace(csv_temporary, csv_path)
    metadata = {
        "version": 1,
        "method": (
            "existing spatial-block split, then seeded round-robin sampling "
            "across 2048 m spatial blocks"
        ),
        "seed": args.seed,
        "block_size_m": args.block_size_m,
        "requested_counts": counts,
        "selected_counts": {
            split: sum(item["split"] == split for item in selected)
            for split in counts
        },
        "ready_candidates_by_split": candidates_by_split,
        "total": len(selected),
        "tiles_file": str(text_path),
        "tiles_file_sha256": sha256_file(text_path),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "source_ready_manifest": str(args.ready_manifest),
        "source_ready_manifest_sha256": sha256_file(args.ready_manifest),
    }
    json_temporary = json_path.with_suffix(".json.tmp")
    json_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(json_temporary, json_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

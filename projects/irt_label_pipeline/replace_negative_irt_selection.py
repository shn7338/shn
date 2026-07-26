#!/usr/bin/env python3
"""Replace negative-coordinate IRT tiles without changing data splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STAGE1_REQUIRED_FILES = (
    "building_height_norm.npy",
    "tx_position_height_norm.npy",
    "tx_distance_norm.npy",
)
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--block-size-m", type=float, default=2048.0)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str, encoding: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding=encoding, newline="")
    os.replace(temporary, path)


def grid_record(
    prepared_root: Path,
    tile: str,
    block_size_m: float,
) -> dict[str, float | int]:
    metadata_path = prepared_root / tile / "metadata.json"
    metadata = load_json(metadata_path)
    grid = metadata["grid"]
    xmin = float(grid["xmin"])
    ymin = float(grid["ymin"])
    xmax = float(grid["xmax"])
    ymax = float(grid["ymax"])
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "block_x": math.floor(((xmin + xmax) / 2.0) / block_size_m),
        "block_y": math.floor(((ymin + ymax) / 2.0) / block_size_m),
    }


def positive_coordinates(record: dict[str, float | int]) -> bool:
    return min(
        float(record["xmin"]),
        float(record["ymin"]),
        float(record["xmax"]),
        float(record["ymax"]),
    ) >= 0.0


def stage1_complete(normalized_root: Path, tile: str) -> bool:
    tile_root = normalized_root / tile
    return all((tile_root / name).is_file() for name in STAGE1_REQUIRED_FILES)


def select_replacements(
    candidates: list[str],
    count: int,
    retained_blocks: Counter[tuple[int, int]],
    grids: dict[str, dict[str, float | int]],
    seed: int,
) -> list[str]:
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for tile in candidates:
        grid = grids[tile]
        block = (int(grid["block_x"]), int(grid["block_y"]))
        groups[block].append(tile)
    generator = np.random.default_rng(seed)
    block_tiebreak = {
        block: float(generator.random()) for block in groups
    }
    for block, tiles in groups.items():
        generator.shuffle(tiles)
    selected: list[str] = []
    selected_per_block: Counter[tuple[int, int]] = Counter()
    while len(selected) < count:
        available = [block for block, tiles in groups.items() if tiles]
        if not available:
            raise RuntimeError(
                f"only found {len(selected)} valid replacements for {count} slots"
            )
        block = min(
            available,
            key=lambda item: (
                retained_blocks[item] + selected_per_block[item],
                selected_per_block[item],
                block_tiebreak[item],
                item,
            ),
        )
        selected.append(groups[block].pop())
        selected_per_block[block] += 1
    return selected


def main() -> int:
    args = parse_args()
    if args.block_size_m <= 0:
        raise ValueError("block size must be positive")
    for path in (
        args.source_selection,
        args.ready_manifest,
        args.normalized_root,
        args.prepared_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_paths = (
        args.output_root / "selection_tiles.txt",
        args.output_root / "selection_tiles.csv",
        args.output_root / "selection_metadata.json",
        args.output_root / "replacement_manifest.csv",
    )
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"refusing to replace selection artifact: {path}")

    with args.source_selection.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 3200:
        raise ValueError(
            f"expected 3200 source rows, found {len(source_rows)}"
        )
    source_tiles = {row["tile"] for row in source_rows}
    if len(source_tiles) != len(source_rows):
        raise ValueError("source selection contains duplicate tiles")

    with args.ready_manifest.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        ready = {
            row["Tile"]
            for row in csv.DictReader(handle)
            if row.get("Status", "").strip().upper() == "READY"
        }

    grids: dict[str, dict[str, float | int]] = {}

    def grid(tile: str) -> dict[str, float | int]:
        if tile not in grids:
            grids[tile] = grid_record(
                args.prepared_root,
                tile,
                args.block_size_m,
            )
        return grids[tile]

    replacements_by_old: dict[str, str] = {}
    replacement_rows: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}
    for split_index, split in enumerate(SPLITS):
        split_rows = [row for row in source_rows if row["split"] == split]
        removed = [
            row for row in split_rows if not positive_coordinates(grid(row["tile"]))
        ]
        retained = [
            row for row in split_rows if positive_coordinates(grid(row["tile"]))
        ]
        retained_blocks = Counter(
            (
                int(grid(row["tile"])["block_x"]),
                int(grid(row["tile"])["block_y"]),
            )
            for row in retained
        )
        split_file = args.normalized_root / f"{split}_tiles.txt"
        split_tiles = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        candidates = []
        for tile in split_tiles:
            if tile not in ready or tile in source_tiles:
                continue
            if not stage1_complete(args.normalized_root, tile):
                continue
            if not positive_coordinates(grid(tile)):
                continue
            candidates.append(tile)
        candidate_counts[split] = len(candidates)
        replacement_tiles = select_replacements(
            candidates,
            len(removed),
            retained_blocks,
            grids,
            int(
                np.random.SeedSequence(
                    [args.seed, split_index]
                ).generate_state(1, dtype=np.uint32)[0]
            ),
        )
        for old_row, new_tile in zip(removed, replacement_tiles, strict=True):
            old_tile = old_row["tile"]
            replacements_by_old[old_tile] = new_tile
            old_grid = grid(old_tile)
            new_grid = grid(new_tile)
            replacement_rows.append(
                {
                    "split": split,
                    "split_index": old_row["split_index"],
                    "old_tile": old_tile,
                    "new_tile": new_tile,
                    "old_xmin": old_grid["xmin"],
                    "old_ymin": old_grid["ymin"],
                    "new_xmin": new_grid["xmin"],
                    "new_ymin": new_grid["ymin"],
                    "new_block_x": new_grid["block_x"],
                    "new_block_y": new_grid["block_y"],
                    "stage1_inputs": "ok",
                }
            )

    output_rows: list[dict[str, Any]] = []
    for row in source_rows:
        tile = replacements_by_old.get(row["tile"], row["tile"])
        tile_grid = grid(tile)
        output_rows.append(
            {
                "tile": tile,
                "split": row["split"],
                "split_index": row["split_index"],
                "block_x": int(tile_grid["block_x"]),
                "block_y": int(tile_grid["block_y"]),
                "block_round": row["block_round"],
            }
        )

    output_tiles = [row["tile"] for row in output_rows]
    if len(output_tiles) != 3200 or len(set(output_tiles)) != 3200:
        raise RuntimeError("replacement selection is not 3200 unique tiles")
    expected_counts = {"train": 2560, "val": 320, "test": 320}
    actual_counts = Counter(row["split"] for row in output_rows)
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(
            f"split counts changed: {dict(actual_counts)} != {expected_counts}"
        )
    for row in output_rows:
        tile = row["tile"]
        if not positive_coordinates(grid(tile)):
            raise RuntimeError(f"negative coordinate remains: {tile}")
        if not stage1_complete(args.normalized_root, tile):
            raise RuntimeError(f"Stage1 inputs are incomplete: {tile}")

    blocks_by_split = {
        split: {
            (int(row["block_x"]), int(row["block_y"]))
            for row in output_rows
            if row["split"] == split
        }
        for split in SPLITS
    }
    overlap_counts = {
        f"{left}_{right}": len(blocks_by_split[left] & blocks_by_split[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"spatial split block overlap: {overlap_counts}")

    text_path, csv_path, metadata_path, replacement_path = output_paths
    atomic_write_text(
        text_path,
        "\n".join(output_tiles) + "\n",
        "utf-8",
    )
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
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
        writer.writerows(output_rows)
    os.replace(csv_temporary, csv_path)
    replacement_temporary = replacement_path.with_suffix(".csv.tmp")
    with replacement_temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(replacement_rows[0]),
        )
        writer.writeheader()
        writer.writerows(replacement_rows)
    os.replace(replacement_temporary, replacement_path)

    metadata = {
        "version": 2,
        "method": (
            "retain every positive-coordinate tile from the v1 selection; "
            "replace only negative-coordinate tiles with READY, Stage1-complete, "
            "positive-coordinate candidates from the same immutable spatial split"
        ),
        "seed": args.seed,
        "block_size_m": args.block_size_m,
        "source_selection": str(args.source_selection),
        "source_selection_sha256": sha256_file(args.source_selection),
        "counts": expected_counts,
        "total": 3200,
        "retained_tiles": 3200 - len(replacement_rows),
        "replaced_tiles": len(replacement_rows),
        "replacements_by_split": dict(
            Counter(row["split"] for row in replacement_rows)
        ),
        "candidate_counts_by_split": candidate_counts,
        "negative_coordinate_tiles_remaining": 0,
        "stage1_input_complete_tiles": 3200,
        "spatial_block_overlap_counts": overlap_counts,
        "tiles_file": str(text_path),
        "tiles_file_sha256": sha256_file(text_path),
        "selection_csv": str(csv_path),
        "selection_csv_sha256": sha256_file(csv_path),
        "replacement_manifest": str(replacement_path),
        "replacement_manifest_sha256": sha256_file(replacement_path),
        "ready_manifest": str(args.ready_manifest),
        "ready_manifest_sha256": sha256_file(args.ready_manifest),
    }
    atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        "utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

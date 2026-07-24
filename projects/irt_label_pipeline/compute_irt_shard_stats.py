#!/usr/bin/env python3
"""Compute streaming train/validation/test statistics for IRT NPY shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return
        batch_count = int(finite.size)
        batch_mean = float(np.mean(finite))
        batch_m2 = float(np.sum((finite - batch_mean) ** 2))
        if self.count:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.m2 += (
                batch_m2
                + delta * delta * self.count * batch_count / total
            )
            self.mean += delta * batch_count / total
            self.count = total
        else:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        self.minimum = min(self.minimum, float(np.min(finite)))
        self.maximum = max(self.maximum, float(np.max(finite)))

    def as_dict(self) -> dict[str, float | int | None]:
        if not self.count:
            return {
                "count": 0,
                "mean_db": None,
                "std_db": None,
                "minimum_db": None,
                "maximum_db": None,
            }
        return {
            "count": self.count,
            "mean_db": self.mean,
            "std_db": math.sqrt(self.m2 / self.count),
            "minimum_db": self.minimum,
            "maximum_db": self.maximum,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    with args.selection_csv.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        split_by_tile = {
            row["tile"]: row["split"] for row in csv.DictReader(handle)
        }
    with args.shard_manifest.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty shard manifest")

    moments = {
        split: {
            "p_iso": RunningMoments(),
            "p_dir": RunningMoments(),
        }
        for split in ("all", "train", "val", "test")
    }
    memmaps: dict[tuple[str, int], np.ndarray] = {}
    split_tile_counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        tile = row["tile"]
        split = split_by_tile.get(tile)
        if split not in split_tile_counts:
            raise ValueError(f"missing or invalid split for {tile}: {split!r}")
        shard = int(row["shard"])
        offset = int(row["offset"])
        for quantity in ("p_iso", "p_dir"):
            key = (quantity, shard)
            if key not in memmaps:
                path = args.shard_root / f"{quantity}_{shard:04d}.npy"
                memmaps[key] = np.load(
                    path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
            values = memmaps[key][offset]
            moments["all"][quantity].update(values)
            moments[split][quantity].update(values)
        split_tile_counts[split] += 1

    link_budget_variance = (
        (35.0 - 10.0) ** 2 / 12.0
        + 2.0 * (20.0 - 10.0) ** 2 / 12.0
        + (10.0 - (-10.0)) ** 2 / 12.0
    )
    statistics = {
        split: {
            quantity: value.as_dict()
            for quantity, value in split_moments.items()
        }
        for split, split_moments in moments.items()
    }
    train_directional = statistics["train"]["p_dir"]
    if (
        train_directional["mean_db"] is None
        or train_directional["std_db"] is None
    ):
        raise RuntimeError("training directional statistics are empty")
    ss_mean = float(train_directional["mean_db"]) + 52.5
    ss_std = math.sqrt(
        float(train_directional["std_db"]) ** 2
        + link_budget_variance
    )
    payload = {
        "version": 1,
        "quantity": (
            "WinProp path gain values read from Antenna Path Loss.txt; "
            "raw dB, finite pixels only"
        ),
        "split_tile_counts": split_tile_counts,
        "statistics": statistics,
        "synthetic_link_budget": {
            "p_tx_dbm": "Uniform(10, 35)",
            "g_tx_db": "Uniform(10, 20)",
            "g_rx_db": "Uniform(10, 20)",
            "insertion_loss_db": "Uniform(-10, 10)",
            "combined_offset_formula": "P_TX + G_TX + G_RX - IL",
            "mean_db": 52.5,
            "variance_db2": link_budget_variance,
        },
        "stage2b_ss_normalization": {
            "mean_db": ss_mean,
            "std_db": ss_std,
            "formula": "(SS_db - mean_db) / std_db",
            "source": (
                "training directional path-gain moments plus independent "
                "synthetic link-budget moments"
            ),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

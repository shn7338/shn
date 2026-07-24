"""Memory-mapped dataset for Stage2-A DPM-to-isotropic-IRT refinement."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


STAGE1_INPUT_FILES = (
    "building_height_norm.npy",
    "tx_position_height_norm.npy",
    "tx_distance_norm.npy",
)


class Stage2AIsoDataset(Dataset):
    def __init__(
        self,
        shard_root: str | Path,
        shard_manifest: str | Path,
        selection_csv: str | Path,
        normalized_root: str | Path,
        split: str,
        augment: bool = False,
        limit: int | None = None,
    ) -> None:
        self.shard_root = Path(shard_root)
        self.normalized_root = Path(normalized_root)
        self.split = split
        self.augment = augment
        with Path(selection_csv).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            split_by_tile = {
                row["tile"]: row["split"] for row in csv.DictReader(handle)
            }
        with Path(shard_manifest).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [
            row for row in rows if split_by_tile.get(row["tile"]) == split
        ]
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"no Stage2-A samples found for split {split!r}")
        self._iso_memmaps: dict[int, np.ndarray] = {}
        self._validate_endpoints()

    def _validate_endpoints(self) -> None:
        for row in {self.rows[0]["tile"]: self.rows[0], self.rows[-1]["tile"]: self.rows[-1]}.values():
            tile_dir = self.normalized_root / row["tile"]
            for filename in STAGE1_INPUT_FILES:
                path = tile_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
            shard = int(row["shard"])
            path = self.shard_root / f"p_iso_{shard:04d}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)

    def _iso_shard(self, index: int) -> np.ndarray:
        if index not in self._iso_memmaps:
            path = self.shard_root / f"p_iso_{index:04d}.npy"
            self._iso_memmaps[index] = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._iso_memmaps[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tile = row["tile"]
        tile_dir = self.normalized_root / tile
        stage1_inputs = np.stack(
            [
                np.load(tile_dir / filename, allow_pickle=False).astype(
                    np.float32,
                    copy=False,
                )
                for filename in STAGE1_INPUT_FILES
            ],
            axis=0,
        )
        if stage1_inputs.shape != (3, 128, 128):
            raise ValueError(
                f"unexpected Stage1 input shape for {tile}: "
                f"{stage1_inputs.shape}"
            )
        if not np.all(np.isfinite(stage1_inputs)):
            raise ValueError(f"non-finite Stage1 input for {tile}")
        if stage1_inputs.min() < -1e-4 or stage1_inputs.max() > 1.0001:
            raise ValueError(f"Stage1 input outside [0, 1] for {tile}")
        shard = int(row["shard"])
        offset = int(row["offset"])
        target = np.asarray(
            self._iso_shard(shard)[offset],
            dtype=np.float32,
        )[None, ...]
        if target.shape != (1, 128, 128):
            raise ValueError(f"unexpected IRT target shape for {tile}: {target.shape}")
        valid_mask = np.isfinite(target)
        if not np.count_nonzero(valid_mask):
            raise ValueError(f"IRT target has no valid pixels for {tile}")
        target = np.where(valid_mask, target, 0.0).astype(np.float32)

        if self.augment:
            rotations = random.randrange(4)
            if rotations:
                stage1_inputs = np.rot90(
                    stage1_inputs,
                    rotations,
                    axes=(-2, -1),
                )
                target = np.rot90(target, rotations, axes=(-2, -1))
                valid_mask = np.rot90(
                    valid_mask,
                    rotations,
                    axes=(-2, -1),
                )
            if random.random() < 0.5:
                stage1_inputs = np.flip(stage1_inputs, axis=-1)
                target = np.flip(target, axis=-1)
                valid_mask = np.flip(valid_mask, axis=-1)
            if random.random() < 0.5:
                stage1_inputs = np.flip(stage1_inputs, axis=-2)
                target = np.flip(target, axis=-2)
                valid_mask = np.flip(valid_mask, axis=-2)

        return {
            "stage1_input": torch.from_numpy(
                np.ascontiguousarray(stage1_inputs)
            ),
            "building": torch.from_numpy(
                np.ascontiguousarray(stage1_inputs[:1])
            ),
            "target_irt_db": torch.from_numpy(np.ascontiguousarray(target)),
            "valid_mask": torch.from_numpy(
                np.ascontiguousarray(valid_mask)
            ),
            "tile": tile,
        }

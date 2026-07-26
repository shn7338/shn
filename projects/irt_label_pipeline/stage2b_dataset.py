"""Memory-mapped directional-SS dataset with online sparse measurements."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from stage2a_dataset import STAGE1_INPUT_FILES


def stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class Stage2BDirectionalDataset(Dataset):
    """Expose four directional labels per tile without loading full shards."""

    def __init__(
        self,
        shard_root: str | Path,
        shard_manifest: str | Path,
        selection_csv: str | Path,
        normalized_root: str | Path,
        split: str,
        ss_mean_db: float,
        ss_std_db: float,
        seed: int,
        stage1_prediction_root: str | Path | None = None,
        augment: bool = False,
        sparse_points_min: int = 1,
        sparse_points_max: int = 200,
        fixed_sparse_points: int | None = None,
        limit_tiles: int | None = None,
    ) -> None:
        self.shard_root = Path(shard_root)
        self.normalized_root = Path(normalized_root)
        self.stage1_prediction_root = (
            Path(stage1_prediction_root)
            if stage1_prediction_root is not None
            else None
        )
        self.split = split
        self.ss_mean_db = float(ss_mean_db)
        self.ss_std_db = float(ss_std_db)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.sparse_points_min = int(sparse_points_min)
        self.sparse_points_max = int(sparse_points_max)
        self.fixed_sparse_points = fixed_sparse_points
        self.epoch = 0
        if self.ss_std_db <= 0:
            raise ValueError("SS normalization standard deviation must be positive")
        if not 1 <= self.sparse_points_min <= self.sparse_points_max:
            raise ValueError("invalid sparse-point range")

        with Path(selection_csv).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            split_by_tile = {
                row["tile"]: row["split"] for row in csv.DictReader(handle)
            }
        with Path(shard_manifest).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            tile_rows = [
                row
                for row in csv.DictReader(handle)
                if split_by_tile.get(row["tile"]) == split
            ]
        if limit_tiles is not None:
            tile_rows = tile_rows[:limit_tiles]
        self.rows = [
            (row, direction_index)
            for row in tile_rows
            for direction_index in range(4)
        ]
        if not self.rows:
            raise ValueError(f"no Stage2-B samples found for split {split!r}")
        self._direction_memmaps: dict[int, np.ndarray] = {}
        self._validate_endpoints()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _validate_endpoints(self) -> None:
        for row, _ in (self.rows[0], self.rows[-1]):
            tile_dir = self.normalized_root / row["tile"]
            for filename in STAGE1_INPUT_FILES:
                path = tile_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
            if self.stage1_prediction_root is not None:
                prediction_path = (
                    self.stage1_prediction_root
                    / row["tile"]
                    / "path_gain_pred_norm.npy"
                )
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
            shard = int(row["shard"])
            path = self.shard_root / f"p_dir_{shard:04d}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)

    def _direction_shard(self, index: int) -> np.ndarray:
        if index not in self._direction_memmaps:
            path = self.shard_root / f"p_dir_{index:04d}.npy"
            self._direction_memmaps[index] = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._direction_memmaps[index]

    def __len__(self) -> int:
        return len(self.rows)

    def _link_budget_offset(self, tile: str, direction: int) -> tuple[float, ...]:
        rng = np.random.default_rng(
            stable_seed(self.seed, "link-budget", tile, direction)
        )
        p_tx = float(rng.uniform(10.0, 35.0))
        g_tx = float(rng.uniform(10.0, 20.0))
        g_rx = float(rng.uniform(10.0, 20.0))
        insertion_loss = float(rng.uniform(-10.0, 10.0))
        return p_tx, g_tx, g_rx, insertion_loss, p_tx + g_tx + g_rx - insertion_loss

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, direction_index = self.rows[index]
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
        stage1_prediction = None
        if self.stage1_prediction_root is not None:
            stage1_prediction = np.load(
                self.stage1_prediction_root
                / tile
                / "path_gain_pred_norm.npy",
                allow_pickle=False,
            ).astype(np.float32, copy=False)[None, ...]
            if stage1_prediction.shape != (1, 128, 128):
                raise ValueError(
                    f"unexpected cached Stage1 prediction shape for {tile}: "
                    f"{stage1_prediction.shape}"
                )
            if not np.all(np.isfinite(stage1_prediction)):
                raise ValueError(
                    f"non-finite cached Stage1 prediction for {tile}"
                )

        shard = int(row["shard"])
        offset = int(row["offset"])
        path_gain = np.asarray(
            self._direction_shard(shard)[offset, direction_index],
            dtype=np.float32,
        )
        if path_gain.shape != (128, 128):
            raise ValueError(
                f"unexpected directional target shape for {tile}: "
                f"{path_gain.shape}"
            )
        valid_mask = np.isfinite(path_gain)
        valid_indices = np.flatnonzero(valid_mask)
        if not valid_indices.size:
            raise ValueError(f"directional target has no valid pixels for {tile}")

        p_tx, g_tx, g_rx, insertion_loss, offset_db = (
            self._link_budget_offset(tile, direction_index)
        )
        target_db = path_gain + offset_db
        target_norm = (target_db - self.ss_mean_db) / self.ss_std_db
        target_norm = np.where(valid_mask, target_norm, 0.0).astype(np.float32)

        sparse_rng = np.random.default_rng(
            stable_seed(
                self.seed,
                "sparse",
                self.split,
                tile,
                direction_index,
                self.epoch if self.split == "train" else 0,
            )
        )
        if self.fixed_sparse_points is None:
            requested_points = int(
                sparse_rng.integers(
                    self.sparse_points_min,
                    self.sparse_points_max + 1,
                )
            )
        else:
            requested_points = int(self.fixed_sparse_points)
        point_count = min(requested_points, int(valid_indices.size))
        selected = sparse_rng.choice(
            valid_indices,
            size=point_count,
            replace=False,
        )
        sparse_norm = np.zeros((128, 128), dtype=np.float32)
        sparse_mask = np.zeros((128, 128), dtype=np.float32)
        sparse_norm.flat[selected] = target_norm.flat[selected]
        sparse_mask.flat[selected] = 1.0

        if self.augment:
            augment_rng = np.random.default_rng(
                stable_seed(
                    self.seed,
                    "augment",
                    tile,
                    direction_index,
                    self.epoch,
                )
            )
            rotations = int(augment_rng.integers(0, 4))
            if rotations:
                stage1_inputs = np.rot90(
                    stage1_inputs, rotations, axes=(-2, -1)
                )
                if stage1_prediction is not None:
                    stage1_prediction = np.rot90(
                        stage1_prediction,
                        rotations,
                        axes=(-2, -1),
                    )
                target_norm = np.rot90(
                    target_norm, rotations, axes=(-2, -1)
                )
                valid_mask = np.rot90(
                    valid_mask, rotations, axes=(-2, -1)
                )
                sparse_norm = np.rot90(
                    sparse_norm, rotations, axes=(-2, -1)
                )
                sparse_mask = np.rot90(
                    sparse_mask, rotations, axes=(-2, -1)
                )
            if bool(augment_rng.integers(0, 2)):
                stage1_inputs = np.flip(stage1_inputs, axis=-1)
                if stage1_prediction is not None:
                    stage1_prediction = np.flip(
                        stage1_prediction, axis=-1
                    )
                target_norm = np.flip(target_norm, axis=-1)
                valid_mask = np.flip(valid_mask, axis=-1)
                sparse_norm = np.flip(sparse_norm, axis=-1)
                sparse_mask = np.flip(sparse_mask, axis=-1)
            if bool(augment_rng.integers(0, 2)):
                stage1_inputs = np.flip(stage1_inputs, axis=-2)
                if stage1_prediction is not None:
                    stage1_prediction = np.flip(
                        stage1_prediction, axis=-2
                    )
                target_norm = np.flip(target_norm, axis=-2)
                valid_mask = np.flip(valid_mask, axis=-2)
                sparse_norm = np.flip(sparse_norm, axis=-2)
                sparse_mask = np.flip(sparse_mask, axis=-2)

        azimuth = int(row[f"azimuth_{direction_index}_deg"])
        result = {
            "stage1_input": torch.from_numpy(
                np.ascontiguousarray(stage1_inputs)
            ),
            "building": torch.from_numpy(
                np.ascontiguousarray(stage1_inputs[:1])
            ),
            "target_ss_norm": torch.from_numpy(
                np.ascontiguousarray(target_norm[None, ...])
            ),
            "sparse_ss_norm": torch.from_numpy(
                np.ascontiguousarray(sparse_norm[None, ...])
            ),
            "sparse_mask": torch.from_numpy(
                np.ascontiguousarray(sparse_mask[None, ...])
            ),
            "valid_mask": torch.from_numpy(
                np.ascontiguousarray(valid_mask[None, ...])
            ),
            "tile": tile,
            "direction_index": direction_index,
            "azimuth_deg": azimuth,
            "sparse_points": point_count,
            "link_budget": torch.tensor(
                [p_tx, g_tx, g_rx, insertion_loss, offset_db],
                dtype=torch.float32,
            ),
        }
        if stage1_prediction is not None:
            result["stage1_prediction_norm"] = torch.from_numpy(
                np.ascontiguousarray(stage1_prediction)
            )
        return result

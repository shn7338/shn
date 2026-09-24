"""PyTorch dataset for the random-base-station inversion Pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def transform_row_col(
    row: float,
    col: float,
    rotations: int,
    mirror: bool,
    size: int = 128,
) -> tuple[float, float]:
    for _ in range(rotations % 4):
        row, col = size - 1.0 - col, row
    if mirror:
        col = size - 1.0 - col
    return row, col


def transform_direction_sin_cos(
    sin_value: float,
    cos_value: float,
    rotations: int,
    mirror: bool,
) -> tuple[float, float]:
    for _ in range(rotations % 4):
        sin_value, cos_value = cos_value, -sin_value
    if mirror:
        cos_value = -cos_value
    return sin_value, cos_value


def transform_map(array: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    transformed = np.rot90(array, rotations, axes=(-2, -1))
    if mirror:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed)


class BSInversionPilotDataset(Dataset[dict[str, Any]]):
    """One item is one site-direction pair from a compact Pilot NPZ."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        seed: int,
        augment: bool = False,
        limit_sites: int | None = None,
        signal_mean_db: float | None = None,
        signal_std_db: float | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        self.root = Path(dataset_root)
        self.split = split
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = 0
        manifest = load_json(self.root / "run_manifest.json")
        self.data_config = load_json(Path(manifest["config"]))
        normalization = load_json(self.root / "normalization.json")
        signal_stats = normalization["directional_signal_strength_db"]
        self.signal_mean_db = float(
            signal_stats["mean"] if signal_mean_db is None else signal_mean_db
        )
        self.signal_std_db = float(
            signal_stats["std"] if signal_std_db is None else signal_std_db
        )
        if self.signal_std_db <= 0.0:
            raise ValueError("signal normalization std must be positive")
        self.power_mean_db = float(
            self.data_config["effective_power"]["normalization_mean_db"]
        )
        self.power_std_db = float(
            self.data_config["effective_power"]["normalization_std_db"]
        )
        entries = [entry for entry in manifest["entries"] if entry["split"] == split]
        if limit_sites is not None:
            if limit_sites < 1:
                raise ValueError("limit_sites must be positive")
            entries = entries[:limit_sites]
        self.entries = entries
        self.rows = [
            (entry, direction)
            for entry in entries
            for direction in range(len(entry["azimuths_deg"]))
        ]
        if not self.rows:
            raise ValueError(f"no samples for split {split}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _augmentation(self, site_id: str, direction: int) -> tuple[int, bool]:
        if not self.augment:
            return 0, False
        rng = np.random.default_rng(
            stable_seed(self.seed, "augment", self.epoch, site_id, direction)
        )
        return int(rng.integers(0, 4)), bool(rng.integers(0, 2))

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry, direction = self.rows[index]
        site_id = entry["site_id"]
        path = self.root / "samples" / self.split / f"{site_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            building = data["building_height_norm"].astype(np.float32)
            target_db = data["directional_signal_strength_db"][direction].astype(
                np.float32
            )
            valid = data["directional_valid_mask"][direction].astype(bool)
            sparse_indices = data["sparse_flat_indices"][direction].astype(np.int64)
            sparse_db_values = data["sparse_signal_strength_db"][direction].astype(
                np.float32
            )
            heatmap = data["tx_location_heatmap"].astype(np.float32)
            row_col = data["tx_row_col_px"].astype(np.float32)
            tx_height_m = float(data["tx_height_m"].item())
            power_db = float(data["effective_power_db"].item())
            direction_sin_cos = data["azimuth_sin_cos"][direction].astype(
                np.float32
            )
            azimuth_deg = float(data["azimuth_deg"][direction])
            iso_db = data["isotropic_path_gain_db"].astype(np.float32)
            iso_valid = data["isotropic_valid_mask"].astype(bool)

        sparse_norm = np.zeros((128, 128), dtype=np.float32)
        sparse_mask = np.zeros((128, 128), dtype=np.float32)
        sparse_norm.flat[sparse_indices] = (
            sparse_db_values - self.signal_mean_db
        ) / self.signal_std_db
        sparse_mask.flat[sparse_indices] = 1.0
        target_norm = np.where(
            valid,
            (target_db - self.signal_mean_db) / self.signal_std_db,
            0.0,
        ).astype(np.float32)
        target_db_filled = np.where(valid, target_db, 0.0).astype(np.float32)
        iso_db_filled = np.where(iso_valid, iso_db, 0.0).astype(np.float32)

        rotations, mirror = self._augmentation(site_id, direction)
        if rotations or mirror:
            building = transform_map(building, rotations, mirror)
            sparse_norm = transform_map(sparse_norm, rotations, mirror)
            sparse_mask = transform_map(sparse_mask, rotations, mirror)
            target_norm = transform_map(target_norm, rotations, mirror)
            target_db_filled = transform_map(target_db_filled, rotations, mirror)
            valid = transform_map(valid, rotations, mirror)
            heatmap = transform_map(heatmap, rotations, mirror)
            iso_db_filled = transform_map(iso_db_filled, rotations, mirror)
            iso_valid = transform_map(iso_valid, rotations, mirror)
            row, col = transform_row_col(
                float(row_col[0]),
                float(row_col[1]),
                rotations,
                mirror,
            )
            row_col = np.asarray([row, col], dtype=np.float32)
            transformed_sin, transformed_cos = transform_direction_sin_cos(
                float(direction_sin_cos[0]),
                float(direction_sin_cos[1]),
                rotations,
                mirror,
            )
            direction_sin_cos = np.asarray(
                [transformed_sin, transformed_cos], dtype=np.float32
            )
            azimuth_deg = math.degrees(
                math.atan2(transformed_sin, transformed_cos)
            ) % 360.0

        tx_x_m = (float(row_col[1]) + 0.5) * 4.0
        tx_y_m = 512.0 - (float(row_col[0]) + 0.5) * 4.0
        model_input = np.stack((building, sparse_norm, sparse_mask), axis=0)
        return {
            "model_input": torch.from_numpy(np.ascontiguousarray(model_input)),
            "building": torch.from_numpy(np.ascontiguousarray(building[None])),
            "sparse_ss_norm": torch.from_numpy(
                np.ascontiguousarray(sparse_norm[None])
            ),
            "sparse_mask": torch.from_numpy(
                np.ascontiguousarray(sparse_mask[None])
            ),
            "target_map_norm": torch.from_numpy(
                np.ascontiguousarray(target_norm[None])
            ),
            "target_map_db": torch.from_numpy(
                np.ascontiguousarray(target_db_filled[None])
            ),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid[None])),
            "target_heatmap": torch.from_numpy(
                np.ascontiguousarray(heatmap[None])
            ),
            "target_row_col_px": torch.from_numpy(row_col.copy()),
            "target_power_norm": torch.tensor(
                (power_db - self.power_mean_db) / self.power_std_db,
                dtype=torch.float32,
            ),
            "target_power_db": torch.tensor(power_db, dtype=torch.float32),
            "target_direction_sin_cos": torch.from_numpy(
                direction_sin_cos.copy()
            ),
            "target_azimuth_deg": torch.tensor(azimuth_deg, dtype=torch.float32),
            "isotropic_path_gain_db": torch.from_numpy(
                np.ascontiguousarray(iso_db_filled[None])
            ),
            "isotropic_valid_mask": torch.from_numpy(
                np.ascontiguousarray(iso_valid[None])
            ),
            "tx_xy_m": torch.tensor([tx_x_m, tx_y_m], dtype=torch.float32),
            "tx_height_m": torch.tensor(tx_height_m, dtype=torch.float32),
            "site_id": site_id,
            "tile": entry["tile"],
            "site_index": int(entry["site_index"]),
            "direction_index": int(direction),
        }


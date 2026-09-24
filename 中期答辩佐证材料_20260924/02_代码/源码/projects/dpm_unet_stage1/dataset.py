"""Dataset for the normalized Stage-1 DPM surrogate arrays."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_FILES = (
    "building_height_norm.npy",
    "tx_position_height_norm.npy",
    "tx_distance_norm.npy",
)
INPUT_MODES = {
    "height": INPUT_FILES[:1],
    "height_tx": INPUT_FILES[:2],
    "height_tx_distance": INPUT_FILES,
}
TARGET_FILE = "path_gain_norm.npy"
MASK_FILE = "path_gain_valid_mask.npy"


def building_edge_map(height: np.ndarray, radius_pixels: int = 1) -> np.ndarray:
    """Return a binary morphological boundary band derived from height > 0."""
    if radius_pixels < 1:
        raise ValueError("radius_pixels must be at least 1")
    building = height > 0
    padded = np.pad(building, radius_pixels, mode="constant", constant_values=False)
    size = 2 * radius_pixels + 1
    dilated = np.zeros_like(building)
    eroded = np.ones_like(building)
    for dy in range(size):
        for dx in range(size):
            view = padded[dy : dy + building.shape[0], dx : dx + building.shape[1]]
            dilated |= view
            eroded &= view
    return (dilated & ~eroded).astype(np.float32)


class DPMTileDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        limit: int | None = None,
        validate_files: bool = True,
        add_building_edge_channel: bool = False,
        edge_channel_radius_pixels: int = 1,
        input_mode: str = "height_tx_distance",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.add_building_edge_channel = add_building_edge_channel
        self.edge_channel_radius_pixels = edge_channel_radius_pixels
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode {input_mode!r}; choose from {tuple(INPUT_MODES)}")
        self.input_mode = input_mode
        split_file = self.root / f"{split}_tiles.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        self.tiles = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit is not None:
            self.tiles = self.tiles[:limit]
        if not self.tiles:
            raise ValueError(f"No tiles in split {split!r}")
        if validate_files:
            self._validate_sample_files()

    def _validate_sample_files(self) -> None:
        # Checking every file would add tens of thousands of metadata operations
        # at every launch. Full generation was already validated; checking the
        # first and last tile catches an incorrect root or incomplete layout.
        for name in {self.tiles[0], self.tiles[-1]}:
            tile_dir = self.root / name
            for filename in (*INPUT_MODES[self.input_mode], TARGET_FILE, MASK_FILE):
                path = tile_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing dataset file: {path}")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.tiles[index]
        tile_dir = self.root / name
        channels = [
            np.load(tile_dir / filename).astype(np.float32, copy=False)
            for filename in INPUT_MODES[self.input_mode]
        ]
        if self.add_building_edge_channel:
            channels.append(building_edge_map(channels[0], self.edge_channel_radius_pixels))
        inputs = np.stack(channels, axis=0)
        target = np.load(tile_dir / TARGET_FILE).astype(np.float32, copy=False)[None, ...]
        mask = np.load(tile_dir / MASK_FILE).astype(np.bool_, copy=False)[None, ...]

        # Rotate/flip all channels, target, and mask together. This is physically
        # valid for the isotropic antennas used to create the DPM labels.
        if self.augment:
            rotations = random.randrange(4)
            if rotations:
                inputs = np.rot90(inputs, rotations, axes=(-2, -1))
                target = np.rot90(target, rotations, axes=(-2, -1))
                mask = np.rot90(mask, rotations, axes=(-2, -1))
            if random.random() < 0.5:
                inputs = np.flip(inputs, axis=-1)
                target = np.flip(target, axis=-1)
                mask = np.flip(mask, axis=-1)
            if random.random() < 0.5:
                inputs = np.flip(inputs, axis=-2)
                target = np.flip(target, axis=-2)
                mask = np.flip(mask, axis=-2)

        # np.rot90/np.flip can produce negative strides, so make the buffers
        # contiguous before converting to torch tensors.
        return {
            "input": torch.from_numpy(np.ascontiguousarray(inputs)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "tile": name,
        }

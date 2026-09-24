#!/usr/bin/env python3
"""Evaluate the frozen three-stage radio-map pipeline under BS errors.

The current pipeline explicitly consumes transmitter location only in Stage1
(``tx_position_height_norm`` and ``tx_distance_norm``).  Transmit power is an
additive dB offset rather than a dedicated model input, while antenna azimuth is
represented only by the directional labels and sparse measurements.  The
experiment therefore evaluates:

* location error by rebuilding the Stage1 TX marker and distance map;
* power error by adding a dB bias to the Stage2-A coarse map passed to Stage2-B;
* direction error by measuring the intrinsic map difference between the four
  Sionna azimuths available for each test tile.

Existing checkpoints and datasets are read-only.  Results are written as JSON,
CSV, and Markdown to a separate experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stage2b_dataset import Stage2BDirectionalDataset, stable_seed
from train_stage2b_directional_ss import (
    build_configured_unet,
    load_frozen_models,
    load_json,
    resolve_device,
    seed_everything,
    sha256_file,
)


DEFAULT_CONFIG = Path(__file__).with_name(
    "config_stage2b_sionna35_depth8_27360_paper_v1.json"
)
DEFAULT_OUTPUT = Path(
    r"D:\桌面\dac\03_runs\experiments\bs_sensitivity_stage123_v1"
)


def parse_number_list(value: str, cast: type[int] | type[float]) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sparse-points", type=int, default=100)
    parser.add_argument(
        "--location-errors",
        default="1,2,4,8",
        help="Comma-separated TX location-error magnitudes in pixels.",
    )
    parser.add_argument(
        "--power-errors",
        default="-5,-3,-1,1,3,5",
        help="Comma-separated coarse-map dB biases.",
    )
    parser.add_argument("--limit-tiles", type=int)
    parser.add_argument("--direction-limit-tiles", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--skip-direction", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def make_base_dataset(
    config: dict[str, Any],
    ss_mean_db: float,
    ss_std_db: float,
    sparse_points: int,
    limit_tiles: int | None,
) -> Stage2BDirectionalDataset:
    return Stage2BDirectionalDataset(
        shard_root=config["shard_root"],
        shard_manifest=config["shard_manifest"],
        selection_csv=config["selection_csv"],
        normalized_root=config["normalized_root"],
        split="test",
        ss_mean_db=ss_mean_db,
        ss_std_db=ss_std_db,
        seed=int(config["seed"]),
        stage1_prediction_root=None,
        augment=False,
        sparse_points_min=sparse_points,
        sparse_points_max=sparse_points,
        fixed_sparse_points=sparse_points,
        limit_tiles=limit_tiles,
    )


def infer_tx_pixel_position(
    distance_norm: np.ndarray,
    marker_row: int,
    marker_col: int,
    resolution_m: float,
    distance_max_m: float,
) -> tuple[float, float]:
    """Recover the sub-pixel TX position from the normalized distance field."""
    distance_pixels_sq = (
        np.expm1(distance_norm * math.log1p(distance_max_m))
        / resolution_m
    ) ** 2

    def coordinate_from_differences(
        axis: int,
        index: int,
        fixed: int,
    ) -> float:
        length = distance_norm.shape[axis]
        if axis == 0:
            value = lambda position: float(
                distance_pixels_sq[position, fixed]
            )
        else:
            value = lambda position: float(
                distance_pixels_sq[fixed, position]
            )
        if 0 < index < length - 1:
            difference = value(index + 1) - value(index - 1)
            return index - difference / 4.0
        if index == 0:
            difference = value(1) - value(0)
            return (1.0 - difference) / 2.0
        difference = value(index) - value(index - 1)
        return index - (difference + 1.0) / 2.0

    tx_row = coordinate_from_differences(0, marker_row, marker_col)
    tx_col = coordinate_from_differences(1, marker_col, marker_row)
    if not (math.isfinite(tx_row) and math.isfinite(tx_col)):
        raise ValueError("failed to infer finite TX position")
    return tx_row, tx_col


def perturb_stage1_location(
    stage1_input: torch.Tensor,
    tile: str,
    error_pixels: int,
    resolution_m: float,
    distance_max_m: float,
) -> tuple[torch.Tensor, bool]:
    if error_pixels <= 0:
        return stage1_input, False
    arrays = stage1_input.numpy().copy()
    tx_map = arrays[1]
    distance_norm = arrays[2]
    flat_index = int(np.argmax(tx_map))
    marker_row, marker_col = np.unravel_index(flat_index, tx_map.shape)
    marker_height = float(tx_map[marker_row, marker_col])
    if marker_height <= 0:
        raise ValueError(f"{tile} has no in-grid TX marker")
    tx_row, tx_col = infer_tx_pixel_position(
        distance_norm,
        marker_row,
        marker_col,
        resolution_m,
        distance_max_m,
    )
    vectors = (
        (-error_pixels, 0),
        (error_pixels, 0),
        (0, -error_pixels),
        (0, error_pixels),
    )
    direction = stable_seed("bs-location-error", tile, error_pixels) % 4
    delta_row, delta_col = vectors[direction]
    shifted_row = marker_row + delta_row
    shifted_col = marker_col + delta_col
    arrays[1].fill(0.0)
    outside = not (
        0 <= shifted_row < tx_map.shape[0]
        and 0 <= shifted_col < tx_map.shape[1]
    )
    if not outside:
        arrays[1, shifted_row, shifted_col] = marker_height

    row_grid = np.arange(tx_map.shape[0], dtype=np.float64)[:, None]
    col_grid = np.arange(tx_map.shape[1], dtype=np.float64)[None, :]
    shifted_distance_m = resolution_m * np.hypot(
        row_grid - (tx_row + delta_row),
        col_grid - (tx_col + delta_col),
    )
    arrays[2] = np.clip(
        np.log1p(shifted_distance_m) / math.log1p(distance_max_m),
        0.0,
        1.0,
    ).astype(np.float32)
    return torch.from_numpy(arrays), outside


class SensitivityDataset(Dataset):
    """Add stage-wise targets and optional TX-location perturbation."""

    def __init__(
        self,
        base: Stage2BDirectionalDataset,
        location_error_pixels: int,
        tx_resolution_m: float,
        tx_distance_max_m: float,
    ) -> None:
        self.base = base
        self.location_error_pixels = int(location_error_pixels)
        self.tx_resolution_m = float(tx_resolution_m)
        self.tx_distance_max_m = float(tx_distance_max_m)
        self._iso_memmaps: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.base)

    def _iso_shard(self, shard: int) -> np.ndarray:
        if shard not in self._iso_memmaps:
            self._iso_memmaps[shard] = np.load(
                self.base.shard_root / f"p_iso_{shard:04d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._iso_memmaps[shard]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        row, _ = self.base.rows[index]
        tile = str(sample["tile"])
        stage1_input, outside = perturb_stage1_location(
            sample["stage1_input"],
            tile,
            self.location_error_pixels,
            self.tx_resolution_m,
            self.tx_distance_max_m,
        )
        sample["stage1_input"] = stage1_input
        sample["tx_marker_outside"] = outside

        tile_dir = self.base.normalized_root / tile
        sample["stage1_target_norm"] = torch.from_numpy(
            np.load(
                tile_dir / "path_gain_norm.npy",
                allow_pickle=False,
            ).astype(np.float32, copy=False)[None, ...]
        )
        sample["stage1_valid_mask"] = torch.from_numpy(
            np.load(
                tile_dir / "path_gain_valid_mask.npy",
                allow_pickle=False,
            ).astype(bool, copy=False)[None, ...]
        )
        shard = int(row["shard"])
        offset = int(row["offset"])
        iso_db = np.asarray(
            self._iso_shard(shard)[offset],
            dtype=np.float32,
        )
        iso_valid = np.isfinite(iso_db)
        sample["stage2a_target_db"] = torch.from_numpy(
            np.ascontiguousarray(
                np.where(iso_valid, iso_db, 0.0)[None, ...]
            )
        )
        sample["stage2a_valid_mask"] = torch.from_numpy(
            np.ascontiguousarray(iso_valid[None, ...])
        )
        return sample


@dataclass
class ErrorSums:
    squared: float = 0.0
    absolute: float = 0.0
    count: int = 0

    def update(self, error: torch.Tensor) -> None:
        if not error.numel():
            return
        self.squared += float(torch.sum(error.square()).item())
        self.absolute += float(torch.sum(error.abs()).item())
        self.count += int(error.numel())

    def metrics(self, prefix: str) -> dict[str, float | int]:
        if not self.count:
            return {
                f"{prefix}_rmse_db": math.nan,
                f"{prefix}_mae_db": math.nan,
                f"{prefix}_pixels": 0,
            }
        return {
            f"{prefix}_rmse_db": math.sqrt(self.squared / self.count),
            f"{prefix}_mae_db": self.absolute / self.count,
            f"{prefix}_pixels": self.count,
        }


@dataclass
class SimilaritySums:
    total: float = 0.0
    samples: int = 0

    def update(self, values: torch.Tensor) -> None:
        finite = values[torch.isfinite(values)]
        self.total += float(torch.sum(finite).item())
        self.samples += int(finite.numel())

    def mean(self) -> float:
        return self.total / self.samples if self.samples else math.nan


def masked_ssim_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    data_range_db: float,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute local SSIM while excluding invalid indoor/unhit pixels."""
    mask = valid_mask.float()
    padding = window_size // 2

    def pool(value: torch.Tensor) -> torch.Tensor:
        return functional.avg_pool2d(
            value,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )

    weight = pool(mask).clamp_min(1e-6)
    mean_prediction = pool(prediction * mask) / weight
    mean_target = pool(target * mask) / weight
    second_prediction = pool(prediction.square() * mask) / weight
    second_target = pool(target.square() * mask) / weight
    cross = pool(prediction * target * mask) / weight
    variance_prediction = (
        second_prediction - mean_prediction.square()
    ).clamp_min(0.0)
    variance_target = (
        second_target - mean_target.square()
    ).clamp_min(0.0)
    covariance = cross - mean_prediction * mean_target
    c1 = (0.01 * data_range_db) ** 2
    c2 = (0.03 * data_range_db) ** 2
    numerator = (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
    )
    denominator = (
        (mean_prediction.square() + mean_target.square() + c1)
        * (variance_prediction + variance_target + c2)
    )
    ssim_map = numerator / denominator.clamp_min(1e-12)
    # Require at least half of each local window to be valid so building edges
    # and unhit regions cannot dominate the structural score.
    valid_centers = valid_mask & (weight >= 0.5)
    flattened_scores: list[torch.Tensor] = []
    for sample in range(prediction.shape[0]):
        selected = valid_centers[sample]
        if not torch.any(selected):
            selected = valid_mask[sample]
        flattened_scores.append(torch.mean(ssim_map[sample][selected]))
    return torch.stack(flattened_scores)


def make_loader(
    dataset: Dataset,
    config: dict[str, Any],
    device: torch.device,
) -> DataLoader:
    workers = int(config["workers"])
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "persistent_workers": False,
    }
    if workers > 0:
        arguments["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(**arguments)


@torch.inference_mode()
def evaluate_condition(
    loader: DataLoader,
    stage1: nn.Module,
    stage2a: nn.Module,
    stage2b: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
    stage1_mean_db: float,
    stage1_std_db: float,
    stage2a_residual_center_db: float,
    stage2a_residual_scale_db: float,
    iso_mean_db: float,
    iso_std_db: float,
    ss_mean_db: float,
    ss_std_db: float,
    coarse_power_error_db: float,
) -> dict[str, float | int]:
    started = time.perf_counter()
    stage1_errors = ErrorSums()
    stage2a_errors = ErrorSums()
    final_errors = ErrorSums()
    unmeasured_errors = ErrorSums()
    measured_errors = ErrorSums()
    final_similarity = SimilaritySums()
    outside_markers = 0
    samples = 0
    for batch in loader:
        stage1_input = batch["stage1_input"].to(device, non_blocking=True)
        building = batch["building"].to(device, non_blocking=True)
        sparse_norm = batch["sparse_ss_norm"].to(device, non_blocking=True)
        sparse_mask = batch["sparse_mask"].to(device, non_blocking=True)
        target_norm = batch["target_ss_norm"].to(device, non_blocking=True)
        final_valid = batch["valid_mask"].to(
            device,
            non_blocking=True,
            dtype=torch.bool,
        )
        stage1_target = batch["stage1_target_norm"].to(
            device,
            non_blocking=True,
        )
        stage1_valid = batch["stage1_valid_mask"].to(
            device,
            non_blocking=True,
            dtype=torch.bool,
        )
        stage2a_target_db = batch["stage2a_target_db"].to(
            device,
            non_blocking=True,
        )
        stage2a_valid = batch["stage2a_valid_mask"].to(
            device,
            non_blocking=True,
            dtype=torch.bool,
        )
        outside_markers += int(batch["tx_marker_outside"].sum().item())
        samples += int(stage1_input.shape[0])
        if channels_last:
            stage1_input = stage1_input.contiguous(
                memory_format=torch.channels_last
            )
            building = building.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            dpm_norm = stage1(stage1_input)
            residual_norm = stage2a(torch.cat((building, dpm_norm), dim=1))
        dpm_db = dpm_norm.float() * stage1_std_db + stage1_mean_db
        corrected_iso_db = (
            dpm_db
            + stage2a_residual_center_db
            + residual_norm.float() * stage2a_residual_scale_db
        )
        coarse_for_final_db = corrected_iso_db + coarse_power_error_db
        coarse_for_final_norm = (
            coarse_for_final_db - iso_mean_db
        ) / iso_std_db
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            prediction_norm = stage2b(
                torch.cat(
                    (
                        building,
                        coarse_for_final_norm,
                        sparse_norm,
                        sparse_mask,
                    ),
                    dim=1,
                )
            )
        prediction_db = prediction_norm.float() * ss_std_db + ss_mean_db
        target_db = target_norm * ss_std_db + ss_mean_db

        stage1_error = (
            dpm_norm.float() - stage1_target
        ) * stage1_std_db
        stage1_errors.update(stage1_error[stage1_valid])
        stage2a_errors.update(
            (corrected_iso_db - stage2a_target_db)[stage2a_valid]
        )
        final_error = prediction_db - target_db
        measured = final_valid & sparse_mask.bool()
        unmeasured = final_valid & ~sparse_mask.bool()
        final_errors.update(final_error[final_valid])
        measured_errors.update(final_error[measured])
        unmeasured_errors.update(final_error[unmeasured])
        final_similarity.update(
            masked_ssim_per_sample(
                prediction_db,
                target_db,
                final_valid,
                data_range_db=6.0 * ss_std_db,
            )
        )
    result: dict[str, float | int] = {
        "samples": samples,
        "tx_marker_outside_samples": outside_markers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result.update(stage1_errors.metrics("stage1"))
    result.update(stage2a_errors.metrics("stage2a"))
    result.update(final_errors.metrics("final"))
    result.update(unmeasured_errors.metrics("final_unmeasured"))
    result.update(measured_errors.metrics("final_measured"))
    result["final_masked_ssim"] = final_similarity.mean()
    result["final_masked_ssim_samples"] = final_similarity.samples
    return result


def circular_separation_degrees(first: float, second: float) -> float:
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


@dataclass
class DirectionBin:
    label: str
    minimum: float
    maximum: float
    squared: float = 0.0
    absolute: float = 0.0
    pixels: int = 0
    separation_sum: float = 0.0
    pairs: int = 0

    def accepts(self, separation: float) -> bool:
        return self.minimum < separation <= self.maximum

    def update(self, separation: float, error: np.ndarray) -> None:
        values = error.astype(np.float64, copy=False)
        self.squared += float(np.sum(values * values))
        self.absolute += float(np.sum(np.abs(values)))
        self.pixels += int(values.size)
        self.separation_sum += float(separation)
        self.pairs += 1

    def result(self) -> dict[str, float | int | str]:
        return {
            "direction_error_bin": self.label,
            "minimum_separation_deg": self.minimum,
            "maximum_separation_deg": self.maximum,
            "mean_separation_deg": self.separation_sum / self.pairs,
            "direction_pairs": self.pairs,
            "joint_valid_pixels": self.pixels,
            "intrinsic_map_rmse_db": math.sqrt(self.squared / self.pixels),
            "intrinsic_map_mae_db": self.absolute / self.pixels,
        }


def evaluate_direction_difference(
    dataset: Stage2BDirectionalDataset,
    limit_tiles: int | None,
) -> list[dict[str, float | int | str]]:
    bins = [
        DirectionBin("about_5_deg", 0.0, 10.0),
        DirectionBin("about_15_deg", 10.0, 20.0),
        DirectionBin("about_30_deg", 20.0, 40.0),
        DirectionBin("40_to_80_deg", 40.0, 80.0),
        DirectionBin("80_to_120_deg", 80.0, 120.0),
        DirectionBin("120_to_180_deg", 120.0, 180.0),
    ]
    tile_rows = [dataset.rows[index][0] for index in range(0, len(dataset), 4)]
    if limit_tiles is not None:
        tile_rows = tile_rows[:limit_tiles]
    memmaps: dict[int, np.ndarray] = {}
    for row in tile_rows:
        shard = int(row["shard"])
        if shard not in memmaps:
            memmaps[shard] = np.load(
                dataset.shard_root / f"p_dir_{shard:04d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        offset = int(row["offset"])
        maps = np.asarray(memmaps[shard][offset], dtype=np.float32)
        azimuths = [float(row[f"azimuth_{index}_deg"]) for index in range(4)]
        for first in range(4):
            for second in range(first + 1, 4):
                separation = circular_separation_degrees(
                    azimuths[first],
                    azimuths[second],
                )
                valid = np.isfinite(maps[first]) & np.isfinite(maps[second])
                if not np.any(valid):
                    continue
                difference = maps[first][valid] - maps[second][valid]
                for current in bins:
                    if current.accepts(separation):
                        current.update(separation, difference)
                        break
    return [current.result() for current in bins if current.pairs]


def read_published_baselines(config: dict[str, Any]) -> dict[str, Any]:
    stage1_dir = Path(config["stage1_model_dir"])
    stage2a_dir = Path(config["stage2a_model_dir"])
    stage2b_dir = Path(config["output_dir"])
    return {
        "stage1": load_json(stage1_dir / "test_metrics.json"),
        "stage2a": load_json(stage2a_dir / "test_metrics.json"),
        "stage2b": load_json(stage2b_dir / "test_metrics.json"),
        "checkpoints": {
            "stage1": {
                "path": str(stage1_dir / "best.pt"),
                "sha256": sha256_file(stage1_dir / "best.pt"),
            },
            "stage2a": {
                "path": str(stage2a_dir / "best.pt"),
                "sha256": sha256_file(stage2a_dir / "best.pt"),
            },
            "stage2b": {
                "path": str(stage2b_dir / "best.pt"),
                "sha256": sha256_file(stage2b_dir / "best.pt"),
            },
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    published = payload["published_baselines"]
    conditions = payload["model_sensitivity"]
    directions = payload["direction_sensitivity"]
    lines = [
        "# Three-stage model BS-sensitivity report",
        "",
        "## Frozen baseline",
        "",
        "| Stage | Test RMSE (dB) | Test MAE (dB) |",
        "|---|---:|---:|",
        f"| Stage1 DPM | {fmt(published['stage1']['rmse_db'])} | {fmt(published['stage1']['mae_db'])} |",
        f"| Stage2-A isotropic | {fmt(published['stage2a']['test']['rmse_db'])} | {fmt(published['stage2a']['test']['mae_db'])} |",
    ]
    stage2b_100 = published["stage2b"]["test_by_sparse_points"]["100"]
    lines.append(
        f"| Stage2-B directional (100 points) | {fmt(stage2b_100['rmse_db'])} | {fmt(stage2b_100['mae_db'])} |"
    )
    lines.extend(
        [
            "",
            "## Reproduced pipeline sensitivity",
            "",
            "| Condition | Error | Stage1 RMSE | Stage2-A RMSE | Final RMSE | Final unmeasured RMSE | masked-SSIM | Delta vs reproduced baseline |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in conditions:
        error = (
            f"{row['location_error_pixels']} px"
            if row["condition_type"] == "location"
            else (
                f"{row['coarse_power_error_db']:+g} dB"
                if row["condition_type"] == "power"
                else "0"
            )
        )
        lines.append(
            "| {condition_id} | {error} | {stage1} | {stage2a} | {final} | {unmeasured} | {ssim} | {delta} |".format(
                condition_id=row["condition_id"],
                error=error,
                stage1=fmt(row["stage1_rmse_db"]),
                stage2a=fmt(row["stage2a_rmse_db"]),
                final=fmt(row["final_rmse_db"]),
                unmeasured=fmt(row["final_unmeasured_rmse_db"]),
                ssim=fmt(row["final_masked_ssim"], 4),
                delta=fmt(row["final_rmse_delta_vs_baseline_db"]),
            )
        )
    if directions:
        lines.extend(
            [
                "",
                "## Intrinsic direction sensitivity",
                "",
                "The current networks have no explicit azimuth input. The table therefore measures the ground-truth map difference between available Sionna directions, using only jointly valid pixels and the same link-budget offset.",
                "",
                "| Angular bin | Mean angle | Direction pairs | Map RMSE difference (dB) | Map MAE difference (dB) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in directions:
            lines.append(
                "| {label} | {angle} deg | {pairs} | {rmse} | {mae} |".format(
                    label=row["direction_error_bin"],
                    angle=fmt(row["mean_separation_deg"], 1),
                    pairs=row["direction_pairs"],
                    rmse=fmt(row["intrinsic_map_rmse_db"]),
                    mae=fmt(row["intrinsic_map_mae_db"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interface conclusion",
            "",
            "- Stage1 explicitly consumes TX location through a marker map and a distance map.",
            "- Stage2-A inherits location dependence through the frozen Stage1 prediction.",
            "- Stage2-B receives no explicit power or azimuth parameter; both are implicit in the absolute sparse signal values and directional target.",
            "- Reusing all three stages with a latent BS branch therefore requires an adapter that converts estimated location into the two Stage1 TX features. Explicit power/azimuth estimates need new conditioning channels or must remain auxiliary-only outputs.",
            "- masked-SSIM uses an 11x11 local window, fixed data range of six training standard deviations, and only windows with at least 50% valid outdoor pixels.",
            "",
        ]
    )
    return "\n".join(lines)


def load_stage2b(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(config["output_dir"]) / "best.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_configured_unet(config, in_channels=4).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model, checkpoint


def condition_specs(
    location_errors: Iterable[int],
    power_errors: Iterable[float],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "condition_id": "baseline_true_bs",
            "condition_type": "baseline",
            "location_error_pixels": 0,
            "coarse_power_error_db": 0.0,
        }
    ]
    specs.extend(
        {
            "condition_id": f"location_{error}px",
            "condition_type": "location",
            "location_error_pixels": int(error),
            "coarse_power_error_db": 0.0,
        }
        for error in location_errors
        if error > 0
    )
    specs.extend(
        {
            "condition_id": f"power_{error:+g}db",
            "condition_type": "power",
            "location_error_pixels": 0,
            "coarse_power_error_db": float(error),
        }
        for error in power_errors
        if error != 0
    )
    return specs


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if args.workers is not None:
        config["workers"] = args.workers
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device or config.get("device", "auto"))
    channels_last = bool(config.get("channels_last", False))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    seed_everything(int(config["seed"]))

    irt_normalization = load_json(Path(config["irt_normalization"]))
    iso_statistics = irt_normalization["statistics"]["train"]["p_iso"]
    ss_statistics = irt_normalization["stage2b_ss_normalization"]
    iso_mean_db = float(iso_statistics["mean_db"])
    iso_std_db = float(iso_statistics["std_db"])
    ss_mean_db = float(ss_statistics["mean_db"])
    ss_std_db = float(ss_statistics["std_db"])
    tx_normalization = load_json(
        Path(config["normalized_root"]) / "tx_feature_normalization.json"
    )
    tx_distance_max_m = float(tx_normalization["distance"]["max_m"])
    tx_resolution_m = 4.0

    print(f"Loading frozen models on {device}...", flush=True)
    stage1, stage2a, frozen = load_frozen_models(
        config,
        device,
        channels_last,
    )
    stage2b, stage2b_checkpoint = load_stage2b(
        config,
        device,
        channels_last,
    )
    published = read_published_baselines(config)
    base_dataset = make_base_dataset(
        config,
        ss_mean_db,
        ss_std_db,
        args.sparse_points,
        args.limit_tiles,
    )
    specs = condition_specs(
        parse_number_list(args.location_errors, int),
        parse_number_list(args.power_errors, float),
    )
    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] Evaluating {spec['condition_id']}...",
            flush=True,
        )
        dataset = SensitivityDataset(
            base_dataset,
            location_error_pixels=int(spec["location_error_pixels"]),
            tx_resolution_m=tx_resolution_m,
            tx_distance_max_m=tx_distance_max_m,
        )
        loader = make_loader(dataset, config, device)
        metrics = evaluate_condition(
            loader,
            stage1,
            stage2a,
            stage2b,
            device,
            amp_enabled,
            channels_last,
            float(frozen["stage1_mean_db"]),
            float(frozen["stage1_std_db"]),
            float(frozen["stage2a_residual_center_db"]),
            float(frozen["stage2a_residual_scale_db"]),
            iso_mean_db,
            iso_std_db,
            ss_mean_db,
            ss_std_db,
            float(spec["coarse_power_error_db"]),
        )
        row = {**spec, **metrics}
        results.append(row)
        print(
            json.dumps(
                {
                    "condition": spec["condition_id"],
                    "stage1_rmse_db": row["stage1_rmse_db"],
                    "stage2a_rmse_db": row["stage2a_rmse_db"],
                    "final_rmse_db": row["final_rmse_db"],
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    baseline_rmse = float(results[0]["final_rmse_db"])
    for row in results:
        row["final_rmse_delta_vs_baseline_db"] = (
            float(row["final_rmse_db"]) - baseline_rmse
        )

    direction_results: list[dict[str, Any]] = []
    if not args.skip_direction:
        print("Evaluating intrinsic direction-map differences...", flush=True)
        direction_results = evaluate_direction_difference(
            base_dataset,
            args.direction_limit_tiles,
        )

    payload = {
        "status": "ok",
        "config": str(config_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sparse_points": args.sparse_points,
        "test_tiles": len(base_dataset) // 4,
        "test_directional_samples": len(base_dataset),
        "published_baselines": published,
        "stage2b_best_epoch": int(stage2b_checkpoint["epoch"]),
        "model_sensitivity": results,
        "direction_sensitivity": direction_results,
        "methodology": {
            "location": "rebuild Stage1 TX marker and log-distance map after a deterministic cardinal shift per tile",
            "power": "add dB bias to the frozen Stage2-A coarse map before Stage2-B normalization",
            "direction": "ground-truth p_dir map difference for azimuth pairs binned by circular angular separation",
            "direction_limit": "no explicit azimuth channel exists in the current network, so model-level azimuth perturbation is not defined",
            "ssim": "11x11 local masked SSIM; fixed data range=6*training SS std; window valid fraction >= 0.5",
        },
    }
    atomic_write_json(output_dir / "results.json", payload)
    write_csv(output_dir / "model_sensitivity.csv", results)
    write_csv(output_dir / "direction_sensitivity.csv", direction_results)
    atomic_write_text(output_dir / "REPORT.md", render_report(payload))
    print(f"Results written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate the frozen Stage1/2-A/2-B pipeline on random-BS Pilot labels."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bs_inversion_dataset import BSInversionPilotDataset, load_json
from bs_parameter_model import BSParameterEstimator
from latent_bs_fusion import (
    build_latent_fusion_features,
    configured_feature_names,
    configured_stage2b_in_channels,
)
from run_sionna_dataset import atomic_write_json, sha256_file
from train_stage2b_directional_ss import (
    build_configured_unet,
    load_frozen_models,
    resolve_device,
    seed_everything,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STAGE2B_CONFIG = SCRIPT_DIR / "config_stage2b_sionna35_depth8_27360_paper_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\桌面\dac\02_inversion\bs_inversion_pilot_v1"),
    )
    parser.add_argument("--stage2b-config", type=Path, default=DEFAULT_STAGE2B_CONFIG)
    parser.add_argument(
        "--stage2b-checkpoint",
        type=Path,
        help="Optional Stage2-B checkpoint override for ablation evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\桌面\dac\03_runs\experiments\bs_inversion_pilot_oracle_pipeline_v1.json"
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit-sites", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--estimator-config",
        type=Path,
        help="If provided, feed estimated rather than true BS location to Stage1.",
    )
    parser.add_argument("--estimator-checkpoint", type=Path)
    return parser.parse_args()


class ErrorSums:
    def __init__(self) -> None:
        self.count = 0
        self.squared = 0.0
        self.absolute = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().double()
        self.count += int(values.numel())
        self.squared += float(torch.sum(values.square()).item())
        self.absolute += float(torch.sum(torch.abs(values)).item())

    def metrics(self, prefix: str) -> dict[str, float | int]:
        return {
            f"{prefix}_pixels": self.count,
            f"{prefix}_rmse_db": math.sqrt(self.squared / self.count),
            f"{prefix}_mae_db": self.absolute / self.count,
        }


def edge_bin_label(edge_distance_m: float) -> str:
    for lower, upper in ((64, 96), (96, 128), (128, 160), (160, 192)):
        if edge_distance_m < upper:
            return f"{lower:03d}-{upper:03d}m"
    return "192-256m"


def center_bin_label(center_distance_m: float) -> str:
    for lower, upper in ((0, 64), (64, 128), (128, 192)):
        if center_distance_m < upper:
            return f"{lower:03d}-{upper:03d}m"
    return "192m-plus"


def update_position_group(
    groups: dict[str, dict[str, Any]],
    label: str,
    iso_error: torch.Tensor,
    final_error: torch.Tensor,
    location_error_px: float | None,
) -> None:
    group = groups.setdefault(
        label,
        {
            "directional_samples": 0,
            "iso": ErrorSums(),
            "final": ErrorSums(),
            "location_errors_px": [],
        },
    )
    group["directional_samples"] += 1
    group["iso"].update(iso_error)
    group["final"].update(final_error)
    if location_error_px is not None:
        group["location_errors_px"].append(float(location_error_px))


def finalize_position_groups(
    groups: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, group in sorted(groups.items()):
        row: dict[str, Any] = {
            "directional_samples": int(group["directional_samples"]),
            **group["iso"].metrics("stage2a_iso"),
            **group["final"].metrics("final"),
        }
        location_errors = np.asarray(group["location_errors_px"], dtype=np.float64)
        if location_errors.size:
            row["location_error_px_mean"] = float(location_errors.mean())
            row["location_error_px_p90"] = float(
                np.percentile(location_errors, 90.0)
            )
        result[label] = row
    return result


def make_stage1_input(
    building: torch.Tensor,
    tx_xy_m: torch.Tensor,
    tx_height_m: torch.Tensor,
    tx_height_cap_m: float,
    distance_max_m: float,
    in_channels: int = 3,
    long_distance_max_m: float | None = None,
) -> torch.Tensor:
    batch, _, rows, cols = building.shape
    device = building.device
    x_centers = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) * 4.0
    y_centers = 512.0 - (
        torch.arange(rows, device=device, dtype=torch.float32) + 0.5
    ) * 4.0
    delta_x = x_centers.view(1, 1, cols) - tx_xy_m[:, 0].view(batch, 1, 1)
    delta_y = y_centers.view(1, rows, 1) - tx_xy_m[:, 1].view(batch, 1, 1)
    distance = torch.sqrt(delta_x.square() + delta_y.square())
    distance_norm = torch.clamp(
        torch.log1p(distance) / math.log1p(distance_max_m), 0.0, 1.0
    )
    tx_map = torch.zeros((batch, rows, cols), device=device, dtype=torch.float32)
    tx_col = torch.floor(tx_xy_m[:, 0] / 4.0).long().clamp(0, cols - 1)
    tx_row = torch.floor((512.0 - tx_xy_m[:, 1]) / 4.0).long().clamp(0, rows - 1)
    tx_map[torch.arange(batch, device=device), tx_row, tx_col] = torch.clamp(
        tx_height_m / tx_height_cap_m, max=1.0
    )
    base = (building, tx_map[:, None], distance_norm[:, None])
    if in_channels == 3:
        return torch.cat(base, dim=1)
    if in_channels != 6:
        raise ValueError(f"unsupported Stage1 input channel count: {in_channels}")
    if long_distance_max_m is None or long_distance_max_m <= distance_max_m:
        raise ValueError("six-channel Stage1 requires a larger long-distance maximum")
    long_distance_norm = torch.clamp(
        torch.log1p(distance) / math.log1p(long_distance_max_m), 0.0, 1.0
    )
    delta_x_norm = delta_x.expand(batch, rows, cols) / 512.0
    delta_y_norm = delta_y.expand(batch, rows, cols) / 512.0
    return torch.cat(
        (
            *base,
            long_distance_norm[:, None],
            delta_x_norm[:, None],
            delta_y_norm[:, None],
        ),
        dim=1,
    )


def sparse_calibrated_iso(
    corrected_iso_db: torch.Tensor,
    target_db: torch.Tensor,
    sparse_mask: torch.Tensor,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for sample in range(corrected_iso_db.shape[0]):
        measured = sparse_mask[sample].bool()
        offset = torch.median(
            target_db[sample][measured] - corrected_iso_db[sample][measured]
        )
        predictions.append(corrected_iso_db[sample] + offset)
    return torch.stack(predictions, dim=0)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    stage2b_config = load_json(args.stage2b_config.resolve())
    seed_everything(int(stage2b_config["seed"]))
    device = resolve_device(stage2b_config["device"])
    amp_enabled = bool(stage2b_config["amp"]) and device.type == "cuda"
    channels_last = bool(stage2b_config["channels_last"]) and device.type == "cuda"
    irt_normalization = load_json(stage2b_config["irt_normalization"])
    iso_stats = irt_normalization["statistics"]["train"]["p_iso"]
    ss_stats = irt_normalization["stage2b_ss_normalization"]
    iso_mean_db = float(iso_stats["mean_db"])
    iso_std_db = float(iso_stats["std_db"])
    ss_mean_db = float(ss_stats["mean_db"])
    ss_std_db = float(ss_stats["std_db"])
    dataset = BSInversionPilotDataset(
        args.dataset_root,
        split=args.split,
        seed=int(stage2b_config["seed"]),
        augment=False,
        limit_sites=args.limit_sites,
        signal_mean_db=ss_mean_db,
        signal_std_db=ss_std_db,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    stage1, stage2a, frozen = load_frozen_models(
        stage2b_config, device, channels_last
    )
    stage2b_path = args.stage2b_checkpoint or (
        Path(stage2b_config["output_dir"]) / "best.pt"
    )
    stage2b_checkpoint = torch.load(
        stage2b_path, map_location=device, weights_only=False
    )
    stage2b_checkpoint_config = stage2b_checkpoint["config"]
    stage2b_in_channels = configured_stage2b_in_channels(
        stage2b_checkpoint_config
    )
    stage2b = build_configured_unet(
        stage2b_checkpoint_config, in_channels=stage2b_in_channels
    ).to(device)
    stage2b.load_state_dict(stage2b_checkpoint["model"], strict=True)
    stage2b.eval().requires_grad_(False)
    if channels_last:
        stage2b.to(memory_format=torch.channels_last)
    estimator: BSParameterEstimator | None = None
    estimator_checkpoint_path: Path | None = None
    estimator_signal_mean_db = 0.0
    estimator_signal_std_db = 1.0
    estimator_power_mean_db = 0.0
    estimator_power_std_db = 1.0
    if args.estimator_config is not None:
        estimator_config = load_json(args.estimator_config.resolve())
        estimator_checkpoint_path = args.estimator_checkpoint or (
            Path(estimator_config["output_dir"]) / "best.pt"
        )
        estimator_checkpoint = torch.load(
            estimator_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        estimator_model_config = estimator_checkpoint["config"]["model"]
        estimator = BSParameterEstimator(
            in_channels=3,
            base_channels=int(estimator_model_config["base_channels"]),
            scalar_hidden_channels=int(
                estimator_model_config["scalar_hidden_channels"]
            ),
            softargmax_temperature=float(
                estimator_model_config["softargmax_temperature"]
            ),
        ).to(device)
        estimator.load_state_dict(estimator_checkpoint["model"], strict=True)
        estimator.eval().requires_grad_(False)
        if channels_last:
            estimator.to(memory_format=torch.channels_last)
        pilot_normalization = load_json(
            Path(estimator_config["dataset_root"]) / "normalization.json"
        )
        estimator_signal_stats = pilot_normalization[
            "directional_signal_strength_db"
        ]
        estimator_signal_mean_db = float(estimator_signal_stats["mean"])
        estimator_signal_std_db = float(estimator_signal_stats["std"])
        estimator_manifest = load_json(
            Path(estimator_config["dataset_root"]) / "run_manifest.json"
        )
        estimator_data_config = load_json(Path(estimator_manifest["config"]))
        estimator_power_config = estimator_data_config["effective_power"]
        estimator_power_mean_db = float(
            estimator_power_config["normalization_mean_db"]
        )
        estimator_power_std_db = float(
            estimator_power_config["normalization_std_db"]
        )
    tx_normalization = load_json(
        Path(stage2b_config["stage1_model_dir"]) / "tx_feature_normalization.json"
    )
    tx_height_cap = float(tx_normalization["transmitter_height"]["cap_m"])
    distance_max = float(tx_normalization["distance"]["max_m"])
    long_distance_max = float(
        tx_normalization.get("long_distance", {}).get(
            "max_m", math.sqrt(2.0) * 512.0
        )
    )
    iso_errors = ErrorSums()
    final_errors = ErrorSums()
    final_unmeasured_errors = ErrorSums()
    baseline_errors = ErrorSums()
    baseline_unmeasured_errors = ErrorSums()
    measured_errors = ErrorSums()
    location_errors_px: list[float] = []
    power_errors_db: list[float] = []
    direction_errors_deg: list[float] = []
    edge_groups: dict[str, dict[str, Any]] = {}
    center_groups: dict[str, dict[str, Any]] = {}
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            building = batch["building"].to(device, non_blocking=True)
            sparse_norm = batch["sparse_ss_norm"].to(device, non_blocking=True)
            sparse_mask = batch["sparse_mask"].to(device, non_blocking=True)
            target_db = batch["target_map_db"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True).bool()
            iso_target = batch["isotropic_path_gain_db"].to(
                device, non_blocking=True
            )
            iso_valid = batch["isotropic_valid_mask"].to(
                device, non_blocking=True
            ).bool()
            true_tx_xy_m = batch["tx_xy_m"].to(device, non_blocking=True)
            tx_height_m = batch["tx_height_m"].to(device, non_blocking=True)
            tx_xy_m = true_tx_xy_m
            estimated_outputs: dict[str, torch.Tensor] | None = None
            batch_location_errors_px: torch.Tensor | None = None
            if estimator is not None:
                sparse_db = sparse_norm.float() * ss_std_db + ss_mean_db
                estimator_sparse_norm = torch.where(
                    sparse_mask.bool(),
                    (sparse_db - estimator_signal_mean_db)
                    / estimator_signal_std_db,
                    0.0,
                )
                estimator_input = torch.cat(
                    (building, estimator_sparse_norm, sparse_mask), dim=1
                )
                if channels_last:
                    estimator_input = estimator_input.contiguous(
                        memory_format=torch.channels_last
                    )
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    estimated_outputs = estimator(estimator_input)
                estimated_row_col = estimated_outputs["row_col_px"].float()
                tx_xy_m = torch.stack(
                    (
                        (estimated_row_col[:, 1] + 0.5) * 4.0,
                        512.0 - (estimated_row_col[:, 0] + 0.5) * 4.0,
                    ),
                    dim=1,
                )
                true_row_col = torch.stack(
                    (
                        (512.0 - true_tx_xy_m[:, 1]) / 4.0 - 0.5,
                        true_tx_xy_m[:, 0] / 4.0 - 0.5,
                    ),
                    dim=1,
                )
                batch_location_errors_px = torch.linalg.vector_norm(
                    estimated_row_col - true_row_col, dim=1
                )
                location_errors_px.extend(
                    batch_location_errors_px.detach().cpu().tolist()
                )
                predicted_power_db = (
                    estimated_outputs["power_norm"].float()
                    * estimator_power_std_db
                    + estimator_power_mean_db
                )
                target_power_db = batch["target_power_db"].to(
                    device, non_blocking=True
                )
                power_errors_db.extend(
                    torch.abs(predicted_power_db - target_power_db)
                    .detach()
                    .cpu()
                    .tolist()
                )
                predicted_direction = estimated_outputs["direction_raw"].float()
                predicted_direction = predicted_direction / torch.linalg.vector_norm(
                    predicted_direction, dim=1, keepdim=True
                ).clamp_min(1e-6)
                target_direction = batch["target_direction_sin_cos"].to(
                    device, non_blocking=True
                ).float()
                target_direction = target_direction / torch.linalg.vector_norm(
                    target_direction, dim=1, keepdim=True
                ).clamp_min(1e-6)
                predicted_angle = torch.atan2(
                    predicted_direction[:, 0], predicted_direction[:, 1]
                )
                target_angle = torch.atan2(
                    target_direction[:, 0], target_direction[:, 1]
                )
                angle_difference = torch.remainder(
                    predicted_angle - target_angle + math.pi, 2.0 * math.pi
                ) - math.pi
                direction_errors_deg.extend(
                    torch.abs(torch.rad2deg(angle_difference)).detach().cpu().tolist()
                )
            stage1_input = make_stage1_input(
                building,
                tx_xy_m,
                tx_height_m,
                tx_height_cap,
                distance_max,
                in_channels=int(frozen.get("stage1_in_channels", 3)),
                long_distance_max_m=long_distance_max,
            )
            if channels_last:
                building = building.contiguous(memory_format=torch.channels_last)
                stage1_input = stage1_input.contiguous(
                    memory_format=torch.channels_last
                )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                dpm_norm = stage1(stage1_input)
                residual_norm = stage2a(torch.cat((building, dpm_norm), dim=1))
            dpm_db = (
                dpm_norm.float() * float(frozen["stage1_std_db"])
                + float(frozen["stage1_mean_db"])
            )
            corrected_iso_db = (
                dpm_db
                + float(frozen["stage2a_residual_center_db"])
                + residual_norm.float() * float(frozen["stage2a_residual_scale_db"])
            )
            coarse_norm = (corrected_iso_db - iso_mean_db) / iso_std_db
            fusion_feature_names = configured_feature_names(
                stage2b_checkpoint_config
            )
            if fusion_feature_names:
                if estimated_outputs is None:
                    raise ValueError(
                        "latent-fusion Stage2-B evaluation requires an estimator"
                    )
                latent_features = build_latent_fusion_features(
                    estimated_outputs,
                    stage2b_checkpoint_config["latent_fusion"],
                    estimator_signal_mean_db=estimator_signal_mean_db,
                    estimator_signal_std_db=estimator_signal_std_db,
                    stage2b_signal_mean_db=ss_mean_db,
                    stage2b_signal_std_db=ss_std_db,
                    estimator_power_mean_db=estimator_power_mean_db,
                    estimator_power_std_db=estimator_power_std_db,
                    rows=int(building.shape[-2]),
                    cols=int(building.shape[-1]),
                )
            else:
                latent_features = building.new_empty(
                    (building.shape[0], 0, building.shape[-2], building.shape[-1])
                )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                stage2b_input = torch.cat(
                    (
                        building,
                        coarse_norm,
                        sparse_norm,
                        sparse_mask,
                        latent_features,
                    ),
                    dim=1,
                )
                if channels_last:
                    stage2b_input = stage2b_input.contiguous(
                        memory_format=torch.channels_last
                    )
                prediction_norm = stage2b(stage2b_input)
            prediction_db = prediction_norm.float() * ss_std_db + ss_mean_db
            baseline_db = sparse_calibrated_iso(
                corrected_iso_db, target_db, sparse_mask
            )
            unmeasured = valid & ~sparse_mask.bool()
            measured = valid & sparse_mask.bool()
            iso_errors.update((corrected_iso_db - iso_target)[iso_valid])
            final_errors.update((prediction_db - target_db)[valid])
            final_unmeasured_errors.update((prediction_db - target_db)[unmeasured])
            measured_errors.update((prediction_db - target_db)[measured])
            baseline_errors.update((baseline_db - target_db)[valid])
            baseline_unmeasured_errors.update((baseline_db - target_db)[unmeasured])
            for sample_index in range(building.shape[0]):
                true_x = float(true_tx_xy_m[sample_index, 0].item())
                true_y = float(true_tx_xy_m[sample_index, 1].item())
                edge_distance = min(true_x, true_y, 512.0 - true_x, 512.0 - true_y)
                center_distance = math.hypot(true_x - 256.0, true_y - 256.0)
                sample_location_error = (
                    float(batch_location_errors_px[sample_index].item())
                    if batch_location_errors_px is not None
                    else None
                )
                iso_error = (corrected_iso_db[sample_index] - iso_target[sample_index])[
                    iso_valid[sample_index]
                ]
                final_error = (prediction_db[sample_index] - target_db[sample_index])[
                    valid[sample_index]
                ]
                update_position_group(
                    edge_groups,
                    edge_bin_label(edge_distance),
                    iso_error,
                    final_error,
                    sample_location_error,
                )
                update_position_group(
                    center_groups,
                    center_bin_label(center_distance),
                    iso_error,
                    final_error,
                    sample_location_error,
                )
            samples += int(building.shape[0])
    report: dict[str, Any] = {
        "version": 1,
        "status": "ok",
        "meaning": (
            "estimated BS position supplied to Stage1"
            if estimator is not None
            else "oracle upper bound: true random-BS position/height supplied to Stage1"
        ),
        "split": args.split,
        "sites": len(dataset.entries),
        "directional_samples": samples,
        "sparse_points": 100,
        "device": str(device),
        "stage1_checkpoint": str(Path(stage2b_config["stage1_model_dir"]) / "best.pt"),
        "stage2a_checkpoint": str(Path(stage2b_config["stage2a_model_dir"]) / "best.pt"),
        "stage2b_checkpoint": str(stage2b_path),
        "stage2b_checkpoint_sha256": sha256_file(stage2b_path),
        "stage2b_input_channels": stage2b_in_channels,
        "stage2b_latent_features": list(
            configured_feature_names(stage2b_checkpoint_config)
        ),
        "estimator_checkpoint": (
            str(estimator_checkpoint_path)
            if estimator_checkpoint_path is not None
            else None
        ),
        "estimator_checkpoint_sha256": (
            sha256_file(estimator_checkpoint_path)
            if estimator_checkpoint_path is not None
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report.update(iso_errors.metrics("oracle_stage2a_iso"))
    report.update(baseline_errors.metrics("sparse_calibrated_iso_baseline"))
    report.update(
        baseline_unmeasured_errors.metrics("sparse_calibrated_iso_baseline_unmeasured")
    )
    report.update(final_errors.metrics("oracle_final"))
    report.update(final_unmeasured_errors.metrics("oracle_final_unmeasured"))
    report.update(measured_errors.metrics("oracle_final_measured"))
    # Generic aliases keep estimated-coordinate reports semantically clear while
    # preserving the original oracle_* keys used by earlier result files.
    report.update(iso_errors.metrics("stage2a_iso"))
    report.update(final_errors.metrics("final"))
    report.update(final_unmeasured_errors.metrics("final_unmeasured"))
    report.update(measured_errors.metrics("final_measured"))
    report["coordinate_source"] = "estimated" if estimator is not None else "oracle"
    report["position_bins"] = {
        "by_minimum_edge_distance": finalize_position_groups(edge_groups),
        "by_center_distance": finalize_position_groups(center_groups),
    }
    report["final_rmse_improvement_vs_baseline_db"] = (
        float(report["sparse_calibrated_iso_baseline_rmse_db"])
        - float(report["oracle_final_rmse_db"])
    )
    if location_errors_px:
        location_array = np.asarray(location_errors_px, dtype=np.float64)
        report.update(
            {
                "estimated_location_error_px_mean": float(location_array.mean()),
                "estimated_location_error_px_median": float(
                    np.median(location_array)
                ),
                "estimated_location_error_px_p90": float(
                    np.percentile(location_array, 90.0)
                ),
                "estimated_location_error_px_rmse": float(
                    np.sqrt(np.mean(np.square(location_array)))
                ),
            }
        )
    if power_errors_db:
        power_array = np.asarray(power_errors_db, dtype=np.float64)
        direction_array = np.asarray(direction_errors_deg, dtype=np.float64)
        report.update(
            {
                "estimated_power_abs_error_db_mean": float(power_array.mean()),
                "estimated_power_abs_error_db_median": float(
                    np.median(power_array)
                ),
                "estimated_power_abs_error_db_p90": float(
                    np.percentile(power_array, 90.0)
                ),
                "estimated_direction_abs_error_deg_mean": float(
                    direction_array.mean()
                ),
                "estimated_direction_abs_error_deg_median": float(
                    np.median(direction_array)
                ),
                "estimated_direction_abs_error_deg_p90": float(
                    np.percentile(direction_array, 90.0)
                ),
            }
        )
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

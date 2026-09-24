#!/usr/bin/env python3
"""Jointly adapt Stage1 and Stage2-A to Pilot random-BS samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bs_inversion_dataset import BSInversionPilotDataset, load_json
from bs_parameter_model import BSParameterEstimator
from evaluate_pilot_oracle_pipeline import make_stage1_input
from run_sionna_dataset import sha256_file
from stage2_models import ResidualUNet
from train_stage2b_directional_ss import (
    build_configured_unet,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class ErrorSums:
    def __init__(self) -> None:
        self.count = 0
        self.squared = 0.0
        self.absolute = 0.0

    def update(self, errors: torch.Tensor) -> None:
        values = errors.detach().double()
        self.count += int(values.numel())
        self.squared += float(torch.sum(values.square()).item())
        self.absolute += float(torch.sum(values.abs()).item())

    def metrics(self) -> dict[str, float | int]:
        if not self.count:
            raise RuntimeError("no valid isotropic pixels")
        return {
            "pixels": self.count,
            "rmse_db": math.sqrt(self.squared / self.count),
            "mae_db": self.absolute / self.count,
        }


def make_loader(
    dataset: BSInversionPilotDataset,
    config: dict[str, Any],
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    workers = int(config["workers"])
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "persistent_workers": False,
    }
    if workers:
        arguments["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(**arguments)


def load_estimator(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, float, float, Path, str]:
    estimator_config = load_json(Path(config["estimator_config"]))
    checkpoint_path = Path(config["estimator_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    model = BSParameterEstimator(
        in_channels=3,
        base_channels=int(model_config["base_channels"]),
        scalar_hidden_channels=int(model_config["scalar_hidden_channels"]),
        softargmax_temperature=float(model_config["softargmax_temperature"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    normalization = load_json(Path(estimator_config["dataset_root"]) / "normalization.json")
    stats = normalization["directional_signal_strength_db"]
    return (
        model,
        float(stats["mean"]),
        float(stats["std"]),
        checkpoint_path,
        sha256_file(checkpoint_path),
    )


def load_initial_models(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    stage1_path = Path(config["initial_stage1_checkpoint"])
    stage2a_path = Path(config["initial_stage2a_checkpoint"])
    stage1_checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
    stage1_base = int(stage1_checkpoint.get("args", {}).get("base_channels", 32))
    source_in_channels = int(
        stage1_checkpoint.get("args", {}).get("in_channels", 3)
    )
    target_in_channels = int(
        config.get("stage1_input_channels", source_in_channels)
    )
    if target_in_channels not in {3, 6}:
        raise ValueError("Stage1 input channels must be 3 or 6")
    stage1 = ResidualUNet(
        in_channels=target_in_channels, base_channels=stage1_base
    ).to(device)
    source_state = stage1_checkpoint["model"]
    target_state = stage1.state_dict()
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if key == "input.block.0.weight" and target_in_channels != source_in_channels:
            target_value.zero_()
            target_value[:, :source_in_channels].copy_(
                source_value.to(device=target_value.device, dtype=target_value.dtype)
            )
        else:
            target_value.copy_(
                source_value.to(device=target_value.device, dtype=target_value.dtype)
            )
    stage1.load_state_dict(target_state, strict=True)
    stage2a_checkpoint = torch.load(stage2a_path, map_location=device, weights_only=False)
    stage2a = build_configured_unet(stage2a_checkpoint["config"], in_channels=2).to(
        device
    )
    stage2a.load_state_dict(stage2a_checkpoint["model"], strict=True)
    if channels_last:
        stage1.to(memory_format=torch.channels_last)
        stage2a.to(memory_format=torch.channels_last)
    stage1_normalization = load_json(Path(config["initial_stage1_model_dir"]) / "normalization.json")
    path_gain = stage1_normalization["statistics"]["path_gain"]
    metadata = {
        "stage1_base_channels": stage1_base,
        "stage1_in_channels": target_in_channels,
        "source_stage1_in_channels": source_in_channels,
        "stage1_mean_db": float(path_gain["mean_db"]),
        "stage1_std_db": float(path_gain["std_db"]),
        "residual_center_db": float(stage2a_checkpoint["residual_center_db"]),
        "residual_scale_db": float(stage2a_checkpoint["residual_scale_db"]),
        "initial_stage1_sha256": sha256_file(stage1_path),
        "initial_stage2a_sha256": sha256_file(stage2a_path),
    }
    return stage1, stage2a, metadata


def run_epoch(
    loader: DataLoader,
    estimator: nn.Module,
    stage1: nn.Module,
    stage2a: nn.Module,
    *,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
    estimator_signal_mean_db: float,
    estimator_signal_std_db: float,
    stage1_mean_db: float,
    stage1_std_db: float,
    residual_center_db: float,
    residual_scale_db: float,
    loss_scale_db: float,
    tx_height_cap_m: float,
    distance_max_m: float,
    stage1_in_channels: int,
    long_distance_max_m: float,
    position_jitter_probability: float,
    position_jitter_std_px: float,
    far_position_loss_multiplier: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip: float,
    progress_every: int,
    phase: str,
    epoch: int,
) -> dict[str, float | int]:
    training = optimizer is not None
    stage1.train(training)
    stage2a.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    errors = ErrorSums()
    location_errors: list[float] = []
    loss_sum = 0.0
    batches = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        building = batch["building"].to(device, non_blocking=True)
        sparse_norm = batch["sparse_ss_norm"].to(device, non_blocking=True)
        sparse_mask = batch["sparse_mask"].to(device, non_blocking=True)
        target_db = batch["isotropic_path_gain_db"].to(device, non_blocking=True)
        valid = batch["isotropic_valid_mask"].to(device, non_blocking=True).bool()
        true_row_col = batch["target_row_col_px"].to(device, non_blocking=True)
        tx_height_m = batch["tx_height_m"].to(device, non_blocking=True)
        sparse_db = (
            sparse_norm.float() * estimator_signal_std_db
            + estimator_signal_mean_db
        )
        estimator_sparse_norm = torch.where(
            sparse_mask.bool(),
            (sparse_db - estimator_signal_mean_db) / estimator_signal_std_db,
            0.0,
        )
        estimator_input = torch.cat((building, estimator_sparse_norm, sparse_mask), dim=1)
        if channels_last:
            estimator_input = estimator_input.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            estimated_row_col = estimator(estimator_input)["row_col_px"].float()
        location_errors.extend(
            torch.linalg.vector_norm(estimated_row_col - true_row_col, dim=1)
            .cpu()
            .tolist()
        )
        input_row_col = estimated_row_col
        if training and position_jitter_probability > 0.0:
            apply_jitter = (
                torch.rand(
                    (input_row_col.shape[0], 1),
                    device=input_row_col.device,
                )
                < position_jitter_probability
            )
            jitter = torch.randn_like(input_row_col) * position_jitter_std_px
            input_row_col = torch.where(
                apply_jitter,
                torch.clamp(input_row_col + jitter, 0.0, 127.0),
                input_row_col,
            )
        tx_xy_m = torch.stack(
            (
                (input_row_col[:, 1] + 0.5) * 4.0,
                512.0 - (input_row_col[:, 0] + 0.5) * 4.0,
            ),
            dim=1,
        )
        stage1_input = make_stage1_input(
            building,
            tx_xy_m,
            tx_height_m,
            tx_height_cap_m,
            distance_max_m,
            in_channels=stage1_in_channels,
            long_distance_max_m=long_distance_max_m,
        )
        if channels_last:
            building = building.contiguous(memory_format=torch.channels_last)
            stage1_input = stage1_input.contiguous(memory_format=torch.channels_last)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            dpm_norm = stage1(stage1_input)
            residual_norm = stage2a(torch.cat((building, dpm_norm), dim=1))
            prediction_db = (
                dpm_norm.float() * stage1_std_db
                + stage1_mean_db
                + residual_center_db
                + residual_norm.float() * residual_scale_db
            )
            normalized_error_map = (prediction_db - target_db) / loss_scale_db
            if training and far_position_loss_multiplier > 1.0:
                squared = normalized_error_map.square() * valid
                per_sample = squared.flatten(1).sum(dim=1) / torch.clamp_min(
                    valid.flatten(1).sum(dim=1), 1
                )
                radius_px = torch.linalg.vector_norm(
                    true_row_col - 63.5, dim=1
                )
                radius_weight = 1.0 + (
                    far_position_loss_multiplier - 1.0
                ) * torch.clamp(radius_px / 68.0, 0.0, 1.0)
                loss = torch.sum(per_sample * radius_weight) / torch.sum(
                    radius_weight
                )
            else:
                normalized_error = normalized_error_map[valid]
                loss = functional.mse_loss(
                    normalized_error,
                    torch.zeros_like(normalized_error),
                    reduction="mean",
                )
        if training:
            assert optimizer is not None and scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(stage1.parameters()) + list(stage2a.parameters()), gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        errors.update(prediction_db[valid] - target_db[valid])
        loss_sum += float(loss.detach().item())
        batches += 1
        if progress_every and (
            batch_index % progress_every == 0 or batch_index == len(loader)
        ):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches_total": len(loader),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    metrics: dict[str, float | int] = {"loss": loss_sum / batches}
    metrics.update(errors.metrics())
    location_array = np.asarray(location_errors, dtype=np.float64)
    metrics.update(
        {
            "location_mean_error_px": float(location_array.mean()),
            "location_p90_error_px": float(np.percentile(location_array, 90.0)),
        }
    )
    return metrics


def save_best_models(
    stage1: nn.Module,
    stage2a: nn.Module,
    *,
    config: dict[str, Any],
    metadata: dict[str, Any],
    estimator_checkpoint: Path,
    estimator_hash: str,
    epoch: int,
    val_metrics: dict[str, Any],
    stage1_output_dir: Path,
    stage2a_output_dir: Path,
) -> tuple[str, str]:
    stage1_path = stage1_output_dir / "best.pt"
    atomic_torch_save(
        stage1_path,
        {
            "version": 1,
            "model": stage1.state_dict(),
            "epoch": epoch,
            "best_val_rmse_db": float(val_metrics["rmse_db"]),
            "args": {
                "base_channels": int(metadata["stage1_base_channels"]),
                "in_channels": int(metadata["stage1_in_channels"]),
                "input_encoding": (
                    "building_txheight_legacydistance_longdistance_dx_dy"
                    if int(metadata["stage1_in_channels"]) == 6
                    else "building_txheight_distance"
                ),
            },
            "config": config,
            "initial_stage1_checkpoint_sha256": metadata[
                "initial_stage1_sha256"
            ],
            "estimator_checkpoint": str(estimator_checkpoint),
            "estimator_checkpoint_sha256": estimator_hash,
        },
    )
    stage1_hash = sha256_file(stage1_path)
    stage2a_path = stage2a_output_dir / "best.pt"
    atomic_torch_save(
        stage2a_path,
        {
            "version": 1,
            "model": stage2a.state_dict(),
            "epoch": epoch,
            "best_val_rmse_db": float(val_metrics["rmse_db"]),
            "best_val_metrics": val_metrics,
            "config": config,
            "residual_center_db": float(metadata["residual_center_db"]),
            "residual_scale_db": float(metadata["residual_scale_db"]),
            "stage1_checkpoint_sha256": stage1_hash,
            "initial_stage2a_checkpoint_sha256": metadata[
                "initial_stage2a_sha256"
            ],
            "estimator_checkpoint": str(estimator_checkpoint),
            "estimator_checkpoint_sha256": estimator_hash,
        },
    )
    return stage1_hash, sha256_file(stage2a_path)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device or str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    datasets = {
        split: BSInversionPilotDataset(
            config["dataset_root"],
            split=split,
            seed=int(config["seed"]),
            augment=split == "train" and bool(config["augmentation"]),
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: make_loader(
            dataset,
            config,
            shuffle=split == "train",
            device=device,
        )
        for split, dataset in datasets.items()
    }
    estimator, estimator_mean, estimator_std, estimator_path, estimator_hash = (
        load_estimator(config, device, channels_last)
    )
    stage1, stage2a, metadata = load_initial_models(config, device, channels_last)
    optimizer = AdamW(
        [
            {
                "params": stage1.parameters(),
                "lr": float(config["stage1_learning_rate"]),
            },
            {
                "params": stage2a.parameters(),
                "lr": float(config["stage2a_learning_rate"]),
            },
        ],
        weight_decay=float(config["weight_decay"]),
    )
    epochs = args.epochs or int(config["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(config.get("minimum_learning_rate", 0.0)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    tx_normalization = load_json(
        Path(config["initial_stage1_model_dir"]) / "tx_feature_normalization.json"
    )
    tx_height_cap_m = float(tx_normalization["transmitter_height"]["cap_m"])
    distance_max_m = float(tx_normalization["distance"]["max_m"])
    long_distance_max_m = float(
        config.get("long_distance_max_m", math.sqrt(2.0) * 512.0)
    )

    stage1_output_dir = Path(config["stage1_output_dir"]).resolve()
    stage2a_output_dir = Path(config["stage2a_output_dir"]).resolve()
    stage1_output_dir.mkdir(parents=True, exist_ok=True)
    stage2a_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(config["initial_stage1_model_dir"]) / "normalization.json",
        stage1_output_dir / "normalization.json",
    )
    output_tx_normalization = dict(tx_normalization)
    if int(metadata["stage1_in_channels"]) == 6:
        output_tx_normalization["input_channels"] = [
            "building_height_norm",
            "tx_position_height_norm",
            "legacy_tx_distance_norm_366m",
            "long_tx_distance_norm_724m",
            "signed_delta_x_norm",
            "signed_delta_y_norm",
        ]
        output_tx_normalization["long_distance"] = {
            "max_m": long_distance_max_m,
            "normalization": "log1p(distance_m) / log1p(max_m), clipped to [0,1]",
        }
        output_tx_normalization["signed_offsets"] = {
            "scale_m": 512.0,
            "delta_x": "(pixel_center_x - estimated_tx_x) / 512",
            "delta_y": "(pixel_center_y - estimated_tx_y) / 512",
        }
    atomic_write_json(
        stage1_output_dir / "tx_feature_normalization.json",
        output_tx_normalization,
    )
    best_stage1_path = stage1_output_dir / "best.pt"
    best_stage2a_path = stage2a_output_dir / "best.pt"
    if (
        (best_stage1_path.exists() or best_stage2a_path.exists())
        and args.resume is None
    ):
        raise FileExistsError("adapted Stage1/Stage2-A best checkpoint already exists")
    run_config = {
        **config,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "estimator_checkpoint_sha256": estimator_hash,
        "split_samples": {key: len(value) for key, value in datasets.items()},
        **metadata,
    }
    atomic_write_json(stage2a_output_dir / "run_config.json", run_config)

    history: list[dict[str, Any]] = []
    best_rmse = math.inf
    best_epoch = 0
    patience_used = 0
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        stage1.load_state_dict(checkpoint["stage1_model"], strict=True)
        stage2a.load_state_dict(checkpoint["stage2a_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_rmse = float(checkpoint["best_val_rmse_db"])
        patience_used = int(checkpoint["patience_used"])
        history = list(checkpoint["history"])

    common = {
        "device": device,
        "amp_enabled": amp_enabled,
        "channels_last": channels_last,
        "estimator_signal_mean_db": estimator_mean,
        "estimator_signal_std_db": estimator_std,
        "stage1_mean_db": float(metadata["stage1_mean_db"]),
        "stage1_std_db": float(metadata["stage1_std_db"]),
        "residual_center_db": float(metadata["residual_center_db"]),
        "residual_scale_db": float(metadata["residual_scale_db"]),
        "loss_scale_db": float(config["loss_scale_db"]),
        "tx_height_cap_m": tx_height_cap_m,
        "distance_max_m": distance_max_m,
        "stage1_in_channels": int(metadata["stage1_in_channels"]),
        "long_distance_max_m": long_distance_max_m,
        "position_jitter_probability": float(
            config.get("position_jitter_probability", 0.0)
        ),
        "position_jitter_std_px": float(config.get("position_jitter_std_px", 0.0)),
        "far_position_loss_multiplier": float(
            config.get("far_position_loss_multiplier", 1.0)
        ),
        "gradient_clip": float(config["gradient_clip"]),
        "progress_every": int(config.get("progress_every_batches", 0)),
    }
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        datasets["train"].set_epoch(epoch)
        train_metrics = run_epoch(
            loaders["train"],
            estimator,
            stage1,
            stage2a,
            optimizer=optimizer,
            scaler=scaler,
            phase="train",
            epoch=epoch,
            **common,
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                loaders["val"],
                estimator,
                stage1,
                stage2a,
                optimizer=None,
                scaler=None,
                phase="val",
                epoch=epoch,
                **common,
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            "stage1_learning_rate": optimizer.param_groups[0]["lr"],
            "stage2a_learning_rate": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        write_history(stage2a_output_dir / "history.csv", history)
        val_rmse = float(val_metrics["rmse_db"])
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            patience_used = 0
            save_best_models(
                stage1,
                stage2a,
                config=config,
                metadata=metadata,
                estimator_checkpoint=estimator_path,
                estimator_hash=estimator_hash,
                epoch=epoch,
                val_metrics=val_metrics,
                stage1_output_dir=stage1_output_dir,
                stage2a_output_dir=stage2a_output_dir,
            )
        else:
            patience_used += 1
        atomic_torch_save(
            stage2a_output_dir / "last_joint.pt",
            {
                "version": 1,
                "stage1_model": stage1.state_dict(),
                "stage2a_model": stage2a.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_rmse_db": best_rmse,
                "patience_used": patience_used,
                "history": history,
                "config": config,
            },
        )
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_rmse_db": train_metrics["rmse_db"],
                    "val_rmse_db": val_rmse,
                    "best_epoch": best_epoch,
                    "best_val_rmse_db": best_rmse,
                    "patience_used": patience_used,
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if patience_used >= int(config["patience"]):
            break

    best_stage1 = torch.load(best_stage1_path, map_location=device, weights_only=False)
    best_stage2a = torch.load(best_stage2a_path, map_location=device, weights_only=False)
    stage1.load_state_dict(best_stage1["model"], strict=True)
    stage2a.load_state_dict(best_stage2a["model"], strict=True)
    with torch.inference_mode():
        test_metrics = run_epoch(
            loaders["test"],
            estimator,
            stage1,
            stage2a,
            optimizer=None,
            scaler=None,
            phase="test",
            epoch=best_epoch,
            **common,
        )
    baseline_rmse = float(config["acceptance"]["baseline_test_rmse_db"])
    improvement = baseline_rmse - float(test_metrics["rmse_db"])
    accepted = improvement >= float(config["acceptance"]["minimum_improvement_db"])
    report = {
        "version": 1,
        "status": "ok" if accepted else "needs_iteration",
        "best_epoch": best_epoch,
        "best_val_rmse_db": best_rmse,
        "test_metrics": test_metrics,
        "baseline_test_rmse_db": baseline_rmse,
        "test_rmse_improvement_db": improvement,
        "accepted_for_stage2b": accepted,
        "stage1_checkpoint": str(best_stage1_path),
        "stage1_checkpoint_sha256": sha256_file(best_stage1_path),
        "stage2a_checkpoint": str(best_stage2a_path),
        "stage2a_checkpoint_sha256": sha256_file(best_stage2a_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(stage2a_output_dir / "test_metrics.json", report)
    atomic_write_json(
        stage2a_output_dir / "training_summary.json",
        {
            **report,
            "epochs_completed": len(history),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

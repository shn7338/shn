#!/usr/bin/env python3
"""Jointly align the late BS estimator heads and V5 latent adapters."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from bs_inversion_dataset import BSInversionPilotDataset, load_json
from bs_parameter_model import BSParameterEstimator
from evaluate_pilot_oracle_pipeline import make_stage1_input
from finetune_pilot_stage2b import (
    ErrorSums,
    atomic_torch_save,
    atomic_write_json,
    make_loader,
    write_history,
)
from latent_bs_fusion import (
    build_latent_fusion_features,
    configured_stage2b_in_channels,
)
from run_sionna_dataset import sha256_file
from stage2_models import LatentAdapterGeo2SigMapUNet
from train_bs_parameter_estimator import circular_error_degrees, compute_losses
from train_stage2b_directional_ss import (
    build_configured_unet,
    load_frozen_models,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def cosine_lambda(total_epochs: int, minimum_ratio: float):
    def schedule(completed_epochs: int) -> float:
        progress = min(max(completed_epochs, 0), total_epochs) / max(total_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return schedule


def load_stage2b(
    checkpoint_path: Path,
    device: torch.device,
    channels_last: bool,
) -> tuple[LatentAdapterGeo2SigMapUNet, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    model = build_configured_unet(
        checkpoint_config,
        in_channels=configured_stage2b_in_channels(checkpoint_config),
    ).to(device)
    if not isinstance(model, LatentAdapterGeo2SigMapUNet):
        raise TypeError("V6 requires a V5 LatentAdapterGeo2SigMapUNet checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_optimization_phase("adapter")
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model, checkpoint_config, checkpoint


def load_trainable_estimator(
    checkpoint_path: Path,
    trainable_modules: tuple[str, ...],
    device: torch.device,
    channels_last: bool,
) -> tuple[BSParameterEstimator, dict[str, Any], dict[str, nn.Module]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    model = BSParameterEstimator(
        in_channels=3,
        base_channels=int(model_config["base_channels"]),
        scalar_hidden_channels=int(model_config["scalar_hidden_channels"]),
        softargmax_temperature=float(model_config["softargmax_temperature"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    selected: dict[str, nn.Module] = {}
    for name in trainable_modules:
        module = getattr(model, name, None)
        if not isinstance(module, nn.Module):
            raise ValueError(f"unknown estimator module for V6: {name!r}")
        module.requires_grad_(True)
        selected[name] = module
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model, checkpoint, selected


def set_estimator_mode(
    estimator: BSParameterEstimator,
    selected_modules: dict[str, nn.Module],
    training: bool,
) -> None:
    estimator.eval()
    if training:
        for module in selected_modules.values():
            module.train(True)


def parameter_groups(
    stage2b: LatentAdapterGeo2SigMapUNet,
    estimator_modules: dict[str, nn.Module],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[Any]]:
    schedule = config["optimization"]
    groups: list[dict[str, Any]] = []
    lambdas: list[Any] = []

    def add_group(name: str, parameters, learning_rate: float, minimum: float) -> None:
        values = list(parameters)
        if not values:
            raise ValueError(f"empty optimizer group: {name}")
        groups.append({"name": name, "params": values, "lr": learning_rate})
        lambdas.append(
            cosine_lambda(int(config["epochs"]), minimum / learning_rate)
        )

    add_group(
        "stage2b_adapters",
        stage2b.adapter_parameters(),
        float(schedule["stage2b_adapter_learning_rate"]),
        float(schedule["stage2b_adapter_minimum_learning_rate"]),
    )
    spatial_names = tuple(str(v) for v in schedule["estimator_spatial_modules"])
    scalar_names = tuple(str(v) for v in schedule["estimator_scalar_modules"])
    add_group(
        "estimator_spatial",
        (
            parameter
            for name in spatial_names
            for parameter in estimator_modules[name].parameters()
        ),
        float(schedule["estimator_spatial_learning_rate"]),
        float(schedule["estimator_spatial_minimum_learning_rate"]),
    )
    add_group(
        "estimator_scalar",
        (
            parameter
            for name in scalar_names
            for parameter in estimator_modules[name].parameters()
        ),
        float(schedule["estimator_scalar_learning_rate"]),
        float(schedule["estimator_scalar_minimum_learning_rate"]),
    )
    return groups, lambdas


class JointMetrics:
    def __init__(self) -> None:
        self.samples = 0
        self.total_loss = 0.0
        self.map_loss = 0.0
        self.auxiliary_loss = 0.0
        self.component_sums = {
            name: 0.0 for name in ("heatmap", "coordinate", "power", "direction", "map")
        }
        self.final_errors = ErrorSums()
        self.unmeasured_errors = ErrorSums()
        self.location_errors: list[float] = []
        self.power_errors: list[float] = []
        self.direction_errors: list[float] = []

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        prediction_db: torch.Tensor,
        total_loss: torch.Tensor,
        map_loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        components: dict[str, torch.Tensor],
        power_mean_db: float,
        power_std_db: float,
    ) -> None:
        batch_size = int(prediction_db.shape[0])
        self.samples += batch_size
        self.total_loss += float(total_loss.detach().item()) * batch_size
        self.map_loss += float(map_loss.detach().item()) * batch_size
        self.auxiliary_loss += float(auxiliary_loss.detach().item()) * batch_size
        for name, value in components.items():
            self.component_sums[name] += float(value.detach().item()) * batch_size
        valid = batch["valid_mask"].bool()
        unmeasured = valid & ~batch["sparse_mask"].bool()
        errors = prediction_db - batch["target_map_db"]
        self.final_errors.update(errors[valid])
        self.unmeasured_errors.update(errors[unmeasured])
        location = torch.linalg.vector_norm(
            outputs["row_col_px"].float() - batch["target_row_col_px"].float(), dim=1
        )
        power_db = outputs["power_norm"].float() * power_std_db + power_mean_db
        power_error = torch.abs(power_db - batch["target_power_db"].float())
        direction_error = circular_error_degrees(
            outputs["direction_raw"], batch["target_direction_sin_cos"]
        )
        self.location_errors.extend(location.detach().cpu().tolist())
        self.power_errors.extend(power_error.detach().cpu().tolist())
        self.direction_errors.extend(direction_error.detach().cpu().tolist())

    def result(self) -> dict[str, float | int]:
        def distribution(values: list[float], prefix: str) -> dict[str, float]:
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{prefix}_mean": float(array.mean()),
                f"{prefix}_p90": float(np.percentile(array, 90.0)),
            }

        result: dict[str, float | int] = {
            "samples": self.samples,
            "loss_total": self.total_loss / self.samples,
            "loss_final_map": self.map_loss / self.samples,
            "loss_auxiliary": self.auxiliary_loss / self.samples,
        }
        for name, value in self.component_sums.items():
            result[f"loss_{name}"] = value / self.samples
        result.update(self.final_errors.metrics("map"))
        result.update(self.unmeasured_errors.metrics("unmeasured"))
        result.update(distribution(self.location_errors, "location_error_px"))
        result.update(distribution(self.power_errors, "power_abs_error_db"))
        result.update(distribution(self.direction_errors, "direction_abs_error_deg"))
        return result


def run_epoch(
    loader,
    estimator: BSParameterEstimator,
    estimator_modules: dict[str, nn.Module],
    stage1: nn.Module,
    stage2a: nn.Module,
    stage2b: LatentAdapterGeo2SigMapUNet,
    *,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
    config: dict[str, Any],
    frozen: dict[str, Any],
    estimator_metadata: dict[str, float],
    normalization: dict[str, float],
    tx_metadata: dict[str, float],
    epoch: int,
    phase: str,
) -> dict[str, float | int]:
    training = optimizer is not None
    set_estimator_mode(estimator, estimator_modules, training)
    stage2b.set_optimization_phase("adapter")
    stage2b.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    metrics = JointMetrics()
    loss_config = config["joint_loss"]
    for batch_index, raw_batch in enumerate(loader, start=1):
        keys = (
            "building", "sparse_ss_norm", "sparse_mask", "target_map_db",
            "target_map_norm", "valid_mask", "target_heatmap", "target_row_col_px",
            "target_power_norm", "target_power_db", "target_direction_sin_cos",
            "tx_height_m",
        )
        batch = {key: raw_batch[key].to(device, non_blocking=True) for key in keys}
        building = batch["building"]
        sparse_norm = batch["sparse_ss_norm"]
        sparse_mask = batch["sparse_mask"]
        sparse_db = (
            sparse_norm.float() * normalization["ss_std_db"]
            + normalization["ss_mean_db"]
        )
        estimator_sparse_norm = torch.where(
            sparse_mask.bool(),
            (sparse_db - estimator_metadata["signal_mean_db"])
            / estimator_metadata["signal_std_db"],
            0.0,
        )
        estimator_input = torch.cat(
            (building, estimator_sparse_norm, sparse_mask), dim=1
        )
        if channels_last:
            estimator_input = estimator_input.contiguous(
                memory_format=torch.channels_last
            )
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            estimated = estimator(estimator_input)

        # Stage1/Stage2A stay frozen and use a detached point coordinate. V6
        # gradients reach the estimator through the differentiable latent maps.
        row_col_for_coarse = estimated["row_col_px"].detach().float()
        tx_xy_m = torch.stack(
            (
                (row_col_for_coarse[:, 1] + 0.5) * 4.0,
                512.0 - (row_col_for_coarse[:, 0] + 0.5) * 4.0,
            ),
            dim=1,
        )
        stage1_input = make_stage1_input(
            building,
            tx_xy_m,
            batch["tx_height_m"],
            tx_metadata["height_cap_m"],
            tx_metadata["distance_max_m"],
            in_channels=int(frozen.get("stage1_in_channels", 3)),
            long_distance_max_m=tx_metadata["long_distance_max_m"],
        )
        if channels_last:
            building = building.contiguous(memory_format=torch.channels_last)
            stage1_input = stage1_input.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
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
        corrected_iso_norm = (
            corrected_iso_db - normalization["iso_mean_db"]
        ) / normalization["iso_std_db"]

        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            latent = build_latent_fusion_features(
                estimated,
                config["latent_fusion"],
                estimator_signal_mean_db=estimator_metadata["signal_mean_db"],
                estimator_signal_std_db=estimator_metadata["signal_std_db"],
                stage2b_signal_mean_db=normalization["ss_mean_db"],
                stage2b_signal_std_db=normalization["ss_std_db"],
                estimator_power_mean_db=estimator_metadata["power_mean_db"],
                estimator_power_std_db=estimator_metadata["power_std_db"],
                rows=int(building.shape[-2]), cols=int(building.shape[-1]),
            )
            stage2b_input = torch.cat(
                (building, corrected_iso_norm, sparse_norm, sparse_mask, latent), dim=1
            )
            if channels_last:
                stage2b_input = stage2b_input.contiguous(
                    memory_format=torch.channels_last
                )
            prediction_norm = stage2b(stage2b_input)
            target_norm = torch.where(
                batch["valid_mask"].bool(),
                (batch["target_map_db"] - normalization["ss_mean_db"])
                / normalization["ss_std_db"],
                0.0,
            )
            unmeasured = batch["valid_mask"].bool() & ~sparse_mask.bool()
            final_map_loss = functional.mse_loss(
                prediction_norm[unmeasured], target_norm[unmeasured]
            )
            estimator_target = {
                "target_heatmap": batch["target_heatmap"],
                "target_row_col_px": batch["target_row_col_px"],
                "target_power_norm": batch["target_power_norm"],
                "target_direction_sin_cos": batch["target_direction_sin_cos"],
                "valid_mask": batch["valid_mask"],
                "target_map_norm": torch.where(
                    batch["valid_mask"].bool(),
                    (batch["target_map_db"] - estimator_metadata["signal_mean_db"])
                    / estimator_metadata["signal_std_db"],
                    0.0,
                ),
            }
            auxiliary_loss, components = compute_losses(
                estimated, estimator_target, loss_config["estimator_component_weights"]
            )
            total_loss = (
                float(loss_config["final_map_weight"]) * final_map_loss
                + float(loss_config["estimator_auxiliary_scale"]) * auxiliary_loss
            )
        if training:
            assert optimizer is not None and scaler is not None
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if batch_index == 1 and bool(config.get("require_gradient_audit", True)):
                audit = {
                    "stage1_frozen": not any(
                        parameter.grad is not None for parameter in stage1.parameters()
                    ),
                    "stage2a_frozen": not any(
                        parameter.grad is not None for parameter in stage2a.parameters()
                    ),
                    "stage2b_backbone_frozen": not any(
                        parameter.grad is not None
                        for parameter in stage2b.backbone.parameters()
                    ),
                    "estimator_frozen_modules_untouched": not any(
                        parameter.grad is not None
                        for parameter in estimator.parameters()
                        if not parameter.requires_grad
                    ),
                }
                for group in optimizer.param_groups:
                    audit[f"gradient_present_{group['name']}"] = any(
                        parameter.grad is not None for parameter in group["params"]
                    )
                if not all(audit.values()):
                    raise RuntimeError(f"V6 gradient audit failed: {audit}")
                print(json.dumps({
                    "event": "gradient_audit", "checks": audit,
                    "cuda_peak_memory_mib": (
                        round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
                        if device.type == "cuda" else None
                    ),
                }, ensure_ascii=False), flush=True)
            trainable = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            nn.utils.clip_grad_norm_(trainable, float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        prediction_db = (
            prediction_norm.detach().float() * normalization["ss_std_db"]
            + normalization["ss_mean_db"]
        )
        metrics.update(
            estimated, batch, prediction_db, total_loss, final_map_loss,
            auxiliary_loss, components,
            estimator_metadata["power_mean_db"], estimator_metadata["power_std_db"],
        )
        progress_every = int(config.get("progress_every_batches", 0))
        if progress_every and (
            batch_index % progress_every == 0 or batch_index == len(loader)
        ):
            print(json.dumps({
                "event": "progress", "phase": phase, "epoch": epoch,
                "batch": batch_index, "batches_total": len(loader),
            }, ensure_ascii=False), flush=True)
    return metrics.result()


def preservation_checks(
    metrics: dict[str, Any], constraints: dict[str, float]
) -> dict[str, bool]:
    return {
        "location": float(metrics["location_error_px_mean"])
        <= float(constraints["maximum_val_location_mean_error_px"]),
        "power": float(metrics["power_abs_error_db_mean"])
        <= float(constraints["maximum_val_power_mae_db"]),
        "direction": float(metrics["direction_abs_error_deg_mean"])
        <= float(constraints["maximum_val_direction_mae_deg"]),
    }


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

    source_normalization = load_json(Path(config["irt_normalization"]))
    iso_stats = source_normalization["statistics"]["train"]["p_iso"]
    ss_stats = source_normalization["stage2b_ss_normalization"]
    normalization = {
        "iso_mean_db": float(iso_stats["mean_db"]),
        "iso_std_db": float(iso_stats["std_db"]),
        "ss_mean_db": float(ss_stats["mean_db"]),
        "ss_std_db": float(ss_stats["std_db"]),
    }
    datasets = {
        split: BSInversionPilotDataset(
            config["dataset_root"], split=split, seed=int(config["seed"]),
            augment=split == "train" and bool(config["augmentation"]),
            signal_mean_db=normalization["ss_mean_db"],
            signal_std_db=normalization["ss_std_db"],
        )
        for split in ("train", "val")
    }
    loaders = {
        split: make_loader(dataset, config, shuffle=split == "train", device=device)
        for split, dataset in datasets.items()
    }
    stage1, stage2a, frozen = load_frozen_models(config, device, channels_last)
    stage2b_path = Path(config["initial_stage2b_checkpoint"])
    stage2b, _, stage2b_checkpoint = load_stage2b(
        stage2b_path, device, channels_last
    )
    estimator_path = Path(config["initial_estimator_checkpoint"])
    optimization = config["optimization"]
    trainable_modules = tuple(
        dict.fromkeys(
            [
                *optimization["estimator_spatial_modules"],
                *optimization["estimator_scalar_modules"],
            ]
        )
    )
    estimator, estimator_checkpoint, estimator_modules = load_trainable_estimator(
        estimator_path, trainable_modules, device, channels_last
    )
    estimator_config = load_json(Path(config["estimator_config"]))
    estimator_norm = load_json(
        Path(estimator_config["dataset_root"]) / "normalization.json"
    )["directional_signal_strength_db"]
    manifest = load_json(Path(estimator_config["dataset_root"]) / "run_manifest.json")
    data_config = load_json(Path(manifest["config"]))
    power_config = data_config["effective_power"]
    estimator_metadata = {
        "signal_mean_db": float(estimator_norm["mean"]),
        "signal_std_db": float(estimator_norm["std"]),
        "power_mean_db": float(power_config["normalization_mean_db"]),
        "power_std_db": float(power_config["normalization_std_db"]),
    }
    tx_norm = load_json(
        Path(config["stage1_model_dir"]) / "tx_feature_normalization.json"
    )
    tx_metadata = {
        "height_cap_m": float(tx_norm["transmitter_height"]["cap_m"]),
        "distance_max_m": float(tx_norm["distance"]["max_m"]),
        "long_distance_max_m": float(
            tx_norm.get("long_distance", {}).get("max_m", math.sqrt(2.0) * 512.0)
        ),
    }

    groups, lambdas = parameter_groups(stage2b, estimator_modules, config)
    optimizer = AdamW(groups, weight_decay=float(config["weight_decay"]))
    scheduler = LambdaLR(optimizer, lr_lambda=lambdas)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_joint_path = output_dir / "best.pt"
    best_stage2b_path = output_dir / "best_stage2b.pt"
    best_estimator_path = output_dir / "best_estimator.pt"
    if best_joint_path.exists() and args.resume is None:
        raise FileExistsError(f"{best_joint_path} already exists; choose a new V6 output")

    history: list[dict[str, Any]] = []
    best_rmse = float(stage2b_checkpoint["best_val_rmse_db"])
    best_epoch = 0
    patience_used = 0
    start_epoch = 1
    if args.resume is not None:
        resume = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        stage2b.load_state_dict(resume["stage2b_model"], strict=True)
        estimator.load_state_dict(resume["estimator_model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        history = list(resume["history"])
        best_rmse = float(resume["best_val_rmse_db"])
        best_epoch = int(resume["best_epoch"])
        patience_used = int(resume["patience_used"])
        start_epoch = int(resume["epoch"]) + 1
    else:
        initial_joint = {
            "version": 1, "epoch": 0,
            "stage2b_model": stage2b.state_dict(),
            "estimator_model": estimator.state_dict(),
            "best_val_rmse_db": best_rmse, "config": config,
        }
        atomic_torch_save(best_joint_path, initial_joint)
        atomic_torch_save(best_stage2b_path, {
            "version": 1, "model": stage2b.state_dict(), "epoch": 0,
            "best_val_rmse_db": best_rmse, "config": config,
        })
        atomic_torch_save(best_estimator_path, {
            "version": 1, "model": estimator.state_dict(), "epoch": 0,
            "config": estimator_checkpoint["config"],
        })

    trainable_counts = {
        str(group["name"]): sum(parameter.numel() for parameter in group["params"])
        for group in optimizer.param_groups
    }
    atomic_write_json(output_dir / "run_config.json", {
        **config, "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device_resolved": str(device),
        "stage2b_checkpoint_sha256": sha256_file(stage2b_path),
        "estimator_checkpoint_sha256": sha256_file(estimator_path),
        "trainable_parameter_counts": trainable_counts,
        "gradient_path": "final map -> V5 latent channels -> estimator; Stage1 coordinate path detached",
        **frozen,
    })

    started = time.perf_counter()
    constraints = config["preservation_constraints"]
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        datasets["train"].set_epoch(epoch)
        train_metrics = run_epoch(
            loaders["train"], estimator, estimator_modules, stage1, stage2a, stage2b,
            optimizer=optimizer, scaler=scaler, device=device,
            amp_enabled=amp_enabled, channels_last=channels_last,
            config=config, frozen=frozen, estimator_metadata=estimator_metadata,
            normalization=normalization, tx_metadata=tx_metadata,
            epoch=epoch, phase="train",
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                loaders["val"], estimator, estimator_modules, stage1, stage2a, stage2b,
                optimizer=None, scaler=None, device=device,
                amp_enabled=amp_enabled, channels_last=channels_last,
                config=config, frozen=frozen, estimator_metadata=estimator_metadata,
                normalization=normalization, tx_metadata=tx_metadata,
                epoch=epoch, phase="val",
            )
        current_lrs = {
            f"learning_rate_{group['name']}": float(group["lr"])
            for group in optimizer.param_groups
        }
        checks = preservation_checks(val_metrics, constraints)
        row = {
            "epoch": epoch, **current_lrs,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_preserves_{key}": value for key, value in checks.items()},
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        val_rmse = float(val_metrics["map_rmse_db"])
        improved = val_rmse < best_rmse and all(checks.values())
        if improved:
            best_rmse = val_rmse
            best_epoch = epoch
            patience_used = 0
            atomic_torch_save(best_joint_path, {
                "version": 1, "epoch": epoch,
                "stage2b_model": stage2b.state_dict(),
                "estimator_model": estimator.state_dict(),
                "best_val_rmse_db": best_rmse,
                "best_val_metrics": val_metrics,
                "preservation_checks": checks, "config": config,
            })
            atomic_torch_save(best_stage2b_path, {
                "version": 1, "model": stage2b.state_dict(), "epoch": epoch,
                "best_val_rmse_db": best_rmse,
                "best_val_metrics": val_metrics, "config": config,
            })
            atomic_torch_save(best_estimator_path, {
                "version": 1, "model": estimator.state_dict(), "epoch": epoch,
                "config": estimator_checkpoint["config"],
                "joint_training_config": config,
                "best_val_metrics": val_metrics,
            })
        else:
            patience_used += 1
        scheduler.step()
        atomic_torch_save(output_dir / "last.pt", {
            "version": 1, "epoch": epoch,
            "stage2b_model": stage2b.state_dict(),
            "estimator_model": estimator.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "history": history,
            "best_val_rmse_db": best_rmse, "best_epoch": best_epoch,
            "patience_used": patience_used, "config": config,
        })
        print(json.dumps({
            "event": "epoch", "epoch": epoch,
            "train_rmse_db": train_metrics["map_rmse_db"],
            "val_rmse_db": val_rmse, "best_val_rmse_db": best_rmse,
            "best_epoch": best_epoch, "preservation_checks": checks,
            "patience_used": patience_used,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }, ensure_ascii=False), flush=True)
        if patience_used >= int(config["patience"]):
            break

    summary = {
        "version": 1, "status": "ok", "best_epoch": best_epoch,
        "best_val_rmse_db": best_rmse, "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "best_joint_checkpoint": str(best_joint_path),
        "best_stage2b_checkpoint": str(best_stage2b_path),
        "best_estimator_checkpoint": str(best_estimator_path),
        "best_stage2b_sha256": sha256_file(best_stage2b_path),
        "best_estimator_sha256": sha256_file(best_estimator_path),
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

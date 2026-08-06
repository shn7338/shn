#!/usr/bin/env python3
"""Train Stage2-B with frozen Stage1 and frozen Stage2-A models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from stage2_models import Geo2SigMapUNet, ResidualUNet, parameter_count
from stage2b_dataset import Stage2BDirectionalDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-limit-tiles", type=int)
    parser.add_argument("--val-limit-tiles", type=int)
    parser.add_argument("--test-limit-tiles", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Stage2BDirectionalDataset,
    config: dict[str, Any],
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
        # Workers are recreated each epoch so dataset.set_epoch() is visible
        # under Windows spawn semantics.
        "persistent_workers": False,
    }
    if workers > 0:
        arguments["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(**arguments)


def build_configured_unet(
    config: dict[str, Any],
    in_channels: int,
) -> nn.Module:
    model_config = config.get("model", {})
    architecture = str(
        model_config.get("architecture", "stable_residual_unet")
    ).lower()
    base_channels = int(
        model_config.get("base_channels", config.get("base_channels", 32))
    )
    if architecture == "geo2sigmap_unet":
        return Geo2SigMapUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            gradient_checkpointing=bool(
                model_config.get("gradient_checkpointing", False)
            ),
            zero_init_output=bool(
                model_config.get("zero_init_output", False)
            ),
        )
    if architecture == "stable_residual_unet":
        return ResidualUNet(
            in_channels=in_channels,
            base_channels=base_channels,
        )
    raise ValueError(f"unsupported U-Net architecture: {architecture!r}")


def build_optimizer(
    config: dict[str, Any],
    model: nn.Module,
) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    arguments = {
        "params": model.parameters(),
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return Adam(**arguments)
    if name == "adamw":
        return AdamW(**arguments)
    raise ValueError(f"unsupported optimizer: {name!r}")


def build_scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> CosineAnnealingLR | None:
    name = str(config.get("scheduler", "cosine")).lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    raise ValueError(f"unsupported scheduler: {name!r}")


def load_frozen_models(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    stage1_path = Path(config["stage1_model_dir"]) / "best.pt"
    stage1_normalization_path = (
        Path(config["stage1_model_dir"]) / "normalization.json"
    )
    stage2a_path = Path(config["stage2a_model_dir"]) / "best.pt"
    for path in (stage1_path, stage1_normalization_path, stage2a_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    stage1_hash = sha256_file(stage1_path)
    expected_stage1 = config["provenance"].get("stage1_checkpoint_sha256")
    if expected_stage1 and stage1_hash.lower() != expected_stage1.lower():
        raise ValueError("Stage1 checkpoint hash mismatch")
    stage1_checkpoint = torch.load(
        stage1_path,
        map_location=device,
        weights_only=False,
    )
    stage1_base = int(
        stage1_checkpoint.get("args", {}).get("base_channels", 32)
    )
    stage1 = ResidualUNet(
        in_channels=3,
        base_channels=stage1_base,
    ).to(device)
    stage1.load_state_dict(stage1_checkpoint["model"], strict=True)

    stage2a_hash = sha256_file(stage2a_path)
    expected_stage2a = config["provenance"].get(
        "stage2a_checkpoint_sha256"
    )
    if expected_stage2a and stage2a_hash.lower() != expected_stage2a.lower():
        raise ValueError("Stage2-A checkpoint hash mismatch")
    stage2a_checkpoint = torch.load(
        stage2a_path,
        map_location=device,
        weights_only=False,
    )
    stage2a_config = stage2a_checkpoint["config"]
    stage2a = build_configured_unet(stage2a_config, in_channels=2).to(device)
    stage2a.load_state_dict(stage2a_checkpoint["model"], strict=True)
    stage2a_stage1_hash = str(
        stage2a_checkpoint.get("stage1_checkpoint_sha256", "")
    )
    if stage2a_stage1_hash.lower() != stage1_hash.lower():
        raise ValueError("Stage2-A was trained with a different Stage1 model")
    if bool(config.get("require_accepted_stage2a", False)):
        stage2a_metrics_path = (
            Path(config["stage2a_model_dir"]) / "test_metrics.json"
        )
        if not stage2a_metrics_path.is_file():
            raise FileNotFoundError(stage2a_metrics_path)
        stage2a_metrics = load_json(stage2a_metrics_path)
        if not bool(stage2a_metrics.get("accepted_for_stage2b", False)):
            raise ValueError("Stage2-A has not passed Stage2-B acceptance")

    for model in (stage1, stage2a):
        model.eval()
        model.requires_grad_(False)
        if channels_last:
            model.to(memory_format=torch.channels_last)
    normalization = load_json(stage1_normalization_path)
    path_gain = normalization["statistics"]["path_gain"]
    metadata: dict[str, Any] = {
        "stage1_mean_db": float(path_gain["mean_db"]),
        "stage1_std_db": float(path_gain["std_db"]),
        "stage2a_residual_center_db": float(
            stage2a_checkpoint.get("residual_center_db", 0.0)
        ),
        "stage2a_residual_scale_db": float(
            stage2a_checkpoint["residual_scale_db"]
        ),
        "stage2a_best_epoch": int(stage2a_checkpoint["epoch"]),
        "stage2a_best_val_rmse_db": float(
            stage2a_checkpoint["best_val_rmse_db"]
        ),
        "stage1_checkpoint_sha256": stage1_hash,
        "stage2a_checkpoint_sha256": stage2a_hash,
    }
    return stage1, stage2a, metadata


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    name: str,
    beta: float,
) -> torch.Tensor:
    if not torch.any(mask):
        raise ValueError("batch contains no valid directional pixels")
    prediction_valid = prediction[mask]
    target_valid = target[mask]
    if name == "masked_mse":
        return functional.mse_loss(
            prediction_valid,
            target_valid,
            reduction="mean",
        )
    if name == "masked_huber":
        return functional.smooth_l1_loss(
            prediction_valid,
            target_valid,
            beta=beta,
            reduction="mean",
        )
    raise ValueError(f"unsupported loss: {name!r}")


def sparse_calibrated_iso_baseline(
    corrected_iso_db: torch.Tensor,
    target_ss_norm: torch.Tensor,
    sparse_mask: torch.Tensor,
    ss_mean_db: float,
    ss_std_db: float,
) -> torch.Tensor:
    """Shift corrected isotropic gain by the median measured SS offset."""
    target_db = target_ss_norm * ss_std_db + ss_mean_db
    predictions: list[torch.Tensor] = []
    for sample in range(corrected_iso_db.shape[0]):
        measured = sparse_mask[sample].bool()
        if torch.any(measured):
            offset = torch.median(
                target_db[sample][measured]
                - corrected_iso_db[sample][measured]
            )
        else:
            offset = corrected_iso_db.new_tensor(52.5)
        predictions.append(corrected_iso_db[sample] + offset)
    return torch.stack(predictions, dim=0)


def run_epoch(
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
    loss_name: str,
    huber_beta: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    accumulation: int,
    gradient_clip: float,
    phase: str | None = None,
    epoch: int | None = None,
    sparse_points_label: int | str | None = None,
    progress_every: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    stage2b.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    batches = 0
    valid_pixels = 0
    squared_error = 0.0
    absolute_error = 0.0
    baseline_squared_error = 0.0
    baseline_absolute_error = 0.0
    sparse_point_sum = 0
    epoch_started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        stage1_input = batch["stage1_input"].to(
            device, non_blocking=True
        )
        building = batch["building"].to(device, non_blocking=True)
        target_norm = batch["target_ss_norm"].to(
            device, non_blocking=True
        )
        sparse_norm = batch["sparse_ss_norm"].to(
            device, non_blocking=True
        )
        sparse_mask = batch["sparse_mask"].to(
            device, non_blocking=True
        )
        valid_mask = batch["valid_mask"].to(
            device, non_blocking=True, dtype=torch.bool
        )
        cached_dpm_norm = batch.get("stage1_prediction_norm")
        if cached_dpm_norm is not None:
            cached_dpm_norm = cached_dpm_norm.to(
                device, non_blocking=True
            )
        sparse_point_sum += int(torch.sum(sparse_mask).item())
        if channels_last:
            stage1_input = stage1_input.contiguous(
                memory_format=torch.channels_last
            )
            building = building.contiguous(
                memory_format=torch.channels_last
            )
            if cached_dpm_norm is not None:
                cached_dpm_norm = cached_dpm_norm.contiguous(
                    memory_format=torch.channels_last
                )
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            dpm_norm = (
                stage1(stage1_input)
                if cached_dpm_norm is None
                else cached_dpm_norm
            )
            stage2a_input = torch.cat((building, dpm_norm), dim=1)
            residual_norm = stage2a(stage2a_input)
        dpm_db = dpm_norm.float() * stage1_std_db + stage1_mean_db
        corrected_iso_db = (
            dpm_db
            + stage2a_residual_center_db
            + residual_norm.float() * stage2a_residual_scale_db
        )
        corrected_iso_norm = (corrected_iso_db - iso_mean_db) / iso_std_db
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            stage2b_input = torch.cat(
                (
                    building,
                    corrected_iso_norm,
                    sparse_norm,
                    sparse_mask,
                ),
                dim=1,
            )
            prediction_norm = stage2b(stage2b_input)
            loss = masked_loss(
                prediction_norm,
                target_norm,
                valid_mask,
                loss_name,
                huber_beta,
            )
        if training:
            assert optimizer is not None and scaler is not None
            scaler.scale(loss / accumulation).backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    stage2b.parameters(), gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        prediction_db = (
            prediction_norm.detach().float() * ss_std_db + ss_mean_db
        )
        target_db = target_norm * ss_std_db + ss_mean_db
        baseline_db = sparse_calibrated_iso_baseline(
            corrected_iso_db,
            target_norm,
            sparse_mask,
            ss_mean_db,
            ss_std_db,
        )
        error = prediction_db[valid_mask] - target_db[valid_mask]
        baseline_error = (
            baseline_db[valid_mask] - target_db[valid_mask]
        )
        count = int(error.numel())
        valid_pixels += count
        squared_error += float(torch.sum(error.square()).item())
        absolute_error += float(torch.sum(error.abs()).item())
        baseline_squared_error += float(
            torch.sum(baseline_error.square()).item()
        )
        baseline_absolute_error += float(
            torch.sum(baseline_error.abs()).item()
        )
        loss_sum += float(loss.detach().item())
        batches += 1
        if progress_every > 0 and (
            batches % progress_every == 0 or batches == len(loader)
        ):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "phase": phase,
                        "epoch": epoch,
                        "sparse_points": sparse_points_label,
                        "batch": batches,
                        "batches_total": len(loader),
                        "percent": round(100.0 * batches / len(loader), 1),
                        "elapsed_seconds": round(
                            time.perf_counter() - epoch_started,
                            1,
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if not batches or not valid_pixels:
        raise RuntimeError("empty epoch")
    rmse_db = math.sqrt(squared_error / valid_pixels)
    mae_db = absolute_error / valid_pixels
    baseline_rmse_db = math.sqrt(baseline_squared_error / valid_pixels)
    baseline_mae_db = baseline_absolute_error / valid_pixels
    return {
        "loss": loss_sum / batches,
        "rmse_db": rmse_db,
        "mae_db": mae_db,
        "sparse_calibrated_iso_rmse_db": baseline_rmse_db,
        "sparse_calibrated_iso_mae_db": baseline_mae_db,
        "rmse_improvement_vs_sparse_calibrated_iso_db": (
            baseline_rmse_db - rmse_db
        ),
        "mae_improvement_vs_sparse_calibrated_iso_db": (
            baseline_mae_db - mae_db
        ),
        "mean_sparse_points": sparse_point_sum / len(loader.dataset),
        "valid_pixels": float(valid_pixels),
    }


def make_dataset(
    config: dict[str, Any],
    split: str,
    ss_mean_db: float,
    ss_std_db: float,
    limit_tiles: int | None,
    fixed_sparse_points: int | None,
) -> Stage2BDirectionalDataset:
    return Stage2BDirectionalDataset(
        shard_root=config["shard_root"],
        shard_manifest=config["shard_manifest"],
        selection_csv=config["selection_csv"],
        normalized_root=config["normalized_root"],
        split=split,
        ss_mean_db=ss_mean_db,
        ss_std_db=ss_std_db,
        seed=int(config["seed"]),
        stage1_prediction_root=config.get("stage1_prediction_root"),
        augment=split == "train" and bool(config["augmentation"]),
        sparse_points_min=int(config["sparse_points_train"][0]),
        sparse_points_max=int(config["sparse_points_train"][1]),
        fixed_sparse_points=fixed_sparse_points,
        limit_tiles=limit_tiles,
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    expected_manifest_hash = config.get("provenance", {}).get(
        "irt_shard_manifest_sha256"
    )
    actual_manifest_hash = sha256_file(Path(config["shard_manifest"]))
    if (
        expected_manifest_hash
        and actual_manifest_hash.lower() != expected_manifest_hash.lower()
    ):
        raise ValueError("IRT shard manifest hash mismatch")
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device or str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    irt_normalization = load_json(Path(config["irt_normalization"]))
    iso_normalization = irt_normalization["statistics"]["train"]["p_iso"]
    iso_mean_db = float(iso_normalization["mean_db"])
    iso_std_db = float(iso_normalization["std_db"])
    ss_normalization = irt_normalization["stage2b_ss_normalization"]
    ss_mean_db = float(ss_normalization["mean_db"])
    ss_std_db = float(ss_normalization["std_db"])
    datasets = {
        "train": make_dataset(
            config,
            "train",
            ss_mean_db,
            ss_std_db,
            args.train_limit_tiles,
            None,
        ),
        "val": make_dataset(
            config,
            "val",
            ss_mean_db,
            ss_std_db,
            args.val_limit_tiles,
            int(config["validation_sparse_points"]),
        ),
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
    stage1, stage2a, frozen_metadata = load_frozen_models(
        config, device, channels_last
    )
    stage2b = build_configured_unet(config, in_channels=4).to(device)
    if channels_last:
        stage2b.to(memory_format=torch.channels_last)
    epochs = args.epochs or int(config["epochs"])
    optimizer = build_optimizer(config, stage2b)
    scheduler = build_scheduler(config, optimizer, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if best_path.exists() and args.resume is None:
        raise FileExistsError(
            f"{best_path} already exists; use --resume or a new output version"
        )
    start_epoch = 1
    best_rmse = math.inf
    best_val_metrics: dict[str, float] = {}
    patience_used = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if "optimizer" not in checkpoint:
            raise ValueError(
                "resume requires last.pt; best.pt is inference-only"
            )
        stage2b.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            saved_scheduler = checkpoint.get("scheduler")
            if saved_scheduler is None:
                raise ValueError("resume checkpoint has no scheduler state")
            scheduler.load_state_dict(saved_scheduler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rmse = float(checkpoint["best_val_rmse_db"])
        best_val_metrics = dict(checkpoint.get("best_val_metrics", {}))
        patience_used = int(checkpoint["patience_used"])
        history = list(checkpoint.get("history", []))

    run_config = {
        **config,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "amp_resolved": amp_enabled,
        "channels_last_resolved": channels_last,
        "iso_mean_db": iso_mean_db,
        "iso_std_db": iso_std_db,
        "ss_mean_db": ss_mean_db,
        "ss_std_db": ss_std_db,
        "irt_shard_manifest_sha256": actual_manifest_hash,
        "stage2b_parameters": parameter_count(stage2b),
        "effective_batch_size": (
            int(config["batch_size"])
            * int(config["gradient_accumulation"])
        ),
        "split_directional_samples": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        **frozen_metadata,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)
    started = time.perf_counter()
    accumulation = int(config["gradient_accumulation"])
    loss_name = str(config["loss"]).lower()
    huber_delta = float(config.get("huber_delta_normalized", 1.0))
    progress_every = int(config.get("progress_every_batches", 25))
    for epoch in range(start_epoch, epochs + 1):
        datasets["train"].set_epoch(epoch)
        train_metrics = run_epoch(
            loaders["train"],
            stage1,
            stage2a,
            stage2b,
            device,
            amp_enabled,
            channels_last,
            float(frozen_metadata["stage1_mean_db"]),
            float(frozen_metadata["stage1_std_db"]),
            float(frozen_metadata["stage2a_residual_center_db"]),
            float(frozen_metadata["stage2a_residual_scale_db"]),
            iso_mean_db,
            iso_std_db,
            ss_mean_db,
            ss_std_db,
            loss_name,
            huber_delta,
            optimizer,
            scaler,
            accumulation,
            float(config["gradient_clip"]),
            "train",
            epoch,
            "random_1_200",
            progress_every,
        )
        val_metrics = run_epoch(
            loaders["val"],
            stage1,
            stage2a,
            stage2b,
            device,
            amp_enabled,
            channels_last,
            float(frozen_metadata["stage1_mean_db"]),
            float(frozen_metadata["stage1_std_db"]),
            float(frozen_metadata["stage2a_residual_center_db"]),
            float(frozen_metadata["stage2a_residual_scale_db"]),
            iso_mean_db,
            iso_std_db,
            ss_mean_db,
            ss_std_db,
            loss_name,
            huber_delta,
            None,
            None,
            accumulation,
            float(config["gradient_clip"]),
            "val",
            epoch,
            int(config["validation_sparse_points"]),
            progress_every,
        )
        if scheduler is not None:
            scheduler.step()
        improved = val_metrics["rmse_db"] < best_rmse
        if improved:
            best_rmse = val_metrics["rmse_db"]
            best_val_metrics = dict(val_metrics)
            patience_used = 0
        else:
            patience_used += 1
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "best_val_rmse_db": best_rmse,
            "improved": improved,
            "patience_used": patience_used,
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        checkpoint = {
            "version": config["version"],
            "epoch": epoch,
            "model": stage2b.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "best_val_rmse_db": best_rmse,
            "best_val_metrics": best_val_metrics,
            "patience_used": patience_used,
            "history": history,
            "config": config,
            "ss_mean_db": ss_mean_db,
            "ss_std_db": ss_std_db,
            **frozen_metadata,
        }
        atomic_torch_save(output_dir / "last.pt", checkpoint)
        if improved:
            best_checkpoint = {
                "version": config["version"],
                "epoch": epoch,
                "model": stage2b.state_dict(),
                "best_val_rmse_db": best_rmse,
                "best_val_metrics": best_val_metrics,
                "config": config,
                "iso_mean_db": iso_mean_db,
                "iso_std_db": iso_std_db,
                "ss_mean_db": ss_mean_db,
                "ss_std_db": ss_std_db,
                **frozen_metadata,
            }
            atomic_torch_save(best_path, best_checkpoint)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if patience_used >= int(config["patience"]):
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    stage2b.load_state_dict(best["model"], strict=True)
    tests: dict[str, Any] = {}
    for sparse_points in config["test_sparse_points"]:
        test_dataset = make_dataset(
            config,
            "test",
            ss_mean_db,
            ss_std_db,
            args.test_limit_tiles,
            int(sparse_points),
        )
        test_loader = make_loader(
            test_dataset, config, shuffle=False, device=device
        )
        tests[str(sparse_points)] = run_epoch(
            test_loader,
            stage1,
            stage2a,
            stage2b,
            device,
            amp_enabled,
            channels_last,
            float(frozen_metadata["stage1_mean_db"]),
            float(frozen_metadata["stage1_std_db"]),
            float(frozen_metadata["stage2a_residual_center_db"]),
            float(frozen_metadata["stage2a_residual_scale_db"]),
            iso_mean_db,
            iso_std_db,
            ss_mean_db,
            ss_std_db,
            loss_name,
            huber_delta,
            None,
            None,
            accumulation,
            float(config["gradient_clip"]),
            "test",
            int(best["epoch"]),
            int(sparse_points),
            progress_every,
        )
    acceptance = config.get("acceptance", {})
    minimum_rmse_improvement = float(
        acceptance.get(
            "minimum_test_rmse_improvement_vs_sparse_calibrated_iso_db",
            0.0,
        )
    )
    require_mae_better = bool(
        acceptance.get("require_test_mae_better", True)
    )
    acceptance_checks: dict[str, Any] = {}
    for sparse_points, metrics in tests.items():
        rmse_improvement = float(
            metrics[
                "rmse_improvement_vs_sparse_calibrated_iso_db"
            ]
        )
        mae_improvement = float(
            metrics[
                "mae_improvement_vs_sparse_calibrated_iso_db"
            ]
        )
        acceptance_checks[sparse_points] = {
            "rmse_improvement_db": rmse_improvement,
            "minimum_required_db": minimum_rmse_improvement,
            "rmse_pass": rmse_improvement >= minimum_rmse_improvement,
            "mae_improvement_db": mae_improvement,
            "mae_pass": (
                not require_mae_better or mae_improvement > 0.0
            ),
        }
    accepted = all(
        check["rmse_pass"] and check["mae_pass"]
        for check in acceptance_checks.values()
    )
    result = {
        "status": "ok" if accepted else "completed_not_accepted",
        "accepted_for_final_evaluation": accepted,
        "best_epoch": int(best["epoch"]),
        "best_val_rmse_db": float(best["best_val_rmse_db"]),
        "best_val_metrics": best.get("best_val_metrics", {}),
        "test_by_sparse_points": tests,
        "acceptance_checks": acceptance_checks,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "input": (
            "[B, frozen Stage2-A isotropic IRT prediction, "
            "sparse directional SS, sparse mask]"
        ),
    }
    atomic_write_json(output_dir / "test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

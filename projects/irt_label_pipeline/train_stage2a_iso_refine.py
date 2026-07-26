#!/usr/bin/env python3
"""Train Stage2-A as a frozen-DPM-to-isotropic-IRT residual refiner."""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from stage2_models import ResidualUNet, parameter_count
from stage2a_dataset import Stage2AIsoDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--device")
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
    dataset: Stage2AIsoDataset,
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
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        arguments["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(**arguments)


def load_stage1(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, float, float, str]:
    model_dir = Path(config["stage1_model_dir"])
    checkpoint_path = model_dir / "best.pt"
    normalization_path = model_dir / "normalization.json"
    expected_hash = config["provenance"]["stage1_checkpoint_sha256"].lower()
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash.lower() != expected_hash:
        raise ValueError(
            f"Stage1 checkpoint hash mismatch: {actual_hash} != {expected_hash}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    saved_args = checkpoint.get("args", {})
    base_channels = int(saved_args.get("base_channels", 32))
    stage1 = ResidualUNet(
        in_channels=3,
        base_channels=base_channels,
    ).to(device)
    stage1.load_state_dict(checkpoint["model"])
    stage1.eval()
    stage1.requires_grad_(False)
    if channels_last:
        stage1 = stage1.to(memory_format=torch.channels_last)
    normalization = load_json(normalization_path)
    path_gain = normalization["statistics"]["path_gain"]
    return (
        stage1,
        float(path_gain["mean_db"]),
        float(path_gain["std_db"]),
        actual_hash,
    )


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if not torch.any(mask):
        raise ValueError("batch contains no valid IRT pixels")
    return functional.smooth_l1_loss(
        prediction[mask],
        target[mask],
        beta=beta,
        reduction="mean",
    )


def run_epoch(
    loader: DataLoader,
    stage1: nn.Module,
    stage2: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
    stage1_mean_db: float,
    stage1_std_db: float,
    residual_scale_db: float,
    huber_beta: float,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    accumulation: int,
    gradient_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    stage2.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    batches = 0
    valid_pixels = 0
    squared_error = 0.0
    absolute_error = 0.0
    baseline_squared_error = 0.0
    baseline_absolute_error = 0.0
    for batch_index, batch in enumerate(loader):
        stage1_input = batch["stage1_input"].to(
            device,
            non_blocking=True,
        )
        building = batch["building"].to(device, non_blocking=True)
        target_db = batch["target_irt_db"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(
            device,
            non_blocking=True,
            dtype=torch.bool,
        )
        cached_dpm_norm = batch.get("stage1_prediction_norm")
        if cached_dpm_norm is not None:
            cached_dpm_norm = cached_dpm_norm.to(
                device,
                non_blocking=True,
            )
        if channels_last:
            stage1_input = stage1_input.contiguous(
                memory_format=torch.channels_last
            )
            building = building.contiguous(memory_format=torch.channels_last)
            if cached_dpm_norm is not None:
                cached_dpm_norm = cached_dpm_norm.contiguous(
                    memory_format=torch.channels_last
                )
        if cached_dpm_norm is None:
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                dpm_norm = stage1(stage1_input)
        else:
            dpm_norm = cached_dpm_norm
        dpm_norm = dpm_norm.detach()
        dpm_db = dpm_norm.float() * stage1_std_db + stage1_mean_db
        residual_target = (target_db - dpm_db) / residual_scale_db
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            stage2_input = torch.cat((building, dpm_norm), dim=1)
            residual_prediction = stage2(stage2_input)
            loss = masked_loss(
                residual_prediction,
                residual_target,
                mask,
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
                    stage2.parameters(),
                    gradient_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        prediction_db = (
            dpm_db + residual_prediction.detach().float() * residual_scale_db
        )
        error = prediction_db[mask] - target_db[mask]
        baseline_error = dpm_db[mask] - target_db[mask]
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
    if not batches or not valid_pixels:
        raise RuntimeError("empty epoch")
    return {
        "loss": loss_sum / batches,
        "rmse_db": math.sqrt(squared_error / valid_pixels),
        "mae_db": absolute_error / valid_pixels,
        "dpm_baseline_rmse_db": math.sqrt(
            baseline_squared_error / valid_pixels
        ),
        "dpm_baseline_mae_db": baseline_absolute_error / valid_pixels,
        "valid_pixels": float(valid_pixels),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(args.device or str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if best_path.exists() and args.resume is None:
        raise FileExistsError(
            f"{best_path} already exists; use --resume or a new versioned output"
        )
    datasets = {
        "train": Stage2AIsoDataset(
            config["shard_root"],
            config["shard_manifest"],
            config["selection_csv"],
            config["normalized_root"],
            "train",
            stage1_prediction_root=config.get("stage1_prediction_root"),
            augment=bool(config["augmentation"]),
            limit=args.train_limit,
        ),
        "val": Stage2AIsoDataset(
            config["shard_root"],
            config["shard_manifest"],
            config["selection_csv"],
            config["normalized_root"],
            "val",
            stage1_prediction_root=config.get("stage1_prediction_root"),
            augment=False,
            limit=args.val_limit,
        ),
        "test": Stage2AIsoDataset(
            config["shard_root"],
            config["shard_manifest"],
            config["selection_csv"],
            config["normalized_root"],
            "test",
            stage1_prediction_root=config.get("stage1_prediction_root"),
            augment=False,
            limit=args.test_limit,
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
    stage1, stage1_mean, stage1_std, stage1_hash = load_stage1(
        config,
        device,
        channels_last,
    )
    stage2 = ResidualUNet(
        in_channels=2,
        base_channels=int(config["base_channels"]),
    ).to(device)
    if channels_last:
        stage2 = stage2.to(memory_format=torch.channels_last)
    optimizer = AdamW(
        stage2.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = args.epochs or int(config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch = 1
    best_rmse = math.inf
    patience_used = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        stage2.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rmse = float(checkpoint["best_val_rmse_db"])
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
        "stage1_mean_db": stage1_mean,
        "stage1_std_db": stage1_std,
        "stage1_checkpoint_sha256": stage1_hash,
        "stage2_parameters": parameter_count(stage2),
        "split_counts": {
            split: len(dataset) for split, dataset in datasets.items()
        },
    }
    atomic_write_json(output_dir / "run_config.json", run_config)
    residual_scale = float(config["residual_scale_db"])
    accumulation = int(config["gradient_accumulation"])
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            loaders["train"],
            stage1,
            stage2,
            device,
            amp_enabled,
            channels_last,
            stage1_mean,
            stage1_std,
            residual_scale,
            float(config["huber_delta_normalized"]),
            optimizer,
            scaler,
            accumulation,
            float(config["gradient_clip"]),
        )
        val_metrics = run_epoch(
            loaders["val"],
            stage1,
            stage2,
            device,
            amp_enabled,
            channels_last,
            stage1_mean,
            stage1_std,
            residual_scale,
            float(config["huber_delta_normalized"]),
            None,
            None,
            accumulation,
            float(config["gradient_clip"]),
        )
        scheduler.step()
        improved = val_metrics["rmse_db"] < best_rmse
        if improved:
            best_rmse = val_metrics["rmse_db"]
            best_epoch = epoch
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
            "model": stage2.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_rmse_db": best_rmse,
            "patience_used": patience_used,
            "history": history,
            "config": config,
            "stage1_checkpoint_sha256": stage1_hash,
            "stage1_mean_db": stage1_mean,
            "stage1_std_db": stage1_std,
            "residual_scale_db": residual_scale,
        }
        atomic_torch_save(output_dir / "last.pt", checkpoint)
        if improved:
            atomic_torch_save(best_path, checkpoint)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if patience_used >= int(config["patience"]):
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    stage2.load_state_dict(best["model"])
    test_metrics = run_epoch(
        loaders["test"],
        stage1,
        stage2,
        device,
        amp_enabled,
        channels_last,
        stage1_mean,
        stage1_std,
        residual_scale,
        float(config["huber_delta_normalized"]),
        None,
        None,
        accumulation,
        float(config["gradient_clip"]),
    )
    result = {
        "status": "ok",
        "best_epoch": int(best["epoch"]),
        "best_val_rmse_db": float(best["best_val_rmse_db"]),
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "formula": (
            "P_iso_IRT_hat = P_iso_DPM_hat + "
            "residual_scale_db * U_IsoRefine(B, P_iso_DPM_hat_norm)"
        ),
    }
    atomic_write_json(output_dir / "test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

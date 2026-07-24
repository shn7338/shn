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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from stage2_models import ResidualUNet, parameter_count
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


def load_frozen_models(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, nn.Module, dict[str, float | str]]:
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
    stage2a = ResidualUNet(
        in_channels=2,
        base_channels=int(stage2a_config["base_channels"]),
    ).to(device)
    stage2a.load_state_dict(stage2a_checkpoint["model"], strict=True)

    for model in (stage1, stage2a):
        model.eval()
        model.requires_grad_(False)
        if channels_last:
            model.to(memory_format=torch.channels_last)
    normalization = load_json(stage1_normalization_path)
    path_gain = normalization["statistics"]["path_gain"]
    metadata: dict[str, float | str] = {
        "stage1_mean_db": float(path_gain["mean_db"]),
        "stage1_std_db": float(path_gain["std_db"]),
        "stage2a_residual_scale_db": float(
            stage2a_checkpoint["residual_scale_db"]
        ),
        "stage1_checkpoint_sha256": stage1_hash,
        "stage2a_checkpoint_sha256": stage2a_hash,
    }
    return stage1, stage2a, metadata


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if not torch.any(mask):
        raise ValueError("batch contains no valid directional pixels")
    return functional.smooth_l1_loss(
        prediction[mask],
        target[mask],
        beta=beta,
        reduction="mean",
    )


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
    stage2a_residual_scale_db: float,
    ss_mean_db: float,
    ss_std_db: float,
    huber_beta: float,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    accumulation: int,
    gradient_clip: float,
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
        sparse_point_sum += int(torch.sum(sparse_mask).item())
        if channels_last:
            stage1_input = stage1_input.contiguous(
                memory_format=torch.channels_last
            )
            building = building.contiguous(
                memory_format=torch.channels_last
            )
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            dpm_norm = stage1(stage1_input)
            stage2a_input = torch.cat((building, dpm_norm), dim=1)
            residual_norm = stage2a(stage2a_input)
        dpm_db = dpm_norm.float() * stage1_std_db + stage1_mean_db
        corrected_iso_db = (
            dpm_db
            + residual_norm.float() * stage2a_residual_scale_db
        )
        corrected_iso_norm = (
            corrected_iso_db - stage1_mean_db
        ) / stage1_std_db
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
            loss = masked_huber(
                prediction_norm,
                target_norm,
                valid_mask,
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
    if not batches or not valid_pixels:
        raise RuntimeError("empty epoch")
    return {
        "loss": loss_sum / batches,
        "rmse_db": math.sqrt(squared_error / valid_pixels),
        "mae_db": absolute_error / valid_pixels,
        "sparse_calibrated_iso_rmse_db": math.sqrt(
            baseline_squared_error / valid_pixels
        ),
        "sparse_calibrated_iso_mae_db": (
            baseline_absolute_error / valid_pixels
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
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device or str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    irt_normalization = load_json(Path(config["irt_normalization"]))
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
    stage2b = ResidualUNet(
        in_channels=4,
        base_channels=int(config["base_channels"]),
    ).to(device)
    if channels_last:
        stage2b.to(memory_format=torch.channels_last)
    optimizer = AdamW(
        stage2b.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = args.epochs or int(config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if best_path.exists() and args.resume is None:
        raise FileExistsError(
            f"{best_path} already exists; use --resume or a new output version"
        )
    start_epoch = 1
    best_rmse = math.inf
    patience_used = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        stage2b.load_state_dict(checkpoint["model"], strict=True)
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
        "ss_mean_db": ss_mean_db,
        "ss_std_db": ss_std_db,
        "stage2b_parameters": parameter_count(stage2b),
        "split_directional_samples": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        **frozen_metadata,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)
    best_epoch = 0
    started = time.perf_counter()
    accumulation = int(config["gradient_accumulation"])
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
            float(frozen_metadata["stage2a_residual_scale_db"]),
            ss_mean_db,
            ss_std_db,
            float(config["huber_delta_normalized"]),
            optimizer,
            scaler,
            accumulation,
            float(config["gradient_clip"]),
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
            float(frozen_metadata["stage2a_residual_scale_db"]),
            ss_mean_db,
            ss_std_db,
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
            "model": stage2b.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_rmse_db": best_rmse,
            "patience_used": patience_used,
            "history": history,
            "config": config,
            "ss_mean_db": ss_mean_db,
            "ss_std_db": ss_std_db,
            **frozen_metadata,
        }
        atomic_torch_save(output_dir / "last.pt", checkpoint)
        if improved:
            atomic_torch_save(best_path, checkpoint)
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
            float(frozen_metadata["stage2a_residual_scale_db"]),
            ss_mean_db,
            ss_std_db,
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
        "test_by_sparse_points": tests,
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

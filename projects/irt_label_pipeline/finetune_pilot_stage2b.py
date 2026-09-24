#!/usr/bin/env python3
"""Fine-tune Stage2-B on Pilot samples using estimated latent BS positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
from latent_bs_fusion import (
    build_latent_fusion_features,
    configured_feature_names,
    configured_stage2b_in_channels,
    transplant_input_channels,
)
from run_sionna_dataset import sha256_file
from train_stage2b_directional_ss import (
    build_configured_unet,
    load_frozen_models,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a last.pt checkpoint, including optimizer state.",
    )
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

    def metrics(self, prefix: str) -> dict[str, float | int]:
        if self.count == 0:
            raise RuntimeError(f"no pixels accumulated for {prefix}")
        return {
            f"{prefix}_pixels": self.count,
            f"{prefix}_rmse_db": math.sqrt(self.squared / self.count),
            f"{prefix}_mae_db": self.absolute / self.count,
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
        # Recreate workers so D4 augmentation sees dataset.set_epoch on Windows.
        "persistent_workers": False,
    }
    if workers:
        arguments["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(**arguments)


def load_estimator(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, dict[str, Any], Path]:
    estimator_config_path = Path(config["estimator_config"])
    estimator_config = load_json(estimator_config_path)
    checkpoint_path = Path(
        config.get(
            "estimator_checkpoint",
            Path(estimator_config["output_dir"]) / "best.pt",
        )
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    estimator = BSParameterEstimator(
        in_channels=3,
        base_channels=int(model_config["base_channels"]),
        scalar_hidden_channels=int(model_config["scalar_hidden_channels"]),
        softargmax_temperature=float(model_config["softargmax_temperature"]),
    ).to(device)
    estimator.load_state_dict(checkpoint["model"], strict=True)
    estimator.eval().requires_grad_(False)
    if channels_last:
        estimator.to(memory_format=torch.channels_last)
    normalization = load_json(Path(estimator_config["dataset_root"]) / "normalization.json")
    signal_stats = normalization["directional_signal_strength_db"]
    manifest = load_json(Path(estimator_config["dataset_root"]) / "run_manifest.json")
    data_config = load_json(Path(manifest["config"]))
    power_config = data_config["effective_power"]
    metadata = {
        "signal_mean_db": float(signal_stats["mean"]),
        "signal_std_db": float(signal_stats["std"]),
        "power_mean_db": float(power_config["normalization_mean_db"]),
        "power_std_db": float(power_config["normalization_std_db"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(estimator_config_path.resolve()),
        "config_sha256": sha256_file(estimator_config_path),
    }
    return estimator, metadata, checkpoint_path


def load_initial_stage2b(
    config: dict[str, Any],
    device: torch.device,
    channels_last: bool,
) -> tuple[nn.Module, Path, str, dict[str, Any]]:
    checkpoint_path = Path(config["initial_stage2b_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint["model"]
    target_in_channels = configured_stage2b_in_channels(config)
    model = build_configured_unet(config, in_channels=target_in_channels).to(device)
    target_state = model.state_dict()
    transplanted, weight_key, source_in_channels, resolved_target_channels = (
        transplant_input_channels(source_state, target_state)
    )
    model.load_state_dict(transplanted, strict=True)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    metadata = {
        "source_in_channels": source_in_channels,
        "target_in_channels": resolved_target_channels,
        "first_convolution_weight_key": weight_key,
        "latent_feature_names": list(configured_feature_names(config)),
        "new_input_planes_initialized_to_zero": bool(
            resolved_target_channels > source_in_channels
        ),
    }
    return model, checkpoint_path, sha256_file(checkpoint_path), metadata


def run_epoch(
    loader: DataLoader,
    estimator: nn.Module,
    stage1: nn.Module,
    stage2a: nn.Module,
    stage2b: nn.Module,
    *,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
    frozen: dict[str, Any],
    estimator_signal_mean_db: float,
    estimator_signal_std_db: float,
    estimator_power_mean_db: float,
    estimator_power_std_db: float,
    iso_mean_db: float,
    iso_std_db: float,
    ss_mean_db: float,
    ss_std_db: float,
    tx_height_cap_m: float,
    distance_max_m: float,
    stage1_in_channels: int,
    long_distance_max_m: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip: float,
    progress_every: int,
    latent_fusion_config: dict[str, Any],
    phase: str,
    epoch: int,
) -> dict[str, float | int]:
    training = optimizer is not None
    stage2b.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    errors = ErrorSums()
    unmeasured_errors = ErrorSums()
    baseline_errors = ErrorSums()
    location_errors: list[float] = []
    loss_sum = 0.0
    batches = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        building = batch["building"].to(device, non_blocking=True)
        sparse_norm = batch["sparse_ss_norm"].to(device, non_blocking=True)
        sparse_mask = batch["sparse_mask"].to(device, non_blocking=True)
        target_db = batch["target_map_db"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).bool()
        true_row_col = batch["target_row_col_px"].to(device, non_blocking=True)
        tx_height_m = batch["tx_height_m"].to(device, non_blocking=True)

        sparse_db = sparse_norm.float() * ss_std_db + ss_mean_db
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
            estimated = estimator(estimator_input)
            row_col = estimated["row_col_px"].float()
        location_errors.extend(
            torch.linalg.vector_norm(row_col - true_row_col, dim=1).cpu().tolist()
        )
        tx_xy_m = torch.stack(
            (
                (row_col[:, 1] + 0.5) * 4.0,
                512.0 - (row_col[:, 0] + 0.5) * 4.0,
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
        with torch.inference_mode(), torch.amp.autocast(
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
        corrected_iso_norm = (corrected_iso_db - iso_mean_db) / iso_std_db
        latent_features = build_latent_fusion_features(
            estimated,
            latent_fusion_config,
            estimator_signal_mean_db=estimator_signal_mean_db,
            estimator_signal_std_db=estimator_signal_std_db,
            stage2b_signal_mean_db=ss_mean_db,
            stage2b_signal_std_db=ss_std_db,
            estimator_power_mean_db=estimator_power_mean_db,
            estimator_power_std_db=estimator_power_std_db,
            rows=int(building.shape[-2]),
            cols=int(building.shape[-1]),
        )
        target_norm = torch.where(
            valid,
            (target_db - ss_mean_db) / ss_std_db,
            0.0,
        )
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
                    latent_features,
                ),
                dim=1,
            )
            if channels_last:
                stage2b_input = stage2b_input.contiguous(
                    memory_format=torch.channels_last
                )
            prediction_norm = stage2b(stage2b_input)
            loss = functional.mse_loss(
                prediction_norm[valid], target_norm[valid], reduction="mean"
            )
        if training:
            assert scaler is not None and optimizer is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(stage2b.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        prediction_db = prediction_norm.detach().float() * ss_std_db + ss_mean_db
        unmeasured = valid & ~sparse_mask.bool()
        errors.update((prediction_db - target_db)[valid])
        unmeasured_errors.update((prediction_db - target_db)[unmeasured])
        baseline_predictions: list[torch.Tensor] = []
        for sample in range(building.shape[0]):
            measured = sparse_mask[sample].bool()
            offset = torch.median(
                target_db[sample][measured] - corrected_iso_db[sample][measured]
            )
            baseline_predictions.append(corrected_iso_db[sample] + offset)
        baseline_db = torch.stack(baseline_predictions)
        baseline_errors.update((baseline_db - target_db)[valid])
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
    metrics.update(errors.metrics("map"))
    metrics.update(unmeasured_errors.metrics("unmeasured"))
    metrics.update(baseline_errors.metrics("baseline"))
    location_array = np.asarray(location_errors, dtype=np.float64)
    metrics.update(
        {
            "location_mean_error_px": float(location_array.mean()),
            "location_p90_error_px": float(np.percentile(location_array, 90.0)),
        }
    )
    return metrics


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
    iso_mean_db = float(iso_stats["mean_db"])
    iso_std_db = float(iso_stats["std_db"])
    ss_mean_db = float(ss_stats["mean_db"])
    ss_std_db = float(ss_stats["std_db"])
    datasets = {
        split: BSInversionPilotDataset(
            config["dataset_root"],
            split=split,
            seed=int(config["seed"]),
            augment=split == "train" and bool(config["augmentation"]),
            signal_mean_db=ss_mean_db,
            signal_std_db=ss_std_db,
        )
        for split in ("train", "val")
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
    stage1, stage2a, frozen = load_frozen_models(config, device, channels_last)
    estimator, estimator_metadata, estimator_checkpoint = load_estimator(
        config, device, channels_last
    )
    stage2b, initial_checkpoint, initial_hash, input_expansion = load_initial_stage2b(
        config, device, channels_last
    )
    optimizer = AdamW(
        stage2b.parameters(),
        lr=float(config["learning_rate"]),
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
        Path(config["stage1_model_dir"]) / "tx_feature_normalization.json"
    )
    tx_height_cap_m = float(tx_normalization["transmitter_height"]["cap_m"])
    distance_max_m = float(tx_normalization["distance"]["max_m"])
    long_distance_max_m = float(
        tx_normalization.get("long_distance", {}).get(
            "max_m", math.sqrt(2.0) * 512.0
        )
    )

    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if best_path.exists() and args.resume is None:
        raise FileExistsError(f"{best_path} already exists; choose a new output version")
    run_config = {
        **config,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "initial_stage2b_checkpoint_sha256": initial_hash,
        "estimator_checkpoint": str(estimator_checkpoint),
        "estimator_checkpoint_sha256": estimator_metadata["checkpoint_sha256"],
        "split_samples": {key: len(value) for key, value in datasets.items()},
        "stage2b_input_expansion": input_expansion,
        **frozen,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)

    history: list[dict[str, Any]] = []
    best_rmse = math.inf
    best_epoch = 0
    patience_used = 0
    start_epoch = 1
    if args.resume is not None:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        for required in ("model", "optimizer", "scheduler", "epoch", "history"):
            if required not in checkpoint:
                raise ValueError(f"resume checkpoint is missing {required!r}: {resume_path}")
        stage2b.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_rmse = float(checkpoint["best_val_rmse_db"])
        patience_used = int(checkpoint.get("patience_used", 0))
        history = list(checkpoint["history"])
        if start_epoch > epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but requested epochs={epochs}"
            )
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        datasets["train"].set_epoch(epoch)
        train_metrics = run_epoch(
            loaders["train"],
            estimator,
            stage1,
            stage2a,
            stage2b,
            device=device,
            amp_enabled=amp_enabled,
            channels_last=channels_last,
            frozen=frozen,
            estimator_signal_mean_db=float(estimator_metadata["signal_mean_db"]),
            estimator_signal_std_db=float(estimator_metadata["signal_std_db"]),
            estimator_power_mean_db=float(estimator_metadata["power_mean_db"]),
            estimator_power_std_db=float(estimator_metadata["power_std_db"]),
            iso_mean_db=iso_mean_db,
            iso_std_db=iso_std_db,
            ss_mean_db=ss_mean_db,
            ss_std_db=ss_std_db,
            tx_height_cap_m=tx_height_cap_m,
            distance_max_m=distance_max_m,
            stage1_in_channels=int(frozen.get("stage1_in_channels", 3)),
            long_distance_max_m=long_distance_max_m,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=float(config["gradient_clip"]),
            progress_every=int(config.get("progress_every_batches", 0)),
            latent_fusion_config=config.get("latent_fusion", {}),
            phase="train",
            epoch=epoch,
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                loaders["val"],
                estimator,
                stage1,
                stage2a,
                stage2b,
                device=device,
                amp_enabled=amp_enabled,
                channels_last=channels_last,
                frozen=frozen,
                estimator_signal_mean_db=float(estimator_metadata["signal_mean_db"]),
                estimator_signal_std_db=float(estimator_metadata["signal_std_db"]),
                estimator_power_mean_db=float(estimator_metadata["power_mean_db"]),
                estimator_power_std_db=float(estimator_metadata["power_std_db"]),
                iso_mean_db=iso_mean_db,
                iso_std_db=iso_std_db,
                ss_mean_db=ss_mean_db,
                ss_std_db=ss_std_db,
                tx_height_cap_m=tx_height_cap_m,
                distance_max_m=distance_max_m,
                stage1_in_channels=int(frozen.get("stage1_in_channels", 3)),
                long_distance_max_m=long_distance_max_m,
                optimizer=None,
                scaler=None,
                gradient_clip=float(config["gradient_clip"]),
                progress_every=int(config.get("progress_every_batches", 0)),
                latent_fusion_config=config.get("latent_fusion", {}),
                phase="val",
                epoch=epoch,
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        val_rmse = float(val_metrics["map_rmse_db"])
        improved = val_rmse < best_rmse
        if improved:
            best_rmse = val_rmse
            best_epoch = epoch
            patience_used = 0
            atomic_torch_save(
                best_path,
                {
                    "version": 1,
                    "model": stage2b.state_dict(),
                    "epoch": epoch,
                    "best_val_rmse_db": best_rmse,
                    "best_val_metrics": val_metrics,
                    "config": config,
                    "initial_stage2b_checkpoint": str(initial_checkpoint),
                    "initial_stage2b_checkpoint_sha256": initial_hash,
                    "estimator_checkpoint": str(estimator_checkpoint),
                    "estimator_checkpoint_sha256": estimator_metadata[
                        "checkpoint_sha256"
                    ],
                    "stage1_checkpoint_sha256": frozen["stage1_checkpoint_sha256"],
                    "stage2a_checkpoint_sha256": frozen["stage2a_checkpoint_sha256"],
                },
            )
        else:
            patience_used += 1
        atomic_torch_save(
            output_dir / "last.pt",
            {
                "version": 1,
                "model": stage2b.state_dict(),
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
                    "train_rmse_db": train_metrics["map_rmse_db"],
                    "val_rmse_db": val_rmse,
                    "best_val_rmse_db": best_rmse,
                    "best_epoch": best_epoch,
                    "patience_used": patience_used,
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if patience_used >= int(config["patience"]):
            break

    summary = {
        "version": 1,
        "status": "ok",
        "best_epoch": best_epoch,
        "best_val_rmse_db": best_rmse,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

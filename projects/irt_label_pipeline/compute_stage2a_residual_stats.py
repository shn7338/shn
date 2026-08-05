#!/usr/bin/env python3
"""Fit train-only DPM-to-Sionna residual normalization for Stage2-A."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from stage2a_dataset import Stage2AIsoDataset
from train_stage2a_iso_refine import (
    atomic_write_json,
    load_json,
    load_stage1,
    resolve_device,
    seed_everything,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_loader(
    dataset: Stage2AIsoDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    prefetch_factor: int,
) -> DataLoader:
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        arguments["prefetch_factor"] = prefetch_factor
    return DataLoader(**arguments)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output_value = args.output or config.get("residual_stats_path")
    if output_value is None:
        raise ValueError("--output or config residual_stats_path is required")
    output_path = Path(output_value).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    seed_everything(int(config["seed"]))
    device = resolve_device(args.device or str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    dataset = Stage2AIsoDataset(
        config["shard_root"],
        config["shard_manifest"],
        config["selection_csv"],
        config["normalized_root"],
        "train",
        stage1_prediction_root=config.get("stage1_prediction_root"),
        augment=False,
        limit=args.limit,
    )
    batch_size = args.batch_size or int(config.get("stats_batch_size", 16))
    workers = (
        args.workers
        if args.workers is not None
        else int(config.get("workers", 2))
    )
    loader = make_loader(
        dataset,
        batch_size,
        workers,
        device,
        int(config.get("prefetch_factor", 2)),
    )
    stage1, stage1_mean, stage1_std, stage1_hash = load_stage1(
        config,
        device,
        channels_last,
    )

    count = 0
    residual_sum = 0.0
    residual_square_sum = 0.0
    residual_absolute_sum = 0.0
    residual_min = math.inf
    residual_max = -math.inf
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        stage1_input = batch["stage1_input"].to(
            device,
            non_blocking=True,
        )
        target_db = batch["target_irt_db"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(
            device,
            non_blocking=True,
            dtype=torch.bool,
        )
        cached_prediction = batch.get("stage1_prediction_norm")
        if cached_prediction is not None:
            dpm_norm = cached_prediction.to(device, non_blocking=True)
        else:
            if channels_last:
                stage1_input = stage1_input.contiguous(
                    memory_format=torch.channels_last
                )
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                dpm_norm = stage1(stage1_input)
        dpm_db = dpm_norm.float() * stage1_std + stage1_mean
        residual = (target_db - dpm_db)[valid_mask].double()
        if not residual.numel():
            continue
        count += int(residual.numel())
        residual_sum += float(residual.sum().item())
        residual_square_sum += float(residual.square().sum().item())
        residual_absolute_sum += float(residual.abs().sum().item())
        residual_min = min(residual_min, float(residual.min().item()))
        residual_max = max(residual_max, float(residual.max().item()))
        if batch_index % 100 == 0 or batch_index == len(loader):
            print(
                json.dumps(
                    {
                        "batches": batch_index,
                        "batches_total": len(loader),
                        "tiles_processed": min(
                            batch_index * batch_size,
                            len(dataset),
                        ),
                        "valid_pixels": count,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if not count:
        raise RuntimeError("no valid residual pixels were found")

    mean = residual_sum / count
    variance = max(residual_square_sum / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if not math.isfinite(std) or std <= 0:
        raise RuntimeError("residual standard deviation is invalid")
    payload = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "full_split": args.limit is None,
        "tiles": len(dataset),
        "valid_pixels": count,
        "residual_db": {
            "definition": "P_iso_Sionna_dB - P_iso_DPM_pred_dB",
            "mean_db": mean,
            "std_db": std,
            "minimum_db": residual_min,
            "maximum_db": residual_max,
            "rmse_from_zero_db": math.sqrt(residual_square_sum / count),
            "mae_from_zero_db": residual_absolute_sum / count,
            "calibrated_constant_baseline_rmse_db": std,
        },
        "source": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "shard_manifest": str(Path(config["shard_manifest"]).resolve()),
            "shard_manifest_sha256": sha256_file(
                Path(config["shard_manifest"])
            ),
            "stage1_checkpoint": str(
                Path(config["stage1_model_dir"]) / "best.pt"
            ),
            "stage1_checkpoint_sha256": stage1_hash,
            "stage1_mean_db": stage1_mean,
            "stage1_std_db": stage1_std,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "batch_size": batch_size,
            "workers": workers,
            "amp": amp_enabled,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train the latent BS location/power/azimuth estimator on the Pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bs_inversion_dataset import BSInversionPilotDataset, load_json
from bs_parameter_model import BSParameterEstimator, parameter_count
from run_sionna_dataset import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize model weights only; optimizer and epoch start fresh.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def circular_error_degrees(
    predicted_sin_cos: torch.Tensor, target_sin_cos: torch.Tensor
) -> torch.Tensor:
    predicted = functional.normalize(predicted_sin_cos.float(), dim=1, eps=1e-6)
    target = functional.normalize(target_sin_cos.float(), dim=1, eps=1e-6)
    predicted_angle = torch.atan2(predicted[:, 0], predicted[:, 1])
    target_angle = torch.atan2(target[:, 0], target[:, 1])
    difference = torch.remainder(
        predicted_angle - target_angle + math.pi, 2.0 * math.pi
    ) - math.pi
    return torch.abs(torch.rad2deg(difference))


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_heatmap = batch["target_heatmap"]
    target_probability = target_heatmap.flatten(1)
    target_probability = target_probability / target_probability.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    location_log_probability = functional.log_softmax(
        outputs["location_logits"].flatten(1), dim=1
    )
    heatmap_loss = functional.kl_div(
        location_log_probability,
        target_probability,
        reduction="batchmean",
    )
    coordinate_loss = functional.smooth_l1_loss(
        outputs["row_col_px"],
        batch["target_row_col_px"],
        beta=2.0,
    ) / 10.0
    power_loss = functional.smooth_l1_loss(
        outputs["power_norm"], batch["target_power_norm"], beta=0.5
    )
    predicted_direction = functional.normalize(
        outputs["direction_raw"], dim=1, eps=1e-6
    )
    target_direction = functional.normalize(
        batch["target_direction_sin_cos"], dim=1, eps=1e-6
    )
    direction_loss = (
        1.0 - torch.sum(predicted_direction * target_direction, dim=1)
    ).mean()
    valid = batch["valid_mask"].bool()
    map_loss = functional.mse_loss(
        outputs["map_norm"][valid], batch["target_map_norm"][valid]
    )
    components = {
        "heatmap": heatmap_loss,
        "coordinate": coordinate_loss,
        "power": power_loss,
        "direction": direction_loss,
        "map": map_loss,
    }
    total = sum(float(weights[name]) * value for name, value in components.items())
    return total, components


class EpochMetrics:
    def __init__(self) -> None:
        self.samples = 0
        self.loss_sums = {
            "total": 0.0,
            "heatmap": 0.0,
            "coordinate": 0.0,
            "power": 0.0,
            "direction": 0.0,
            "map": 0.0,
        }
        self.location_errors: list[float] = []
        self.power_errors: list[float] = []
        self.direction_errors: list[float] = []
        self.map_pixels = 0
        self.map_squared_db = 0.0
        self.map_absolute_db = 0.0

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        total_loss: torch.Tensor,
        components: dict[str, torch.Tensor],
        power_mean_db: float,
        power_std_db: float,
        signal_std_db: float,
    ) -> None:
        batch_size = int(outputs["power_norm"].shape[0])
        self.samples += batch_size
        self.loss_sums["total"] += float(total_loss.detach().item()) * batch_size
        for name, value in components.items():
            self.loss_sums[name] += float(value.detach().item()) * batch_size
        location = torch.linalg.vector_norm(
            outputs["row_col_px"].float() - batch["target_row_col_px"].float(),
            dim=1,
        )
        predicted_power_db = (
            outputs["power_norm"].float() * power_std_db + power_mean_db
        )
        power_error = torch.abs(
            predicted_power_db - batch["target_power_db"].float()
        )
        direction_error = circular_error_degrees(
            outputs["direction_raw"], batch["target_direction_sin_cos"]
        )
        self.location_errors.extend(location.detach().cpu().tolist())
        self.power_errors.extend(power_error.detach().cpu().tolist())
        self.direction_errors.extend(direction_error.detach().cpu().tolist())
        valid = batch["valid_mask"].bool()
        map_error_db = (
            outputs["map_norm"].float() - batch["target_map_norm"].float()
        ) * signal_std_db
        values = map_error_db[valid].double()
        self.map_pixels += int(values.numel())
        self.map_squared_db += float(torch.sum(values.square()).item())
        self.map_absolute_db += float(torch.sum(torch.abs(values)).item())

    @staticmethod
    def distribution(values: list[float], prefix: str) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_mean": float(np.mean(array)),
            f"{prefix}_median": float(np.median(array)),
            f"{prefix}_p90": float(np.percentile(array, 90.0)),
            f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(array)))),
        }

    def result(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {"samples": self.samples}
        for name, value in self.loss_sums.items():
            result[f"loss_{name}"] = value / self.samples
        result.update(self.distribution(self.location_errors, "location_error_px"))
        result.update(self.distribution(self.power_errors, "power_abs_error_db"))
        result.update(self.distribution(self.direction_errors, "direction_abs_error_deg"))
        result.update(
            {
                "map_valid_pixels": self.map_pixels,
                "map_rmse_db": math.sqrt(self.map_squared_db / self.map_pixels),
                "map_mae_db": self.map_absolute_db / self.map_pixels,
            }
        )
        return result


def run_epoch(
    loader: DataLoader,
    model: BSParameterEstimator,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    channels_last: bool,
    weights: dict[str, float],
    gradient_clip: float,
    power_mean_db: float,
    power_std_db: float,
    signal_std_db: float,
    progress_every: int,
    phase: str,
    epoch: int,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    metrics = EpochMetrics()
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, start=1):
        tensor_keys = (
            "model_input",
            "target_map_norm",
            "valid_mask",
            "target_heatmap",
            "target_row_col_px",
            "target_power_norm",
            "target_power_db",
            "target_direction_sin_cos",
        )
        for key in tensor_keys:
            batch[key] = batch[key].to(device, non_blocking=True)
        inputs = batch["model_input"]
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model(inputs)
                total_loss, components = compute_losses(outputs, batch, weights)
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics.update(
            outputs,
            batch,
            total_loss,
            components,
            power_mean_db,
            power_std_db,
            signal_std_db,
        )
        if progress_every > 0 and batch_index % progress_every == 0:
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(loader),
                        "loss": float(total_loss.detach().item()),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return metrics.result()


def make_loader(
    dataset: BSInversionPilotDataset,
    config: dict[str, Any],
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    workers = int(config["workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=int(config["prefetch_factor"]) if workers > 0 else None,
        drop_last=False,
    )


def prefixed(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_report(
    output_dir: Path, result: dict[str, Any], acceptance: dict[str, bool]
) -> None:
    metrics = result["test_metrics"]
    report = f"""# 基站参数估计 Pilot v1

状态：{'通过预设门槛' if all(acceptance.values()) else '未完全通过预设门槛'}

- 位置平均误差：{metrics['location_error_px_mean']:.3f} 像素（{metrics['location_error_px_mean'] * 4.0:.3f} m）
- 位置 P90：{metrics['location_error_px_p90']:.3f} 像素
- 有效功率 MAE：{metrics['power_abs_error_db_mean']:.3f} dB
- 方位角 MAE：{metrics['direction_abs_error_deg_mean']:.3f}°
- 方位角 P90：{metrics['direction_abs_error_deg_p90']:.3f}°
- 辅助完整图 RMSE：{metrics['map_rmse_db']:.3f} dB

门槛：位置平均误差 ≤2 像素、功率 MAE ≤3 dB、方向 MAE ≤15°。
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = load_json(config_path)
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.smoke:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(str(config["output_dir"]) + f"_smoke_{stamp}")
        epochs = 2
        limits = {"train": 16, "val": 8, "test": 8}
        runtime_config = {
            **config,
            "batch_size": min(8, int(config["batch_size"])),
            "workers": 0,
        }
    else:
        output_dir = Path(config["output_dir"])
        epochs = int(config["epochs"])
        limits = {"train": None, "val": None, "test": None}
        runtime_config = dict(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    if best_path.exists() and args.resume is None and not args.smoke:
        raise FileExistsError(
            f"formal best checkpoint already exists; pass --resume explicitly: {best_path}"
        )
    datasets = {
        split: BSInversionPilotDataset(
            config["dataset_root"],
            split=split,
            seed=int(config["seed"]),
            augment=split == "train" and bool(config["augmentation"]),
            limit_sites=limits[split],
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: make_loader(
            dataset,
            runtime_config,
            shuffle=split == "train",
            device=device,
        )
        for split, dataset in datasets.items()
    }
    model_config = config["model"]
    model = BSParameterEstimator(
        in_channels=3,
        base_channels=int(model_config["base_channels"]),
        scalar_hidden_channels=int(model_config["scalar_hidden_channels"]),
        softargmax_temperature=float(model_config["softargmax_temperature"]),
    ).to(device)
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    init_checkpoint_hash: str | None = None
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(init_checkpoint["model"], strict=True)
        init_checkpoint_hash = sha256_file(args.init_checkpoint)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch = 1
    best_val_objective = math.inf
    best_val_metrics: dict[str, Any] = {}
    patience_used = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_objective = float(checkpoint["best_val_objective"])
        best_val_metrics = checkpoint.get("best_val_metrics", {})
        patience_used = int(checkpoint.get("patience_used", 0))
        history = list(checkpoint.get("history", []))
    normalization = load_json(Path(config["dataset_root"]) / "normalization.json")
    signal_std_db = float(normalization["directional_signal_strength_db"]["std"])
    power_mean_db = datasets["train"].power_mean_db
    power_std_db = datasets["train"].power_std_db
    run_config = {
        **runtime_config,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output_dir_resolved": str(output_dir),
        "device_resolved": str(device),
        "amp_resolved": amp_enabled,
        "channels_last_resolved": channels_last,
        "parameter_count": parameter_count(model),
        "split_samples": {split: len(dataset) for split, dataset in datasets.items()},
        "signal_mean_db": datasets["train"].signal_mean_db,
        "signal_std_db": signal_std_db,
        "power_mean_db": power_mean_db,
        "power_std_db": power_std_db,
        "init_checkpoint": (
            str(args.init_checkpoint.resolve())
            if args.init_checkpoint is not None
            else None
        ),
        "init_checkpoint_sha256": init_checkpoint_hash,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)
    weights = {key: float(value) for key, value in config["loss_weights"].items()}
    for epoch in range(start_epoch, epochs + 1):
        datasets["train"].set_epoch(epoch)
        train_metrics = run_epoch(
            loaders["train"],
            model,
            device,
            optimizer,
            scaler,
            amp_enabled,
            channels_last,
            weights,
            float(config["gradient_clip"]),
            power_mean_db,
            power_std_db,
            signal_std_db,
            int(config["progress_every_batches"]),
            "train",
            epoch,
        )
        val_metrics = run_epoch(
            loaders["val"],
            model,
            device,
            None,
            None,
            amp_enabled,
            channels_last,
            weights,
            float(config["gradient_clip"]),
            power_mean_db,
            power_std_db,
            signal_std_db,
            0,
            "val",
            epoch,
        )
        scheduler.step()
        objective = float(val_metrics["loss_total"])
        improved = objective < best_val_objective
        if improved:
            best_val_objective = objective
            best_val_metrics = val_metrics
            patience_used = 0
        else:
            patience_used += 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_val_objective": best_val_objective,
            "patience_used": patience_used,
            **prefixed(train_metrics, "train"),
            **prefixed(val_metrics, "val"),
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)
        checkpoint = {
            "version": int(config["version"]),
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_objective": best_val_objective,
            "best_val_metrics": best_val_metrics,
            "patience_used": patience_used,
            "history": history,
            "config": config,
            "run_config": run_config,
        }
        atomic_torch_save(output_dir / "last.pt", checkpoint)
        if improved:
            atomic_torch_save(best_path, checkpoint)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if patience_used >= int(config["patience"]):
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    test_metrics = run_epoch(
        loaders["test"],
        model,
        device,
        None,
        None,
        amp_enabled,
        channels_last,
        weights,
        float(config["gradient_clip"]),
        power_mean_db,
        power_std_db,
        signal_std_db,
        0,
        "test",
        int(best["epoch"]),
    )
    thresholds = config["acceptance"]
    acceptance = {
        "location_mean_le_threshold": float(test_metrics["location_error_px_mean"])
        <= float(thresholds["maximum_location_mean_error_px"]),
        "power_mae_le_threshold": float(test_metrics["power_abs_error_db_mean"])
        <= float(thresholds["maximum_power_mae_db"]),
        "direction_mae_le_threshold": float(
            test_metrics["direction_abs_error_deg_mean"]
        )
        <= float(thresholds["maximum_direction_mae_deg"]),
    }
    result = {
        "version": int(config["version"]),
        "status": "accepted" if all(acceptance.values()) else "needs_iteration",
        "best_epoch": int(best["epoch"]),
        "best_val_metrics": best["best_val_metrics"],
        "test_metrics": test_metrics,
        "acceptance_checks": acceptance,
        "thresholds": thresholds,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_dir / "test_metrics.json", result)
    write_report(output_dir, result, acceptance)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

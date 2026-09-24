"""Train the Stage-1 U-Net surrogate for WinProp DPM path-gain maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import DPMTileDataset
from metrics import DBMetrics, masked_loss
from model import DPMUNet, parameter_count


def boundary_band(height_channel: torch.Tensor, radius_pixels: int) -> torch.Tensor:
    """Binary band on both sides of building boundaries."""
    building = height_channel > 1e-6
    kernel = radius_pixels * 2 + 1
    dilated = F.max_pool2d(building.float(), kernel_size=kernel, stride=1, padding=radius_pixels) > 0
    eroded = -F.max_pool2d(-building.float(), kernel_size=kernel, stride=1, padding=radius_pixels) > 0.5
    return dilated & ~eroded


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def create_loader(dataset: DPMTileDataset, batch_size: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
        drop_last=shuffle,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gain_std_db: float,
    loss_kind: str,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    channels_last: bool,
    edge_weight: float,
    edge_radius_pixels: int,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    metrics = DBMetrics()
    edge_metrics = DBMetrics()
    non_edge_metrics = DBMetrics()
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        edge = boundary_band(inputs[:, 0:1], edge_radius_pixels)
        pixel_weight = 1.0 + (edge_weight - 1.0) * edge.to(torch.float32)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(inputs)
                loss = masked_loss(prediction, target, mask, loss_kind, pixel_weight)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().item())
        batches += 1
        metrics.update(prediction.detach(), target, mask, gain_std_db)
        edge_metrics.update(prediction.detach(), target, mask & edge, gain_std_db)
        non_edge_metrics.update(prediction.detach(), target, mask & ~edge, gain_std_db)
    if batches == 0:
        raise RuntimeError("No batches were processed")
    edge_result = edge_metrics.compute()
    non_edge_result = non_edge_metrics.compute()
    return {
        "loss": total_loss / batches,
        **metrics.compute(),
        "edge_mae_db": edge_result["mae_db"],
        "edge_rmse_db": edge_result["rmse_db"],
        "non_edge_mae_db": non_edge_result["mae_db"],
        "non_edge_rmse_db": non_edge_result["rmse_db"],
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_rmse: float,
    best_edge_rmse: float,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_rmse_db": best_rmse,
        "best_val_edge_rmse_db": best_edge_rmse,
        "args": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--building-edge-channel", action="store_true", help="Append an explicit building-boundary input channel")
    parser.add_argument(
        "--input-mode",
        choices=("height", "height_tx", "height_tx_distance"),
        default="height_tx_distance",
        help="Input-channel ablation mode",
    )
    parser.add_argument("--edge-channel-radius-pixels", type=int, default=1, help="Radius of the explicit input edge channel")
    parser.add_argument("--edge-weight", type=float, default=1.0, help="Loss weight for valid pixels in the building-boundary band")
    parser.add_argument("--edge-radius-pixels", type=int, default=2, help="Boundary loss radius; 2 pixels equals 8 m")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-limit", type=int, help="Limit train tiles for a smoke test")
    parser.add_argument("--val-limit", type=int, help="Limit validation tiles for a smoke test")
    parser.add_argument("--max-train-batches", type=int, help="Stop each train epoch after N batches")
    parser.add_argument("--max-val-batches", type=int, help="Stop each validation epoch after N batches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    use_channels_last = device.type == "cuda"
    if device.type == "cuda":
        # All tiles are 128x128, so cuDNN can safely benchmark once and retain
        # the fastest convolution kernels for every following batch.
        torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output_dir / name).exists() for name in ("history.csv", "last.pt", "best.pt")):
        raise FileExistsError(
            f"Training outputs already exist in {args.output_dir}. "
            "Use a new --output-dir or pass --resume <last.pt>."
        )
    if args.edge_weight < 1.0 or args.edge_radius_pixels < 1 or args.edge_channel_radius_pixels < 1:
        raise ValueError("Edge weights/radii must be positive, and --edge-weight must be at least 1")

    normalization = json.loads((args.data_root / "normalization.json").read_text(encoding="utf-8"))
    gain_stats = normalization["statistics"]["path_gain"]
    gain_std_db = float(gain_stats["std_db"])
    gain_mean_db = float(gain_stats["mean_db"])
    train_dataset = DPMTileDataset(
        args.data_root,
        "train",
        augment=not args.no_augment,
        limit=args.train_limit,
        add_building_edge_channel=args.building_edge_channel,
        edge_channel_radius_pixels=args.edge_channel_radius_pixels,
        input_mode=args.input_mode,
    )
    val_dataset = DPMTileDataset(
        args.data_root,
        "val",
        augment=False,
        limit=args.val_limit,
        add_building_edge_channel=args.building_edge_channel,
        edge_channel_radius_pixels=args.edge_channel_radius_pixels,
        input_mode=args.input_mode,
    )
    train_loader = create_loader(train_dataset, args.batch_size, args.workers, True, device)
    val_loader = create_loader(val_dataset, args.batch_size, args.workers, False, device)

    base_input_channels = {"height": 1, "height_tx": 2, "height_tx_distance": 3}[args.input_mode]
    input_channels = base_input_channels + int(args.building_edge_channel)
    model = DPMUNet(in_channels=input_channels, base_channels=args.base_channels).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.03)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch, best_rmse, best_edge_rmse, bad_epochs = 1, math.inf, math.inf, 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rmse = float(checkpoint.get("best_val_rmse_db", math.inf))
        best_edge_rmse = float(checkpoint.get("best_val_edge_rmse_db", math.inf))

    run_config = {
        **vars(args),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "resume": str(args.resume) if args.resume else None,
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": use_amp,
        "channels_last": use_channels_last,
        "train_tiles": len(train_dataset),
        "val_tiles": len(val_dataset),
        "parameters": parameter_count(model),
        "input_channels": input_channels,
        "path_gain_mean_db": gain_mean_db,
        "path_gain_std_db": gain_std_db,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(run_config, ensure_ascii=False, indent=2, default=str), flush=True)

    history_path = args.output_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(
            model, train_loader, device, gain_std_db, args.loss, optimizer, scaler,
            use_amp, use_channels_last, args.edge_weight, args.edge_radius_pixels, args.max_train_batches,
        )
        with torch.no_grad():
            val_result = run_epoch(
                model, val_loader, device, gain_std_db, args.loss, None, None,
                use_amp, use_channels_last, args.edge_weight, args.edge_radius_pixels, args.max_val_batches,
            )
        scheduler.step()
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_mae_db": train_result["mae_db"],
            "train_rmse_db": train_result["rmse_db"],
            "val_loss": val_result["loss"],
            "val_mae_db": val_result["mae_db"],
            "val_rmse_db": val_result["rmse_db"],
            "val_edge_mae_db": val_result["edge_mae_db"],
            "val_edge_rmse_db": val_result["edge_rmse_db"],
            "val_non_edge_mae_db": val_result["non_edge_mae_db"],
            "val_non_edge_rmse_db": val_result["non_edge_rmse_db"],
            "seconds": elapsed,
        }
        append_history(history_path, row)
        improved = val_result["rmse_db"] < best_rmse
        edge_improved = val_result["edge_rmse_db"] < best_edge_rmse
        if improved:
            best_rmse = val_result["rmse_db"]
            bad_epochs = 0
        else:
            bad_epochs += 1
        if edge_improved:
            best_edge_rmse = val_result["edge_rmse_db"]
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, best_rmse, best_edge_rmse, args,
        )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, best_rmse, best_edge_rmse, args,
            )
        if edge_improved:
            save_checkpoint(
                args.output_dir / "best_edge.pt", model, optimizer, scheduler, scaler,
                epoch, best_rmse, best_edge_rmse, args,
            )
        print(
            f"epoch={epoch:03d} train_loss={train_result['loss']:.5f} "
            f"val_mae={val_result['mae_db']:.3f}dB val_rmse={val_result['rmse_db']:.3f}dB "
            f"edge_rmse={val_result['edge_rmse_db']:.3f}dB "
            f"best={best_rmse:.3f}dB best_edge={best_edge_rmse:.3f}dB time={elapsed:.1f}s",
            flush=True,
        )
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without validation improvement.", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

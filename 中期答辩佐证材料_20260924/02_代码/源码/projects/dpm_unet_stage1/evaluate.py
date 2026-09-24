"""Evaluate a trained checkpoint on validation or held-out test tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DPMTileDataset
from metrics import DBMetrics, masked_loss
from model import DPMUNet


def boundary_band(height_channel: torch.Tensor, radius_pixels: int) -> torch.Tensor:
    building = height_channel > 1e-6
    kernel = radius_pixels * 2 + 1
    dilated = F.max_pool2d(building.float(), kernel_size=kernel, stride=1, padding=radius_pixels) > 0
    eroded = -F.max_pool2d(-building.float(), kernel_size=kernel, stride=1, padding=radius_pixels) > 0.5
    return dilated & ~eroded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    add_edge_channel = bool(saved_args.get("building_edge_channel", False))
    input_mode = saved_args.get("input_mode", "height_tx_distance")
    base_input_channels = {"height": 1, "height_tx": 2, "height_tx_distance": 3}[input_mode]
    input_channels = base_input_channels + int(add_edge_channel)
    edge_radius = int(saved_args.get("edge_radius_pixels", 2))
    edge_weight = float(saved_args.get("edge_weight", 1.0))
    model = DPMUNet(in_channels=input_channels, base_channels=int(saved_args.get("base_channels", 32))).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = DPMTileDataset(
        args.data_root,
        args.split,
        augment=False,
        limit=args.limit,
        add_building_edge_channel=add_edge_channel,
        edge_channel_radius_pixels=int(saved_args.get("edge_channel_radius_pixels", 1)),
        input_mode=input_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    normalization = json.loads((args.data_root / "normalization.json").read_text(encoding="utf-8"))
    gain_std = float(normalization["statistics"]["path_gain"]["std_db"])
    amp = device.type == "cuda"
    total_loss, batches = 0.0, 0
    metrics = DBMetrics()
    edge_metrics = DBMetrics()
    non_edge_metrics = DBMetrics()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            if device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            edge = boundary_band(inputs[:, 0:1], edge_radius)
            pixel_weight = 1.0 + (edge_weight - 1.0) * edge.float()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(inputs)
                loss = masked_loss(prediction, target, mask, saved_args.get("loss", "huber"), pixel_weight)
            total_loss += float(loss.item())
            batches += 1
            metrics.update(prediction, target, mask, gain_std)
            edge_metrics.update(prediction, target, mask & edge, gain_std)
            non_edge_metrics.update(prediction, target, mask & ~edge, gain_std)
    edge_result = edge_metrics.compute()
    non_edge_result = non_edge_metrics.compute()
    result = {
        "split": args.split,
        "tiles": len(dataset),
        "loss": total_loss / batches,
        **metrics.compute(),
        "edge_radius_pixels": edge_radius,
        "edge_radius_m": edge_radius * 4,
        "edge_mae_db": edge_result["mae_db"],
        "edge_rmse_db": edge_result["rmse_db"],
        "non_edge_mae_db": non_edge_result["mae_db"],
        "non_edge_rmse_db": non_edge_result["rmse_db"],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render prediction, DPM target, and error for one tile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import building_edge_map
from model import DPMUNet


def colorize(values: np.ndarray, vmin: float, vmax: float, palette: np.ndarray) -> Image.Image:
    """Map a floating array to RGB without depending on a GUI plotting stack."""
    valid = np.isfinite(values)
    denominator = max(vmax - vmin, 1e-6)
    normalized = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / denominator, 0, 1)
    position = normalized * (len(palette) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(palette) - 1)
    fraction = (position - lower)[..., None]
    rgb = palette[lower] * (1 - fraction) + palette[upper] * fraction
    rgb = rgb.astype(np.uint8)
    rgb[~valid] = (45, 45, 45)
    return Image.fromarray(rgb, mode="RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
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
    model = DPMUNet(input_channels, int(saved_args.get("base_channels", 32))).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tile_dir = args.data_root / args.tile
    all_arrays = {
        "height": np.load(tile_dir / "building_height_norm.npy"),
        "tx": np.load(tile_dir / "tx_position_height_norm.npy"),
        "distance": np.load(tile_dir / "tx_distance_norm.npy"),
    }
    input_arrays = {
        "height": [all_arrays["height"]],
        "height_tx": [all_arrays["height"], all_arrays["tx"]],
        "height_tx_distance": [all_arrays["height"], all_arrays["tx"], all_arrays["distance"]],
    }[input_mode]
    if add_edge_channel:
        input_arrays.append(
            building_edge_map(input_arrays[0], int(saved_args.get("edge_channel_radius_pixels", 1)))
        )
    x = np.stack(input_arrays, axis=0).astype(np.float32)
    target_norm = np.load(tile_dir / "path_gain_norm.npy").astype(np.float32)
    mask = np.load(tile_dir / "path_gain_valid_mask.npy").astype(bool)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        inputs = torch.from_numpy(x[None]).to(device)
        if device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        prediction_norm = model(inputs).squeeze().float().cpu().numpy()
    normalization = json.loads((args.data_root / "normalization.json").read_text(encoding="utf-8"))
    stats = normalization["statistics"]["path_gain"]
    mean, std = float(stats["mean_db"]), float(stats["std_db"])
    prediction_db = prediction_norm * std + mean
    target_db = target_norm * std + mean
    target_db[~mask] = np.nan
    error_db = prediction_db - target_db

    signal_valid = target_db[np.isfinite(target_db)]
    signal_min, signal_max = np.percentile(signal_valid, (2, 98))
    error_limit = max(float(np.nanpercentile(np.abs(error_db), 98)), 1.0)
    sequential = np.array([(30, 20, 100), (30, 130, 210), (30, 200, 130), (245, 220, 60), (210, 45, 30)], dtype=np.float32)
    diverging = np.array([(25, 60, 180), (120, 190, 245), (245, 245, 245), (245, 155, 105), (180, 25, 35)], dtype=np.float32)
    panels = [
        colorize(x[0], 0.0, 1.0, sequential),
        colorize(target_db, float(signal_min), float(signal_max), sequential),
        colorize(prediction_db, float(signal_min), float(signal_max), sequential),
        colorize(error_db, -error_limit, error_limit, diverging),
    ]
    titles = [
        "Building height (0..1)",
        f"DPM target ({signal_min:.1f}..{signal_max:.1f} dB)",
        "U-Net prediction (same scale)",
        f"Error (+/-{error_limit:.1f} dB)",
    ]
    scale, title_height = 3, 30
    panel_size = 128 * scale
    canvas = Image.new("RGB", (panel_size * len(panels), panel_size + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        enlarged = panel.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST)
        x_offset = index * panel_size
        canvas.paste(enlarged, (x_offset, title_height))
        draw.text((x_offset + 8, 8), title, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

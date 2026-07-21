"""Shared inference utilities for the Stage-1 DPM surrogate model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import INPUT_MODES, building_edge_map
from model import DPMUNet


@dataclass
class LoadedModel:
    model: DPMUNet
    device: torch.device
    checkpoint_path: Path
    normalization_path: Path
    input_mode: str
    input_files: tuple[str, ...]
    add_edge_channel: bool
    edge_channel_radius_pixels: int
    base_channels: int
    path_gain_mean_db: float
    path_gain_std_db: float
    warmed_up: bool = False


def resolve_model_files(
    model_dir: Path | None,
    checkpoint: Path | None,
    normalization: Path | None,
) -> tuple[Path, Path]:
    if model_dir is not None:
        model_dir = model_dir.resolve()
        checkpoint = checkpoint or model_dir / "best.pt"
        normalization = normalization or model_dir / "normalization.json"
    if checkpoint is None or normalization is None:
        raise ValueError("Provide --model-dir, or both --checkpoint and --normalization")
    checkpoint = checkpoint.resolve()
    normalization = normalization.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not normalization.is_file():
        raise FileNotFoundError(f"Missing normalization file: {normalization}")
    return checkpoint, normalization


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_model(
    checkpoint_path: Path,
    normalization_path: Path,
    device_name: str = "auto",
) -> LoadedModel:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    input_mode = str(saved_args.get("input_mode", "height_tx_distance"))
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported checkpoint input_mode: {input_mode!r}")

    add_edge_channel = bool(saved_args.get("building_edge_channel", False))
    edge_radius = int(saved_args.get("edge_channel_radius_pixels", 1))
    base_channels = int(saved_args.get("base_channels", 32))
    input_files = tuple(INPUT_MODES[input_mode])
    input_channels = len(input_files) + int(add_edge_channel)

    model = DPMUNet(input_channels, base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)

    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    path_gain = normalization["statistics"]["path_gain"]
    mean_db = float(path_gain["mean_db"])
    std_db = float(path_gain["std_db"])
    if not np.isfinite(mean_db) or not np.isfinite(std_db) or std_db <= 0:
        raise ValueError("Invalid path-gain mean/std in normalization.json")

    return LoadedModel(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        input_mode=input_mode,
        input_files=input_files,
        add_edge_channel=add_edge_channel,
        edge_channel_radius_pixels=edge_radius,
        base_channels=base_channels,
        path_gain_mean_db=mean_db,
        path_gain_std_db=std_db,
    )


def load_input_arrays(tile_dir: Path, loaded: LoadedModel) -> tuple[np.ndarray, list[dict[str, Any]]]:
    channels: list[np.ndarray] = []
    channel_stats: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None

    for filename in loaded.input_files:
        path = tile_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing inference input: {path}")
        array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D array in {path}, got shape {array.shape}")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(f"Input shape mismatch in {path}: {array.shape} != {expected_shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Input contains NaN or infinity: {path}")

        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < -1e-4 or maximum > 1.0001:
            raise ValueError(
                f"{path} is outside the expected normalized range [0, 1] "
                f"(observed {minimum:.6g}..{maximum:.6g}); do not pass raw arrays"
            )
        channels.append(array)
        channel_stats.append(
            {"file": filename, "min": minimum, "max": maximum, "mean": float(array.mean())}
        )

    if loaded.add_edge_channel:
        channels.append(building_edge_map(channels[0], loaded.edge_channel_radius_pixels))
    return np.stack(channels, axis=0), channel_stats


def load_optional_mask(mask_path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None:
        return None
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing output mask: {mask_path}")
    mask = np.load(mask_path, allow_pickle=False).astype(bool, copy=False)
    if mask.shape != shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} != {shape}")
    return mask


def predict_tile(
    tile_dir: Path,
    loaded: LoadedModel,
    mask_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tile_dir = tile_dir.resolve()
    if not tile_dir.is_dir():
        raise NotADirectoryError(f"Missing tile directory: {tile_dir}")

    inputs, channel_stats = load_input_arrays(tile_dir, loaded)
    mask = load_optional_mask(mask_path, inputs.shape[-2:])
    tensor = torch.from_numpy(inputs[None]).to(loaded.device)
    if loaded.device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
        torch.cuda.synchronize(loaded.device)

    def forward() -> np.ndarray:
        with torch.inference_mode(), torch.autocast(
            device_type=loaded.device.type,
            dtype=torch.float16,
            enabled=loaded.device.type == "cuda",
        ):
            return loaded.model(tensor).squeeze(0).squeeze(0).float().cpu().numpy()

    warmup_ms = 0.0
    if not loaded.warmed_up:
        warmup_started = time.perf_counter()
        forward()
        if loaded.device.type == "cuda":
            torch.cuda.synchronize(loaded.device)
        warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
        loaded.warmed_up = True

    started = time.perf_counter()
    prediction_norm = forward()
    if loaded.device.type == "cuda":
        torch.cuda.synchronize(loaded.device)
    inference_ms = (time.perf_counter() - started) * 1000.0

    prediction_db = (
        prediction_norm * loaded.path_gain_std_db + loaded.path_gain_mean_db
    ).astype(np.float32, copy=False)
    prediction_norm = prediction_norm.astype(np.float32, copy=False)
    if mask is not None:
        prediction_norm = prediction_norm.copy()
        prediction_db = prediction_db.copy()
        prediction_norm[~mask] = np.nan
        prediction_db[~mask] = np.nan

    finite_db = prediction_db[np.isfinite(prediction_db)]
    if finite_db.size == 0:
        raise ValueError("The optional mask removed every predicted pixel")
    metadata: dict[str, Any] = {
        "tile_dir": str(tile_dir),
        "checkpoint": str(loaded.checkpoint_path),
        "normalization": str(loaded.normalization_path),
        "device": str(loaded.device),
        "input_mode": loaded.input_mode,
        "input_files": list(loaded.input_files),
        "input_channels": channel_stats,
        "array_shape": list(prediction_db.shape),
        "mask": str(mask_path.resolve()) if mask_path is not None else None,
        "valid_output_pixels": int(finite_db.size),
        "path_gain_mean_db": loaded.path_gain_mean_db,
        "path_gain_std_db": loaded.path_gain_std_db,
        "prediction_db": {
            "min": float(finite_db.min()),
            "max": float(finite_db.max()),
            "mean": float(finite_db.mean()),
            "p02": float(np.percentile(finite_db, 2)),
            "p98": float(np.percentile(finite_db, 98)),
        },
        "warmup_ms": warmup_ms,
        "inference_ms": inference_ms,
    }
    source_metadata = tile_dir / "metadata.json"
    if source_metadata.is_file():
        try:
            metadata["source_metadata"] = json.loads(source_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata["source_metadata_warning"] = f"Could not parse {source_metadata}"
    return prediction_norm, prediction_db, metadata


def _colorize(values: np.ndarray, vmin: float, vmax: float, palette: np.ndarray) -> Image.Image:
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


def render_prediction(
    prediction_db: np.ndarray,
    output_path: Path,
    palette_name: str = "color",
    scale: int = 4,
) -> tuple[float, float]:
    finite = prediction_db[np.isfinite(prediction_db)]
    vmin, vmax = (float(value) for value in np.percentile(finite, (2, 98)))
    if palette_name == "gray":
        palette = np.array([(0, 0, 0), (255, 255, 255)], dtype=np.float32)
    elif palette_name == "color":
        palette = np.array(
            [(30, 20, 100), (30, 130, 210), (30, 200, 130), (245, 220, 60), (210, 45, 30)],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown palette: {palette_name}")

    image = _colorize(prediction_db, vmin, vmax, palette)
    height, width = prediction_db.shape
    title_height = 32
    canvas = Image.new("RGB", (width * scale, height * scale + title_height), "white")
    canvas.paste(
        image.resize((width * scale, height * scale), resample=Image.Resampling.NEAREST),
        (0, title_height),
    )
    ImageDraw.Draw(canvas).text(
        (8, 9),
        f"Predicted DPM path gain ({vmin:.1f}..{vmax:.1f} dB)",
        fill="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return vmin, vmax


def render_prediction_with_buildings(
    prediction_db: np.ndarray,
    tile_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Render a publication-style grayscale map with black buildings and TX."""
    height = np.load(tile_dir / "building_height_norm.npy", allow_pickle=False)
    tx_map = np.load(tile_dir / "tx_position_height_norm.npy", allow_pickle=False)
    if height.shape != prediction_db.shape or tx_map.shape != prediction_db.shape:
        raise ValueError("Building/TX arrays do not match the prediction shape")

    rows, cols = prediction_db.shape
    resolution_m = 4.0
    metadata_path = tile_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolution_m = float(source_metadata.get("grid", {}).get("resolution", resolution_m))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    extent = (0.0, cols * resolution_m, 0.0, rows * resolution_m)

    finite = prediction_db[np.isfinite(prediction_db)]
    vmin, vmax = (float(value) for value in np.percentile(finite, (2, 98)))
    building = height > 0

    gray_palette = np.array([(0, 0, 0), (255, 255, 255)], dtype=np.float32)
    map_rgb = np.asarray(_colorize(prediction_db, vmin, vmax, gray_palette)).copy()
    map_rgb[building] = (0, 0, 0)

    scale = 4
    map_width, map_height = cols * scale, rows * scale
    left, top, right, bottom = 72, 48, 118, 64
    canvas = Image.new(
        "RGB",
        (left + map_width + right, top + map_height + bottom),
        "white",
    )
    map_image = Image.fromarray(map_rgb, mode="RGB").resize(
        (map_width, map_height), resample=Image.Resampling.NEAREST
    )
    canvas.paste(map_image, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (left, top, left + map_width - 1, top + map_height - 1),
        outline="black",
        width=2,
    )

    draw.text((left + map_width // 2 - 130, 15), "Predicted DPM path gain", fill="black")
    x_max_m, y_max_m = extent[1], extent[3]
    tick_count = 4
    for index in range(tick_count + 1):
        fraction = index / tick_count
        x = int(round(left + fraction * map_width))
        y = int(round(top + map_height - fraction * map_height))
        x_value = fraction * x_max_m
        y_value = fraction * y_max_m
        draw.line((x, top + map_height, x, top + map_height + 6), fill="black", width=1)
        draw.text((x - 12, top + map_height + 9), f"{x_value:.0f}", fill="black")
        draw.line((left - 6, y, left, y), fill="black", width=1)
        draw.text((left - 42, y - 6), f"{y_value:.0f}", fill="black")
    draw.text((left + map_width // 2 - 15, top + map_height + 38), "x (m)", fill="black")

    y_label = Image.new("RGB", (45, 18), "white")
    ImageDraw.Draw(y_label).text((0, 2), "y (m)", fill="black")
    y_label = y_label.rotate(90, expand=True, fillcolor="white")
    canvas.paste(y_label, (10, top + map_height // 2 - y_label.height // 2))

    colorbar_x = left + map_width + 28
    colorbar_width = 22
    gradient = np.linspace(255, 0, map_height, dtype=np.uint8)[:, None]
    gradient_rgb = np.repeat(gradient[:, :, None], 3, axis=2)
    gradient_rgb = np.repeat(gradient_rgb, colorbar_width, axis=1)
    canvas.paste(Image.fromarray(gradient_rgb, mode="RGB"), (colorbar_x, top))
    draw.rectangle(
        (colorbar_x, top, colorbar_x + colorbar_width, top + map_height),
        outline="black",
        width=1,
    )
    draw.text((colorbar_x - 8, top - 20), "dB", fill="black")
    for index in range(tick_count + 1):
        fraction = index / tick_count
        y = int(round(top + fraction * map_height))
        value = vmax - fraction * (vmax - vmin)
        draw.line((colorbar_x + colorbar_width, y, colorbar_x + colorbar_width + 5, y), fill="black")
        draw.text((colorbar_x + colorbar_width + 8, y - 6), f"{value:.1f}", fill="black")

    legend_y = top + map_height + 38
    draw.rectangle((left + map_width - 115, legend_y, left + map_width - 99, legend_y + 12), fill="black")
    draw.text((left + map_width - 94, legend_y), "Buildings", fill="black")

    tx_location: dict[str, float | int] | None = None
    if np.any(tx_map > 0):
        tx_row, tx_col = np.unravel_index(int(np.argmax(tx_map)), tx_map.shape)
        tx_x_m = (float(tx_col) + 0.5) * resolution_m
        tx_y_m = (float(rows - tx_row) - 0.5) * resolution_m
        centre_x = left + (float(tx_col) + 0.5) * scale
        centre_y = top + (float(tx_row) + 0.5) * scale
        outer_radius, inner_radius = 10.0, 4.5
        star = []
        for point in range(10):
            angle = -np.pi / 2 + point * np.pi / 5
            radius = outer_radius if point % 2 == 0 else inner_radius
            star.append((centre_x + radius * np.cos(angle), centre_y + radius * np.sin(angle)))
        draw.polygon(star, fill="white", outline="black")
        draw.text((centre_x + 11, centre_y - 7), "TX", fill="white", stroke_width=2, stroke_fill="black")
        tx_location = {
            "row": int(tx_row),
            "col": int(tx_col),
            "local_x_m": tx_x_m,
            "local_y_m": tx_y_m,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "file": str(output_path),
        "palette": "gray",
        "renderer": "Pillow",
        "building_style": "black fill",
        "building_pixels": int(building.sum()),
        "resolution_m": resolution_m,
        "p02_vmin_db": vmin,
        "p98_vmax_db": vmax,
        "tx": tx_location,
    }


def save_outputs(
    output_dir: Path,
    prediction_norm: np.ndarray,
    prediction_db: np.ndarray,
    metadata: dict[str, Any],
    save_png: bool,
    palette_name: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_path = output_dir / "path_gain_pred_norm.npy"
    db_path = output_dir / "path_gain_pred_db.npy"
    metadata_path = output_dir / "inference_metadata.json"
    np.save(norm_path, prediction_norm.astype(np.float32, copy=False), allow_pickle=False)
    np.save(db_path, prediction_db.astype(np.float32, copy=False), allow_pickle=False)

    outputs = {"normalized": norm_path, "db": db_path, "metadata": metadata_path}
    if save_png:
        png_path = output_dir / "path_gain_prediction.png"
        vmin, vmax = render_prediction(prediction_db, png_path, palette_name)
        building_png_path = output_dir / "path_gain_prediction_with_buildings.png"
        building_visualization = render_prediction_with_buildings(
            prediction_db,
            Path(metadata["tile_dir"]),
            building_png_path,
        )
        metadata["visualization"] = {
            "file": str(png_path),
            "palette": palette_name,
            "p02_vmin_db": vmin,
            "p98_vmax_db": vmax,
            "with_buildings": building_visualization,
        }
        outputs["png"] = png_path
        outputs["png_with_buildings"] = building_png_path
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs

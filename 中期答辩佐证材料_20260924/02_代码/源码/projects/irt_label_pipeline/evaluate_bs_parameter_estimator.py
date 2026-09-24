#!/usr/bin/env python3
"""Evaluate a BS estimator with validation-selected heatmap coordinate readout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from bs_inversion_dataset import BSInversionPilotDataset, load_json
from bs_parameter_model import BSParameterEstimator
from run_sionna_dataset import atomic_write_json, sha256_file
from train_bs_parameter_estimator import circular_error_degrees, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the config output directory for side-by-side evaluations.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def heatmap_argmax(logits: torch.Tensor) -> torch.Tensor:
    batch, _, rows, cols = logits.shape
    indices = torch.argmax(logits.flatten(1), dim=1)
    row = torch.div(indices, cols, rounding_mode="floor").float()
    col = torch.remainder(indices, cols).float()
    return torch.stack((row, col), dim=1)


def heatmap_local_centroid(
    logits: torch.Tensor,
    radius: int = 4,
    temperature: float = 1.0,
) -> torch.Tensor:
    peaks = heatmap_argmax(logits)
    coordinates: list[torch.Tensor] = []
    _, _, rows, cols = logits.shape
    for sample in range(logits.shape[0]):
        peak_row = int(peaks[sample, 0].item())
        peak_col = int(peaks[sample, 1].item())
        row_min = max(0, peak_row - radius)
        row_max = min(rows, peak_row + radius + 1)
        col_min = max(0, peak_col - radius)
        col_max = min(cols, peak_col + radius + 1)
        patch = logits[sample, 0, row_min:row_max, col_min:col_max].float()
        probability = torch.softmax(patch.flatten() / temperature, dim=0)
        row_grid, col_grid = torch.meshgrid(
            torch.arange(row_min, row_max, device=logits.device, dtype=torch.float32),
            torch.arange(col_min, col_max, device=logits.device, dtype=torch.float32),
            indexing="ij",
        )
        coordinates.append(
            torch.stack(
                (
                    torch.sum(probability * row_grid.flatten()),
                    torch.sum(probability * col_grid.flatten()),
                )
            )
        )
    return torch.stack(coordinates, dim=0)


def distribution(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p90": float(np.percentile(array, 90.0)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(array)))),
        f"{prefix}_maximum": float(np.max(array)),
    }


def evaluate_split(
    model: BSParameterEstimator,
    dataset: BSInversionPilotDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp_enabled: bool,
    channels_last: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    locations: dict[str, list[float]] = {
        "global_softargmax": [],
        "peak_argmax": [],
        "local_centroid": [],
    }
    power_errors: list[float] = []
    direction_errors: list[float] = []
    map_squared = 0.0
    map_absolute = 0.0
    map_pixels = 0
    details: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["model_input"].to(device, non_blocking=True)
            target_coord = batch["target_row_col_px"].to(device, non_blocking=True)
            target_power = batch["target_power_db"].to(device, non_blocking=True)
            target_direction = batch["target_direction_sin_cos"].to(
                device, non_blocking=True
            )
            target_map = batch["target_map_norm"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True).bool()
            if channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(inputs)
            coordinates = {
                "global_softargmax": outputs["row_col_px"].float(),
                "peak_argmax": heatmap_argmax(outputs["location_logits"]),
                "local_centroid": heatmap_local_centroid(
                    outputs["location_logits"], radius=4, temperature=1.0
                ),
            }
            batch_location_errors = {
                name: torch.linalg.vector_norm(value - target_coord, dim=1)
                for name, value in coordinates.items()
            }
            for name, values in batch_location_errors.items():
                locations[name].extend(values.detach().cpu().tolist())
            predicted_power = (
                outputs["power_norm"].float() * dataset.power_std_db
                + dataset.power_mean_db
            )
            power_error = torch.abs(predicted_power - target_power)
            direction_error = circular_error_degrees(
                outputs["direction_raw"], target_direction
            )
            power_errors.extend(power_error.detach().cpu().tolist())
            direction_errors.extend(direction_error.detach().cpu().tolist())
            map_error_db = (
                outputs["map_norm"].float() - target_map.float()
            ) * dataset.signal_std_db
            map_values = map_error_db[valid].double()
            map_squared += float(torch.sum(map_values.square()).item())
            map_absolute += float(torch.sum(torch.abs(map_values)).item())
            map_pixels += int(map_values.numel())
            for sample in range(inputs.shape[0]):
                row: dict[str, Any] = {
                    "site_id": batch["site_id"][sample],
                    "direction_index": int(batch["direction_index"][sample]),
                    "true_row_px": float(target_coord[sample, 0].item()),
                    "true_col_px": float(target_coord[sample, 1].item()),
                    "power_abs_error_db": float(power_error[sample].item()),
                    "direction_abs_error_deg": float(direction_error[sample].item()),
                }
                for name, value in coordinates.items():
                    row[f"{name}_row_px"] = float(value[sample, 0].item())
                    row[f"{name}_col_px"] = float(value[sample, 1].item())
                    row[f"{name}_error_px"] = float(
                        batch_location_errors[name][sample].item()
                    )
                details.append(row)
    metrics: dict[str, Any] = {
        "samples": len(dataset),
        "sites": len(dataset.entries),
        "map_valid_pixels": map_pixels,
        "map_rmse_db": math.sqrt(map_squared / map_pixels),
        "map_mae_db": map_absolute / map_pixels,
    }
    for name, values in locations.items():
        metrics.update(distribution(values, f"location_{name}_error_px"))
    metrics.update(distribution(power_errors, "power_abs_error_db"))
    metrics.update(distribution(direction_errors, "direction_abs_error_deg"))
    return metrics, details


def write_details(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config = load_json(args.config.resolve())
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or output_dir / "best.pt"
    device = resolve_device(str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    model = BSParameterEstimator(
        in_channels=3,
        base_channels=int(model_config["base_channels"]),
        scalar_hidden_channels=int(model_config["scalar_hidden_channels"]),
        softargmax_temperature=float(model_config["softargmax_temperature"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    datasets = {
        split: BSInversionPilotDataset(
            config["dataset_root"],
            split=split,
            seed=int(config["seed"]),
            augment=False,
        )
        for split in ("val", "test")
    }
    val_metrics, val_details = evaluate_split(
        model,
        datasets["val"],
        device,
        args.batch_size,
        args.workers,
        amp_enabled,
        channels_last,
    )
    methods = ("global_softargmax", "peak_argmax", "local_centroid")
    selected_method = min(
        methods,
        key=lambda name: float(val_metrics[f"location_{name}_error_px_mean"]),
    )
    test_metrics, test_details = evaluate_split(
        model,
        datasets["test"],
        device,
        args.batch_size,
        args.workers,
        amp_enabled,
        channels_last,
    )
    thresholds = config["acceptance"]
    acceptance = {
        "location_mean_le_threshold": float(
            test_metrics[f"location_{selected_method}_error_px_mean"]
        )
        <= float(thresholds["maximum_location_mean_error_px"]),
        "power_mae_le_threshold": float(test_metrics["power_abs_error_db_mean"])
        <= float(thresholds["maximum_power_mae_db"]),
        "direction_mae_le_threshold": float(
            test_metrics["direction_abs_error_deg_mean"]
        )
        <= float(thresholds["maximum_direction_mae_deg"]),
    }
    result = {
        "version": 1,
        "status": "accepted" if all(acceptance.values()) else "needs_iteration",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "coordinate_method_selection_split": "val",
        "selected_coordinate_method": selected_method,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "acceptance_checks": acceptance,
        "thresholds": thresholds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_dir / "coordinate_readout_metrics.json", result)
    write_details(output_dir / "val_parameter_predictions.csv", val_details)
    write_details(output_dir / "test_parameter_predictions.csv", test_details)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

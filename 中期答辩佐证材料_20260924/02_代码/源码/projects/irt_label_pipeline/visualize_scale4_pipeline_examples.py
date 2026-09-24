#!/usr/bin/env python3
"""Create an honest legacy-to-complete-V3 Scale-4 comparison cache."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bs_inversion_dataset import BSInversionPilotDataset, load_json
from bs_parameter_model import BSParameterEstimator
from evaluate_pilot_oracle_pipeline import make_stage1_input
from latent_bs_fusion import (
    build_latent_fusion_features,
    configured_feature_names,
    configured_stage2b_in_channels,
)
from run_sionna_dataset import atomic_write_json, sha256_file
from train_stage2b_directional_ss import (
    build_configured_unet,
    load_frozen_models,
    resolve_device,
    seed_everything,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-config",
        type=Path,
        default=SCRIPT_DIR / "config_stage2b_position_encoding_diagnostic_v2.json",
    )
    parser.add_argument(
        "--after-config",
        type=Path,
        default=SCRIPT_DIR
        / "config_stage2b_position_encoding_scale4_diagnostic_v3.json",
    )
    parser.add_argument(
        "--estimator-config",
        type=Path,
        default=SCRIPT_DIR / "config_bs_parameter_estimator_scale4_v2.json",
    )
    parser.add_argument(
        "--estimator-checkpoint",
        type=Path,
        default=Path(r"D:\桌面\dac\models\bs_parameter_estimator_scale4_v2\best.pt"),
    )
    parser.add_argument(
        "--before-estimator-checkpoint",
        type=Path,
        help="Optional estimator override used only for the before prediction.",
    )
    parser.add_argument(
        "--after-estimator-checkpoint",
        type=Path,
        help="Optional estimator override used only for the after prediction.",
    )
    parser.add_argument(
        "--before-stage2b-checkpoint",
        type=Path,
        default=Path(
            r"D:\桌面\dac\models\stage2b_bs_inversion_pilot_finetune_v2\best.pt"
        ),
        help="Legacy Stage2-B checkpoint used only for the before prediction.",
    )
    parser.add_argument(
        "--stage2b-checkpoint",
        type=Path,
        default=Path(
            r"D:\桌面\dac\models\stage2b_position_encoding_scale4_v3\best.pt"
        ),
        help="Complete Stage2-B V3 checkpoint used for the after prediction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\桌面\dac\experiments\scale4_visual_comparison_v3"),
    )
    parser.add_argument("--candidates-per-bin", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Save selected arrays without importing the Matplotlib renderer again.",
    )
    return parser.parse_args()


def center_bin(center_distance_m: float) -> str:
    if center_distance_m < 64.0:
        return "000-064m"
    if center_distance_m < 128.0:
        return "064-128m"
    if center_distance_m < 192.0:
        return "128-192m"
    return "192m-plus"


def load_estimator(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    channels_last: bool,
) -> tuple[BSParameterEstimator, float, float, float, float]:
    config = load_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]["model"]
    model = BSParameterEstimator(
        in_channels=3,
        base_channels=int(model_config["base_channels"]),
        scalar_hidden_channels=int(model_config["scalar_hidden_channels"]),
        softargmax_temperature=float(model_config["softargmax_temperature"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    stats = load_json(Path(config["dataset_root"]) / "normalization.json")[
        "directional_signal_strength_db"
    ]
    manifest = load_json(Path(config["dataset_root"]) / "run_manifest.json")
    data_config = load_json(Path(manifest["config"]))
    power_config = data_config["effective_power"]
    return (
        model,
        float(stats["mean"]),
        float(stats["std"]),
        float(power_config["normalization_mean_db"]),
        float(power_config["normalization_std_db"]),
    )


def load_stage2b(
    checkpoint_path: Path,
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    model = build_configured_unet(
        checkpoint_config,
        in_channels=configured_stage2b_in_channels(checkpoint_config),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return model, checkpoint_config


def load_pair(
    config_path: Path,
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any], dict[str, float]]:
    config = load_json(config_path)
    stage1, stage2a, metadata = load_frozen_models(config, device, channels_last)
    tx_stats = load_json(
        Path(config["stage1_model_dir"]) / "tx_feature_normalization.json"
    )
    tx_values = {
        "height_cap_m": float(tx_stats["transmitter_height"]["cap_m"]),
        "distance_max_m": float(tx_stats["distance"]["max_m"]),
        "long_distance_max_m": float(
            tx_stats.get("long_distance", {}).get(
                "max_m", math.sqrt(2.0) * 512.0
            )
        ),
    }
    return stage1, stage2a, metadata, tx_values


def predict_pair(
    batch: dict[str, Any],
    *,
    stage1: torch.nn.Module,
    stage2a: torch.nn.Module,
    pair_metadata: dict[str, Any],
    tx_values: dict[str, float],
    stage2b: torch.nn.Module,
    stage2b_config: dict[str, Any],
    estimator: BSParameterEstimator,
    estimator_mean_db: float,
    estimator_std_db: float,
    estimator_power_mean_db: float,
    estimator_power_std_db: float,
    iso_mean_db: float,
    iso_std_db: float,
    ss_mean_db: float,
    ss_std_db: float,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    building = batch["building"].to(device, non_blocking=True)
    sparse_norm = batch["sparse_ss_norm"].to(device, non_blocking=True)
    sparse_mask = batch["sparse_mask"].to(device, non_blocking=True)
    tx_height_m = batch["tx_height_m"].to(device, non_blocking=True)
    sparse_db = sparse_norm.float() * ss_std_db + ss_mean_db
    estimator_sparse_norm = torch.where(
        sparse_mask.bool(),
        (sparse_db - estimator_mean_db) / estimator_std_db,
        0.0,
    )
    estimator_input = torch.cat(
        (building, estimator_sparse_norm, sparse_mask), dim=1
    )
    if channels_last:
        estimator_input = estimator_input.contiguous(
            memory_format=torch.channels_last
        )
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        estimated = estimator(estimator_input)
    estimated_row_col = estimated["row_col_px"].float()
    tx_xy_m = torch.stack(
        (
            (estimated_row_col[:, 1] + 0.5) * 4.0,
            512.0 - (estimated_row_col[:, 0] + 0.5) * 4.0,
        ),
        dim=1,
    )
    stage1_input = make_stage1_input(
        building,
        tx_xy_m,
        tx_height_m,
        tx_values["height_cap_m"],
        tx_values["distance_max_m"],
        in_channels=int(pair_metadata["stage1_in_channels"]),
        long_distance_max_m=tx_values["long_distance_max_m"],
    )
    if channels_last:
        building = building.contiguous(memory_format=torch.channels_last)
        stage1_input = stage1_input.contiguous(memory_format=torch.channels_last)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        dpm_norm = stage1(stage1_input)
        residual_norm = stage2a(torch.cat((building, dpm_norm), dim=1))
    dpm_db = (
        dpm_norm.float() * float(pair_metadata["stage1_std_db"])
        + float(pair_metadata["stage1_mean_db"])
    )
    corrected_iso_db = (
        dpm_db
        + float(pair_metadata["stage2a_residual_center_db"])
        + residual_norm.float()
        * float(pair_metadata["stage2a_residual_scale_db"])
    )
    coarse_norm = (corrected_iso_db - iso_mean_db) / iso_std_db
    latent_names = configured_feature_names(stage2b_config)
    if latent_names:
        latent_features = build_latent_fusion_features(
            estimated,
            stage2b_config["latent_fusion"],
            estimator_signal_mean_db=estimator_mean_db,
            estimator_signal_std_db=estimator_std_db,
            stage2b_signal_mean_db=ss_mean_db,
            stage2b_signal_std_db=ss_std_db,
            estimator_power_mean_db=estimator_power_mean_db,
            estimator_power_std_db=estimator_power_std_db,
            rows=int(building.shape[-2]),
            cols=int(building.shape[-1]),
        )
    else:
        latent_features = building.new_empty(
            (building.shape[0], 0, building.shape[-2], building.shape[-1])
        )
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        stage2b_input = torch.cat(
            (building, coarse_norm, sparse_norm, sparse_mask, latent_features), dim=1
        )
        if channels_last:
            stage2b_input = stage2b_input.contiguous(
                memory_format=torch.channels_last
            )
        prediction_norm = stage2b(stage2b_input)
    prediction_db = prediction_norm.float() * ss_std_db + ss_mean_db
    return prediction_db, estimated_row_col


def draw_examples(
    rows: list[dict[str, Any]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal_values = np.concatenate(
        [
            row[key][row["valid"]]
            for row in rows
            for key in ("target", "before", "after")
        ]
    )
    signal_vmin, signal_vmax = np.percentile(signal_values, (1.0, 99.0))
    error_values = np.concatenate(
        [np.abs(row["after"] - row["target"])[row["valid"]] for row in rows]
    )
    error_vmax = max(8.0, float(np.percentile(error_values, 95.0)))
    # Matplotlib's constrained-layout engine is unstable on this Windows
    # runtime for a large grid with shared colorbars. Use explicit spacing.
    figure, axes = plt.subplots(
        len(rows), 5, figsize=(22.0, 4.1 * len(rows)), constrained_layout=False
    )
    for row_index, row in enumerate(rows):
        axis = axes[row_index, 0]
        axis.imshow(row["building"], cmap="Greys", vmin=0.0, vmax=1.0)
        sparse_rows, sparse_cols = np.nonzero(row["sparse_mask"])
        axis.scatter(
            sparse_cols,
            sparse_rows,
            c=row["sparse_db"][row["sparse_mask"]],
            s=9,
            cmap="turbo",
            vmin=signal_vmin,
            vmax=signal_vmax,
            linewidths=0.0,
        )
        true_row, true_col = row["true_row_col"]
        estimated_row, estimated_col = row["estimated_row_col"]
        axis.scatter(
            [true_col], [true_row], marker="*", s=180, c="red", edgecolors="white",
            linewidths=0.8, label="True BS"
        )
        axis.scatter(
            [estimated_col], [estimated_row], marker="x", s=100, c="cyan",
            linewidths=2.0, label="Estimated BS"
        )
        axis.set_title("Input: buildings + 100 sparse points")
        axis.set_ylabel(
            f"{row['bin']}\n{row['site_id']} / az {row['azimuth_deg']:.0f} deg\n"
            f"BS error {row['location_error_px']:.2f} px"
        )
        if row_index == 0:
            axis.legend(loc="lower left", fontsize=8)

        for column, key, title in (
            (1, "target", "Ground truth"),
            (2, "before", f"Legacy pipeline / RMSE {row['before_rmse_db']:.2f} dB"),
            (3, "after", f"Complete V3 / RMSE {row['after_rmse_db']:.2f} dB"),
        ):
            masked = np.ma.masked_where(~row["valid"], row[key])
            image = axes[row_index, column].imshow(
                masked, cmap="turbo", vmin=signal_vmin, vmax=signal_vmax
            )
            axes[row_index, column].set_title(title)
        absolute_error = np.ma.masked_where(
            ~row["valid"], np.abs(row["after"] - row["target"])
        )
        error_image = axes[row_index, 4].imshow(
            absolute_error, cmap="magma", vmin=0.0, vmax=error_vmax
        )
        axes[row_index, 4].set_title(
            f"Complete V3 absolute error / MAE {row['after_mae_db']:.2f} dB"
        )
        for column in range(5):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])

    figure.colorbar(
        image,
        ax=axes[:, 1:4].ravel().tolist(),
        label="Signal strength (dB)",
        shrink=0.82,
    )
    figure.colorbar(
        error_image,
        ax=axes[:, 4].ravel().tolist(),
        label="Absolute error (dB)",
        shrink=0.82,
    )
    figure.suptitle(
        "Scale-4 representative test samples: legacy pipeline vs complete V3 pipeline",
        fontsize=16,
    )
    figure.subplots_adjust(
        left=0.055, right=0.93, bottom=0.035, top=0.94, wspace=0.14, hspace=0.26
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_training_curve(history_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.genfromtxt(history_path, delimiter=",", names=True)
    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    axis.plot(rows["epoch"], rows["train_rmse_db"], marker="o", label="Train")
    axis.plot(rows["epoch"], rows["val_rmse_db"], marker="o", label="Validation")
    axis.axhline(7.53716491575115, color="gray", linestyle="--", label="Before: 7.537 dB")
    axis.axhline(7.33716491575115, color="green", linestyle=":", label="Stage2B gate: 7.337 dB")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Isotropic map RMSE (dB)")
    axis.set_title("Stage1 + Stage2A Scale-4 fine-tuning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    after_config = load_json(args.after_config.resolve())
    seed_everything(int(after_config["seed"]))
    device = resolve_device(str(after_config["device"]))
    amp_enabled = bool(after_config["amp"]) and device.type == "cuda"
    channels_last = bool(after_config["channels_last"]) and device.type == "cuda"
    irt_normalization = load_json(after_config["irt_normalization"])
    iso_stats = irt_normalization["statistics"]["train"]["p_iso"]
    ss_stats = irt_normalization["stage2b_ss_normalization"]
    iso_mean_db, iso_std_db = float(iso_stats["mean_db"]), float(iso_stats["std_db"])
    ss_mean_db, ss_std_db = float(ss_stats["mean_db"]), float(ss_stats["std_db"])
    dataset = BSInversionPilotDataset(
        after_config["dataset_root"],
        split="test",
        seed=int(after_config["seed"]),
        augment=False,
        signal_mean_db=ss_mean_db,
        signal_std_db=ss_std_db,
    )
    wanted: dict[str, list[int]] = {
        "000-064m": [],
        "064-128m": [],
        "128-192m": [],
        "192m-plus": [],
    }
    for index, (entry, direction) in enumerate(dataset.rows):
        if direction != 0:
            continue
        x_m, y_m = map(float, entry["tx_xy_m"])
        label = center_bin(math.hypot(x_m - 256.0, y_m - 256.0))
        if len(wanted[label]) < args.candidates_per_bin:
            wanted[label].append(index)
    candidate_indices = [index for label in wanted for index in wanted[label]]
    if any(len(indices) < args.candidates_per_bin for indices in wanted.values()):
        raise ValueError("not enough candidates in every radial bin")
    loader = DataLoader(
        Subset(dataset, candidate_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    before_estimator_path = (
        args.before_estimator_checkpoint.resolve()
        if args.before_estimator_checkpoint is not None
        else args.estimator_checkpoint.resolve()
    )
    after_estimator_path = (
        args.after_estimator_checkpoint.resolve()
        if args.after_estimator_checkpoint is not None
        else args.estimator_checkpoint.resolve()
    )
    (
        before_estimator,
        before_estimator_mean_db,
        before_estimator_std_db,
        before_estimator_power_mean_db,
        before_estimator_power_std_db,
    ) = load_estimator(
        args.estimator_config.resolve(),
        before_estimator_path,
        device,
        channels_last,
    )
    if after_estimator_path == before_estimator_path:
        after_estimator = before_estimator
        after_estimator_mean_db = before_estimator_mean_db
        after_estimator_std_db = before_estimator_std_db
        after_estimator_power_mean_db = before_estimator_power_mean_db
        after_estimator_power_std_db = before_estimator_power_std_db
    else:
        (
            after_estimator,
            after_estimator_mean_db,
            after_estimator_std_db,
            after_estimator_power_mean_db,
            after_estimator_power_std_db,
        ) = load_estimator(
            args.estimator_config.resolve(),
            after_estimator_path,
            device,
            channels_last,
        )
    before = load_pair(args.before_config.resolve(), device, channels_last)
    after = load_pair(args.after_config.resolve(), device, channels_last)
    after_stage2b_path = args.stage2b_checkpoint.resolve()
    before_stage2b_path = (
        args.before_stage2b_checkpoint.resolve()
        if args.before_stage2b_checkpoint is not None
        else after_stage2b_path
    )
    before_stage2b, before_stage2b_config = load_stage2b(
        before_stage2b_path, device, channels_last
    )
    after_stage2b, after_stage2b_config = load_stage2b(
        after_stage2b_path, device, channels_last
    )

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            common = {
                "batch": batch,
                "iso_mean_db": iso_mean_db,
                "iso_std_db": iso_std_db,
                "ss_mean_db": ss_mean_db,
                "ss_std_db": ss_std_db,
                "device": device,
                "amp_enabled": amp_enabled,
                "channels_last": channels_last,
            }
            before_prediction, estimated_row_col = predict_pair(
                stage1=before[0],
                stage2a=before[1],
                pair_metadata=before[2],
                tx_values=before[3],
                stage2b=before_stage2b,
                stage2b_config=before_stage2b_config,
                estimator=before_estimator,
                estimator_mean_db=before_estimator_mean_db,
                estimator_std_db=before_estimator_std_db,
                estimator_power_mean_db=before_estimator_power_mean_db,
                estimator_power_std_db=before_estimator_power_std_db,
                **common,
            )
            after_prediction, _ = predict_pair(
                stage1=after[0],
                stage2a=after[1],
                pair_metadata=after[2],
                tx_values=after[3],
                stage2b=after_stage2b,
                stage2b_config=after_stage2b_config,
                estimator=after_estimator,
                estimator_mean_db=after_estimator_mean_db,
                estimator_std_db=after_estimator_std_db,
                estimator_power_mean_db=after_estimator_power_mean_db,
                estimator_power_std_db=after_estimator_power_std_db,
                **common,
            )
            target = batch["target_map_db"].numpy()[:, 0]
            valid = batch["valid_mask"].numpy()[:, 0].astype(bool)
            before_np = before_prediction.detach().cpu().numpy()[:, 0]
            after_np = after_prediction.detach().cpu().numpy()[:, 0]
            estimated_np = estimated_row_col.detach().cpu().numpy()
            true_np = batch["target_row_col_px"].numpy()
            building_np = batch["building"].numpy()[:, 0]
            sparse_mask_np = batch["sparse_mask"].numpy()[:, 0].astype(bool)
            sparse_db_np = (
                batch["sparse_ss_norm"].numpy()[:, 0] * ss_std_db + ss_mean_db
            )
            tx_xy_np = batch["tx_xy_m"].numpy()
            for sample_index in range(target.shape[0]):
                mask = valid[sample_index]
                before_error = before_np[sample_index][mask] - target[sample_index][mask]
                after_error = after_np[sample_index][mask] - target[sample_index][mask]
                x_m, y_m = tx_xy_np[sample_index]
                records.append(
                    {
                        "bin": center_bin(math.hypot(float(x_m) - 256.0, float(y_m) - 256.0)),
                        "site_id": batch["site_id"][sample_index],
                        "azimuth_deg": float(batch["target_azimuth_deg"][sample_index]),
                        "building": building_np[sample_index],
                        "sparse_mask": sparse_mask_np[sample_index],
                        "sparse_db": sparse_db_np[sample_index],
                        "target": target[sample_index],
                        "valid": mask,
                        "before": before_np[sample_index],
                        "after": after_np[sample_index],
                        "true_row_col": true_np[sample_index],
                        "estimated_row_col": estimated_np[sample_index],
                        "location_error_px": float(
                            np.linalg.norm(estimated_np[sample_index] - true_np[sample_index])
                        ),
                        "before_rmse_db": float(np.sqrt(np.mean(before_error**2))),
                        "after_rmse_db": float(np.sqrt(np.mean(after_error**2))),
                        "after_mae_db": float(np.mean(np.abs(after_error))),
                    }
                )

    selected: list[dict[str, Any]] = []
    for label in wanted:
        candidates = [row for row in records if row["bin"] == label]
        median = float(np.median([row["after_rmse_db"] for row in candidates]))
        selected.append(min(candidates, key=lambda row: abs(row["after_rmse_db"] - median)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "representative_before_after.png"
    curve_path = args.output_dir / "training_curve.png"
    arrays_path = args.output_dir / "selected_samples.npz"
    arrays: dict[str, np.ndarray] = {}
    for index, row in enumerate(selected):
        for key in (
            "building",
            "sparse_mask",
            "sparse_db",
            "target",
            "valid",
            "before",
            "after",
            "true_row_col",
            "estimated_row_col",
        ):
            arrays[f"sample_{index}_{key}"] = np.asarray(row[key])
    np.savez_compressed(arrays_path, **arrays)
    history_path = Path(after_config["stage2a_model_dir"]) / "history.csv"
    if not args.skip_render:
        draw_examples(selected, comparison_path)
        draw_training_curve(history_path, curve_path)
    summary = {
        "version": 2,
        "comparison_scope": "legacy pipeline vs complete V3 pipeline (Stage1 + Stage2A + Stage2B)",
        "cache_semantics": "sample_i_before is the legacy prediction; sample_i_after is the complete Stage2B V3 prediction",
        "selection": "direction-0 sample nearest the median complete-V3 RMSE among the first 64 test sites in each radial bin",
        "before_stage1_checkpoint": str(Path(load_json(args.before_config)["stage1_model_dir"]) / "best.pt"),
        "after_stage1_checkpoint": str(Path(after_config["stage1_model_dir"]) / "best.pt"),
        "after_stage1_sha256": sha256_file(Path(after_config["stage1_model_dir"]) / "best.pt"),
        "after_stage2a_sha256": sha256_file(Path(after_config["stage2a_model_dir"]) / "best.pt"),
        "before_stage2b_checkpoint": str(before_stage2b_path),
        "before_stage2b_sha256": sha256_file(before_stage2b_path),
        "before_stage2b_input_channels": configured_stage2b_in_channels(
            before_stage2b_config
        ),
        "after_stage2b_checkpoint": str(after_stage2b_path),
        "after_stage2b_sha256": sha256_file(after_stage2b_path),
        "after_stage2b_input_channels": configured_stage2b_in_channels(
            after_stage2b_config
        ),
        "after_stage2b_latent_features": list(
            configured_feature_names(after_stage2b_config)
        ),
        "before_estimator_checkpoint": str(before_estimator_path),
        "after_estimator_checkpoint": str(after_estimator_path),
        "comparison_png": str(comparison_path),
        "training_curve_png": str(curve_path),
        "selected_arrays": str(arrays_path),
        "samples": [
            {
                key: row[key]
                for key in (
                    "bin",
                    "site_id",
                    "azimuth_deg",
                    "location_error_px",
                    "before_rmse_db",
                    "after_rmse_db",
                    "after_mae_db",
                )
            }
            for row in selected
        ],
    }
    atomic_write_json(args.output_dir / "visualization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-infer and visualize one fixed held-out sample with formal Scale-4 V3."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_stage2b_position_encoding_scale4_v3.json"
DEFAULT_ESTIMATOR_CONFIG = SCRIPT_DIR / "config_bs_parameter_estimator_scale4_v2.json"
DEFAULT_REPORT = Path(
    r"D:\桌面\dac\03_runs\experiments\bs_inversion_scale4_stage2b_v3_test_bins.json"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\桌面\dac\03_runs\experiments\single_test_v3_evaluation"
)
DEFAULT_SITE_ID = "tile_000222_site03"
DEFAULT_DIRECTION = 0
OUTPUT_NAMES = (
    "single_test_v3_evaluation.png",
    "single_test_v3_metrics.json",
    "single_test_v3_arrays.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--estimator-config", type=Path, default=DEFAULT_ESTIMATOR_CONFIG
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--site-id", default=DEFAULT_SITE_ID)
    parser.add_argument("--direction", type=int, default=DEFAULT_DIRECTION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sample_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    if target.shape != prediction.shape or target.shape != valid.shape:
        raise ValueError("target, prediction, and valid mask must have identical shapes")
    target_values = target[valid].astype(np.float64)
    prediction_values = prediction[valid].astype(np.float64)
    if target_values.size == 0:
        raise ValueError("valid mask contains no pixels")
    if not np.isfinite(target_values).all() or not np.isfinite(
        prediction_values
    ).all():
        raise ValueError("valid target and prediction pixels must be finite")
    error = prediction_values - target_values
    absolute = np.abs(error)
    target_centered = target_values - target_values.mean()
    prediction_centered = prediction_values - prediction_values.mean()
    denominator = float(
        np.sqrt(np.sum(target_centered**2) * np.sum(prediction_centered**2))
    )
    pearson_r = (
        float(np.sum(target_centered * prediction_centered) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    return {
        "valid_pixels": int(target_values.size),
        "mae_db": float(absolute.mean()),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "bias_db": float(error.mean()),
        "p90_absolute_error_db": float(np.percentile(absolute, 90.0)),
        "max_absolute_error_db": float(absolute.max()),
        "pearson_r": pearson_r,
    }


def compute_display_limits(
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, list[float] | float]:
    signal = np.concatenate((target[valid], prediction[valid])).astype(np.float64)
    error = np.abs(
        prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    )
    if (
        signal.size == 0
        or not np.isfinite(signal).all()
        or not np.isfinite(error).all()
    ):
        raise ValueError("display-limit inputs must contain finite valid pixels")
    signal_vmin, signal_vmax = np.percentile(signal, (1.0, 99.0))
    return {
        "signal_db": [float(signal_vmin), float(signal_vmax)],
        "error_p99_db": max(
            float(np.percentile(error, 99.0)), np.finfo(float).eps
        ),
    }


def find_sample_index(
    rows: Sequence[tuple[dict[str, Any], int]],
    site_id: str,
    direction: int,
) -> int:
    matches = [
        index
        for index, (entry, row_direction) in enumerate(rows)
        if entry["site_id"] == site_id
        and int(row_direction) == int(direction)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one sample for {site_id} direction {direction}, "
            f"found {len(matches)}"
        )
    return matches[0]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance(
    config: dict[str, Any],
    report: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, str]:
    expected_report_paths = {
        "stage1": Path(report["stage1_checkpoint"]),
        "stage2a": Path(report["stage2a_checkpoint"]),
        "stage2b": Path(report["stage2b_checkpoint"]),
        "estimator": Path(report["estimator_checkpoint"]),
    }
    expected_config_paths = {
        "stage1": Path(config["stage1_model_dir"]) / "best.pt",
        "stage2a": Path(config["stage2a_model_dir"]) / "best.pt",
        "estimator": Path(config["estimator_checkpoint"]),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.resolve() != expected_report_paths[name].resolve():
            raise ValueError(
                f"{name} checkpoint path does not match the formal V3 report"
            )
        if (
            name in expected_config_paths
            and path.resolve() != expected_config_paths[name].resolve()
        ):
            raise ValueError(f"{name} checkpoint path does not match the V3 config")

    actual = {name: sha256_file(path) for name, path in paths.items()}
    expected_hashes = {
        "stage1": config["provenance"]["stage1_checkpoint_sha256"],
        "stage2a": config["provenance"]["stage2a_checkpoint_sha256"],
        "stage2b": report["stage2b_checkpoint_sha256"],
        "estimator": report["estimator_checkpoint_sha256"],
    }
    for name, expected in expected_hashes.items():
        if actual[name].lower() != str(expected).lower():
            raise ValueError(
                f"{name} checkpoint SHA-256 does not match formal provenance"
            )
    config_estimator_hash = config["provenance"][
        "estimator_checkpoint_sha256"
    ]
    if actual["estimator"].lower() != str(config_estimator_hash).lower():
        raise ValueError(
            "estimator checkpoint SHA-256 does not match the V3 config"
        )
    return actual


def guard_outputs(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / name
        for name in OUTPUT_NAMES
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"output exists; rerun with --overwrite: {joined}"
        )


def validate_sample_arrays(
    arrays: dict[str, np.ndarray], azimuth_deg: float
) -> None:
    required = (
        "building",
        "sparse_mask",
        "sparse_db",
        "target_db",
        "valid_mask",
        "prediction_db",
    )
    for name in required:
        if arrays[name].shape != (128, 128):
            raise ValueError(
                f"{name} must have shape (128, 128), got {arrays[name].shape}"
            )
    if int(arrays["sparse_mask"].sum()) != 100:
        raise ValueError("fixed sample must contain exactly 100 sparse points")
    if not math.isclose(float(azimuth_deg), 12.0, abs_tol=1e-5):
        raise ValueError(
            "fixed sample direction 0 must have azimuth 12 degrees, "
            f"got {azimuth_deg}"
        )
    if int(arrays["valid_mask"].sum()) == 0:
        raise ValueError("valid mask contains no pixels")
    for name in ("building", "target_db", "prediction_db"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite required values")
    if not np.isfinite(arrays["sparse_db"][arrays["sparse_mask"]]).all():
        raise ValueError("sparse signal values must be finite")


def augment_evidence_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in arrays.items()}
    result["signed_error_db"] = result["prediction_db"] - result["target_db"]
    result["absolute_error_db"] = np.abs(result["signed_error_db"])
    return result


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_formal_v3_inference(
    config_path: Path,
    estimator_config_path: Path,
    report_path: Path,
    site_id: str,
    direction: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from torch.utils.data._utils.collate import default_collate

    from bs_inversion_dataset import BSInversionPilotDataset
    from train_stage2b_directional_ss import resolve_device, seed_everything
    from visualize_scale4_pipeline_examples import (
        load_estimator,
        load_pair,
        load_stage2b,
        predict_pair,
    )

    config_path = config_path.resolve()
    estimator_config_path = estimator_config_path.resolve()
    report_path = report_path.resolve()
    config = load_json(config_path)
    report = load_json(report_path)
    paths = {
        "stage1": Path(report["stage1_checkpoint"]).resolve(),
        "stage2a": Path(report["stage2a_checkpoint"]).resolve(),
        "stage2b": Path(report["stage2b_checkpoint"]).resolve(),
        "estimator": Path(report["estimator_checkpoint"]).resolve(),
    }
    hashes = validate_provenance(config, report, paths)
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    channels_last = bool(config["channels_last"]) and device.type == "cuda"

    irt_normalization = load_json(Path(config["irt_normalization"]))
    iso_stats = irt_normalization["statistics"]["train"]["p_iso"]
    ss_stats = irt_normalization["stage2b_ss_normalization"]
    iso_mean_db = float(iso_stats["mean_db"])
    iso_std_db = float(iso_stats["std_db"])
    ss_mean_db = float(ss_stats["mean_db"])
    ss_std_db = float(ss_stats["std_db"])

    dataset = BSInversionPilotDataset(
        config["dataset_root"],
        split="test",
        seed=int(config["seed"]),
        augment=False,
        signal_mean_db=ss_mean_db,
        signal_std_db=ss_std_db,
    )
    sample_index = find_sample_index(dataset.rows, site_id, direction)
    batch = default_collate([dataset[sample_index]])

    (
        estimator,
        estimator_mean,
        estimator_std,
        estimator_power_mean,
        estimator_power_std,
    ) = load_estimator(
        estimator_config_path,
        paths["estimator"],
        device,
        channels_last,
    )
    stage1, stage2a, pair_metadata, tx_values = load_pair(
        config_path, device, channels_last
    )
    stage2b, stage2b_config = load_stage2b(
        paths["stage2b"], device, channels_last
    )
    with torch.inference_mode():
        prediction, estimated_row_col = predict_pair(
            batch,
            stage1=stage1,
            stage2a=stage2a,
            pair_metadata=pair_metadata,
            tx_values=tx_values,
            stage2b=stage2b,
            stage2b_config=stage2b_config,
            estimator=estimator,
            estimator_mean_db=estimator_mean,
            estimator_std_db=estimator_std,
            estimator_power_mean_db=estimator_power_mean,
            estimator_power_std_db=estimator_power_std,
            iso_mean_db=iso_mean_db,
            iso_std_db=iso_std_db,
            ss_mean_db=ss_mean_db,
            ss_std_db=ss_std_db,
            device=device,
            amp_enabled=amp_enabled,
            channels_last=channels_last,
        )

    sparse_mask = batch["sparse_mask"][0, 0].numpy().astype(bool)
    arrays = {
        "building": batch["building"][0, 0].numpy().astype(np.float32),
        "sparse_mask": sparse_mask,
        "sparse_db": (
            batch["sparse_ss_norm"][0, 0].numpy() * ss_std_db + ss_mean_db
        ).astype(np.float32),
        "target_db": batch["target_map_db"][0, 0]
        .numpy()
        .astype(np.float32),
        "valid_mask": batch["valid_mask"][0, 0].numpy().astype(bool),
        "prediction_db": prediction[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "true_row_col_px": batch["target_row_col_px"][0]
        .numpy()
        .astype(np.float32),
        "estimated_row_col_px": estimated_row_col[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }
    azimuth_deg = float(batch["target_azimuth_deg"][0])
    validate_sample_arrays(arrays, azimuth_deg)
    runtime = {
        "device": str(device),
        "amp_enabled": amp_enabled,
        "channels_last": channels_last,
        "checkpoint_sha256": hashes,
        "sample_index": sample_index,
        "signal_mean_db": ss_mean_db,
        "signal_std_db": ss_std_db,
    }
    sample = {
        "site_id": site_id,
        "direction_index": int(direction),
        "azimuth_deg": azimuth_deg,
        "tile": batch["tile"][0],
        "site_index": int(batch["site_index"][0]),
    }
    return sample, arrays, {
        "config": config,
        "report": report,
        "runtime": runtime,
    }


def row_col_to_xy_m(row_col: np.ndarray) -> tuple[float, float]:
    return (
        (float(row_col[1]) + 0.5) * 4.0,
        512.0 - (float(row_col[0]) + 0.5) * 4.0,
    )


def style_map_axis(axis: Any) -> None:
    axis.set(
        xlabel="East (m)",
        ylabel="North (m)",
        xlim=(0.0, 512.0),
        ylim=(0.0, 512.0),
    )
    axis.set_xticks([0, 128, 256, 384, 512])
    axis.set_yticks([0, 128, 256, 384, 512])
    axis.set_aspect("equal")


def render_figure(
    arrays: dict[str, np.ndarray],
    sample: dict[str, Any],
    metrics: dict[str, float | int],
    aggregate: dict[str, Any],
    limits: dict[str, list[float] | float],
    figure_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from PIL import Image

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    signal_vmin, signal_vmax = limits["signal_db"]
    error_limit = float(limits["error_p99_db"])
    valid = arrays["valid_mask"]
    signal_cmap = mpl.colormaps["viridis"].copy()
    signal_cmap.set_bad("#A6A6A6")
    signed_cmap = mpl.colormaps["RdBu_r"].copy()
    signed_cmap.set_bad("#A6A6A6")
    absolute_cmap = mpl.colormaps["magma"].copy()
    absolute_cmap.set_bad("#A6A6A6")

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.4, 8.6),
        layout="constrained",
    )
    figure.get_layout_engine().set(rect=(0.02, 0.055, 0.98, 0.94))

    input_axis = axes[0, 0]
    input_axis.imshow(
        arrays["building"],
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=(0, 512, 0, 512),
        origin="upper",
        interpolation="nearest",
    )
    sparse_rows, sparse_cols = np.nonzero(arrays["sparse_mask"])
    input_axis.scatter(
        (sparse_cols + 0.5) * 4.0,
        512.0 - (sparse_rows + 0.5) * 4.0,
        c=arrays["sparse_db"][arrays["sparse_mask"]],
        s=16,
        cmap=signal_cmap,
        vmin=signal_vmin,
        vmax=signal_vmax,
        marker="o",
        edgecolors="white",
        linewidths=0.25,
    )
    true_x, true_y = row_col_to_xy_m(arrays["true_row_col_px"])
    estimated_x, estimated_y = row_col_to_xy_m(
        arrays["estimated_row_col_px"]
    )
    input_axis.scatter(
        true_x,
        true_y,
        marker="*",
        s=190,
        color="#D55E00",
        edgecolors="white",
        linewidths=0.9,
        zorder=4,
    )
    input_axis.scatter(
        estimated_x,
        estimated_y,
        marker="x",
        s=90,
        color="#0072B2",
        linewidths=2.0,
        zorder=5,
    )
    input_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                markersize=11,
                markerfacecolor="#D55E00",
                markeredgecolor="white",
                label="True BS",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="",
                markersize=8,
                color="#0072B2",
                markeredgewidth=2,
                label="Estimated BS",
            ),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    input_axis.set_title(
        "A  Input: buildings + 100 sparse samples", loc="left"
    )

    signal_images = []
    signal_panels = (
        (axes[0, 1], "target_db", "B  Ground truth"),
        (
            axes[0, 2],
            "prediction_db",
            "C  V3 prediction\n"
            f"RMSE {metrics['rmse_db']:.2f} dB | "
            f"MAE {metrics['mae_db']:.2f} dB",
        ),
    )
    for axis, key, title in signal_panels:
        image = axis.imshow(
            np.ma.masked_where(~valid, arrays[key]),
            cmap=signal_cmap,
            vmin=signal_vmin,
            vmax=signal_vmax,
            extent=(0, 512, 0, 512),
            origin="upper",
            interpolation="nearest",
        )
        signal_images.append(image)
        axis.set_title(title, loc="left")

    signed_image = axes[1, 0].imshow(
        np.ma.masked_where(~valid, arrays["signed_error_db"]),
        cmap=signed_cmap,
        norm=TwoSlopeNorm(
            vmin=-error_limit,
            vcenter=0.0,
            vmax=error_limit,
        ),
        extent=(0, 512, 0, 512),
        origin="upper",
        interpolation="nearest",
    )
    axes[1, 0].set_title(
        f"D  Signed error | bias {metrics['bias_db']:+.2f} dB",
        loc="left",
    )
    absolute_image = axes[1, 1].imshow(
        np.ma.masked_where(~valid, arrays["absolute_error_db"]),
        cmap=absolute_cmap,
        vmin=0.0,
        vmax=error_limit,
        extent=(0, 512, 0, 512),
        origin="upper",
        interpolation="nearest",
    )
    axes[1, 1].set_title(
        "E  Absolute error\n"
        f"P90 {metrics['p90_absolute_error_db']:.2f} dB | "
        f"max {metrics['max_absolute_error_db']:.2f} dB",
        loc="left",
    )

    summary_axis = axes[1, 2]
    summary_axis.axis("off")
    gate = aggregate["gate"]
    summary_axis.set_title("F  Numeric summary", loc="left")
    summary_axis.text(
        0.03,
        0.96,
        "Fixed held-out sample\n"
        f"Valid pixels: {metrics['valid_pixels']:,}\n"
        f"MAE: {metrics['mae_db']:.3f} dB\n"
        f"RMSE: {metrics['rmse_db']:.3f} dB\n"
        f"Bias: {metrics['bias_db']:+.3f} dB\n"
        f"P90 |error|: {metrics['p90_absolute_error_db']:.3f} dB\n"
        f"Max |error|: {metrics['max_absolute_error_db']:.3f} dB\n"
        f"Pearson r: {metrics['pearson_r']:.3f}",
        transform=summary_axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.7,
        linespacing=1.2,
        color="#222222",
    )
    summary_axis.text(
        0.03,
        0.45,
        "Complete test (4,096 maps)\n"
        f"MAE: {aggregate['mae_db']:.3f} dB\n"
        f"RMSE: {aggregate['rmse_db']:.3f} dB",
        transform=summary_axis.transAxes,
        va="top",
        ha="left",
        fontsize=9.7,
        linespacing=1.2,
        color="#222222",
    )
    summary_axis.text(
        0.03,
        0.24,
        "FAIL — Gate not passed",
        transform=summary_axis.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color="#B2182B",
    )
    summary_axis.text(
        0.03,
        0.17,
        f"RMSE ≤ {gate['maximum_rmse_db']:.2f} dB required\n"
        f"Improvement ≥ {gate['minimum_improvement_db']:.2f} dB required\n"
        f"Actual improvement: {gate['actual_improvement_db']:.3f} dB\n"
        "Gray = invalid; excluded from sample metrics.",
        transform=summary_axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        linespacing=1.05,
        color="#333333",
    )

    map_axes = (
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
    )
    for axis in map_axes:
        style_map_axis(axis)
    figure.colorbar(
        signal_images[-1],
        ax=axes[0, :],
        orientation="horizontal",
        fraction=0.055,
        pad=0.08,
        extend="both",
        label=(
            "Signal strength (dB); shared 1st–99th percentile display range"
        ),
    )
    figure.colorbar(
        signed_image,
        ax=axes[1, 0],
        fraction=0.047,
        pad=0.04,
        extend="both",
        label="V3 prediction - truth (dB); P99 limits",
    )
    figure.colorbar(
        absolute_image,
        ax=axes[1, 1],
        fraction=0.047,
        pad=0.04,
        extend="max",
        label="Absolute error (dB); P99 upper limit",
    )
    figure.suptitle(
        "Scale-4 V3 signal-map evaluation on one fixed held-out test sample",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        f"Sample {sample['site_id']} | direction {sample['direction_index']} | "
        f"azimuth {sample['azimuth_deg']:.0f}° | 128×128 pixels | "
        "4 m resolution",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    figure.savefig(
        figure_path,
        dpi=220,
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)
    with Image.open(figure_path) as rendered:
        rendered.convert("RGB").save(figure_path, dpi=(220, 220))


def build_aggregate_result(
    config: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    if int(report["directional_samples"]) != 4096 or int(
        report["sites"]
    ) != 1024:
        raise ValueError(
            "formal V3 report must describe 1,024 sites and 4,096 maps"
        )
    baseline_rmse_db = float(
        config["acceptance"]["baseline_final_test_rmse_db"]
    )
    rmse_db = float(report["final_rmse_db"])
    maximum_rmse_db = float(
        config["acceptance"]["maximum_test_rmse_db"]
    )
    minimum_improvement_db = float(
        config["acceptance"]["minimum_improvement_db"]
    )
    actual_improvement_db = baseline_rmse_db - rmse_db
    return {
        "sites": int(report["sites"]),
        "maps": int(report["directional_samples"]),
        "mae_db": float(report["final_mae_db"]),
        "rmse_db": rmse_db,
        "gate": {
            "baseline_rmse_db": baseline_rmse_db,
            "maximum_rmse_db": maximum_rmse_db,
            "minimum_improvement_db": minimum_improvement_db,
            "actual_improvement_db": actual_improvement_db,
            "passed": bool(
                rmse_db <= maximum_rmse_db
                and actual_improvement_db >= minimum_improvement_db
            ),
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    guard_outputs(output_dir, args.overwrite)
    sample, arrays, provenance = run_formal_v3_inference(
        args.config,
        args.estimator_config,
        args.report,
        args.site_id,
        args.direction,
    )
    arrays = augment_evidence_arrays(arrays)
    metrics = sample_metrics(
        arrays["target_db"],
        arrays["prediction_db"],
        arrays["valid_mask"],
    )
    if not all(np.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("sample metrics must all be finite")
    limits = compute_display_limits(
        arrays["target_db"],
        arrays["prediction_db"],
        arrays["valid_mask"],
    )
    config = provenance["config"]
    report = provenance["report"]
    aggregate = build_aggregate_result(config, report)
    if aggregate["gate"]["passed"]:
        raise ValueError(
            "formal V3 gate was expected to be unpassed; report/config changed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / OUTPUT_NAMES[0]
    metrics_path = output_dir / OUTPUT_NAMES[1]
    arrays_path = output_dir / OUTPUT_NAMES[2]
    render_figure(
        arrays,
        sample,
        metrics,
        aggregate,
        limits,
        figure_path,
    )
    atomic_save_npz(arrays_path, arrays)

    from PIL import Image

    with Image.open(figure_path) as image:
        figure_metadata = {
            "mode": image.mode,
            "size_px": list(image.size),
            "dpi": list(image.info.get("dpi", (0, 0))),
        }
    result = {
        "version": 1,
        "status": "completed",
        "sample": sample,
        "sample_metrics": metrics,
        "complete_test": aggregate,
        "display_limits": limits,
        "valid_mask_definition": (
            "directional_valid_mask; buildings and unhit outdoor pixels "
            "are excluded"
        ),
        "transformations": {
            "metric_filter": "valid_mask only",
            "signal_display": "shared target/prediction 1st–99th percentiles",
            "signed_error_display": "zero-centred symmetric P99 absolute error",
            "absolute_error_display": "0 to P99 absolute error",
            "interpolation": "nearest; no smoothing or upsampling claim",
        },
        "provenance": {
            "config": str(args.config.resolve()),
            "report": str(args.report.resolve()),
            "dataset_root": str(config["dataset_root"]),
            "checkpoints": {
                name: {
                    "path": str(
                        Path(report[f"{name}_checkpoint"]).resolve()
                    ),
                    "sha256": provenance["runtime"]["checkpoint_sha256"][
                        name
                    ],
                }
                for name in ("stage1", "stage2a", "stage2b", "estimator")
            },
            "runtime": {
                key: value
                for key, value in provenance["runtime"].items()
                if key != "checkpoint_sha256"
            },
        },
        "outputs": {
            "figure": str(figure_path),
            "metrics": str(metrics_path),
            "arrays": str(arrays_path),
            "figure_metadata": figure_metadata,
        },
        "destination": (
            "general project evaluation figure; no publisher-specific "
            "compliance is claimed"
        ),
        "alt_text": (
            "Six-panel evaluation of formal Scale-4 V3 on held-out sample "
            "tile_000222_site03: input buildings and 100 sparse "
            "measurements, ground truth, V3 prediction, signed error, "
            "absolute error, and numeric summary. Invalid pixels are gray. "
            f"The sample RMSE is {metrics['rmse_db']:.3f} dB and the "
            f"complete-test RMSE is {aggregate['rmse_db']:.3f} dB, so the "
            "predeclared gate is not passed."
        ),
    }
    atomic_write_json(metrics_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

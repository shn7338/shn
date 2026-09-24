#!/usr/bin/env python3
"""Render an honest one-sample V4-to-V5 training evaluation figure.

The script consumes the immutable arrays cached by
``visualize_scale4_pipeline_examples.py``.  It does not rerun inference or
alter the source arrays.  Signal maps share one colour scale, clipped limits
are explicitly recorded, and the complete-test RMSE values are shown beside
the selected sample so that one visually favourable tile cannot stand in for
the aggregate result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


DEFAULT_SOURCE_DIR = Path(
    r"D:\桌面\dac\03_runs\experiments\scale4_visual_comparison_v5"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\桌面\dac\03_runs\experiments\single_test_training_evaluation"
)
DEFAULT_REPORTS = {
    "V3": Path(
        r"D:\桌面\dac\03_runs\experiments\bs_inversion_scale4_stage2b_v3_test_bins.json"
    ),
    "V4": Path(
        r"D:\桌面\dac\03_runs\experiments\bs_inversion_scale4_stage2b_latent_fusion_v4_test_bins.json"
    ),
    "V5": Path(
        r"D:\桌面\dac\03_runs\experiments\bs_inversion_scale4_stage2b_multiscale_adapter_v5_test_bins.json"
    ),
    "V6": Path(
        r"D:\桌面\dac\03_runs\experiments\bs_inversion_scale4_joint_estimator_stage2b_v6_test_bins.json"
    ),
}
GATE_PATH = Path(
    r"D:\桌面\dac\03_runs\experiments\bs_inversion_scale4_stage2b_multiscale_adapter_v5_gate.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing selected_samples.npz and visualization_summary.json",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=3,
        help="Cached representative sample index (default: far-center sample 3)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    target_values = target[valid].astype(np.float64)
    prediction_values = prediction[valid].astype(np.float64)
    error = prediction_values - target_values
    absolute = np.abs(error)
    target_centered = target_values - target_values.mean()
    prediction_centered = prediction_values - prediction_values.mean()
    pearson_denominator = float(
        np.sqrt(
            np.sum(target_centered * target_centered)
            * np.sum(prediction_centered * prediction_centered)
        )
    )
    pearson_r = (
        float(np.sum(target_centered * prediction_centered)) / pearson_denominator
        if pearson_denominator > 0.0
        else float("nan")
    )
    return {
        "valid_pixels": int(valid.sum()),
        "mae_db": float(absolute.mean()),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "bias_db": float(error.mean()),
        "p90_absolute_error_db": float(np.percentile(absolute, 90.0)),
        # Compute explicitly instead of calling np.corrcoef.  The sigmap
        # environment's BLAS-backed covariance path crashes on this Windows
        # host (0xc06d007f), while these bounded elementwise reductions are
        # numerically equivalent for Pearson's r.
        "pearson_r": pearson_r,
        "within_3_db_percent": float(np.mean(absolute <= 3.0) * 100.0),
        "within_5_db_percent": float(np.mean(absolute <= 5.0) * 100.0),
        "within_10_db_percent": float(np.mean(absolute <= 10.0) * 100.0),
    }


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    arrays_path = source_dir / "selected_samples.npz"
    summary_path = source_dir / "visualization_summary.json"
    summary = load_json(summary_path)
    if not 0 <= args.sample_index < len(summary["samples"]):
        raise ValueError(
            f"sample index {args.sample_index} is outside 0..{len(summary['samples']) - 1}"
        )

    cached = np.load(arrays_path)
    prefix = f"sample_{args.sample_index}_"
    building = cached[prefix + "building"].astype(np.float32)
    sparse_mask = cached[prefix + "sparse_mask"].astype(bool)
    sparse_db = cached[prefix + "sparse_db"].astype(np.float32)
    target = cached[prefix + "target"].astype(np.float32)
    valid = cached[prefix + "valid"].astype(bool)
    v4_prediction = cached[prefix + "before"].astype(np.float32)
    v5_prediction = cached[prefix + "after"].astype(np.float32)
    true_row_col = cached[prefix + "true_row_col"].astype(np.float32)
    estimated_row_col = cached[prefix + "estimated_row_col"].astype(np.float32)
    sample_info = summary["samples"][args.sample_index]

    v4_metrics = sample_metrics(target, v4_prediction, valid)
    v5_metrics = sample_metrics(target, v5_prediction, valid)
    error = v5_prediction - target
    absolute_error = np.abs(error)
    update = v5_prediction - v4_prediction

    signal_values = np.concatenate(
        (target[valid], v4_prediction[valid], v5_prediction[valid])
    ).astype(np.float64)
    signal_vmin, signal_vmax = [
        float(value) for value in np.percentile(signal_values, (1.0, 99.0))
    ]
    error_limit = max(
        1.0, float(np.percentile(np.abs(error[valid]).astype(np.float64), 99.0))
    )
    update_limit = max(
        0.1, float(np.percentile(np.abs(update[valid]).astype(np.float64), 99.0))
    )

    aggregate_reports = {name: load_json(path) for name, path in DEFAULT_REPORTS.items()}
    aggregate_rmse = {
        name: float(report["final_rmse_db"])
        for name, report in aggregate_reports.items()
    }
    gate = load_json(GATE_PATH)
    acceptance_threshold = float(gate["checks"]["overall_rmse"]["maximum_db"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16.0, 8.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.08]},
    )
    # Reserve a dedicated footer band so provenance and the scientific
    # conclusion never cover axis labels at the delivered size.
    figure.get_layout_engine().set(rect=(0.015, 0.075, 0.985, 0.945))

    # Panel A: model inputs, with colour plus marker redundancy.
    input_axis = axes[0, 0]
    input_axis.imshow(building, cmap="Greys", vmin=0.0, vmax=1.0)
    sparse_rows, sparse_cols = np.nonzero(sparse_mask)
    sparse_plot = input_axis.scatter(
        sparse_cols,
        sparse_rows,
        c=sparse_db[sparse_mask],
        s=12,
        cmap="viridis",
        vmin=signal_vmin,
        vmax=signal_vmax,
        marker="o",
        linewidths=0.2,
        edgecolors="white",
    )
    input_axis.scatter(
        [true_row_col[1]],
        [true_row_col[0]],
        marker="*",
        s=190,
        color="#D55E00",
        edgecolors="white",
        linewidths=0.9,
        zorder=4,
    )
    input_axis.scatter(
        [estimated_row_col[1]],
        [estimated_row_col[0]],
        marker="x",
        s=90,
        color="#0072B2",
        linewidths=2.0,
        zorder=5,
    )
    input_axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="*", linestyle="", markersize=11,
                markerfacecolor="#D55E00", markeredgecolor="white", label="True BS"
            ),
            Line2D(
                [0], [0], marker="x", linestyle="", markersize=8,
                color="#0072B2", markeredgewidth=2, label="Estimated BS"
            ),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    input_axis.set_title("A  Input: buildings + 100 sparse samples", loc="left")

    signal_images = []
    for axis, array, title in (
        (axes[0, 1], target, "B  Ground truth"),
        (
            axes[0, 2],
            v4_prediction,
            f"C  V4 prediction | RMSE {v4_metrics['rmse_db']:.2f} dB",
        ),
        (
            axes[0, 3],
            v5_prediction,
            f"D  V5 prediction | RMSE {v5_metrics['rmse_db']:.2f} dB",
        ),
    ):
        image = axis.imshow(
            np.ma.masked_where(~valid, array),
            cmap="viridis",
            vmin=signal_vmin,
            vmax=signal_vmax,
            interpolation="nearest",
        )
        signal_images.append(image)
        axis.set_title(title, loc="left")

    signed_error_image = axes[1, 0].imshow(
        np.ma.masked_where(~valid, error),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit),
        interpolation="nearest",
    )
    axes[1, 0].set_title(
        f"E  V5 signed error | bias {v5_metrics['bias_db']:+.2f} dB", loc="left"
    )

    absolute_error_image = axes[1, 1].imshow(
        np.ma.masked_where(~valid, absolute_error),
        cmap="magma",
        vmin=0.0,
        vmax=error_limit,
        interpolation="nearest",
    )
    axes[1, 1].set_title(
        f"F  V5 absolute error | P90 {v5_metrics['p90_absolute_error_db']:.2f} dB",
        loc="left",
    )

    update_image = axes[1, 2].imshow(
        np.ma.masked_where(~valid, update),
        cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-update_limit, vcenter=0.0, vmax=update_limit),
        interpolation="nearest",
    )
    axes[1, 2].set_title(
        f"G  Training update: V5 - V4 | P99 ±{update_limit:.2f} dB", loc="left"
    )

    aggregate_axis = axes[1, 3]
    names = list(aggregate_rmse)
    values = [aggregate_rmse[name] for name in names]
    colors = ["#999999", "#56B4E9", "#009E73", "#CC79A7"]
    x_positions = np.arange(len(names), dtype=np.float64)
    aggregate_axis.plot(
        x_positions,
        values,
        color="#4D4D4D",
        linewidth=1.5,
        linestyle="-",
        zorder=2,
    )
    aggregate_axis.scatter(
        x_positions,
        values,
        c=colors,
        s=85,
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )
    aggregate_axis.axhline(
        acceptance_threshold,
        color="#D55E00",
        linestyle="--",
        linewidth=1.5,
        label=f"Predeclared gate: {acceptance_threshold:.2f} dB",
    )
    for x_position, value in zip(x_positions, values):
        aggregate_axis.text(
            x_position,
            value + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    aggregate_axis.set_ylim(6.78, 6.98)
    aggregate_axis.set_xticks(x_positions, names)
    aggregate_axis.set_ylabel("Full test RMSE (dB)")
    aggregate_axis.set_title("H  Complete-test result (4096 maps)", loc="left")
    aggregate_axis.grid(axis="y", alpha=0.25)
    aggregate_axis.legend(loc="lower left", fontsize=8, framealpha=0.9)

    for axis in axes.flat[:7]:
        axis.set_xticks([0, 32, 64, 96, 127], [0, 128, 256, 384, 508])
        axis.set_yticks([0, 32, 64, 96, 127], [512, 384, 256, 128, 4])
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        axis.set_aspect("equal")

    figure.colorbar(
        signal_images[-1],
        ax=[axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3]],
        label="Signal strength (dB); 1st–99th percentile shown",
        orientation="horizontal",
        fraction=0.055,
        pad=0.08,
        extend="both",
    )
    figure.colorbar(
        signed_error_image,
        ax=axes[1, 0],
        label="Prediction - truth (dB); 99th percentile limits",
        fraction=0.046,
        pad=0.04,
        extend="both",
    )
    figure.colorbar(
        absolute_error_image,
        ax=axes[1, 1],
        label="Absolute error (dB); P99 upper limit",
        fraction=0.046,
        pad=0.04,
        extend="max",
    )
    figure.colorbar(
        update_image,
        ax=axes[1, 2],
        label="V5 - V4 (dB); 99th percentile limits",
        fraction=0.046,
        pad=0.04,
        extend="both",
    )

    sample_delta = float(v4_metrics["rmse_db"]) - float(v5_metrics["rmse_db"])
    aggregate_delta = aggregate_rmse["V4"] - aggregate_rmse["V5"]
    figure.suptitle(
        "Scale-4 signal-map training evaluation on one representative held-out test sample",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.014,
        (
            f"Sample {sample_info['site_id']} | azimuth {sample_info['azimuth_deg']:.0f}° | "
            f"center-distance bin {sample_info['bin']} | valid pixels {v5_metrics['valid_pixels']:,} | "
            f"V5 MAE {v5_metrics['mae_db']:.2f} dB, RMSE {v5_metrics['rmse_db']:.2f} dB, "
            f"r={v5_metrics['pearson_r']:.3f}; sample ΔRMSE {sample_delta:+.3f} dB.\n"
            f"Full test V4→V5 improves {aggregate_delta:.3f} dB; V6 adds 0.000 dB and the "
            "predeclared gate is not passed."
        ),
        ha="center",
        va="bottom",
        fontsize=9.3,
        linespacing=1.35,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "single_test_v4_to_v5_evaluation.png"
    figure.savefig(
        figure_path,
        dpi=220,
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)
    # Matplotlib writes an RGBA PNG even with a fully opaque white canvas.
    # Deliver a genuine RGB file so downstream viewers cannot reinterpret the
    # background and alter apparent contrast.
    with Image.open(figure_path) as rendered:
        rendered.convert("RGB").save(figure_path, dpi=(220, 220))

    result = {
        "version": 1,
        "status": "completed",
        "sample_selection": (
            "cached direction-0 representative from the 192m-plus bin; among the four "
            "radial-bin representatives, its V5 RMSE is closest to the complete-test V5 RMSE"
        ),
        "sample_index": args.sample_index,
        "sample": sample_info,
        "v4_sample_metrics": v4_metrics,
        "v5_sample_metrics": v5_metrics,
        "sample_rmse_improvement_v4_to_v5_db": sample_delta,
        "complete_test_rmse_db": aggregate_rmse,
        "complete_test_rmse_improvement_v4_to_v5_db": aggregate_delta,
        "v6_improvement_vs_v5_db": aggregate_rmse["V5"] - aggregate_rmse["V6"],
        "predeclared_gate_rmse_db": acceptance_threshold,
        "predeclared_gate_passed": bool(gate["accepted"]),
        "display_limits": {
            "signal_1st_99th_percentile_db": [signal_vmin, signal_vmax],
            "signed_and_absolute_error_99th_percentile_db": error_limit,
            "v5_minus_v4_update_99th_percentile_db": update_limit,
            "out_of_range_policy": "saturated and marked by extended colorbar ends",
        },
        "source_arrays": str(arrays_path),
        "source_summary": str(summary_path),
        "source_reports": {name: str(path) for name, path in DEFAULT_REPORTS.items()},
        "source_gate": str(GATE_PATH),
        "output_figure": str(figure_path),
        "alt_text": (
            "Eight-panel evaluation of one held-out Scale-4 test map. The top row "
            "shows building and sparse-signal inputs, ground truth, and V4/V5 "
            "predictions. The bottom row shows signed error, absolute error, the "
            "V5-minus-V4 update, and full-test RMSE for V3 through V6. V5 slightly "
            "improves over V4, V6 is unchanged, and all remain above the 6.82 dB gate."
        ),
    }
    result_path = output_dir / "single_test_v4_to_v5_metrics.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

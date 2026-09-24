#!/usr/bin/env python3
"""Render the cached legacy-to-complete-V3 Scale-4 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"D:\桌面\dac\experiments\scale4_visual_comparison_v3"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            r"D:\桌面\dac\models\stage2a_position_encoding_scale4_v3\history.csv"
        ),
    )
    return parser.parse_args()


def load_rows(input_dir: Path) -> tuple[list[dict], dict]:
    summary = json.loads(
        (input_dir / "visualization_summary.json").read_text(encoding="utf-8")
    )
    data = np.load(input_dir / "selected_samples.npz", allow_pickle=False)
    rows: list[dict] = []
    array_keys = (
        "building",
        "sparse_mask",
        "sparse_db",
        "target",
        "valid",
        "before",
        "after",
        "true_row_col",
        "estimated_row_col",
    )
    for index, metadata in enumerate(summary["samples"]):
        row = dict(metadata)
        for key in array_keys:
            row[key] = data[f"sample_{index}_{key}"]
        row["sparse_mask"] = row["sparse_mask"].astype(bool)
        row["valid"] = row["valid"].astype(bool)
        rows.append(row)
    return rows, summary


def draw_examples(rows: list[dict], output_path: Path) -> None:
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
            [true_col],
            [true_row],
            marker="*",
            s=180,
            c="red",
            edgecolors="white",
            linewidths=0.8,
            label="True BS",
        )
        axis.scatter(
            [estimated_col],
            [estimated_row],
            marker="x",
            s=100,
            c="cyan",
            linewidths=2.0,
            label="Estimated BS",
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
    signal_colorbar_axis = figure.add_axes((0.925, 0.56, 0.012, 0.30))
    error_colorbar_axis = figure.add_axes((0.925, 0.14, 0.012, 0.30))
    figure.colorbar(image, cax=signal_colorbar_axis, label="Signal strength (dB)")
    figure.colorbar(
        error_image, cax=error_colorbar_axis, label="Absolute error (dB)"
    )
    figure.suptitle(
        "Scale-4 representative test samples: legacy pipeline vs complete V3 pipeline",
        fontsize=16,
    )
    figure.subplots_adjust(
        left=0.055, right=0.905, bottom=0.035, top=0.94, wspace=0.14, hspace=0.26
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_training_curve(history_path: Path, output_path: Path) -> None:
    rows = np.genfromtxt(
        history_path, delimiter=",", names=True, encoding="utf-8-sig"
    )
    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    axis.plot(rows["epoch"], rows["train_rmse_db"], marker="o", label="Train")
    axis.plot(rows["epoch"], rows["val_rmse_db"], marker="o", label="Validation")
    axis.axhline(
        7.53716491575115, color="gray", linestyle="--", label="Before: 7.537 dB"
    )
    axis.axhline(
        7.33716491575115,
        color="green",
        linestyle=":",
        label="Stage2B gate: 7.337 dB",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Isotropic map RMSE (dB)")
    axis.set_title("Stage1 + Stage2A Scale-4 fine-tuning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    rows, summary = load_rows(args.input_dir)
    comparison_path = args.input_dir / "representative_before_after.png"
    curve_path = args.input_dir / "training_curve.png"
    draw_examples(rows, comparison_path)
    draw_training_curve(args.history, curve_path)
    print(
        json.dumps(
            {
                "comparison_png": str(comparison_path),
                "training_curve_png": str(curve_path),
                "samples": summary["samples"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render V3-versus-V4 maps and metrics without importing PyTorch."""

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
    parser.add_argument("--visualization-dir", type=Path, required=True)
    parser.add_argument(
        "--before-evaluation", "--v3-evaluation", dest="before_evaluation",
        type=Path, required=True
    )
    parser.add_argument(
        "--after-evaluation", "--v4-evaluation", dest="after_evaluation",
        type=Path, required=True
    )
    parser.add_argument(
        "--history", "--v4-history", dest="history", type=Path, required=True
    )
    parser.add_argument("--before-label", default="V3")
    parser.add_argument("--after-label", default="V4")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(directory: Path) -> list[dict]:
    summary = load_json(directory / "visualization_summary.json")
    arrays = np.load(directory / "selected_samples.npz")
    rows: list[dict] = []
    for index, metadata in enumerate(summary["samples"]):
        row = dict(metadata)
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
            row[key] = arrays[f"sample_{index}_{key}"]
        row["sparse_mask"] = row["sparse_mask"].astype(bool)
        row["valid"] = row["valid"].astype(bool)
        rows.append(row)
    return rows


def render_examples(
    rows: list[dict], output_path: Path, before_label: str, after_label: str
) -> None:
    signal_values = np.concatenate(
        [row[key][row["valid"]] for row in rows for key in ("target", "before", "after")]
    )
    signal_vmin, signal_vmax = np.percentile(signal_values, (1.0, 99.0))
    errors = np.concatenate(
        [
            np.abs(row[key] - row["target"])[row["valid"]]
            for row in rows
            for key in ("before", "after")
        ]
    )
    error_vmax = max(8.0, float(np.percentile(errors, 95.0)))
    figure, axes = plt.subplots(
        len(rows), 6, figsize=(24.5, 4.05 * len(rows)), constrained_layout=False
    )
    for row_index, row in enumerate(rows):
        input_axis = axes[row_index, 0]
        input_axis.imshow(row["building"], cmap="Greys", vmin=0.0, vmax=1.0)
        sparse_rows, sparse_cols = np.nonzero(row["sparse_mask"])
        input_axis.scatter(
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
        input_axis.scatter(
            [true_col], [true_row], marker="*", s=180, c="red", edgecolors="white",
            linewidths=0.8, label="True BS"
        )
        input_axis.scatter(
            [estimated_col], [estimated_row], marker="x", s=95, c="cyan",
            linewidths=2.0, label="Estimated BS"
        )
        input_axis.set_title("Building + 100 sparse points")
        input_axis.set_ylabel(
            f"{row['bin']}\n{row['site_id']} / az {row['azimuth_deg']:.0f} deg\n"
            f"BS error {row['location_error_px']:.2f} px"
        )
        if row_index == 0:
            input_axis.legend(loc="lower left", fontsize=8)

        for column, key, title in (
            (1, "target", "Ground truth"),
            (2, "before", f"{before_label} / RMSE {row['before_rmse_db']:.2f} dB"),
            (3, "after", f"{after_label} / RMSE {row['after_rmse_db']:.2f} dB"),
        ):
            masked = np.ma.masked_where(~row["valid"], row[key])
            signal_image = axes[row_index, column].imshow(
                masked, cmap="turbo", vmin=signal_vmin, vmax=signal_vmax
            )
            axes[row_index, column].set_title(title)

        for column, key, title in (
            (4, "before", f"{before_label} absolute error"),
            (5, "after", f"{after_label} absolute error"),
        ):
            absolute_error = np.ma.masked_where(
                ~row["valid"], np.abs(row[key] - row["target"])
            )
            error_image = axes[row_index, column].imshow(
                absolute_error, cmap="magma", vmin=0.0, vmax=error_vmax
            )
            axes[row_index, column].set_title(title)
        for column in range(6):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])

    figure.colorbar(
        signal_image,
        ax=axes[:, 1:4].ravel().tolist(),
        label="Signal strength (dB)",
        shrink=0.82,
    )
    figure.colorbar(
        error_image,
        ax=axes[:, 4:6].ravel().tolist(),
        label="Absolute error (dB)",
        shrink=0.82,
    )
    figure.suptitle(
        f"Scale-4 test examples: {before_label} vs {after_label}",
        fontsize=16,
    )
    figure.subplots_adjust(
        left=0.052, right=0.93, bottom=0.035, top=0.94, wspace=0.13, hspace=0.25
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render_training_curve(
    history_path: Path, output_path: Path, after_label: str
) -> None:
    rows = np.genfromtxt(
        history_path, delimiter=",", names=True, encoding="utf-8-sig"
    )
    rows = np.atleast_1d(rows)
    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    axis.plot(rows["epoch"], rows["train_map_rmse_db"], marker="o", label="Train")
    axis.plot(rows["epoch"], rows["val_map_rmse_db"], marker="o", label="Validation")
    best_index = int(np.argmin(rows["val_map_rmse_db"]))
    axis.scatter(
        [rows["epoch"][best_index]],
        [rows["val_map_rmse_db"][best_index]],
        s=90,
        marker="*",
        label=f"Best validation: {rows['val_map_rmse_db'][best_index]:.3f} dB",
        zorder=4,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Map RMSE (dB)")
    axis.set_title(f"Stage2B {after_label} training curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render_position_metrics(
    before: dict,
    after: dict,
    output_path: Path,
    before_label: str,
    after_label: str,
) -> None:
    labels = ["Overall", "000-064m", "064-128m", "128-192m", "192m-plus"]
    before_bins = before["position_bins"]["by_center_distance"]
    after_bins = after["position_bins"]["by_center_distance"]
    before_values = [before["final_rmse_db"]] + [before_bins[label]["final_rmse_db"] for label in labels[1:]]
    after_values = [after["final_rmse_db"]] + [after_bins[label]["final_rmse_db"] for label in labels[1:]]
    x = np.arange(len(labels))
    width = 0.37
    figure, axis = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    bars_before = axis.bar(x - width / 2, before_values, width, label=before_label, color="#7d8da6")
    bars_after = axis.bar(x + width / 2, after_values, width, label=after_label, color="#2c9c69")
    axis.bar_label(bars_before, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(bars_after, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Test RMSE (dB, lower is better)")
    axis.set_title(
        f"{before_label} vs {after_label} test RMSE by base-station distance from map center"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    lower = min(before_values + after_values)
    upper = max(before_values + after_values)
    axis.set_ylim(max(0.0, lower - 0.45), upper + 0.45)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    directory = args.visualization_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rows = load_rows(directory)
    before = load_json(args.before_evaluation.resolve())
    after = load_json(args.after_evaluation.resolve())
    slug = f"{args.before_label.lower()}_vs_{args.after_label.lower()}"
    representative_path = directory / f"representative_{slug}.png"
    training_path = directory / f"training_curve_{args.after_label.lower()}.png"
    position_path = directory / f"test_rmse_by_position_{slug}.png"
    render_examples(rows, representative_path, args.before_label, args.after_label)
    render_training_curve(args.history.resolve(), training_path, args.after_label)
    render_position_metrics(
        before, after, position_path, args.before_label, args.after_label
    )
    print(json.dumps({
        "representative": str(representative_path),
        "training_curve": str(training_path),
        "position_metrics": str(position_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-infer and visualize one fixed held-out sample with formal Scale-4 V3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_stage2b_position_encoding_scale4_v3.json"
DEFAULT_ESTIMATOR_CONFIG = SCRIPT_DIR / "config_bs_parameter_estimator_scale4_v2.json"
DEFAULT_REPORT = Path(
    r"D:\桌面\dac\experiments\bs_inversion_scale4_stage2b_v3_test_bins.json"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\桌面\dac\experiments\single_test_v3_evaluation"
)
DEFAULT_SITE_ID = "tile_000222_site03"
DEFAULT_DIRECTION = 0
OUTPUT_NAMES = (
    "single_test_v3_evaluation.png",
    "single_test_v3_metrics.json",
    "single_test_v3_arrays.npz",
)


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

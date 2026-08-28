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

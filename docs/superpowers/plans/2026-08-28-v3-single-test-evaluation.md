# V3 Single-Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-infer the fixed held-out sample `tile_000222_site03`, direction 0, with the formal Stage2B V3 checkpoint and export a truthful 2×3 evaluation figure plus reproducible numeric evidence.

**Architecture:** Add one focused CLI that validates the formal V3 provenance, locates the fixed test item, reuses the established Scale-4 model-loading and `predict_pair` forward path, computes metrics only on `valid_mask`, and exports arrays, JSON, and an opaque PNG to a new directory. Keep metric, limit, sample-selection, and output-guard logic as pure functions covered by `unittest`; exercise the real GPU/CPU inference and render as a separate integration run.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Matplotlib object-oriented API, Pillow, built-in `unittest`, existing `irt_label_pipeline` loaders.

**Execution note (2026-08-28):** Use `C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe` for both inference and rendering. The `sigmap` environment can run inference but its NumPy linear-algebra DLL terminates Matplotlib with Windows exception `0xc06d007f`. Compute the acceptance improvement from the predeclared config baseline (`baseline_final_test_rmse_db - final_rmse_db`); the report's 5.252 dB improvement refers to a different sparse baseline.

---

## File structure

- Create `projects/irt_label_pipeline/render_single_test_v3_evaluation.py`: formal V3 provenance validation, fixed-sample inference, valid-pixel metrics, 2×3 rendering, and atomic evidence export.
- Create `projects/irt_label_pipeline/tests/test_render_single_test_v3_evaluation.py`: dependency-light unit tests for valid-mask metrics, percentile limits, unique sample lookup, provenance rejection, and overwrite protection.
- Preserve `projects/irt_label_pipeline/render_single_test_training_evaluation.py`: the previous V4→V5 figure remains untouched.
- Produce, but do not add to Git, `D:\桌面\dac\experiments\single_test_v3_evaluation\single_test_v3_evaluation.png`, `single_test_v3_metrics.json`, and `single_test_v3_arrays.npz`.

### Task 1: Lock down the numeric and selection contract

**Files:**
- Create: `projects/irt_label_pipeline/tests/test_render_single_test_v3_evaluation.py`
- Create: `projects/irt_label_pipeline/render_single_test_v3_evaluation.py`

- [ ] **Step 1: Write failing tests for valid-mask metrics, limits, and exact sample selection**

Create the test module with these imports and cases:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from render_single_test_v3_evaluation import (
    compute_display_limits,
    find_sample_index,
    guard_outputs,
    sample_metrics,
    validate_provenance,
)


class V3EvaluationContractTests(unittest.TestCase):
    def test_sample_metrics_use_only_valid_pixels(self) -> None:
        target = np.asarray([[0.0, 10.0], [20.0, 999.0]], dtype=np.float32)
        prediction = np.asarray([[1.0, 8.0], [24.0, -999.0]], dtype=np.float32)
        valid = np.asarray([[True, True], [True, False]])

        metrics = sample_metrics(target, prediction, valid)

        np.testing.assert_allclose(metrics["mae_db"], 7.0 / 3.0)
        np.testing.assert_allclose(metrics["rmse_db"], np.sqrt(7.0))
        np.testing.assert_allclose(metrics["bias_db"], 1.0)
        np.testing.assert_allclose(metrics["p90_absolute_error_db"], 3.6)
        np.testing.assert_allclose(metrics["max_absolute_error_db"], 4.0)
        self.assertEqual(metrics["valid_pixels"], 3)

    def test_display_limits_ignore_invalid_extremes(self) -> None:
        target = np.asarray([[0.0, 10.0], [20.0, 9999.0]], dtype=np.float32)
        prediction = np.asarray([[2.0, 8.0], [24.0, -9999.0]], dtype=np.float32)
        valid = np.asarray([[True, True], [True, False]])

        limits = compute_display_limits(target, prediction, valid)

        expected_signal = np.percentile([0.0, 10.0, 20.0, 2.0, 8.0, 24.0], [1.0, 99.0])
        np.testing.assert_allclose(limits["signal_db"], expected_signal)
        np.testing.assert_allclose(limits["error_p99_db"], np.percentile([2.0, 2.0, 4.0], 99.0))

    def test_find_sample_index_requires_one_exact_site_direction(self) -> None:
        rows = [
            ({"site_id": "tile_000222_site03"}, 0),
            ({"site_id": "tile_000222_site03"}, 1),
            ({"site_id": "tile_000999_site00"}, 0),
        ]
        self.assertEqual(find_sample_index(rows, "tile_000222_site03", 0), 0)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_sample_index(rows, "missing", 0)
```

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -m unittest projects.irt_label_pipeline.tests.test_render_single_test_v3_evaluation -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'render_single_test_v3_evaluation'`.

- [ ] **Step 3: Add the CLI constants and pure numeric functions**

Create `render_single_test_v3_evaluation.py` with the fixed defaults, `sample_metrics`, `compute_display_limits`, and `find_sample_index`:

```python
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
DEFAULT_REPORT = Path(r"D:\桌面\dac\experiments\bs_inversion_scale4_stage2b_v3_test_bins.json")
DEFAULT_OUTPUT_DIR = Path(r"D:\桌面\dac\experiments\single_test_v3_evaluation")
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
    if not np.isfinite(target_values).all() or not np.isfinite(prediction_values).all():
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
    error = np.abs(prediction[valid].astype(np.float64) - target[valid].astype(np.float64))
    if signal.size == 0 or not np.isfinite(signal).all() or not np.isfinite(error).all():
        raise ValueError("display-limit inputs must contain finite valid pixels")
    signal_vmin, signal_vmax = np.percentile(signal, (1.0, 99.0))
    return {
        "signal_db": [float(signal_vmin), float(signal_vmax)],
        "error_p99_db": max(float(np.percentile(error, 99.0)), np.finfo(float).eps),
    }


def find_sample_index(
    rows: Sequence[tuple[dict[str, Any], int]],
    site_id: str,
    direction: int,
) -> int:
    matches = [
        index
        for index, (entry, row_direction) in enumerate(rows)
        if entry["site_id"] == site_id and int(row_direction) == int(direction)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one sample for {site_id} direction {direction}, found {len(matches)}"
        )
    return matches[0]
```

- [ ] **Step 4: Run the tests and confirm only the not-yet-implemented imports fail**

Run the same `unittest` command.

Expected: import error naming `guard_outputs` or `validate_provenance`; the numeric functions themselves are now defined.

### Task 2: Enforce provenance and explicit-overwrite safety

**Files:**
- Modify: `projects/irt_label_pipeline/tests/test_render_single_test_v3_evaluation.py`
- Modify: `projects/irt_label_pipeline/render_single_test_v3_evaluation.py`

- [ ] **Step 1: Add failing tests for hash/path validation and output protection**

Append these methods inside `V3EvaluationContractTests`:

```python
    def test_validate_provenance_rejects_stage2b_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage1_dir = root / "stage1"
            stage2a_dir = root / "stage2a"
            stage1_dir.mkdir()
            stage2a_dir.mkdir()
            paths = {
                "stage1": stage1_dir / "best.pt",
                "stage2a": stage2a_dir / "best.pt",
                "stage2b": root / "stage2b.pt",
                "estimator": root / "estimator.pt",
            }
            for name, path in paths.items():
                path.write_bytes(name.encode("ascii"))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            config = {
                "stage1_model_dir": str(stage1_dir),
                "stage2a_model_dir": str(stage2a_dir),
                "estimator_checkpoint": str(paths["estimator"]),
                "provenance": {
                    "stage1_checkpoint_sha256": digest(paths["stage1"]),
                    "stage2a_checkpoint_sha256": digest(paths["stage2a"]),
                    "estimator_checkpoint_sha256": digest(paths["estimator"]),
                },
            }
            report = {
                "stage1_checkpoint": str(paths["stage1"]),
                "stage2a_checkpoint": str(paths["stage2a"]),
                "stage2b_checkpoint": str(paths["stage2b"]),
                "stage2b_checkpoint_sha256": "0" * 64,
                "estimator_checkpoint": str(paths["estimator"]),
                "estimator_checkpoint_sha256": digest(paths["estimator"]),
            }
            with self.assertRaisesRegex(ValueError, "stage2b.*SHA-256"):
                validate_provenance(config, report, paths)

    def test_guard_outputs_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "single_test_v3_metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                guard_outputs(output_dir, overwrite=False)
            guard_outputs(output_dir, overwrite=True)
```

- [ ] **Step 2: Run the tests and verify the new safety cases fail**

Run the focused `unittest` command.

Expected: `ImportError` or failing cases because `validate_provenance` and `guard_outputs` are not implemented.

- [ ] **Step 3: Implement source hashing, exact report-path checks, and overwrite guard**

Add the following functions below `find_sample_index`:

```python
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
            raise ValueError(f"{name} checkpoint path does not match the formal V3 report")
        if name in expected_config_paths and path.resolve() != expected_config_paths[name].resolve():
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
            raise ValueError(f"{name} checkpoint SHA-256 does not match formal provenance")
    config_estimator_hash = config["provenance"]["estimator_checkpoint_sha256"]
    if actual["estimator"].lower() != str(config_estimator_hash).lower():
        raise ValueError("estimator checkpoint SHA-256 does not match the V3 config")
    return actual


def guard_outputs(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output exists; rerun with --overwrite: {joined}")
```

- [ ] **Step 4: Run the unit suite and verify all contract tests pass**

Run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -m unittest projects.irt_label_pipeline.tests.test_render_single_test_v3_evaluation -v
```

Expected: five tests, all `ok`.

- [ ] **Step 5: Commit the contract and safety layer**

```powershell
git add -- 'projects/irt_label_pipeline/render_single_test_v3_evaluation.py' 'projects/irt_label_pipeline/tests/test_render_single_test_v3_evaluation.py'
git commit -m 'test: define V3 evaluation evidence contract'
```

### Task 3: Add fixed-sample formal V3 inference and evidence serialization

**Files:**
- Modify: `projects/irt_label_pipeline/render_single_test_v3_evaluation.py`
- Modify: `projects/irt_label_pipeline/tests/test_render_single_test_v3_evaluation.py`

- [ ] **Step 1: Add a failing test for the fixed sample-array contract**

Import `validate_sample_arrays` and append this case:

```python
    def test_validate_sample_arrays_checks_shape_angle_and_sparse_count(self) -> None:
        arrays = {
            "building": np.zeros((128, 128), dtype=np.float32),
            "sparse_mask": np.zeros((128, 128), dtype=bool),
            "sparse_db": np.zeros((128, 128), dtype=np.float32),
            "target_db": np.zeros((128, 128), dtype=np.float32),
            "valid_mask": np.ones((128, 128), dtype=bool),
            "prediction_db": np.ones((128, 128), dtype=np.float32),
        }
        arrays["sparse_mask"].flat[:100] = True
        validate_sample_arrays(arrays, azimuth_deg=12.0)
        arrays["sparse_mask"].flat[100] = True
        with self.assertRaisesRegex(ValueError, "100 sparse"):
            validate_sample_arrays(arrays, azimuth_deg=12.0)
```

- [ ] **Step 2: Run the focused suite and verify it fails on the missing validator**

Expected: import error naming `validate_sample_arrays`.

- [ ] **Step 3: Implement argument parsing and sample-array validation**

Add `parse_args`, JSON loading, and the validator:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--estimator-config", type=Path, default=DEFAULT_ESTIMATOR_CONFIG)
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


def validate_sample_arrays(arrays: dict[str, np.ndarray], azimuth_deg: float) -> None:
    required = ("building", "sparse_mask", "sparse_db", "target_db", "valid_mask", "prediction_db")
    for name in required:
        if arrays[name].shape != (128, 128):
            raise ValueError(f"{name} must have shape (128, 128), got {arrays[name].shape}")
    if int(arrays["sparse_mask"].sum()) != 100:
        raise ValueError("fixed sample must contain exactly 100 sparse points")
    if not math.isclose(float(azimuth_deg), 12.0, abs_tol=1e-5):
        raise ValueError(f"fixed sample direction 0 must have azimuth 12 degrees, got {azimuth_deg}")
    if int(arrays["valid_mask"].sum()) == 0:
        raise ValueError("valid mask contains no pixels")
    for name in ("building", "target_db", "prediction_db"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite required values")
    if not np.isfinite(arrays["sparse_db"][arrays["sparse_mask"]]).all():
        raise ValueError("sparse signal values must be finite")
```

- [ ] **Step 4: Implement the established formal V3 forward path without using the stale cache**

Add a lazy-imported `run_formal_v3_inference` function so unit tests do not initialize CUDA merely by importing the module:

```python
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
    from visualize_scale4_pipeline_examples import load_estimator, load_pair, load_stage2b, predict_pair

    config = load_json(config_path.resolve())
    report = load_json(report_path.resolve())
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
    iso_mean_db, iso_std_db = float(iso_stats["mean_db"]), float(iso_stats["std_db"])
    ss_mean_db, ss_std_db = float(ss_stats["mean_db"]), float(ss_stats["std_db"])
    dataset = BSInversionPilotDataset(
        config["dataset_root"], split="test", seed=int(config["seed"]), augment=False,
        signal_mean_db=ss_mean_db, signal_std_db=ss_std_db,
    )
    sample_index = find_sample_index(dataset.rows, site_id, direction)
    batch = default_collate([dataset[sample_index]])
    estimator, estimator_mean, estimator_std, estimator_power_mean, estimator_power_std = load_estimator(
        estimator_config_path.resolve(), paths["estimator"], device, channels_last
    )
    stage1, stage2a, pair_metadata, tx_values = load_pair(config_path.resolve(), device, channels_last)
    stage2b, stage2b_config = load_stage2b(paths["stage2b"], device, channels_last)
    with torch.inference_mode():
        prediction, estimated_row_col = predict_pair(
            batch,
            stage1=stage1, stage2a=stage2a, pair_metadata=pair_metadata,
            tx_values=tx_values, stage2b=stage2b, stage2b_config=stage2b_config,
            estimator=estimator, estimator_mean_db=estimator_mean,
            estimator_std_db=estimator_std,
            estimator_power_mean_db=estimator_power_mean,
            estimator_power_std_db=estimator_power_std,
            iso_mean_db=iso_mean_db, iso_std_db=iso_std_db,
            ss_mean_db=ss_mean_db, ss_std_db=ss_std_db,
            device=device, amp_enabled=amp_enabled, channels_last=channels_last,
        )
    sparse_mask = batch["sparse_mask"][0, 0].numpy().astype(bool)
    arrays = {
        "building": batch["building"][0, 0].numpy().astype(np.float32),
        "sparse_mask": sparse_mask,
        "sparse_db": (
            batch["sparse_ss_norm"][0, 0].numpy() * ss_std_db + ss_mean_db
        ).astype(np.float32),
        "target_db": batch["target_map_db"][0, 0].numpy().astype(np.float32),
        "valid_mask": batch["valid_mask"][0, 0].numpy().astype(bool),
        "prediction_db": prediction[0, 0].detach().cpu().numpy().astype(np.float32),
        "true_row_col_px": batch["target_row_col_px"][0].numpy().astype(np.float32),
        "estimated_row_col_px": estimated_row_col[0].detach().cpu().numpy().astype(np.float32),
    }
    azimuth_deg = float(batch["target_azimuth_deg"][0])
    validate_sample_arrays(arrays, azimuth_deg)
    runtime = {
        "device": str(device), "amp_enabled": amp_enabled, "channels_last": channels_last,
        "checkpoint_sha256": hashes, "sample_index": sample_index,
        "signal_mean_db": ss_mean_db, "signal_std_db": ss_std_db,
    }
    sample = {
        "site_id": site_id, "direction_index": int(direction), "azimuth_deg": azimuth_deg,
        "tile": batch["tile"][0], "site_index": int(batch["site_index"][0]),
    }
    return sample, arrays, {"config": config, "report": report, "runtime": runtime}
```

- [ ] **Step 5: Add signed/absolute errors and atomic NPZ/JSON writers**

Add:

```python
def augment_evidence_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
```

- [ ] **Step 6: Run unit tests and syntax compilation**

Run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -m unittest projects.irt_label_pipeline.tests.test_render_single_test_v3_evaluation -v
```

Expected: six tests, all `ok`.

Run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -m py_compile 'projects\irt_label_pipeline\render_single_test_v3_evaluation.py'
```

Expected: exit code 0 and no output.

- [ ] **Step 7: Commit fixed-sample V3 inference**

```powershell
git add -- 'projects/irt_label_pipeline/render_single_test_v3_evaluation.py' 'projects/irt_label_pipeline/tests/test_render_single_test_v3_evaluation.py'
git commit -m 'feat: re-infer fixed test sample with formal V3'
```

### Task 4: Render the approved 2×3 figure and metrics panel

**Files:**
- Modify: `projects/irt_label_pipeline/render_single_test_v3_evaluation.py`

- [ ] **Step 1: Add coordinate and map helpers with explicit missing-data styling**

Add helpers that convert pixel coordinates to metres and apply identical map axes:

```python
def row_col_to_xy_m(row_col: np.ndarray) -> tuple[float, float]:
    return (float(row_col[1]) + 0.5) * 4.0, 512.0 - (float(row_col[0]) + 0.5) * 4.0


def style_map_axis(axis: Any) -> None:
    axis.set(xlabel="East (m)", ylabel="North (m)", xlim=(0.0, 512.0), ylim=(0.0, 512.0))
    axis.set_xticks([0, 128, 256, 384, 512])
    axis.set_yticks([0, 128, 256, 384, 512])
    axis.set_aspect("equal")
```

- [ ] **Step 2: Implement the six-panel renderer**

Implement `render_figure(arrays, sample, metrics, aggregate, limits, figure_path)` with Matplotlib's object-oriented API and `layout="constrained"`:

```python
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

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.titlesize": 11, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    signal_vmin, signal_vmax = limits["signal_db"]
    error_limit = float(limits["error_p99_db"])
    valid = arrays["valid_mask"]
    signal_cmap = mpl.colormaps["viridis"].copy()
    signal_cmap.set_bad("#A6A6A6")
    signed_cmap = mpl.colormaps["RdBu_r"].copy()
    signed_cmap.set_bad("#A6A6A6")
    absolute_cmap = mpl.colormaps["magma"].copy()
    absolute_cmap.set_bad("#A6A6A6")
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 8.6), layout="constrained")
    figure.get_layout_engine().set(rect=(0.02, 0.055, 0.98, 0.94))

    input_axis = axes[0, 0]
    input_axis.imshow(arrays["building"], cmap="Greys", vmin=0.0, vmax=1.0,
                      extent=(0, 512, 0, 512), origin="upper", interpolation="nearest")
    sparse_rows, sparse_cols = np.nonzero(arrays["sparse_mask"])
    input_axis.scatter((sparse_cols + 0.5) * 4.0, 512.0 - (sparse_rows + 0.5) * 4.0,
                       c=arrays["sparse_db"][arrays["sparse_mask"]], s=16,
                       cmap=signal_cmap, vmin=signal_vmin, vmax=signal_vmax,
                       marker="o", edgecolors="white", linewidths=0.25)
    true_x, true_y = row_col_to_xy_m(arrays["true_row_col_px"])
    estimated_x, estimated_y = row_col_to_xy_m(arrays["estimated_row_col_px"])
    input_axis.scatter(true_x, true_y, marker="*", s=190, color="#D55E00",
                       edgecolors="white", linewidths=0.9, zorder=4)
    input_axis.scatter(estimated_x, estimated_y, marker="x", s=90, color="#0072B2",
                       linewidths=2.0, zorder=5)
    input_axis.legend(handles=[
        Line2D([0], [0], marker="*", linestyle="", markersize=11,
               markerfacecolor="#D55E00", markeredgecolor="white", label="True BS"),
        Line2D([0], [0], marker="x", linestyle="", markersize=8,
               color="#0072B2", markeredgewidth=2, label="Estimated BS"),
    ], loc="lower left", fontsize=8, framealpha=0.9)
    input_axis.set_title("A  Input: buildings + 100 sparse samples", loc="left")

    signal_images = []
    for axis, key, title in (
        (axes[0, 1], "target_db", "B  Ground truth"),
        (axes[0, 2], "prediction_db",
         f"C  V3 prediction | RMSE {metrics['rmse_db']:.2f}, MAE {metrics['mae_db']:.2f} dB"),
    ):
        image = axis.imshow(np.ma.masked_where(~valid, arrays[key]), cmap=signal_cmap,
                            vmin=signal_vmin, vmax=signal_vmax,
                            extent=(0, 512, 0, 512), origin="upper", interpolation="nearest")
        signal_images.append(image)
        axis.set_title(title, loc="left")

    signed_image = axes[1, 0].imshow(
        np.ma.masked_where(~valid, arrays["signed_error_db"]), cmap=signed_cmap,
        norm=TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit),
        extent=(0, 512, 0, 512), origin="upper", interpolation="nearest")
    axes[1, 0].set_title(f"D  Signed error | bias {metrics['bias_db']:+.2f} dB", loc="left")
    absolute_image = axes[1, 1].imshow(
        np.ma.masked_where(~valid, arrays["absolute_error_db"]), cmap=absolute_cmap,
        vmin=0.0, vmax=error_limit, extent=(0, 512, 0, 512),
        origin="upper", interpolation="nearest")
    axes[1, 1].set_title(
        f"E  Absolute error | P90 {metrics['p90_absolute_error_db']:.2f}, max {metrics['max_absolute_error_db']:.2f} dB",
        loc="left")

    summary_axis = axes[1, 2]
    summary_axis.axis("off")
    gate = aggregate["gate"]
    summary_axis.set_title("F  Numeric summary", loc="left")
    summary_axis.text(0.03, 0.95,
        "Fixed held-out sample\n"
        f"Valid pixels: {metrics['valid_pixels']:,}\n"
        f"MAE: {metrics['mae_db']:.3f} dB\nRMSE: {metrics['rmse_db']:.3f} dB\n"
        f"Bias: {metrics['bias_db']:+.3f} dB\nP90 |error|: {metrics['p90_absolute_error_db']:.3f} dB\n"
        f"Max |error|: {metrics['max_absolute_error_db']:.3f} dB\nPearson r: {metrics['pearson_r']:.3f}\n\n"
        "Complete test (4,096 maps)\n"
        f"MAE: {aggregate['mae_db']:.3f} dB\nRMSE: {aggregate['rmse_db']:.3f} dB\n\n"
        f"✗ Gate not passed\nRMSE must be ≤ {gate['maximum_rmse_db']:.2f} dB\n"
        f"Improvement must be ≥ {gate['minimum_improvement_db']:.2f} dB",
        transform=summary_axis.transAxes, va="top", ha="left", fontsize=10,
        linespacing=1.25, color="#222222")
    summary_axis.text(0.03, 0.15, "Invalid pixels are shown in gray and excluded from metrics.",
                      transform=summary_axis.transAxes, va="top", fontsize=8.5, color="#555555")

    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        style_map_axis(axis)
    figure.colorbar(signal_images[-1], ax=axes[0, :], orientation="horizontal",
                    fraction=0.055, pad=0.08, extend="both",
                    label="Signal strength (dB); shared 1st–99th percentile display range")
    figure.colorbar(signed_image, ax=axes[1, 0], fraction=0.047, pad=0.04,
                    extend="both", label="V3 prediction - truth (dB); P99 limits")
    figure.colorbar(absolute_image, ax=axes[1, 1], fraction=0.047, pad=0.04,
                    extend="max", label="Absolute error (dB); P99 upper limit")
    figure.suptitle("Scale-4 V3 signal-map evaluation on one fixed held-out test sample",
                    fontsize=16, fontweight="bold")
    figure.text(0.5, 0.012,
                f"Sample {sample['site_id']} | direction {sample['direction_index']} | "
                f"azimuth {sample['azimuth_deg']:.0f}° | 128×128 pixels | 4 m resolution",
                ha="center", va="bottom", fontsize=9.5)
    figure.savefig(figure_path, dpi=220, facecolor="white", transparent=False)
    plt.close(figure)
    with Image.open(figure_path) as rendered:
        rendered.convert("RGB").save(figure_path, dpi=(220, 220))
```

- [ ] **Step 3: Implement aggregate gate reporting and the CLI orchestration**

Add `main` so the report values are checked exactly, all three outputs are written to the new directory, and the JSON records alt text, limits, hashes, paths, sample identity, gate checks, and PNG metadata. The gate must be computed as both `final_rmse_db <= maximum_test_rmse_db` and `(baseline_final_test_rmse_db - final_rmse_db) >= minimum_improvement_db`; for the formal report it must evaluate to `False`.

The entry point must follow this exact order:

```python
def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    guard_outputs(output_dir, args.overwrite)
    sample, arrays, provenance = run_formal_v3_inference(
        args.config, args.estimator_config, args.report, args.site_id, args.direction
    )
    arrays = augment_evidence_arrays(arrays)
    metrics = sample_metrics(arrays["target_db"], arrays["prediction_db"], arrays["valid_mask"])
    if not all(np.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("sample metrics must all be finite")
    limits = compute_display_limits(arrays["target_db"], arrays["prediction_db"], arrays["valid_mask"])
    config, report = provenance["config"], provenance["report"]
    if int(report["directional_samples"]) != 4096 or int(report["sites"]) != 1024:
        raise ValueError("formal V3 report must describe 1,024 sites and 4,096 maps")
    aggregate = {
        "sites": int(report["sites"]), "maps": int(report["directional_samples"]),
        "mae_db": float(report["final_mae_db"]), "rmse_db": float(report["final_rmse_db"]),
        "gate": {
            "maximum_rmse_db": float(config["acceptance"]["maximum_test_rmse_db"]),
            "minimum_improvement_db": float(config["acceptance"]["minimum_improvement_db"]),
            "actual_improvement_db": float(config["acceptance"]["baseline_final_test_rmse_db"]) - float(report["final_rmse_db"]),
        },
    }
    aggregate["gate"]["passed"] = bool(
        aggregate["rmse_db"] <= aggregate["gate"]["maximum_rmse_db"]
        and aggregate["gate"]["actual_improvement_db"] >= aggregate["gate"]["minimum_improvement_db"]
    )
    if aggregate["gate"]["passed"]:
        raise ValueError("formal V3 gate was expected to be unpassed; report/config changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / OUTPUT_NAMES[0]
    arrays_path = output_dir / OUTPUT_NAMES[2]
    render_figure(arrays, sample, metrics, aggregate, limits, figure_path)
    atomic_save_npz(arrays_path, arrays)
    from PIL import Image
    with Image.open(figure_path) as image:
        figure_metadata = {"mode": image.mode, "size_px": list(image.size), "dpi": list(image.info.get("dpi", (0, 0)))}
    result = {
        "version": 1, "status": "completed", "sample": sample,
        "sample_metrics": metrics, "complete_test": aggregate,
        "display_limits": limits,
        "valid_mask_definition": "directional_valid_mask; buildings and unhit outdoor pixels excluded",
        "provenance": {
            "config": str(args.config.resolve()), "report": str(args.report.resolve()),
            "dataset_root": str(config["dataset_root"]),
            "checkpoints": {
                name: {"path": str(Path(report[f"{name}_checkpoint"]).resolve()),
                       "sha256": provenance["runtime"]["checkpoint_sha256"][name]}
                for name in ("stage1", "stage2a", "stage2b", "estimator")
            },
            "runtime": {key: value for key, value in provenance["runtime"].items() if key != "checkpoint_sha256"},
        },
        "outputs": {"figure": str(figure_path), "arrays": str(arrays_path), "figure_metadata": figure_metadata},
        "alt_text": (
            "Six-panel evaluation of formal Scale-4 V3 on held-out sample tile_000222_site03: "
            "input buildings and 100 sparse measurements, ground truth, V3 prediction, signed error, "
            "absolute error, and numeric summary. Invalid pixels are gray. The complete-test RMSE "
            f"is {aggregate['rmse_db']:.3f} dB, so the predeclared gate is not passed."
        ),
    }
    atomic_write_json(output_dir / OUTPUT_NAMES[1], result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests and syntax compilation again**

Expected: six tests pass; `py_compile` exits 0.

- [ ] **Step 5: Commit the approved renderer**

```powershell
git add -- 'projects/irt_label_pipeline/render_single_test_v3_evaluation.py'
git commit -m 'feat: render V3 single-test evaluation figure'
```

### Task 5: Run formal inference and audit the delivered evidence

**Files:**
- Generate: `D:\桌面\dac\experiments\single_test_v3_evaluation\single_test_v3_evaluation.png`
- Generate: `D:\桌面\dac\experiments\single_test_v3_evaluation\single_test_v3_metrics.json`
- Generate: `D:\桌面\dac\experiments\single_test_v3_evaluation\single_test_v3_arrays.npz`

- [ ] **Step 1: Execute the fixed-sample formal V3 inference**

Run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' 'projects\irt_label_pipeline\render_single_test_v3_evaluation.py' --overwrite
```

Expected: exit code 0; printed JSON has `sample.site_id = tile_000222_site03`, `sample.direction_index = 0`, `sample.azimuth_deg = 12.0`, `sample_metrics.valid_pixels = 6214`, finite `max_absolute_error_db`, `complete_test.maps = 4096`, `complete_test.rmse_db = 6.946207715784639`, `complete_test.mae_db = 4.101779706371985`, the configured RMSE threshold `6.85`, and `complete_test.gate.passed = false`.

- [ ] **Step 2: Independently recompute every sample metric from the saved NPZ**

Run:

```powershell
$check = @'
import json
import math
from pathlib import Path
import sys
import numpy as np

pipeline = Path(r"C:\Users\pc\Documents\大创\projects\irt_label_pipeline")
sys.path.insert(0, str(pipeline))
from render_single_test_v3_evaluation import sample_metrics

root = Path(r"D:\桌面\dac\experiments\single_test_v3_evaluation")
saved = json.loads((root / "single_test_v3_metrics.json").read_text(encoding="utf-8"))
with np.load(root / "single_test_v3_arrays.npz", allow_pickle=False) as arrays:
    recomputed = sample_metrics(arrays["target_db"], arrays["prediction_db"], arrays["valid_mask"].astype(bool))
    for key, expected in saved["sample_metrics"].items():
        assert math.isclose(float(recomputed[key]), float(expected), rel_tol=0.0, abs_tol=1e-9), key
    np.testing.assert_allclose(arrays["signed_error_db"], arrays["prediction_db"] - arrays["target_db"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(arrays["absolute_error_db"], np.abs(arrays["signed_error_db"]), rtol=0.0, atol=0.0)
    assert int(arrays["sparse_mask"].sum()) == 100
print("metric recomputation: PASS")
'@
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -c $check
```

Expected: prints `metric recomputation: PASS`; no mismatches.

- [ ] **Step 3: Verify aggregate evidence and formal checkpoint identity**

Run:

```powershell
$check = @'
import json
from pathlib import Path

root = Path(r"D:\桌面\dac\experiments\single_test_v3_evaluation")
metrics = json.loads((root / "single_test_v3_metrics.json").read_text(encoding="utf-8"))
report = json.loads(Path(r"D:\桌面\dac\experiments\bs_inversion_scale4_stage2b_v3_test_bins.json").read_text(encoding="utf-8"))
assert metrics["complete_test"]["rmse_db"] == report["final_rmse_db"]
assert metrics["complete_test"]["mae_db"] == report["final_mae_db"]
assert metrics["complete_test"]["maps"] == report["directional_samples"] == 4096
assert metrics["provenance"]["checkpoints"]["stage2b"]["sha256"] == report["stage2b_checkpoint_sha256"]
assert metrics["complete_test"]["gate"]["passed"] is False
print("aggregate/provenance: PASS")
'@
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -c $check
```

Expected: prints `aggregate/provenance: PASS`.

- [ ] **Step 4: Inspect raster metadata at the delivered file**

Run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -c "from PIL import Image; p=r'D:\桌面\dac\experiments\single_test_v3_evaluation\single_test_v3_evaluation.png'; im=Image.open(p); print(im.mode, im.size, im.info.get('dpi')); assert im.mode == 'RGB'; assert min(im.info.get('dpi',(0,0))) >= 180"
```

Expected: `RGB`, a nonzero pixel size near 3168×1892, DPI at least 180, and exit code 0.

- [ ] **Step 5: Visually inspect the PNG at original detail**

Open the local PNG with the image viewer and verify: all six titles are legible; B and C share one signal scale; invalid regions are gray rather than white; signed error is centered at 0; colorbar extensions are visible; East/North axes are equal; true/estimated BS markers are distinguishable by shape; the F panel says `Gate not passed`; no title, label, colorbar, footer, or metric line is clipped or overlapping.

Expected: no visual defects. If a defect is found, adjust only layout/styling, rerun Tasks 4–5 checks, and retain unchanged numeric arrays.

- [ ] **Step 6: Run the final regression suite and review the diff**

Run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python312\python.exe' -m unittest projects.irt_label_pipeline.tests.test_render_single_test_v3_evaluation -v
```

Expected: all tests pass.

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Review `git status --short` and stage only the new V3 script/test/plan; preserve every unrelated user change.

- [ ] **Step 7: Commit any final QA-only adjustments**

```powershell
git add -- 'projects/irt_label_pipeline/render_single_test_v3_evaluation.py' 'projects/irt_label_pipeline/tests/test_render_single_test_v3_evaluation.py' 'docs/superpowers/plans/2026-08-28-v3-single-test-evaluation.md'
git commit -m 'test: verify V3 single-test evaluation artifacts'
```

Expected: commit contains only the V3 evaluation implementation, tests, and this plan; generated evidence remains in the separate experiment directory.

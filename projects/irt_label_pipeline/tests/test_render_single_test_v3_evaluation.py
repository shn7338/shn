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

from render_single_test_v3_evaluation import (  # noqa: E402
    atomic_save_npz,
    atomic_write_json,
    augment_evidence_arrays,
    build_aggregate_result,
    compute_display_limits,
    find_sample_index,
    guard_outputs,
    render_figure,
    row_col_to_xy_m,
    sample_metrics,
    validate_sample_arrays,
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

        expected_signal = np.percentile(
            [0.0, 10.0, 20.0, 2.0, 8.0, 24.0], [1.0, 99.0]
        )
        np.testing.assert_allclose(limits["signal_db"], expected_signal)
        np.testing.assert_allclose(
            limits["error_p99_db"], np.percentile([2.0, 2.0, 4.0], 99.0)
        )

    def test_find_sample_index_requires_one_exact_site_direction(self) -> None:
        rows = [
            ({"site_id": "tile_000222_site03"}, 0),
            ({"site_id": "tile_000222_site03"}, 1),
            ({"site_id": "tile_000999_site00"}, 0),
        ]
        self.assertEqual(find_sample_index(rows, "tile_000222_site03", 0), 0)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_sample_index(rows, "missing", 0)

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

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

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
            (output_dir / "single_test_v3_metrics.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                guard_outputs(output_dir, overwrite=False)
            guard_outputs(output_dir, overwrite=True)

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

    def test_augment_evidence_arrays_preserves_sources_and_adds_errors(self) -> None:
        arrays = {
            "target_db": np.asarray([[1.0, 4.0]], dtype=np.float32),
            "prediction_db": np.asarray([[3.0, 1.0]], dtype=np.float32),
        }

        augmented = augment_evidence_arrays(arrays)

        np.testing.assert_array_equal(
            augmented["signed_error_db"], [[2.0, -3.0]]
        )
        np.testing.assert_array_equal(
            augmented["absolute_error_db"], [[2.0, 3.0]]
        )
        self.assertNotIn("signed_error_db", arrays)

    def test_atomic_writers_round_trip_json_and_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "evidence.json"
            npz_path = root / "evidence.npz"

            atomic_write_json(json_path, {"status": "完成", "value": 3})
            atomic_save_npz(
                npz_path,
                {"value": np.asarray([1.0, 2.0], dtype=np.float32)},
            )

            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["status"],
                "完成",
            )
            with np.load(npz_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["value"], [1.0, 2.0])
            self.assertFalse((root / "evidence.json.tmp").exists())
            self.assertFalse((root / "evidence.npz.tmp").exists())

    def test_row_col_to_xy_m_uses_pixel_centres_and_north_up(self) -> None:
        self.assertEqual(
            row_col_to_xy_m(np.asarray([0.0, 0.0], dtype=np.float32)),
            (2.0, 510.0),
        )
        self.assertEqual(
            row_col_to_xy_m(np.asarray([127.0, 127.0], dtype=np.float32)),
            (510.0, 2.0),
        )

    def test_render_figure_writes_opaque_rgb_png(self) -> None:
        from PIL import Image

        rows, cols = np.indices((128, 128), dtype=np.float32)
        target = -100.0 + rows * 0.1 + cols * 0.05
        prediction = target + np.sin(cols / 12.0).astype(np.float32)
        valid = np.ones((128, 128), dtype=bool)
        valid[:8, :8] = False
        sparse_mask = np.zeros((128, 128), dtype=bool)
        sparse_mask.flat[:100] = True
        arrays = augment_evidence_arrays(
            {
                "building": np.zeros((128, 128), dtype=np.float32),
                "sparse_mask": sparse_mask,
                "sparse_db": target.copy(),
                "target_db": target,
                "valid_mask": valid,
                "prediction_db": prediction,
                "true_row_col_px": np.asarray([20.0, 30.0], dtype=np.float32),
                "estimated_row_col_px": np.asarray(
                    [22.0, 33.0], dtype=np.float32
                ),
            }
        )
        metrics = sample_metrics(target, prediction, valid)
        limits = compute_display_limits(target, prediction, valid)
        aggregate = {
            "mae_db": 4.101779706371985,
            "rmse_db": 6.946207715784639,
            "gate": {
                "maximum_rmse_db": 6.85,
                "minimum_improvement_db": 0.1,
                "actual_improvement_db": 0.022,
                "passed": False,
            },
        }
        sample = {
            "site_id": "tile_000222_site03",
            "direction_index": 0,
            "azimuth_deg": 12.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figure.png"

            render_figure(arrays, sample, metrics, aggregate, limits, output)

            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (3168, 1892))
                self.assertGreaterEqual(min(image.info["dpi"]), 180.0)

    def test_aggregate_gate_uses_predeclared_v3_acceptance_baseline(self) -> None:
        config = {
            "acceptance": {
                "baseline_final_test_rmse_db": 6.968589598108601,
                "maximum_test_rmse_db": 6.85,
                "minimum_improvement_db": 0.1,
            }
        }
        report = {
            "sites": 1024,
            "directional_samples": 4096,
            "final_mae_db": 4.101779706371985,
            "final_rmse_db": 6.946207715784639,
            "final_rmse_improvement_vs_baseline_db": 5.252158639535005,
        }

        aggregate = build_aggregate_result(config, report)

        self.assertAlmostEqual(
            aggregate["gate"]["actual_improvement_db"],
            6.968589598108601 - 6.946207715784639,
        )
        self.assertFalse(aggregate["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()

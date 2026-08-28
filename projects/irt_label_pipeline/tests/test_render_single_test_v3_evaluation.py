from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from render_single_test_v3_evaluation import (  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()

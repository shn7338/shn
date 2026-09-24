#!/usr/bin/env python3
"""Create a compact audit report for the latent-BS Pilot experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_ROOT = Path(r"D:\桌面\dac\03_runs\experiments")
DEFAULT_MODEL_ROOT = Path(r"D:\桌面\dac\03_runs\models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "bs_inversion_pilot_v2_summary.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root = args.experiment_root
    paths = {
        "original_oracle": root / "bs_inversion_pilot_oracle_pipeline_v1_rerun.json",
        "original_estimated": root / "bs_inversion_pilot_estimated_pipeline_v1.json",
        "stage2b_v1_oracle": root
        / "bs_inversion_pilot_oracle_pipeline_finetuned_stage2b_v1.json",
        "stage2b_v1_estimated": root
        / "bs_inversion_pilot_estimated_pipeline_finetuned_stage2b_v1.json",
        "adapted_iso_old_stage2b": root
        / "bs_inversion_pilot_adapted_iso_old_stage2b_v1.json",
        "final_oracle": root
        / "bs_inversion_pilot_oracle_pipeline_finetuned_all_v2.json",
        "final_estimated": root
        / "bs_inversion_pilot_estimated_pipeline_finetuned_all_v2.json",
        "estimator": args.model_root
        / "bs_parameter_estimator_pilot_v1"
        / "test_metrics.json",
        "stage2b_v1_training": args.model_root
        / "stage2b_bs_inversion_pilot_finetune_v1"
        / "training_summary.json",
        "isotropic_training": args.model_root
        / "stage2a_bs_inversion_pilot_finetune_v1"
        / "training_summary.json",
        "stage2b_v2_training": args.model_root
        / "stage2b_bs_inversion_pilot_finetune_v2"
        / "training_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input reports:\n" + "\n".join(missing))
    reports = {name: load_json(path) for name, path in paths.items()}
    original_oracle = reports["original_oracle"]
    original_estimated = reports["original_estimated"]
    stage2b_v1_oracle = reports["stage2b_v1_oracle"]
    stage2b_v1_estimated = reports["stage2b_v1_estimated"]
    adapted_iso_old_stage2b = reports["adapted_iso_old_stage2b"]
    final_oracle = reports["final_oracle"]
    final_estimated = reports["final_estimated"]
    estimator = reports["estimator"]
    estimator_test = estimator["test_metrics"]
    isotropic_training = reports["isotropic_training"]
    stage2b_v2_training = reports["stage2b_v2_training"]

    original_estimated_rmse = float(original_estimated["oracle_final_rmse_db"])
    stage2b_v1_estimated_rmse = float(
        stage2b_v1_estimated["oracle_final_rmse_db"]
    )
    adapted_iso_old_stage2b_rmse = float(
        adapted_iso_old_stage2b["oracle_final_rmse_db"]
    )
    final_estimated_rmse = float(final_estimated["final_rmse_db"])
    final_oracle_rmse = float(final_oracle["final_rmse_db"])
    full_gain = original_estimated_rmse - final_estimated_rmse
    gain_after_v1 = stage2b_v1_estimated_rmse - final_estimated_rmse
    isotropic_chain_gain = (
        stage2b_v1_estimated_rmse - adapted_iso_old_stage2b_rmse
    )
    stage2b_v2_gain = adapted_iso_old_stage2b_rmse - final_estimated_rmse
    final_location_penalty = final_estimated_rmse - final_oracle_rmse
    summary: dict[str, Any] = {
        "version": 2,
        "status": "ok",
        "task": "Infer latent BS parameters internally from building height and sparse signals, then reconstruct the full signal map.",
        "inference_inputs": [
            "building_height_norm",
            "100 sparse signal-strength values",
            "sparse measurement mask",
        ],
        "test_set": {
            "sites": int(final_estimated["sites"]),
            "directional_samples": int(final_estimated["directional_samples"]),
        },
        "latent_parameter_estimator": {
            "location_mean_error_px": float(
                final_estimated["estimated_location_error_px_mean"]
            ),
            "location_median_error_px": float(
                final_estimated["estimated_location_error_px_median"]
            ),
            "location_p90_error_px": float(
                final_estimated["estimated_location_error_px_p90"]
            ),
            "power_mae_db": float(estimator_test["power_abs_error_db_mean"]),
            "direction_mae_deg": float(
                estimator_test["direction_abs_error_deg_mean"]
            ),
            "checkpoint": final_estimated["estimator_checkpoint"],
            "checkpoint_sha256": final_estimated["estimator_checkpoint_sha256"],
        },
        "end_to_end_test_rmse_db": {
            "original_models_oracle_location": float(
                original_oracle["oracle_final_rmse_db"]
            ),
            "original_models_estimated_location": original_estimated_rmse,
            "stage2b_v1_only_oracle_location": float(
                stage2b_v1_oracle["oracle_final_rmse_db"]
            ),
            "stage2b_v1_only_estimated_location": stage2b_v1_estimated_rmse,
            "adapted_isotropic_old_stage2b_estimated_location": (
                adapted_iso_old_stage2b_rmse
            ),
            "fully_adapted_oracle_location": final_oracle_rmse,
            "fully_adapted_estimated_location": final_estimated_rmse,
            "fully_adapted_estimated_unmeasured": float(
                final_estimated["final_unmeasured_rmse_db"]
            ),
            "fully_adapted_sparse_baseline": float(
                final_estimated["sparse_calibrated_iso_baseline_rmse_db"]
            ),
        },
        "derived": {
            "full_gain_vs_original_estimated_db": full_gain,
            "gain_vs_stage2b_v1_pipeline_db": gain_after_v1,
            "isotropic_chain_direct_gain_db": isotropic_chain_gain,
            "stage2b_v2_additional_gain_db": stage2b_v2_gain,
            "final_location_penalty_db": final_location_penalty,
            "gain_vs_sparse_baseline_db": float(
                final_estimated["final_rmse_improvement_vs_baseline_db"]
            ),
            "full_pipeline_acceptance_passed": gain_after_v1 >= 0.2,
        },
        "final_models": {
            "estimator": {
                "checkpoint": final_estimated["estimator_checkpoint"],
                "sha256": final_estimated["estimator_checkpoint_sha256"],
            },
            "stage1": {
                "checkpoint": isotropic_training["stage1_checkpoint"],
                "sha256": isotropic_training["stage1_checkpoint_sha256"],
            },
            "stage2a": {
                "checkpoint": isotropic_training["stage2a_checkpoint"],
                "sha256": isotropic_training["stage2a_checkpoint_sha256"],
                "best_epoch": int(isotropic_training["best_epoch"]),
            },
            "stage2b": {
                "checkpoint": final_estimated["stage2b_checkpoint"],
                "sha256": final_estimated["stage2b_checkpoint_sha256"],
                "best_epoch": int(stage2b_v2_training["best_epoch"]),
            },
        },
        "decision": {
            "current_pipeline_is_viable": True,
            "reason": "All three map stages were successfully reused and adapted; inference does not require user-supplied BS parameters.",
            "next_priority": "Improve location robustness or estimator localization because the remaining oracle gap is now material relative to Stage2-B v2's incremental gain.",
        },
        "source_reports": {name: str(path) for name, path in paths.items()},
    }
    output = args.output.resolve()
    atomic_write(output, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    markdown = f"""# 隐式基站参数 Pilot v2 实验摘要

## 结论

只输入建筑高度、100 个稀疏信号值和掩码的端到端方案可行。完整适配后测试集 RMSE 为 **{final_estimated_rmse:.3f} dB**，稀疏校准基线为 **{final_estimated['sparse_calibrated_iso_baseline_rmse_db']:.3f} dB**。

## 对照结果

| 模型组合 | 基站位置 | 测试 RMSE (dB) |
|---|---|---:|
| 原三级模型 | 网络估计 | {original_estimated_rmse:.3f} |
| 仅适配 Stage2-B v1 | 网络估计 | {stage2b_v1_estimated_rmse:.3f} |
| 新 Stage1/2-A＋旧 Stage2-B | 网络估计 | {adapted_iso_old_stage2b_rmse:.3f} |
| 全部适配 v2 | 网络估计 | {final_estimated_rmse:.3f} |
| 全部适配 v2 | 真实 | {final_oracle_rmse:.3f} |

- 相对原模型累计提升：**{full_gain:.3f} dB**。
- 相对上一版完整管线提升：**{gain_after_v1:.3f} dB**。
- Stage1/2-A 的直接贡献：**{isotropic_chain_gain:.3f} dB**。
- Stage2-B v2 的附加贡献：**{stage2b_v2_gain:.3f} dB**。
- 预测位置相对真实位置的最终损失：**{final_location_penalty:.3f} dB**。
- 相对稀疏校准基线的收益：**{final_estimated['final_rmse_improvement_vs_baseline_db']:.3f} dB**。

## 隐变量估计误差

- 位置：平均 {final_estimated['estimated_location_error_px_mean']:.3f} px，中位数 {final_estimated['estimated_location_error_px_median']:.3f} px，P90 {final_estimated['estimated_location_error_px_p90']:.3f} px。
- 有效功率 MAE：{estimator_test['power_abs_error_db_mean']:.3f} dB。
- 方位角 MAE：{estimator_test['direction_abs_error_deg_mean']:.3f}°。

## 下一步

当前预测位置与真实位置仍相差 {final_location_penalty:.3f} dB，已经大于 Stage2-B v2 的附加收益。下一步优先增强定位鲁棒性，或对 Stage1/2-A 加入位置不确定性训练。
"""
    atomic_write(output.with_suffix(".md"), markdown)
    manifest = {
        "version": 2,
        "status": "ready_for_pilot_inference",
        "inputs": summary["inference_inputs"],
        "models_in_order": summary["final_models"],
        "normalization": {
            "pilot": str(args.model_root.parent / "bs_inversion_pilot_v1" / "normalization.json"),
            "stage1": str(Path(isotropic_training["stage1_checkpoint"]).parent / "normalization.json"),
            "stage1_tx_features": str(Path(isotropic_training["stage1_checkpoint"]).parent / "tx_feature_normalization.json"),
            "stage2b": r"D:\桌面\dac\04_simulation\sionna\dac_sionna_35ghz_depth8_27360_v3\normalization_irt.json",
        },
        "test_metrics": {
            "rmse_db": final_estimated_rmse,
            "mae_db": float(final_estimated["final_mae_db"]),
            "unmeasured_rmse_db": float(final_estimated["final_unmeasured_rmse_db"]),
        },
        "audit_report": str(output),
    }
    atomic_write(
        output.with_name("bs_inversion_pipeline_v2_manifest.json"),
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

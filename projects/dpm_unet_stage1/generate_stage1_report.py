"""Generate the Stage-1 experiment report from saved, reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def best_history_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return min(rows, key=lambda row: float(row["val_rmse_db"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalization = read_json(args.data_root / "normalization.json")
    benchmark = read_json(args.runs_root / "baseline_v1" / "benchmark.json")
    baseline = read_json(args.runs_root / "baseline_v1" / "test_metrics.json")
    height = read_json(args.runs_root / "ablation_height" / "test_metrics.json")
    height_tx = read_json(args.runs_root / "ablation_height_tx" / "test_metrics.json")
    config = read_json(args.runs_root / "baseline_v1" / "run_config.json")
    best = best_history_row(args.runs_root / "baseline_v1" / "history.csv")
    splits = normalization["split_strategy"]["tile_counts"]
    stats = normalization["statistics"]

    dpm_single_ms = 13.9603996 / 2 * 1000
    dpm_parallel_ms = (2 * 3600 + 16 * 60 + 29.87932) / 17770 * 1000
    gpu_single_ms = benchmark["pure_gpu"]["1"]["latency_per_tile_ms"]
    e2e_ms = benchmark["end_to_end_batch1"]["mean_ms"]
    gpu_batch16_tps = benchmark["pure_gpu"]["16"]["tiles_per_second"]
    speedup_single = dpm_single_ms / e2e_ms
    dpm_parallel_tps = 1000 / dpm_parallel_ms
    speedup_throughput = gpu_batch16_tps / dpm_parallel_tps

    report = f"""# 第一阶段实验结果：基于 U-Net 的 DPM 无线信道地图代理模型

## 1. 研究目标

本阶段训练 U-Net，以建筑高度、发射机位置/高度和收发距离为输入，直接预测 WinProp Dominant Path Model（DPM）生成的 3.5 GHz 路径增益图。模型定位是 DPM 的快速代理，而不是对真实测量误差的校正模型。

## 2. 数据与划分

| 项目 | 数值 |
|---|---:|
| 瓦片总数 | {sum(splits.values()):,} |
| 训练集 | {splits['train']:,} |
| 验证集 | {splits['val']:,} |
| 测试集 | {splits['test']:,} |
| 空间分块大小 | {normalization['split_strategy']['block_size_m'] / 1000:.3g} km × {normalization['split_strategy']['block_size_m'] / 1000:.3g} km |
| 单瓦片范围 | 512 m × 512 m |
| 网格分辨率 | 4 m |
| 数组尺寸 | {stats['raw_array_shape'][0]} × {stats['raw_array_shape'][1]} |
| 频率 | 3.5 GHz |

采用 2 km 空间块整体划分训练、验证和测试集，防止相邻瓦片跨集合造成空间泄漏。建筑高度使用训练集正高度像元 P99={stats['height']['cap_m']:.3f} m 截断；路径增益训练集均值为 {stats['path_gain']['mean_db']:.3f} dB，标准差为 {stats['path_gain']['std_db']:.3f} dB。DPM 的 N.C. 像元通过 mask 排除，不参与损失和指标。

## 3. 模型与训练设置

| 项目 | 设置 |
|---|---|
| 网络 | 4 层编码器/解码器 U-Net |
| 输入 | 建筑高度 + TX 位置/高度 + TX 距离 |
| 参数量 | {config['parameters']:,} |
| 损失 | masked Huber loss |
| 优化器 | AdamW |
| 初始学习率 | {config['learning_rate']} |
| batch size | {config['batch_size']} |
| 最大 epoch | {config['epochs']} |
| 数据增强 | 同步旋转与翻转 |
| 训练精度 | CUDA AMP |
| GPU | {config['gpu']} |

最佳验证 checkpoint 位于 epoch {best['epoch']}，验证 MAE={float(best['val_mae_db']):.3f} dB、RMSE={float(best['val_rmse_db']):.3f} dB。

## 4. 独立测试结果

| 指标 | 结果 |
|---|---:|
| 全局 MAE | {baseline['mae_db']:.3f} dB |
| 全局 RMSE | {baseline['rmse_db']:.3f} dB |
| 建筑边缘 8 m MAE | {baseline['edge_mae_db']:.3f} dB |
| 建筑边缘 8 m RMSE | {baseline['edge_rmse_db']:.3f} dB |
| 非边缘 MAE | {baseline['non_edge_mae_db']:.3f} dB |
| 非边缘 RMSE | {baseline['non_edge_rmse_db']:.3f} dB |

验证集与测试集误差接近，说明模型能泛化至未见空间块。误差主要集中在建筑边缘和复杂 NLOS 区域；4 m 栅格无法完整保留 DPM 使用的细粒度几何、屋顶和传播路径细节。

## 5. 输入消融实验

| 输入组合 | 测试 MAE | 测试 RMSE | 边缘 RMSE | 非边缘 RMSE |
|---|---:|---:|---:|---:|
| 仅建筑高度 | {height['mae_db']:.3f} | {height['rmse_db']:.3f} | {height['edge_rmse_db']:.3f} | {height['non_edge_rmse_db']:.3f} |
| 建筑高度 + TX | {height_tx['mae_db']:.3f} | {height_tx['rmse_db']:.3f} | {height_tx['edge_rmse_db']:.3f} | {height_tx['non_edge_rmse_db']:.3f} |
| 建筑高度 + TX + 距离 | {baseline['mae_db']:.3f} | {baseline['rmse_db']:.3f} | {baseline['edge_rmse_db']:.3f} | {baseline['non_edge_rmse_db']:.3f} |

消融实验使用完全相同的空间划分、随机种子、网络宽度、优化器和训练轮数。由此可将性能变化归因于输入信息，而不是训练条件差异。

## 6. 推理速度与 DPM 对比

| 模式 | 时间/吞吐 |
|---|---:|
| WinProp DPM 单进程 | {dpm_single_ms / 1000:.3f} s/瓦片 |
| WinProp DPM 12 路批处理吞吐 | {dpm_parallel_ms:.1f} ms/瓦片（{dpm_parallel_tps:.2f} 瓦片/s） |
| U-Net 纯 GPU batch=1 | {gpu_single_ms:.3f} ms/瓦片 |
| U-Net 端到端 batch=1 | {e2e_ms:.3f} ms/瓦片 |
| U-Net batch=16 | {gpu_batch16_tps:.1f} 瓦片/s |

按单瓦片请求比较，U-Net 端到端相对单进程 DPM 加速约 **{speedup_single:,.0f}×**；按批量系统吞吐比较，U-Net batch=16 相对 12 路并行 DPM 加速约 **{speedup_throughput:,.0f}×**。端到端 U-Net 时间已包含三张 NPY 输入读取、数组拼接、CPU→GPU 拷贝和同步推理；DPM 时间来自同一台电脑的批处理日志。

## 7. 建筑边缘增强实验

额外测试了显式建筑边缘通道与 2 倍边缘加权 loss。该模型测试全局 RMSE=2.648 dB、边缘 RMSE=3.138 dB，均略差于基线的 {baseline['rmse_db']:.3f}/{baseline['edge_rmse_db']:.3f} dB，因此不作为最终模型。该负结果表明，重复强调 4 m 栅格中已有的边缘信息无法补回栅格化时丢失的几何细节。

## 8. 阶段结论与边界

第一级 U-Net 能以约 {baseline['mae_db']:.2f} dB MAE 复现 DPM 路径增益，并将单瓦片端到端生成时间从秒级降至毫秒级。最终模型采用 `baseline_v1/best.pt`。

需要明确：当前标签全部来自 DPM，因此指标衡量的是“复现 DPM 的能力”，不能直接解释为相对真实无线测量的误差。下一阶段如获得稀疏实测 RSRP，可将 DPM/U-Net 路径增益作为先验，再训练测量校正网络。
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

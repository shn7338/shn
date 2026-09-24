# 当前训练路线：Sionna 数据与 Stage 2

这个目录的 `irt_label_pipeline` 名称沿用早期 WinProp IRT 试验；**目录名不是当前训练标签的来源**。当前 Stage 2 的正式标签来自 Sionna RT，WinProp IRT2/IRT6 文件保留用于历史对照、诊断和追溯，不与当前 Sionna 标签混合。

| 阶段 | 当前使用的数据或模型 | 作用与状态 |
| --- | --- | --- |
| Stage 1 | `01_data/normalized_4m_v1`、`03_runs/models/stage1_dpm_unet_baseline_v1` | 用 WinProp **DPM** 数据训练代理模型，预测 DPM 路径增益；这里的 DPM 与旧 WinProp **IRT** 是不同的仿真标签。 |
| Stage 2 标签 | `04_simulation/sionna/dac_sionna_35ghz_depth8_27360_v3` | 3.5 GHz、depth 8 的 Sionna 数据集，27,360 个瓦片已经生成并完成分片与校验；v1/v2 为保留的失败或停止轮次。 |
| Stage 2A | `config_stage2a_sionna35_depth8_27360_paper_v1.json` | 冻结 Stage 1，以 Sionna 各向同性图与 DPM 预测的差值训练修正网络；全量测试记录标为已通过预设门槛。 |
| Stage 2B | `config_stage2b_sionna35_depth8_27360_paper_v1.json` | 以 Sionna 方向图生成稀疏信号值，结合 Stage 2A 预测训练方向信号地图网络；全量测试记录标为已通过预设门槛。 |
| 随机基站 Scale-4 / V5 | `02_inversion/bs_inversion_scale4_v1`、`config_stage2b_multiscale_adapter_scale4_v5.json` | 另一组由 Sionna RT 生成的随机基站数据与 V5 模型实验；V5 已选作展示，但其严格升级门槛记录为未通过。 |

**不要把这些结果直接混在一起比较。**固定中心基站的 27,360 瓦片路线与随机基站 Scale-4 路线使用不同数据和任务设置。Stage 1 的 WinProp DPM 标签也不等于旧 WinProp IRT 标签。

## 证据和代码入口

- Sionna v3 数据：[`04_simulation/sionna/`](../../04_simulation/sionna/README.md)；运行配置 `config_sionna201_35ghz_depth8_27360_v3.json`，生成脚本 `run_sionna_dataset.py`，启动脚本 `start_sionna_35ghz_all_tiles.ps1`。
- Stage 2A：`train_stage2a_iso_refine.py`、`start_stage2a_sionna35_training.ps1`；[全量测试指标](../../中期答辩佐证材料_20260924/01_数据记录/05_训练评估记录/stage2a_iso_refine_sionna35_depth8_paper_v1/test_metrics.json) 标记 `accepted_for_stage2b: true`。
- Stage 2B：`train_stage2b_directional_ss.py`、`start_stage2b_sionna35_training.ps1`；[全量测试指标](../../中期答辩佐证材料_20260924/01_数据记录/05_训练评估记录/stage2b_directional_ss_sionna35_depth8_paper_v1/test_metrics.json) 标记 `accepted_for_final_evaluation: true`。
- Scale-4 / V5：`config_bs_inversion_scale4_v1.json` 明确记录 Sionna RT 数据生成方式；[`V5 门槛记录`](../../中期答辩佐证材料_20260924/01_数据记录/05_训练评估记录/版本对比/bs_inversion_scale4_stage2b_multiscale_adapter_v5_gate.json) 保留实际判断。
- [同目录 README](README.md) 的 WinProp IRT2/IRT6 大段记录属于早期路线；其中的 `irt_*` 文件名和 `irt_normalization` 配置键是沿用的历史命名。判断实际训练来源应查看配置里的 `shard_root`、`dataset_root` 和测试指标。

GitHub 只收录配置、代码与部分指标。完整 Sionna 分片和模型权重在本机，克隆仓库后不能直接复现实验。

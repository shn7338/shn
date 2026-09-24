# 当前数据、模型与结果（2026-09-24 核对）

## 固定中心基站路线

Stage 1 DPM 基线 → Sionna Stage 2A 各向同性修正 → Stage 2B 方向信号图重建。

| 阶段 | 数据与配置 | 状态/结果 |
| --- | --- | --- |
| Stage 1 基线 | WinProp DPM 数据 `01_data/normalized_4m_v1`；[代码说明](../dpm_unet_stage1/README.md) | 预测 DPM 路径增益；与旧 WinProp IRT 标签不同 |
| Stage 2 标签 | [Sionna 27,360 瓦片 v3](../../04_simulation/sionna/dac_sionna_35ghz_depth8_27360_v3/README.md)；[生成配置](config_sionna201_35ghz_depth8_27360_v3.json) | 3.5 GHz、depth 8；27,360/27,360 完成，train/val/test = 21,789/2,840/2,731 |
| Stage 2A | [训练配置](config_stage2a_sionna35_depth8_27360_paper_v1.json) | 冻结 Stage 1，学习 Sionna 各向同性图减去 DPM 预测的残差；最佳 epoch 109，测试 RMSE 5.512 dB，accepted_for_stage2b=true |
| Stage 2B | [训练配置](config_stage2b_sionna35_depth8_27360_paper_v1.json) | 建筑+Stage 2A 输出+稀疏方向信号+mask；实际停在最佳 epoch 154（配置上限 200），50/100/200 点 RMSE = 5.981/5.765/5.503 dB，accepted_for_final_evaluation=true |

[Stage 2A 测试记录](../../中期答辩佐证材料_20260924/01_数据记录/05_训练评估记录/stage2a_iso_refine_sionna35_depth8_paper_v1/test_metrics.json)和 [Stage 2B 测试记录](../../中期答辩佐证材料_20260924/01_数据记录/05_训练评估记录/stage2b_directional_ss_sionna35_depth8_paper_v1/test_metrics.json)已上传。

旧 WinProp IRT2/IRT6 未作为当前 Stage 2 标签。Sionna 全量 v1/v2 分别只完成 31/169 个瓦片，已经停止或被拒绝；3,200 瓦片计划实际仅完成 3 个。其余目录保留作过程证据，不代表都完成了全量生产。

## 随机基站路线与 V5

[数据说明](../../02_inversion/README.md)：Pilot 有 384 个建筑场景、1,536 个站点；Scale-4 有 1,536 个场景、6,144 个站点、30,720 张传播图和 24,576 个方向样本。这里“场景”是一块 512 m 瓦片，不是单栋建筑。两套数据均由 Sionna RT 2.0.1 生成；Scale-4 按到中心的径向区间采样位置，Pilot 按四象限采样。

V5 管线：建筑高度图、稀疏信号值与掩码 → 基站参数估计器 → 六通道位置编码 Stage 1 / Stage 2A → 带七类隐变量、多尺度空间适配器及 FiLM 的 Stage 2B → 完整方向信号图。

[V5 配置](config_stage2b_multiscale_adapter_scale4_v5.json)使用 `bs_parameter_estimator_scale4_v2`、`stage1_position_encoding_scale4_v3`、`stage2a_position_encoding_scale4_v3`，由 V4 初始化。随机基站 Stage 1/2A 已联合适配 Sionna 各向同性目标，不能直接等同于固定基站 DPM 基线。该随机基站 Stage 2A 的 `accepted_for_stage2b=false`；后续实验配置显式使用 `require_accepted_stage2a=false`，与上面的固定基站通过状态不同。

| 版本 | 主要变化 | Scale-4 原始测试 RMSE | 严格门槛 |
| --- | --- | --- | --- |
| V3 | 六通道位置编码的上游模型；Stage 2B 仍为四通道 | 6.946 dB | 后续对照基线 |
| V4 | Stage 2B 拼接七类估计器隐变量，扩为 11 通道 | 6.922 dB | 未通过 |
| V5 | 多尺度空间适配器、标量 FiLM、两阶段微调 | 6.897 dB | 未通过；已由用户选作展示/采用版本 |
| V6 | 联合微调 V5 与估计器后端 | 6.897 dB | 未通过，最佳结果仍为 epoch 0 |

原始判定：[V4](../../03_runs/experiments/bs_inversion_scale4_stage2b_latent_fusion_v4_gate.json)、[V5](../../03_runs/experiments/bs_inversion_scale4_stage2b_multiscale_adapter_v5_gate.json)、[V6](../../03_runs/experiments/bs_inversion_scale4_joint_estimator_stage2b_v6_gate.json)。**选择展示 V5 不等于通过预设升级门槛。**固定基站 Stage 2B 的 5.765 dB 与上述随机基站指标不能直接比较。

## 归一化、坐标和实测边界

- V5 的 `irt_normalization` 指向固定基站 Sionna v3 统计，用于 Stage 2B 的尺度；估计器输入用其 checkpoint 元数据中的统计。不要仅因数据在 Scale-4 目录就统一替换成 Scale-4 默认均值/标准差。
- 当前随机基站代码使用局部 x 向东、y 向北；row=0 为北、col=0 为西；方位角 0° 向东、90° 向北。早期 `first try/` 的 CLI 则采用 0° 向北、顺时针角度，不能原样混用。
- 一张方向样本对应单站点的一个方向；四个方向样本不等于多扇区同时叠加。手机数据需按目标小区/频点筛选，输入点与独立评估点分开。
- 现有手机 CSV 属于试采/处理流程调试材料，不能写成已完成正式实测校准或校园全图验证；仿真信号值也不能直接等同于实测 RSRP。

当前代码见 [README](README.md)，已上传的结果见 [模型索引](../../03_runs/models/README.md)。完整仿真分片、训练权重和环境在本机；仅克隆 GitHub 不足以完整复现。

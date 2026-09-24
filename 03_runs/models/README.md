# 模型目录与版本状态

固定基站路线与随机基站路线应分别比较。V5 是已选择的展示版本；严格门槛未通过的记录照实保留。

| 模型 | 用途与状态 |
| --- | --- |
| [bs_parameter_estimator_pilot_v1](bs_parameter_estimator_pilot_v1/README.md) | Pilot 基站参数估计器：从建筑、稀疏信号与掩码估计位置、有效功率、方向和辅助信号图。测试记录为 needs_iteration。 |
| [bs_parameter_estimator_pilot_v1_smoke_20260809_173225](bs_parameter_estimator_pilot_v1_smoke_20260809_173225/README.md) | Pilot 估计器的短程 smoke 运行，只验证训练/评估流程；测试记录为 needs_iteration。 |
| [bs_parameter_estimator_scale4_v2](bs_parameter_estimator_scale4_v2/README.md) | Scale-4 基站参数估计器，测试记录为 accepted；后续 V3/V4/V5 管线使用该估计器。此 accepted 只针对估计器自己的门槛。 |
| [joint_estimator_stage2b_scale4_v6](joint_estimator_stage2b_scale4_v6/README.md) | V6 联合微调 V5 和估计器后端。训练完成 3 个 epoch，最佳 checkpoint 仍为 epoch 0；没有获得超过 V5 的有效提升，严格门槛未通过。 |
| [stage1_bs_inversion_pilot_finetune_v1](stage1_bs_inversion_pilot_finetune_v1/README.md) | Pilot 随机基站适配中的 Stage 1 微调产物，与配套 Stage 2A 联合适配，用于后续 Pilot Stage 2B。 |
| [stage1_dpm_unet_baseline_v1](stage1_dpm_unet_baseline_v1/README.md) | 固定基站 Stage 1 DPM 代理基线：三通道建筑/TX/距离输入，独立测试 RMSE 2.613 dB、MAE 1.383 dB；误差参照 WinProp DPM 标签。 |
| [stage1_position_encoding_pilot_v2](stage1_position_encoding_pilot_v2/README.md) | Pilot 的六通道位置编码 Stage 1，新增长距离、带符号 Δx/Δy 等位置信息；与同名 Stage 2A 配套。 |
| [stage1_position_encoding_scale4_v3](stage1_position_encoding_scale4_v3/README.md) | Scale-4 的六通道位置编码 Stage 1；与 Stage 2A v3 配套，供随机基站 V3–V5 使用。不要与固定基站三通道 DPM 基线混用。 |
| [stage2a_bs_inversion_pilot_finetune_v1](stage2a_bs_inversion_pilot_finetune_v1/README.md) | Pilot 随机基站各向同性图修正模型；本轮测试 accepted_for_stage2b=true。 |
| [stage2a_iso_refine_sionna35_depth8_paper_v1](stage2a_iso_refine_sionna35_depth8_paper_v1/README.md) | 固定基站 Sionna v3 全量 Stage 2A：修正冻结 Stage 1 的 DPM 预测。最佳 epoch 109，测试 RMSE 5.512 dB，accepted_for_stage2b=true。 |
| [stage2a_position_encoding_pilot_v2](stage2a_position_encoding_pilot_v2/README.md) | Pilot 位置编码方案的 Stage 2A；测试记录 needs_iteration，accepted_for_stage2b=false。 |
| [stage2a_position_encoding_scale4_v3](stage2a_position_encoding_scale4_v3/README.md) | Scale-4 位置编码方案的 Stage 2A；测试 RMSE 7.385 dB，accepted_for_stage2b=false。后续随机基站实验配置允许使用本 checkpoint，因此不能把固定基站 Stage 2A 的通过结论套到这里。 |
| [stage2b_bs_inversion_pilot_finetune_v1](stage2b_bs_inversion_pilot_finetune_v1/README.md) | Pilot Stage 2B 首轮微调，冻结估计器和上游模型，仍使用四通道输入。属于随机基站适配实验。 |
| [stage2b_bs_inversion_pilot_finetune_v2](stage2b_bs_inversion_pilot_finetune_v2/README.md) | Pilot Stage 2B 第二轮微调，使用已做随机位置适配的 Stage 1/2A。这里的 v2 与 Scale-4 V3–V6 不同数据规模。 |
| [stage2b_directional_ss_sionna35_depth8_paper_v1](stage2b_directional_ss_sionna35_depth8_paper_v1/README.md) | 固定基站 Sionna v3 全量 Stage 2B：建筑、修正各向同性图、稀疏信号、掩码四通道。配置上限 200 epoch，实际停在最佳 epoch 154；50/100/200 点测试 RMSE 为 5.981/5.765/5.503 dB，accepted_for_final_evaluation=true。 |
| [stage2b_latent_fusion_scale4_v4](stage2b_latent_fusion_scale4_v4/README.md) | Scale-4 V4：把估计器的七类隐变量加入 Stage 2B，输入由 4 通道扩到 11 通道。原始测试 RMSE 6.922 dB，严格升级门槛未通过。 |
| [stage2b_multiscale_adapter_scale4_v5](stage2b_multiscale_adapter_scale4_v5/README.md) | Scale-4 V5：在 V4 上加入多尺度空间适配器与标量 FiLM，采用两阶段微调。用户已选作展示/采用版本；原始测试 RMSE 6.897 dB，但严格升级门槛未通过。输入仍为建筑、稀疏信号和掩码；基站参数由估计器推断。 |
| [stage2b_position_encoding_scale4_v3](stage2b_position_encoding_scale4_v3/README.md) | Scale-4 V3：使用适应随机位置的 Stage 1/2A，Stage 2B 主体仍为四通道。原始测试 RMSE 6.946 dB，是 V4/V5/V6 对照基线。 |

模型权重均未上传；克隆仓库不能直接加载训练完成的模型。配置入口在 `projects/irt_label_pipeline/`，本目录中的 Markdown 模型卡和报告可能保留实验时的路径。

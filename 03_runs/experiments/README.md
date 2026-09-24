# 评估、版本比较与可视化

本目录顶层 JSON/Markdown 保存随机基站实验配置、分组指标、V4–V6 严格升级门槛等。子目录保存敏感性分析、估计器比较、单样本与 V3–V6 可视化；大规模数组、图片缓存保存在本机，GitHub 主要收录顶层轻量记录和子目录 Markdown。

V5 的原始测试 RMSE 6.897 dB，是已选择的展示版本，但 [V5 门槛](bs_inversion_scale4_stage2b_multiscale_adapter_v5_gate.json) 未通过；[V4 门槛](bs_inversion_scale4_stage2b_latent_fusion_v4_gate.json)、[V6 门槛](bs_inversion_scale4_joint_estimator_stage2b_v6_gate.json) 也未通过。固定基站 5.765 dB 的结果使用另一套数据，不能直接比较。

`scale4_visual_comparison_v3` 已于 2026-08-28 修正，after 使用真正的 Stage 2B V3；带 `legacy_stage2b_v2` 的目录是旧缓存备份，不应用来代表完整 V3。核对图像时以该轮 `visualization_summary.json` 中的 checkpoint 哈希为准。

完整模型说明见 [模型索引](../models/README.md)。

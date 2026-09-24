# 训练与实验记录

| 子目录 | 内容 |
| --- | --- |
| `E_stage/` | Stage 1 DPM 预测及 WinProp/Sionna 的 Stage 2 smoke 试验，收录配置和评估摘要 |
| `experiments/` | 基站反演与信号图实验配置、结果汇总、门槛记录 |
| `inference_demo/` | 本机推断输入与输出；仅保留目录说明 |
| `models/` | 模型权重和模型说明；权重留在本机 |
| `stage1_dpm_unet_runs/` | Stage 1 DPM U-Net 训练记录与源代码 |

当前固定基站 Stage 2A/2B 的正式训练使用 `04_simulation/sionna/dac_sionna_35ghz_depth8_27360_v3`；随机基站 V5 使用 `02_inversion/bs_inversion_scale4_v1`。旧 WinProp IRT 运行不作为这两组训练标签。详细区别见 [当前 Sionna 路线](../projects/irt_label_pipeline/SIONNA_CURRENT.md)。

本机结果约 8 GiB，其中 `.pt`、`.pth` 等权重和大量预测数组未上传。仅凭仓库不能直接运行已训练模型。

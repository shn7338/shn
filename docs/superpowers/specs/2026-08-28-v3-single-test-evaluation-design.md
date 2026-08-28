# V3 单测试样本评估图设计

## 目标

为 Scale-4 信号地图模型的正式 Stage2B V3 检查点重新生成一张单模型测试集可视化。新版图只评价 V3，不显示 V4、V5 或 V6，并沿用既有 V5 图中的同一测试样本，以保证区域、稀疏测量点和有效掩膜一致。

## 固定证据

- 测试样本：`tile_000222_site03`。
- 方向：该样本的第 0 个方向，方位角 12°。
- 稀疏输入：100 个信号采样点及其独立掩膜。
- 区域：512 m × 512 m，128 × 128 像元，4 m 分辨率。
- Stage1：`D:\桌面\dac\models\stage1_position_encoding_scale4_v3\best.pt`。
- Stage2A：`D:\桌面\dac\models\stage2a_position_encoding_scale4_v3\best.pt`。
- Stage2B：`D:\桌面\dac\models\stage2b_position_encoding_scale4_v3\best.pt`。
- 基站参数估计器：`D:\桌面\dac\models\bs_parameter_estimator_scale4_v2\best.pt`。
- 完整测试集报告：`D:\桌面\dac\experiments\bs_inversion_scale4_stage2b_v3_test_bins.json`，包含 1,024 个站点和 4,096 张方向图。
- V3 完整测试集指标：RMSE 6.9462077158 dB，MAE 4.1017797064 dB。
- V3 预设整体 RMSE 门槛：6.85 dB；最终图必须注明未通过。

旧的 `scale4_visual_comparison_v3/selected_samples.npz` 不作为 V3 Stage2B 预测证据，因为该缓存虽然记录了 Stage1/Stage2A 的 V3 改动，Stage2B 仍指向旧的 pilot 检查点。必须用上述正式 Stage2B V3 检查点对固定样本重新推理。

## 图形布局

采用用户确认的 2 × 3 均衡布局：

1. **A 输入**：建筑高度底图、100 个稀疏信号点、真实基站位置和估计基站位置。
2. **B 测试真值**：方向性信号强度真值。
3. **C V3 预测**：正式 V3 检查点的预测，并在标题显示单图 RMSE 和 MAE。
4. **D 有符号误差**：`V3预测 - 真值`，以 0 dB 为发散色阶中心，标题显示 Bias。
5. **E 绝对误差**：显示绝对误差，标题显示 P90 和最大绝对误差。
6. **F 指标摘要**：列出单样本有效像元数、MAE、RMSE、Bias、P90、最大绝对误差、Pearson r，以及完整测试集的 4,096 图 MAE/RMSE和门禁状态。

## 颜色、掩膜和坐标

- 真值和 V3 预测必须使用相同的顺序色阶和相同的 1%–99% 分位显示范围。
- 超出显示范围的值通过色条延伸端明确标记，不修改原始数组。
- 有符号误差以 0 dB 为中心，采用对称的 P99 绝对误差范围。
- 绝对误差采用 0 到 P99 的顺序色阶，并用延伸端表示更大异常值。
- 无效像元必须显示为中性灰色，不能与零误差的白色混淆。
- 输入稀疏点使用与信号图一致的数值色阶；真实与估计基站使用不同形状和直接图例。
- 所有地图使用 East/North 米制坐标，保持等比例显示。

## 数据流

1. 从 `D:\桌面\dac\bs_inversion_scale4_v1` 的 test split 定位 `tile_000222_site03` 和方向 0。
2. 加载建筑、稀疏信号、采样掩膜、方向性真值、有效掩膜和真实基站参数。
3. 加载冻结的基站估计器、Stage1、Stage2A和正式 Stage2B V3。
4. 使用与完整测试集评估脚本相同的归一化和前向链路生成 V3 预测。
5. 仅在方向性 `valid_mask` 内计算指标；建筑和未命中的室外像元不参与。
6. 保存原始推理数组或压缩证据缓存、指标 JSON 和最终 PNG，记录所有检查点路径及哈希。

## 输出

新建独立目录：

`D:\桌面\dac\experiments\single_test_v3_evaluation`

至少包含：

- `single_test_v3_evaluation.png`
- `single_test_v3_metrics.json`
- `single_test_v3_arrays.npz`

项目内新增可复现脚本，命名为：

`projects/irt_label_pipeline/render_single_test_v3_evaluation.py`

不得覆盖原有 V5 评估图、指标或源测试数组。

## 异常处理

- 任一检查点、归一化文件、样本或必需字段不存在时立即失败。
- 检查样本 ID、方向角、数组形状、有效掩膜和稀疏点数量。
- 检查正式 Stage2B V3 检查点路径和哈希与测试报告一致。
- 若预测、真值或有效像元指标出现非有限值，停止导出。
- 已存在输出时只允许脚本显式覆盖自身输出；不得修改源缓存。

## 验证

- 对脚本运行 Python 语法检查。
- 复算单样本 MAE、RMSE、Bias、P90、最大绝对误差和相关系数。
- 核对完整测试集 MAE/RMSE与正式 V3 报告一致。
- 人工查看最终 PNG，确认标题、色条、坐标、灰色无效区和指标面板无裁切或重叠。
- 检查最终 PNG 为不透明 RGB、至少 180 DPI，并记录像素尺寸。
- 输出 JSON 包含简短替代文本、显示范围和溯源信息。

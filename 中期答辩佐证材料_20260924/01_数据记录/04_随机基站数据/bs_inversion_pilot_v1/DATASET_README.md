# 随机基站反演 Pilot v1

状态：完整生成并通过 SHA-256 验证。

- 建筑：384（train/val/test = 256/64/64），按原空间块划分，建筑不会跨集合泄漏。
- 基站站点：1536（每个建筑四象限各 1 个位置）。
- 传播图：7680（每站点 1 张各向同性路径增益 + 4 张方向图）。
- 方向训练样本：6144（train/val/test = 4096/1024/1024）。
- 稀疏点：每张方向图 100 个有效室外点，无噪声、无异常点。

## 目录

- `samples/<split>/<site_id>.npz`：模型所需数组。
- `samples/<split>/<site_id>.json`：字段、单位、参数和文件哈希。
- `completion/`：逐站点完成记录，支持断点续跑。
- `run_manifest.json`：全部随机位置、功率、方向和随机种子。
- `site_index.csv`：便于表格查看的站点索引。
- `normalization.json`：仅由训练集计算的 Pilot 归一化统计。
- `VALIDATION_REPORT.json`：独立全量样本验证结果。

## 单个 NPZ 的主要字段

- `building_height_m`, `building_height_norm`：`[128,128]` 建筑输入。
- `isotropic_path_gain_db`：`[128,128]`，供现有 Stage2-A/Stage2-B 接口复用。
- `directional_signal_strength_db`：`[4,128,128]` 完整信号图真值。
- `sparse_flat_indices`, `sparse_signal_strength_db`：`[4,100]` 稀疏测点；在线用 `row=index//128, col=index%128` 还原掩码。
- `tx_location_heatmap`：`[128,128]`，高斯标准差 1.5 像素。
- `tx_xy_m`, `tx_row_col_px`, `tx_height_m`：真实位置和高度。
- `effective_power_db`：可辨识的合并链路预算偏移。
- `azimuth_deg`, `azimuth_sin_cos`：4 个真实方向标签，后者顺序为 `[sin, cos]`。

方向信号图满足：

`directional_signal_strength_db = directional_path_gain_db + effective_power_db`

因此未重复存储方向路径增益，需要时直接相减恢复。

## 归一化选择

重新训练 Pilot 模型时，从原始 dB 字段按 `normalization.json` 处理：

`signal_norm = (signal_db - -58.996194940320) / 20.544459047423`

`sparse_signal_strength_norm_reference` 使用旧 Stage2-B 的归一化（mean=-57.86771677215448 dB，std=19.549090075885392 dB），只用于直接兼容现有 Stage2-B。若重新训练，应从 `sparse_signal_strength_db` 按 Pilot 统计在线归一化。

坐标约定：局部 x 向东、y 向北；数组 row=0 为北、col=0 为西；方位角 0° 向东、90° 向北。

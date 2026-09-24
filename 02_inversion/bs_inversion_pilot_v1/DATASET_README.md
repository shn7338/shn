# 随机基站反演数据：bs_inversion_pilot_v1

数据配置：`bs_inversion_random_location_pilot_v1`；生成器：Sionna RT 2.0.1。
状态以 `FINAL_REPORT.json` 的 `complete_and_verified` 和本机 `VALIDATION_REPORT.json` 为准。

- 建筑场景（瓦片）：384（train/val/test = 256/64/64）；这里统计的是场景数，不是单体建筑栋数，沿用原空间块划分。
- 基站站点：1536（train/val/test = 1024/256/256）；每个建筑场景四象限各采样 1 个站点。
- 传播图：7680（每站点 1 张各向同性图，另有各方向图）。
- 方向训练样本：6144（train/val/test = 4096/1024/1024）。
- 稀疏点：每张方向图 100 个有效室外点；噪声标准差 0.0 dB，异常点概率 0.0。
- 每站点的四张方向图是不同方位角的仿真样本，不代表四扇区同时发射的叠加图。

## 目录

- `samples/<split>/<site_id>.npz`：模型所需数组。
- `samples/<split>/<site_id>.json`：字段、单位、参数和文件哈希。
- `completion/`：逐站点完成记录，支持断点续跑。
- `run_manifest.json`：全部随机位置、功率、方向和随机种子。
- `site_index.csv`：便于表格查看的站点索引。
- `normalization.json`：仅由本数据集训练部分计算的统计，以及兼容已有模型的参考参数。
- `FINAL_REPORT.json`：生成完成数量。
- `VALIDATION_REPORT.json`：独立全量样本验证结果。

GitHub 本目录收录说明、最终报告和归一化参数；完整 NPZ、manifest 和逐站点记录在本机。部分验证与样本记录另存于仓库的中期答辩佐证材料。

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

`BSInversionPilotDataset` 默认从原始 dB 字段按本目录 `normalization.json` 处理：

`signal_norm = (signal_db - (-58.996194940320)) / 20.544459047423`

`sparse_signal_strength_norm_reference` 使用已有 Stage 2B 的参考归一化（mean=-57.86771677215448 dB，std=19.549090075885392 dB）。**实际模型应使用训练时的统计**：V5 训练脚本通过 `irt_normalization` 读取固定基站 Sionna v3 的统计，并显式覆盖数据集默认值；估计器输入使用其 checkpoint 元数据中的统计。不能把上述数据集默认公式无条件用于所有模型。

历史 `normalization.json` 的 `source` 字符串可能沿用 Pilot 命名，判断数据规模应结合本目录最终报告与样本清单；已有统计数值保持原记录。

坐标约定：局部 x 向东、y 向北；数组 row=0 为北、col=0 为西；方位角 0° 向东、90° 向北。

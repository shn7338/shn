# Stage 1 归一化数据

WinProp DPM 标签及建筑/TX 输入，512 m 场景、4 m 网格、128×128。共 27,360 瓦片，train/val/test = 21,789/2,840/2,731，按 2,048 m 空间块划分。归一化统计只由训练集拟合。

GitHub 收录 `normalization.json`、TX 归一化参数、划分 CSV/TXT 和顶层汇总；逐瓦片 NPY 在本机。参见 [归一化脚本说明](../../scripts/README_normalize_winprop_dataset.md)、[TX 特征说明](../../scripts/README_prepare_tx_features.md)。这套 DPM 标签用于 Stage 1 基线；当前 Stage 2 使用单独生成的 Sionna 标签。

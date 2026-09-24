# 对齐后的 4 m 数据

`prepare_winprop_tiles.py` 将建筑高度与 WinProp DPM 路径增益对齐到每瓦片的 128×128 网格，保存高度、路径增益和 metadata；不可计算像元保留为 NaN，后续由 mask 排除。路径增益 dB 不能直接当作手机 RSRP。

GitHub 收录 `batch_summary.csv`；逐瓦片数组和 metadata 在本机。具体步骤见 [预处理说明](../../scripts/README_prepare_winprop_tiles.md)。

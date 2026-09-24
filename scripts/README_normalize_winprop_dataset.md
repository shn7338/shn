# 划分、归一化与有效像元 mask

该脚本只读取 `prepared_4m` 中的原始数组，输出到一个全新的目录，不修改原始 `.npy`。

默认将相邻的 2 km × 2 km 空间块整体分入训练、验证或测试集（目标比例为 80% / 10% / 10%），从而降低相邻瓦片跨集合带来的空间泄漏。

归一化参数严格只从训练瓦片拟合：

- 建筑高度：仅正高度像元的训练集 P99 作为截断高度，再缩放到 `[0, 1]`；
- 路径增益：仅训练集中有限 dB 值的均值和标准差；
- WinProp `N.C.` / `NaN`：`path_gain_norm.npy` 对应位置填 0，另由 `path_gain_valid_mask.npy` 标记为 `False`。训练损失必须使用此 mask 忽略它们。

运行命令：

```powershell
conda activate sigmap
python D:\桌面\dac\scripts\normalize_winprop_dataset.py `
  --input-root D:\桌面\dac\01_data\prepared_4m `
  --output-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --workers 4
```

输出根目录会包含 `normalization.json`、`splits.csv`、三个 `*_tiles.txt` 文件，以及每个瓦片的三个归一化数组。中断后用相同命令重跑即可跳过已有的瓦片输出；统计参数和 split 是确定性的。

完整运行前可以用 `--dry-run` 只生成并检查数据划分。

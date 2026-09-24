# 第一级 DPM 代理 U-Net：TX 特征生成

该脚本为每个已归一化瓦片生成两个额外输入通道：

- `tx_position_height_norm.npy`：除 TX 像元外均为 0，TX 像元的值为训练集归一化后的发射机高度；
- `tx_distance_norm.npy`：每个像元中心到 TX 的二维距离，经过 `log1p` 归一化。

TX 坐标、高度、频率和接收机高度由每个瓦片对应的 WinProp `Path Loss.txt` 文件头读取。归一化上限只由 `splits.csv` 中的训练瓦片计算，并保存为 `tx_feature_normalization.json`。

运行：

```powershell
conda activate sigmap
python D:\桌面\dac\scripts\prepare_tx_features.py `
  --raw-root D:\桌面\dac\01_data\prepared_4m `
  --normalized-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --workers 4
```

第一级 DPM 代理模型的输入可定义为：

```text
[building_height_norm, tx_position_height_norm, tx_distance_norm]
```

监督标签为 `path_gain_norm.npy`，并必须使用 `path_gain_valid_mask.npy` 屏蔽 WinProp 的 `N.C.` 像元。

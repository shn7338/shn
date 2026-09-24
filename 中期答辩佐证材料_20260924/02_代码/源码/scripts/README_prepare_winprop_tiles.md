# WinProp 与建筑高度图对齐批处理

`prepare_winprop_tiles.py` 会遍历 `tile_*` 文件夹，并为每个瓦片输出：

- `building_height_4m.npy`：建筑高度（米），无建筑为 0；
- `path_gain_4m.npy`：WinProp `Path Loss.txt` 中的原始 dB 数值；`N.C.` 点保留为 `NaN`；
- `metadata.json`：范围、分辨率、CRS、空值数及来源；
- 根输出目录下的 `batch_summary.csv`：每个瓦片的处理状态。

数组永远采用 `array[row, col]`，其中 `row=0` 为北侧、`col=0` 为西侧。脚本从每个 WinProp TXT 文件读取 `LOWER_LEFT`、`UPPER_RIGHT` 和 `RESOLUTION`，而不是猜测网格大小。

## 安装

推荐使用 Miniconda/Anaconda 的 conda-forge 包，Windows 上最稳定：

```powershell
conda create -n sigmap python=3.12 -y
conda activate sigmap
conda install -c conda-forge numpy geopandas rasterio -y
```

也可以使用 pip：

```powershell
python -m pip install -r C:\Users\pc\Documents\大创\scripts\prepare_winprop_tiles_requirements.txt
```

## 先做 10 个瓦片的试运行

```powershell
python C:\Users\pc\Documents\大创\scripts\prepare_winprop_tiles.py `
  --input-root D:\桌面\dac\512mdata `
  --output-root D:\桌面\dac\prepared_4m `
  --limit 10 `
  --workers 1
```

确认 `D:\桌面\dac\prepared_4m\tile_000001\` 中有三个输出文件后，再跑全量。脚本默认续跑：已有三个完整输出文件的瓦片会跳过。

默认只读取每个瓦片的 `<tile名称>_result1` 文件夹，避免把 `*_globaltest` 之类的测试结果混进训练集；你的 `tile_000002` 正好有这种额外目录。若将来目录命名不同，再通过 `--result-dir-suffix` 指定。

## 全量运行

先从 4 个并行进程开始；磁盘和 CPU 富余时再试 6 或 8。不要同时启动第二个同样的批处理进程。

```powershell
python C:\Users\pc\Documents\大创\scripts\prepare_winprop_tiles.py `
  --input-root D:\桌面\dac\512mdata `
  --output-root D:\桌面\dac\prepared_4m `
  --workers 4
```

中断后用同一条命令重新运行即可续跑。只有需要重做已完成瓦片时才加入 `--overwrite`。

## 输出含义与功率

`path_gain_4m.npy` 保存的是 WinProp `Path Loss.txt` 的原始 dB 输出，不自动当作 RSRP。WinProp 的 `N.C.`（not computable）点会保存为 `NaN`，训练时应通过 mask 忽略它们，不能替换成 0。你当前的 tile_000001 配置为 0.001 W（0 dBm）和各向同性天线，因此 Power 与 Path Loss 数值相同。若之后改变发射功率，应在训练标签或推理后单独加入相应的 dB 偏置，而不要改变建筑高度图。

## 失败处理

所有失败瓦片会保留在 `batch_summary.csv` 的 `status=error` 行及 `error` 列中。常见原因是缺少 Shapefile、一个瓦片含多份 Path Loss 文件、缺少 `HEIGHT_M` 字段，或坐标系/范围不重叠。

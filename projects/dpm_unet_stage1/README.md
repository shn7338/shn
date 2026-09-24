# 第一级 DPM 代理 U-Net

该模型学习：

```text
[建筑高度图, TX位置/高度图, TX距离图] -> WinProp DPM 路径增益图
```

DPM 图只作为标签，不作为输入，因此模型学到的是 DPM 的快速代理，而不是复制输入。数据沿用 `normalized_4m_v1` 的 2 km 空间分块划分。

## 标准目录

- 规范源码：`D:\桌面\dac\05_legacy_code\code\dpm_unet_stage1`
- 正式部署模型：`D:\桌面\dac\03_runs\models\stage1_dpm_unet_baseline_v1`
- 训练数据：`D:\桌面\dac\01_data\normalized_4m_v1`
- 完整实验结果：`D:\桌面\dac\03_runs\stage1_dpm_unet_runs`

部署模型目录包含 `best.pt`、训练配置、归一化参数、独立测试指标、速度结果和模型说明。推理必须使用模型包内的归一化参数，不能用待预测数据重新计算。

## 文件

- `dataset.py`：读取三通道输入、标签和有效像元 mask；
- `model.py`：三输入、一输出 U-Net；
- `train.py`：混合精度训练、验证、早停和断点保存；
- `evaluate.py`：在验证集或测试集上报告 dB MAE/RMSE；
- `predict.py`：输出单瓦片预测对比图。
- `inference.py`：无标签推理的共享加载、校验、反标准化和输出逻辑；
- `prepare_inference_inputs.py`：从原始建筑高度、网格和 TX 参数生成三个推理输入；
- `infer.py`：预测单个未知瓦片，不需要 DPM 标签；
- `batch_infer.py`：模型只加载一次，批量预测未知瓦片。
- `visualize_split.py`：按训练/验证/测试清单批量生成带建筑的灰度预测图。

## 环境

当前电脑已经安装 PyTorch 2.13.0 + CUDA 13.0。若以后重建环境，可使用：

```powershell
conda activate sigmap
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

安装后检查：

```powershell
conda activate sigmap
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 冒烟训练

先确认代码、GPU和数据加载都能运行：

```powershell
cd D:\桌面\dac\05_legacy_code\code\dpm_unet_stage1
python train.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --output-dir D:\桌面\dac\03_runs\stage1_dpm_unet_runs\smoke `
  --epochs 1 `
  --batch-size 2 `
  --workers 0 `
  --base-channels 8 `
  --train-limit 8 `
  --val-limit 4 `
  --max-train-batches 2 `
  --max-val-batches 1
```

## 正式训练

RTX 5070 Laptop 8 GB 建议从 batch size 16 开始：

```powershell
cd D:\桌面\dac\05_legacy_code\code\dpm_unet_stage1
python train.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --output-dir D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1 `
  --epochs 60 `
  --batch-size 16 `
  --workers 4 `
  --base-channels 32 `
  --learning-rate 3e-4 `
  --patience 12
```

若显存不足，将 `--batch-size 16` 改成 `8`。最佳模型保存在 `best.pt`，最后一个 epoch 保存在 `last.pt`，逐 epoch 指标保存在 `history.csv`。

## 建筑边缘增强实验

该实验不修改原始数据。程序从建筑高度图在线生成第 4 个建筑边缘通道，并对建筑边缘两侧 8 m 内的有效像元使用 2 倍损失权重：

```powershell
python train.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --output-dir D:\桌面\dac\03_runs\stage1_dpm_unet_runs\edge_v1 `
  --epochs 60 `
  --batch-size 16 `
  --workers 4 `
  --base-channels 32 `
  --learning-rate 3e-4 `
  --patience 12 `
  --building-edge-channel `
  --edge-channel-radius-pixels 1 `
  --edge-weight 2.0 `
  --edge-radius-pixels 2
```

训练日志和 `history.csv` 会额外记录 `val_edge_rmse_db` 和 `val_non_edge_rmse_db`。程序保存：

- `best.pt`：全局验证 RMSE 最优；
- `best_edge.pt`：8 m 建筑边缘带验证 RMSE 最优；
- `last.pt`：最后完成的 epoch。

改进是否有效，应同时比较全局、8 m 建筑边缘带和非边缘区域三个测试 RMSE，不能只挑其中一个指标。

断点续训：

```powershell
python train.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --output-dir D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1 `
  --epochs 60 `
  --batch-size 16 `
  --workers 4 `
  --resume D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1\last.pt
```

## 最终测试

模型和超参数定型后再运行一次测试集：

```powershell
python evaluate.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --checkpoint D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1\best.pt `
  --split test `
  --batch-size 16 `
  --workers 4
```

## 单瓦片可视化

下面的 `predict.py` 用于实验对比，会读取 DPM 真值和有效像元 mask：

```powershell
python predict.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --checkpoint D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1\best.pt `
  --tile tile_000001 `
  --output D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1\tile_000001.png
```

## 真实单瓦片推理（不需要 DPM 标签）

### 从原始建筑高度生成输入

如果手中是未归一化的 `building_height_4m.npy`，以及包含 `grid` 边界的 `metadata.json`，先运行：

```powershell
python prepare_inference_inputs.py `
  --building-height D:\path\to\building_height_4m.npy `
  --grid-metadata D:\path\to\metadata.json `
  --model-dir D:\桌面\dac\03_runs\models\stage1_dpm_unet_baseline_v1 `
  --tx-x 438528.0 `
  --tx-y 4373248.0 `
  --tx-height-m 23.0 `
  --frequency-mhz 3500 `
  --receiver-height-m 2 `
  --output-dir D:\path\to\new_tile
```

TX 坐标必须与网格使用相同的投影坐标系。程序会拒绝非 128x128、非 4 m、频率或接收机高度不匹配的输入，避免在模型适用范围之外静默产生错误结果。

### 执行预测

待预测瓦片目录只需要三个已经按训练参数归一化的输入：

```text
building_height_norm.npy
tx_position_height_norm.npy
tx_distance_norm.npy
```

运行：

```powershell
conda activate sigmap
cd D:\桌面\dac\05_legacy_code\code\dpm_unet_stage1

python infer.py `
  --model-dir D:\桌面\dac\03_runs\models\stage1_dpm_unet_baseline_v1 `
  --tile-dir D:\path\to\new_tile `
  --output-dir D:\path\to\prediction_output `
  --palette color
```

如只希望输出黑白可视化，将 `--palette color` 改为 `--palette gray`。这只改变 PNG 显示，不改变模型数值。

输出：

- `path_gain_pred_norm.npy`：标准化空间的模型输出；
- `path_gain_pred_db.npy`：反标准化后的 DPM 路径增益，单位 dB；
- `path_gain_prediction.png`：不覆盖建筑的纯路径增益可视化；
- `path_gain_prediction_with_buildings.png`：灰度路径增益、黑色建筑、TX标记、米制坐标轴和dB色条；
- `inference_metadata.json`：输入范围、模型、归一化、耗时和预测范围。

若已经通过像元中心法生成室外有效区域 mask，可额外传入：

```powershell
--mask D:\path\to\outdoor_valid_mask.npy
```

不要用 `building_height_norm.npy > 0` 直接替代该 mask，因为建筑高度输入使用 `all_touched=True`，会额外包含只被建筑边缘碰到的像元。

## 真实批量推理

输入根目录应包含多个 `tile_*` 子目录，每个子目录包含上述三个输入 NPY：

```powershell
python batch_infer.py `
  --model-dir D:\桌面\dac\03_runs\models\stage1_dpm_unet_baseline_v1 `
  --input-root D:\path\to\new_tiles `
  --output-root D:\path\to\batch_predictions `
  --save-png
```

批量模式只加载和预热模型一次，适合正式生成信道地图。默认已有输出会跳过；需要重算时增加 `--overwrite`。不需要 PNG 时省略 `--save-png`，可以节省磁盘和绘图时间。

首次启动进程会有 CUDA 和模型加载冷启动，稳定前向推理约为数毫秒/瓦片。`inference_metadata.json` 将 `warmup_ms` 与 `inference_ms` 分开记录。

## 批量可视化验证集

只生成灰度路径增益、黑色建筑、TX标记、米制坐标轴和dB色条，不保存预测NPY和普通彩色图：

```powershell
python visualize_split.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --model-dir D:\桌面\dac\03_runs\models\stage1_dpm_unet_baseline_v1 `
  --split val `
  --output-dir D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1\val_visualizations
```

程序从 `val_tiles.txt` 读取全部2,840张验证瓦片。默认跳过已存在图片，可安全断点续跑；需要强制重画时增加 `--overwrite`。正式运行前可以用 `--limit 3` 试画三张。

## 推理速度测试

```powershell
python benchmark.py `
  --data-root D:\桌面\dac\01_data\normalized_4m_v1 `
  --checkpoint D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1\best.pt `
  --output D:\桌面\dac\03_runs\stage1_dpm_unet_runs\baseline_v1\benchmark.json
```

结果同时包含纯 GPU batch=1/16/32 延迟，以及包含 NPY 读取、CPU 到 GPU 拷贝的单瓦片端到端延迟。

## 输入消融实验

依次训练“仅建筑高度”和“建筑高度 + TX”，再复用完整模型的测试结果：

```powershell
.\run_ablations.ps1
```

三个实验使用同一数据划分、随机种子、网络宽度、训练轮数和优化器。输出分别位于 `ablation_height`、`ablation_height_tx` 和 `baseline_v1`，每个目录的 `test_metrics.json` 可直接用于结果表格。

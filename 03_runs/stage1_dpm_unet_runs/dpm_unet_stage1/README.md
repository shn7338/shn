# Stage 1 历史运行代码副本

本目录保留当时的实现。当前源码、规范目录和安装命令以 [主代码 README](../../../projects/dpm_unet_stage1/README.md) 为准；下文含迁移前路径，仅供追溯。

---

# 第一级 DPM 代理 U-Net

该模型学习：

```text
[建筑高度图, TX位置/高度图, TX距离图] -> WinProp DPM 路径增益图
```

DPM 图只作为标签，不作为输入，因此模型学到的是 DPM 的快速代理，而不是复制输入。数据沿用 `normalized_4m_v1` 的 2 km 空间分块划分。

## 文件

- `dataset.py`：读取三通道输入、标签和有效像元 mask；
- `model.py`：三输入、一输出 U-Net；
- `train.py`：混合精度训练、验证、早停和断点保存；
- `evaluate.py`：在验证集或测试集上报告 dB MAE/RMSE；
- `predict.py`：输出单瓦片预测对比图。

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
cd C:\Users\pc\Documents\大创\projects\dpm_unet_stage1
python train.py `
  --data-root D:\桌面\dac\normalized_4m_v1 `
  --output-dir D:\桌面\dac\stage1_dpm_unet_runs\smoke `
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
cd C:\Users\pc\Documents\大创\projects\dpm_unet_stage1
python train.py `
  --data-root D:\桌面\dac\normalized_4m_v1 `
  --output-dir D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1 `
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
  --data-root D:\桌面\dac\normalized_4m_v1 `
  --output-dir D:\桌面\dac\stage1_dpm_unet_runs\edge_v1 `
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
  --data-root D:\桌面\dac\normalized_4m_v1 `
  --output-dir D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1 `
  --epochs 60 `
  --batch-size 16 `
  --workers 4 `
  --resume D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1\last.pt
```

## 最终测试

模型和超参数定型后再运行一次测试集：

```powershell
python evaluate.py `
  --data-root D:\桌面\dac\normalized_4m_v1 `
  --checkpoint D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1\best.pt `
  --split test `
  --batch-size 16 `
  --workers 4
```

## 单瓦片可视化

```powershell
python predict.py `
  --data-root D:\桌面\dac\normalized_4m_v1 `
  --checkpoint D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1\best.pt `
  --tile tile_000001 `
  --output D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1\tile_000001.png
```

## 推理速度测试

```powershell
python benchmark.py `
  --data-root D:\桌面\dac\normalized_4m_v1 `
  --checkpoint D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1\best.pt `
  --output D:\桌面\dac\stage1_dpm_unet_runs\baseline_v1\benchmark.json
```

结果同时包含纯 GPU batch=1/16/32 延迟，以及包含 NPY 读取、CPU 到 GPU 拷贝的单瓦片端到端延迟。

## 输入消融实验

依次训练“仅建筑高度”和“建筑高度 + TX”，再复用完整模型的测试结果：

```powershell
.\run_ablations.ps1
```

三个实验使用同一数据划分、随机种子、网络宽度、训练轮数和优化器。输出分别位于 `ablation_height`、`ablation_height_tx` 和 `baseline_v1`，每个目录的 `test_metrics.json` 可直接用于结果表格。

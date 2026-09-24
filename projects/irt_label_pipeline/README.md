# Stage 2 与随机基站反演：当前代码入口

本目录沿用 `irt_label_pipeline` 历史名称，**当前 Stage 2 标签来自 Sionna RT**。先阅读 [当前数据、模型与结果](SIONNA_CURRENT.md)。旧 WinProp IRT2/IRT6 仅用于历史对照；Stage 1 原始 DPM 基线仍以 WinProp DPM 为标签。

## 两条已开展的训练路线

| 路线 | 数据 | 当前结果 |
| --- | --- | --- |
| 固定中心基站 | Sionna 3.5 GHz、depth 8、27,360 瓦片 v3 | Stage 2A 测试 RMSE 5.512 dB；Stage 2B 在 100 稀疏点下 5.765 dB，均通过各自记录的门槛 |
| 未知位置的随机基站 | Sionna 随机基站 Pilot → Scale-4 | 位置/功率/方向估计器 + Stage 1/2A/2B；V5 已选作展示，Scale-4 原始测试 RMSE 6.897 dB，严格升级门槛未通过 |

两条路线的数据分布和任务不同，不能用 5.765 与 6.897 直接判断哪个版本更好。上述指标均为仿真测试结果，手机试采尚不能作为模型正式实测验证。

## 代码分工

| 文件或文件组 | 用途 |
| --- | --- |
| `build_sionna_scene.py`、`run_sionna_paper_pilot.py`、`validate_sionna_paper_pilot.py` | 建立 Sionna 场景、逐场景仿真与验证 |
| `run_sionna_dataset.py`、`start_sionna_35ghz_all_tiles.ps1` | 固定基站 v3 批量生成、续跑、状态和分片整理 |
| `stage2a_dataset.py`、`stage2b_dataset.py`、`stage2_models.py` | 固定基站数据加载、稀疏采样和 Stage 2 模型 |
| `train_stage2a_iso_refine.py`、`train_stage2b_directional_ss.py` | 固定基站 Sionna Stage 2A/2B 训练 |
| `run_bs_inversion_pilot.py`、`validate_bs_inversion_pilot.py` | 根据所选配置生成/验证 Pilot 或 Scale-4 随机基站数据；文件名 Pilot 同样用于 Scale-4 |
| `bs_inversion_dataset.py`、`bs_parameter_model.py`、`train_bs_parameter_estimator.py` | 随机基站样本加载、参数估计器及其训练 |
| `finetune_pilot_stage1_stage2a.py`、`finetune_pilot_stage2b.py` | 随机位置适配、位置编码以及 Stage 2B 微调 |
| `latent_bs_fusion.py`、`train_stage2b_latent_adapter_v5.py` | V4 隐变量特征、V5 多尺度适配器与 FiLM 训练 |
| `train_joint_estimator_stage2b_v6.py` | V6 估计器与信号图网络联合微调 |
| `evaluate_*.py`、`assess_*.py`、`render_*.py`、`visualize_*.py` | 原始指标、门槛判断及图像生成；训练完成与门槛通过是两种状态 |
| `config_*.json`、`start_*.ps1`、`watch_*.ps1` | 各轮参数、启动和监看；使用前核对配置内的数据来源、权重和输出目录 |
| `run_winprop_direct_pilot.py`、`preprocess_irt.py`、`compare_sionna_winprop.py` 等 | 历史 WinProp IRT 仿真、处理与对照 |
| `build_irt_shards.py`、`verify_irt_shards.py` | 分片制作/校验等共用工具；带 IRT 的名称不能单独用来判断标签来源 |

## 阅读顺序和运行前提

1. 看 [SIONNA_CURRENT.md](SIONNA_CURRENT.md) 确认自己要读的是固定基站还是随机基站路线。
2. 看对应配置里的 `shard_root` / `dataset_root`、checkpoint 和归一化文件；指标入口见 [模型索引](../../03_runs/models/README.md)。
3. 仿真环境依赖在 [requirements_sionna201_paper.txt](requirements_sionna201_paper.txt)；训练使用独立的 PyTorch 环境，见 [Stage 1 环境说明](../dpm_unet_stage1/README.md)。
4. 数据数组、模型权重及 Python 环境未上传 GitHub。配置与 PowerShell 脚本仍有原电脑的绝对路径，队友需取得数据/权重并按自己的路径设置；源码本身以本目录及 `projects/dpm_unet_stage1/` 为入口。
5. 历史生成命令与详细物理参数保存在 [实验过程记录](EXPERIMENT_HISTORY.md)。已完成实验的原始 JSON/哈希记录保留原样。

`irt_normalization`、`normalization_irt.json`、部分历史指标中的 “IRT” 是沿用名称。例如 V5 配置的 `irt_normalization` 实际指向 Sionna v3。判断来源应读路径、配置和生成记录。

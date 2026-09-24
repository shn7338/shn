# 大创项目导览

本目录汇集无线信号地图项目的代码、数据准备记录、仿真实验、模型训练结果和中期答辩材料。先看 [仿真标签与 Stage 2 流程](projects/irt_label_pipeline/README.md)、[Stage 1 DPM 代理模型](projects/dpm_unet_stage1/README.md) 或 [中期答辩材料索引](中期答辩佐证材料_20260924/00_先看这里.md)。不同版本的实验记录都被保留，**目录存在不表示该轮实验已经达到目标**。

项目的大致流程是：建筑与 WinProp 数据经 `scripts/` 处理为 4 m 瓦片，Stage 1 模型预测 DPM 路径增益；WinProp 或 Sionna 生成后续仿真标签，`projects/irt_label_pipeline/` 负责 Stage 2 训练与评估。北交大场景的早期试验另见 `bjtu_osm_buildings/` 和 `first try/`。

下表列出实际目录；每个 `tile_*`、检查点和临时文件不逐一展开。

| 目录 | 内容 |
| --- | --- |
| [`01_data/`](01_data/README.md) | WinProp 原始数据、天线方向图、预处理数据和预测准备记录 |
| [`02_inversion/`](02_inversion/README.md) | 随机基站位置与参数反演的数据集和工作目录 |
| [`03_runs/`](03_runs/README.md) | 模型、训练、推断、评估及展示结果 |
| [`04_simulation/`](04_simulation/README.md) | IRT、Sionna、WinProp 的多轮仿真运行 |
| [`05_legacy_code/`](05_legacy_code/README.md) | 原 D/E 盘的旧代码与第三方工具副本 |
| [`06_reference/`](06_reference/README.md) | 开题、参考文献、归档和单样本评估图片 |
| [`07_backup/`](07_backup/README.md) | 历史检查点及恢复材料 |
| [`08_runtime/`](08_runtime/README.md) | 本机 Sionna Python 运行环境 |
| [`09_local_artifacts/`](09_local_artifacts/README.md) | 诊断、中间结果、临时文件和工作缓存 |
| [`bjtu_osm_buildings/`](bjtu_osm_buildings/README.md) | 北交大 OSM 建筑轮廓与高度整理代码 |
| [`first try/`](first%20try/README.md) | 早期 Geo2SigMap/Sionna 场景与信号地图试验 |
| `projects/` | 主要模型、仿真与训练代码 |
| `scripts/` | WinProp 数据制作、检查、修复和预处理脚本 |
| `tools/` | 地理数据、文献、佐证材料和迁移辅助工具 |
| `docs/` | 迁移记录、环境记录、设计与实施计划 |
| `presentations/` | V5 展示稿、流程图与逐页预览 |
| [`中期答辩佐证材料_20260924/`](中期答辩佐证材料_20260924/00_先看这里.md) | 数据记录、代码快照与来源校验清单 |

完整迁移路径、旧路径兼容入口及 E 盘备份位置见 [目录迁移记录](docs/目录迁移记录_20260924.md)。

根目录的 `README.md` 是本导览，`.gitignore` 控制哪些本机数据和生成文件不进入 Git，`.gitattributes` 记录 Git 文件处理规则。

## 数据、模型与仿真目录

### `01_data/`：数据与数据准备

| 子目录 | 存放内容 |
| --- | --- |
| `512mdata/` | 512 m 瓦片的 WinProp 建筑数据库、仿真输入/输出；顶层 `tile_index.csv` 和生成报告说明瓦片范围。 |
| `antenna_patterns/` | 各向同性及方向性天线的 `.apa` 方向图、方位角记录。 |
| `normalized_4m_v1/` | 4 m 分辨率的训练/验证/测试划分、归一化参数、TX 特征摘要和逐瓦片数组。 |
| [`prediction_records/`](01_data/prediction_records/README.md) | 27,360 个 ODB 的可用清单、502 个替换记录、汇总和 10 瓦片抽样检查名单。 |
| `prepared_4m/` | 对齐后的建筑高度图与 WinProp 路径增益图，`batch_summary.csv` 记录处理情况。 |
| `raw_datasets/` | 原始地理数据包及后续清洗使用的资料。 |

### `02_inversion/`：随机基站反演

| 子目录 | 存放内容 |
| --- | --- |
| `bs_inversion_pilot_v1/` | pilot 数据集、说明、归一化参数及最终报告。 |
| `bs_inversion_scale4_v1/` | Scale-4 数据集和对应的说明、汇总。 |
| `E_work/` | 原 E 盘 `dac_bs_inversion_scale4_v1_work` 工作目录；迁移时为空。 |

### `03_runs/`：训练和评估

| 子目录 | 存放内容 |
| --- | --- |
| `E_stage/` | 原 E 盘 Stage 1 预测，以及 Stage 2A/2B 小规模 smoke 训练的配置、历史和指标。 |
| `experiments/` | 基站反演、位置编码、V3–V6、敏感性和单样本实验的配置、门槛记录及可视化。 |
| `inference_demo/` | 单样本/批量推断的输入、预处理输入与输出。 |
| `models/` | 各轮 Stage 1、Stage 2 和基站参数估计模型的权重、模型卡与结果记录。 |
| `stage1_dpm_unet_runs/` | Stage 1 基线、建筑边缘增强与高度通道消融的训练结果；`dpm_unet_stage1/` 是当时使用的代码副本。 |

`E_stage/` 下，`dac_stage1_dpm_predictions_positive3200_v2` 是 3,200 瓦片预测；两个 `dac_stage2a_*` 和两个 `dac_stage2b_*` 目录分别记录 WinProp pilot 与 Sionna paper 参数下的 smoke 训练。`stage1_dpm_unet_runs/` 下的 `baseline_v1`、`edge_v1`、`ablation_height` 和 `ablation_height_tx` 分别保存基线、边缘增强及消融结果。

`experiments/` 下的 `bs_sensitivity_*` 是参数敏感性分析，`estimator_scale4_*` 对比基站估计器，`single_test_*` 保存单测试样本评估，`scale4_visual_comparison_v3/v4/v5/v6` 保存各版本可视化；带 `smoke`、`legacy` 的目录是试运行或旧版对照。根目录的 JSON/Markdown 保存配置、分组指标和升级门槛。

`models/` 按模型族分目录：`bs_parameter_estimator_*` 是基站参数估计；`stage1_dpm_unet_baseline_v1` 是 DPM 代理基线；`stage1_bs_inversion_*`、`stage1_position_encoding_*` 是 Stage 1 输入方案；`stage2a_iso_refine_*`、`stage2a_bs_inversion_*`、`stage2a_position_encoding_*` 是 Stage 2A；`stage2b_directional_ss_*`、`stage2b_bs_inversion_*`、`stage2b_position_encoding_*`、`stage2b_latent_fusion_scale4_v4`、`stage2b_multiscale_adapter_scale4_v5` 是 Stage 2B；`joint_estimator_stage2b_scale4_v6` 是联合估计方案。各目录中若有 `REPORT.md`、`MODEL_CARD.md`，可先看其结果摘要。

### `04_simulation/`：仿真运行

| 子目录 | 包含的运行及用途 |
| --- | --- |
| `irt/` | `dac_irt_v2_8bounce` 保存早期八次交互尝试；`dac_irt_v2_winprop2020_max6` 保存 WinProp 2020 上限相关尝试。 |
| `sionna/` | `dac_sionna_35ghz_pilot`、`dac_sionna_paper_pilot`、`dac_sionna_paper8_v1` 是先导与论文参数试验；`dac_sionna_35ghz_depth8_3200_v1` 是 3,200 瓦片轮次；`dac_sionna_35ghz_depth8_27360_v1/v2/v3` 是全量数据的连续三轮记录；`dac_sionna_winprop_irt2_alignment` 用于两种仿真结果的对齐比较。 |
| `winprop/` 直接仿真 | `dac_winprop_irt2_direct_v1`、`dac_winprop_irt2_direct_pilot32_v1` 是直接仿真与 32 瓦片 pilot；`dac_winprop_irt2_direct_3200_v1`、`dac_winprop_irt2_direct_3200_positive_v2` 是 3,200 瓦片及正坐标替换轮次；`dac_winprop_irt6_direct_pilot_positive_v3` 是 IRT6 pilot。 |
| `winprop/` 诊断与验证 | `dac_winprop_irt2_ray_diagnostic_v1`、`ray100_diagnostic_v1`、`ray500_diagnostic_v1/v2`、`no_ray_cap_diagnostic` 排查射线设置；`pattern_gain0_diagnostic`、`verified_pattern_diagnostic_v1`、`verified_pattern_validation_v1`、`roundtrip_pattern_validation_v3` 检查天线方向图与增益。以上省略了相同的 `dac_winprop_irt2_` 前缀。 |

运行目录顶层的 `progress.json`、`run_manifest.json`、`selection_metadata.json` 等用于追溯配置和进度。诊断、停止及效果不佳的轮次也保留着，不能只凭目录名判断结果通过验证。

### `05_legacy_code/` 至 `09_local_artifacts/`

| 目录 | 子目录用途 |
| --- | --- |
| `05_legacy_code/code/` | 原 D 盘旧代码，主要有 `dpm_unet_stage1/` 的历史副本。 |
| `05_legacy_code/github code/` | 原本独立的 Geo2SigMap 代码仓库副本，作为历史/第三方代码留在本机。 |
| `05_legacy_code/dac_tools/` | 原 E 盘 `geo2sigmap/`、`geo2sigmap_full/` 工具副本。 |
| `06_reference/开题/`、`参考文献/`、`归档/` | 开题文件、阅读材料及历史归档；根目录另有一张代表性评估图片。 |
| `07_backup/dac_checkpoints/` | 历史 Git bundle、恢复记录及仿真前检查点；不是当前训练输入。 |
| `08_runtime/dac_sionna_rt_env/` | 本机 Sionna Python 虚拟环境和依赖。 |
| `09_local_artifacts/prediction_*/` | ODB 复检、重建、诊断、备份及 smoke 测试的本机输出。 |
| `09_local_artifacts/_render/`、`_work/`、`pptmaster_staging/`、`presentation_inspection/` | 文档与展示材料的临时工作文件。 |
| `09_local_artifacts/outputs/`、`tmp/`、隐藏 `.tmp_*` | 程序生成的输出与缓存。 |

## 代码放在哪里、做什么

### `projects/`：模型与仿真主代码

| 子目录或文件组 | 作用 |
| --- | --- |
| [`projects/dpm_unet_stage1/`](projects/dpm_unet_stage1/README.md) | `dataset.py`、`model.py` 定义数据和 DPM 代理 U-Net；`train.py`、`evaluate.py`、`benchmark.py` 训练与评估；`prepare_inference_inputs.py`、`infer.py`、`batch_infer.py` 做单样本和批量预测；`visualize_split.py`、`generate_stage1_report.py` 出图和报告。 |
| [`projects/irt_label_pipeline/`](projects/irt_label_pipeline/README.md) 的仿真文件 | `run_winprop_direct_pilot.py`、`run_sionna_dataset.py`、`run_sionna_paper_pilot.py` 生成/运行标签；`preprocess_irt.py`、`build_irt_shards.py`、`verify_irt_shards.py` 处理和校验数据；`select_spatial_irt_tiles.py`、`replace_negative_irt_selection.py` 负责选样与修正。 |
| `projects/irt_label_pipeline/` 的模型文件 | `stage2a_dataset.py`、`stage2b_dataset.py` 读取训练数据；`stage2_models.py`、`bs_parameter_model.py` 定义模型；`train_stage2a_iso_refine.py`、`train_stage2b_directional_ss.py`、`train_stage2b_latent_adapter_v5.py`、`train_joint_estimator_stage2b_v6.py`、`train_bs_parameter_estimator.py` 训练各方案。 |
| `projects/irt_label_pipeline/` 的评估与配置 | `assess_*.py`、`evaluate_*.py`、`render_*.py` 汇总指标与生成展示图；`config_*.json` 固定实验参数；`start_*.ps1`、`watch_*.ps1` 用于启动和监看；`tests/` 放专项测试。 |
| `projects/dachuang_kaiti_defense_ppt169_20260531/` | 早期开题答辩 PPT 的本机生成工作目录，属于产物，未纳入 GitHub。 |

`projects/` 是队友在 GitHub 上阅读当前工作代码的入口。`03_runs/stage1_dpm_unet_runs/dpm_unet_stage1/` 和 `05_legacy_code/code/dpm_unet_stage1/` 是此前运行或迁移保留的副本；一些旧命令仍引用这些本机路径。

### `scripts/`：WinProp 数据流水线

| 文件组 | 作用 |
| --- | --- |
| `clean_winprop_shapefile.py`、`generate_winprop_odb.ps1`、`batch_generate_odb_*.ps1`、`generate_winprop_antenna_patterns.py` | 清理建筑矢量、制作 ODB 与天线方向图。 |
| `batch_check_winprop_odb.ps1`、`start_wallman_check_parallel.ps1`、`show_wallman_check_status.ps1`、`recheck_prediction_warning_odb.ps1`、`rebuild_prediction_worker.ps1`、`start_prediction_odb_rebuild.ps1`、`promote_prediction_ready_odb.ps1` | 检查 ODB、处理警告、重建和正式替换。 |
| `batch_winprop_path_loss*.ps1`、`run_winprop_prediction_smoke.ps1` | 批量生成 WinProp 路径增益，或用代表性瓦片做快速验证。 |
| `prepare_winprop_tiles.py`、`normalize_winprop_dataset.py`、`prepare_tx_features.py` | 对齐建筑/路径增益图、划分数据、归一化并生成发射机特征；同目录的 `README_*.md` 给出具体运行方式。 |
| `repair_dac_json_paths.py` | 目录迁移后修正部分 JSON 中的本机路径。 |
| `logs/`、`__pycache__/` | 本机运行日志与 Python 缓存，不作为源代码。 |

### `tools/`：辅助程序

| 文件组 | 作用 |
| --- | --- |
| `clean_beijing_wallman.py`、`deduplicate_wallman_shapefile.py`、`split_wallman_512m_tiles.py`、`append_regions_512m_tiles.py`、`filter_tiles_by_union_coverage.py` | 清洗地理建筑数据、去重、切成 512 m 瓦片、补充地区和筛选覆盖。 |
| `winprop_convert_oda_to_odb.py`、`winprop_odb_converter/` | WinProp ODA/ODB 转换脚本及其预留工作目录。 |
| `analyze_cnki_pdfs.py`、`extract_pdf_pages.py`、`render_pdf_page.py`、`cnki-controlled-chrome/` | 文献检索/下载辅助、PDF 内容与页面处理；Chrome 的 `profile/`、`node_modules/` 是本机文件。 |
| `build_midterm_evidence.py` | 从已有成果复制轻量样本，生成答辩佐证包与来源校验清单。 |
| `rewrite_dac_paths.py` | 记录并执行 2026-09-24 目录合并后的活动代码路径迁移。 |
| `resume_http_download.ps1`、`wait_for_ablation_report.ps1` | 断点续传及等待消融实验报告的辅助脚本。 |

### 早期场景代码、文档与展示

| 目录 | 子目录与文件用途 |
| --- | --- |
| [`bjtu_osm_buildings/`](bjtu_osm_buildings/README.md) | `main.py` 从 OSM 提取校园建筑轮廓和高度信息；`tests/` 放检查；`output/` 放 CSV、GeoJSON 和地图；`docs/superpowers/{plans,specs}/` 留设计记录。 |
| [`first try/`](first%20try/README.md) | `scripts/generate_bjtu_scene.py` 建场景，`run_bjtu_radio_map.py` 跑 Sionna 图，`run_bjtu_cascaded_unet.py` 演示上游模型，`generate_bjtu_figures.py`、`export_bjtu_scene_glb.py` 出图和导出场景；`scenes/`、`outputs/`、`models/`、`cache/`、`geo2sigmap-upstream/` 是本机输入/输出或上游副本，`.venv/` 是环境。 |
| `docs/environment/`、`docs/superpowers/{plans,specs}/` | 环境记录，以及场景构建和单样本评估的设计、实施计划；`docs/目录迁移记录_20260924.md` 说明路径变更。 |
| `presentations/V5模型展示/` | V5 展示 PPTX、SVG/PNG 流程图；同名子目录放逐页 PNG 和拼图预览。 |
| `中期答辩佐证材料_20260924/01_数据记录/` | `01_校园手机试采`、`02_WinProp与预处理`、`03_Sionna合成数据`、`04_随机基站数据`、`05_训练评估记录` 五类记录。 |
| `中期答辩佐证材料_20260924/02_代码/` | `源码/`、`E盘工具源码/`、代码索引、源码快照和构建脚本。 |
| `中期答辩佐证材料_20260924/03_E盘实验记录/` | 33 个原 E 盘目录的记录样本、`过程总览.md` 和 `E盘目录总览.csv`；材料根目录的 `来源与校验清单.csv` 记录来源及 SHA256。 |

## 旧路径与本机缓存

本机根目录还可看到 `512mdata`、`models`、`experiments`、`code`、`prediction_recheck` 等旧名字。它们是**指向上面实际目录的隐藏 Windows 目录连接**，供旧脚本读取；不是另一份数据，也不会随 GitHub 克隆。`.git/` 是版本历史；`.agents/`、`.codex/`、`.pytest_cache/`、`_render/`、`_work/` 等是本机工具配置、缓存或兼容入口。详情见 [目录迁移记录](docs/目录迁移记录_20260924.md)。

## GitHub 分享范围

仓库包含可阅读的项目代码、配置、中期答辩佐证材料，以及 `01_data/`、`02_inversion/`、`03_runs/`、`04_simulation/` 等目录的说明和部分轻量实验记录。本机 `01_data/` 约 79 GiB，完整数据、训练权重、大规模仿真输出、备份和 Python 环境没有上传。**队友可以了解项目和实验过程，但只克隆 GitHub 仓库不能直接完整复现实验。**部分历史清单仍含原电脑的绝对路径，运行前需要按新电脑的数据位置调整。

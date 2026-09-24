# 大创项目目录

本目录是合并后的主工作目录。先看 [`projects/irt_label_pipeline/README.md`](projects/irt_label_pipeline/README.md) 了解仿真标签与模型流程；中期答辩已有材料从 [`中期答辩佐证材料_20260924/00_先看这里.md`](中期答辩佐证材料_20260924/00_先看这里.md) 开始。

| 目录 | 内容 |
| --- | --- |
| `projects/`、`scripts/`、`tools/` | 当前项目代码、配置与数据处理脚本 |
| `bjtu_osm_buildings/`、`first try/` | 早期场景构建和试验代码 |
| `01_data/` | WinProp 原始数据、天线方向图、原始与预处理数据，以及 `prediction_records/` 中的预测准备记录 |
| `02_inversion/` | 随机基站反演数据集及工作目录 |
| `03_runs/` | 模型权重、训练、推断及实验结果 |
| `04_simulation/` | 原 E 盘的 Sionna、WinProp、IRT 多轮仿真资料 |
| `05_legacy_code/` | 原 D 盘 `code`、`github code` 和原 E 盘 `dac_tools` |
| `06_reference/` | 开题材料、参考文献和单样本评估图片 |
| `07_backup/`、`08_runtime/` | E 盘历史备份与 Sionna Python 环境 |
| `09_local_artifacts/` | 诊断输出和本机临时产物 |
| `presentations/` | V5 展示稿和流程图 |
| `中期答辩佐证材料_20260924/` | 数据记录、代码快照与来源校验清单 |

具体迁移路径、旧路径兼容入口及备份位置见 [`docs/目录迁移记录_20260924.md`](docs/目录迁移记录_20260924.md)。原有数据和模型文件名保留，代码及配置中的目录路径已按新位置修改。

## GitHub 分享范围

仓库包含当前项目代码、配置、中期答辩佐证材料，以及 `01_data/`、`02_inversion/`、`03_runs/`、`04_simulation/` 等目录的说明和部分轻量记录。每个目录的 `README.md` 说明收录范围。本机约 79 GiB 的原始数据、模型权重、大规模仿真结果、备份和运行环境没有纳入普通 Git；这些文件需要通过其他方式提供给队友。部分历史记录仍含本机绝对路径，复现实验时需按新电脑的位置调整。

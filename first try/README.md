# 北京交通大学 Geo2SigMap / Sionna RT 地图

本项目按论文 *Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic
Databases* 的场景与射线追踪阶段，为北京交通大学主校区建立一块可运行的无线数字孪生地图。

> 这是早期 3.66 GHz 场景及上游权重演示。当前自训练模型采用 3.5 GHz Sionna RT 2.0.1 数据；请从 [当前路线](../projects/irt_label_pipeline/SIONNA_CURRENT.md) 阅读 V5。这里的 Sionna 1.2.2、方向角约定与依赖属于独立旧实验。

## 方法对应

| 项目 | 本项目设置 | 论文设置 |
| --- | --- | --- |
| 区域 | 北交大主校区中心 `39.9504404 N, 116.3360690 E` | 测试区域中心部署 BS |
| 窗口 | `512 m x 512 m` | `512 m x 512 m` |
| 栅格分辨率 | `4 m`，即 `128 x 128` | `4 m`，即 `128 x 128` |
| 载频 | `3.66 GHz` | `3.66 GHz` |
| TX 高度 | 建筑最高点上方 `5 m` | `max(B) + 5 m` |
| RX 高度 | `2 m` | `2 m` |
| TX 阵列 | `iso` 及朝北的 `tr38901` 方向性示例 | 各向同性及方向性天线 |
| 传播机制 | LOS、镜面反射、绕射 | 反射、绕射 |
| 最大深度/射线数 | `8` / `7,000,000` | `8` / `7,000,000` |

建筑轮廓来自 OpenStreetMap，`scene.xml` 和 PLY 网格由当时固定版本的 Geo2SigMap Python
管线生成，路径增益图由 Sionna RT 1.2.2 计算。该固定版本的高度辅助函数会忽略仅含
`building:levels` 的对象；`scripts/generate_bjtu_scene.py` 对这点做了本地修正，
并为没有高度信息的建筑固定随机种子，以便结果可重现。

由于没有提供北交大真实基站的位置、方向和下倾角，方向性示例采用论文的中心基站
假设，并按上游 notebook 的方位角转换规则设置为朝北；它是可复现实验场景，不是
校园运营网络的部署复刻。

## 运行

从仓库根目录先执行 `Set-Location "first try"`，在本目录独立环境中运行。`requirements.txt` 固定了上游提交，生成数据和权重未上传。此脚本按楼层数 × 3.5 m 估算高度；`bjtu_osm_buildings/` 的整理工具按 × 3 m 估算，两者是不同试验规则。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\scripts\generate_bjtu_scene.py
.\.venv\Scripts\python.exe .\scripts\run_bjtu_radio_map.py
.\.venv\Scripts\python.exe .\scripts\run_bjtu_radio_map.py --antenna-pattern tr38901 --azimuth-deg 0 --output .\outputs\BJTU_Main_512\directional_north
```

运行上游发布的级联 U-Net 权重前，先下载并解压模型：

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
Invoke-WebRequest -Uri "https://github.com/functions-lab/geo2sigmap/releases/download/v1.0.0/geo2sigmap_pretrained_weights.zip" -OutFile .\models\geo2sigmap_pretrained_weights.zip
Expand-Archive .\models\geo2sigmap_pretrained_weights.zip .\models\geo2sigmap_pretrained_weights -Force
.\.venv\Scripts\python.exe .\scripts\run_bjtu_cascaded_unet.py
```

快速调试时可减少射线数：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_bjtu_radio_map.py --samples 100000
```

## 输出

- `scenes/BJTU_Main_512/scene.xml`: 可由 Sionna RT 加载的 3D 场景。
- `scenes/BJTU_Main_512/2D_Building_Height_Map_4m.png`: 建筑高度图。
- `outputs/BJTU_Main_512/iso/bjtu_radio_map.png`: 建筑、路径增益和合成信号强度图。
- `outputs/BJTU_Main_512/directional_north/bjtu_radio_map.png`: 朝北方向性天线的合成 SS 输入图。
- `outputs/BJTU_Main_512/iso/*.npy`: 可用于后续训练或分析的数值矩阵。
- `outputs/BJTU_Main_512/iso/metadata.json`: 仿真参数记录。
- `outputs/BJTU_Main_512/cascaded_unet/bjtu_cascaded_unet_map.png`: 发布权重的两阶段 U-Net 演示结果。

当前输出是论文管线中的 OSM 场景与 Sionna 合成路径增益/信号强度阶段。论文中的
第二阶段级联 U-Net 已可用上游 release 权重和从方向性 Sionna 合成图抽取的 100 个稀疏
点演示；但若要将结果解释为经过校准的北交大真实 RSRP 全图，仍需匹配频率与目标小区的实测数据、信号口径/链路预算校准及独立留出测点验证；仅替换稀疏输入还不能证明达到真实 RSRP 精度。

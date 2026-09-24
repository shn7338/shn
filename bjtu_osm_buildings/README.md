# 北京交通大学 OSM 建筑高度数据整理工具

这个小项目从 [OpenStreetMap](https://www.openstreetmap.org/) 的
[Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) 获取北京交通大学
建筑轮廓和高度相关标签，并生成 CSV、GeoJSON 与交互式 HTML 地图。输出可以继续
用于无线信道地图构建、QGIS 检查或 Sionna 场景准备。

## 查询范围

程序默认覆盖 OSM 上的两个校园边界：

| 校区 | OSM 对象 |
| --- | --- |
| 北京交通大学主校区 | `way/266512538` |
| 北京交通大学东校区 | `way/266297360` |

程序会先尝试题目中要求的名称 area 查询：

```overpass
area["name"="北京交通大学"]->.searchArea;
```

如果该查询没有建筑或失败，则自动使用上表两个校园 boundary way 转换成 area 后
合并查询。本项目早期运行时，OSM 名称 area 没有直接返回校园建筑，因此一般会看到运行输出
显示查询方式为 `campus_boundary_ways`。

所有路径都会查询两种建筑对象，并请求轮廓 geometry：

```overpass
way["building"](...);
relation["building"](...);
out body geom;
```

## 安装

在 PowerShell 中从仓库根目录执行：

```powershell
cd .\bjtu_osm_buildings
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

这里的 `python` 应指向可用的 Python 解释器。

## 运行

默认抓取主校区与东校区，并把结果写入 `output/`：

```powershell
.\.venv\Scripts\python.exe .\main.py
```

如果你想人工指定研究区域，可使用 Overpass 的 `south,west,north,east` 顺序输入
纬经度 bbox。传入 bbox 后，程序会直接查询这个矩形而不再尝试校园 area：

```powershell
.\.venv\Scripts\python.exe .\main.py --bbox "39.9438,116.3299,39.9542,116.3438"
```

常用可选参数：

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --output-dir .\output `
  --overpass-url "https://overpass-api.de/api/interpreter" `
  --request-timeout 90 `
  --retries 3
```

如果 Overpass 暂时超时，程序会自动重试。公共 Overpass 服务可能忙碌，请避免短
时间高频反复请求。

## 分类规则

每栋建筑只属于一个类别，判断顺序非常重要：

| `category` | 条件 | 高度处理 |
| --- | --- | --- |
| `has_height` | 存在 `height` 标签 | 保留原值，并尽量解析为米制 `height_m` |
| `levels_only` | 没有 `height`，但存在 `building:levels` | 可解析时计算 `estimated_height = building:levels * 3` 米 |
| `missing_height_and_levels` | 前两类信息都没有 | 标注为“缺少高度和楼层数信息” |

例如一栋建筑同时有 `height=18 m` 和 `building:levels=6`，它只进入
`has_height`，不会在楼层估算类中再次出现。

注意：`estimated_height` 是为了模型准备而提供的粗略估算，不是 OSM 记录的真实
建筑高度。每条记录的 `height_note` 会明确写出这一点。

## 输出文件

程序默认在 `output/` 目录生成：

| 文件 | 说明 |
| --- | --- |
| `buildings_with_height.csv` | 含 `height` 标签的建筑 |
| `buildings_with_levels_only.csv` | 仅含楼层数的建筑 |
| `buildings_without_height_or_levels.csv` | 高度与楼层均缺失的建筑 |
| `all_buildings_classified.csv` | 所有建筑，并带 `category` 字段 |
| `buildings_classified.geojson` | 保留面轮廓的 GIS 数据 |
| `bjtu_building_height_map.html` | 可点击查看字段的分类交互地图 |

CSV 与 GeoJSON 的主要字段包括：

| 字段 | 说明 |
| --- | --- |
| `osm_id`, `osm_type` | OSM 对象编号和 `way`/`relation` 类型 |
| `name` | 原始名称；缺失时为 `未命名建筑_<osm_id>` |
| `building` | OSM 建筑类型 |
| `height`, `height_m` | 原始高度和可解析的米制高度 |
| `building:levels`, `estimated_height` | 原始楼层数与三米每层估算值 |
| `height_source`, `height_note`, `category` | 高度来源及分类说明 |
| `centroid_lat`, `centroid_lon` | 建筑轮廓中心点 |
| `geometry_status` | geometry 是否成功构造 |
| `tags`, `geometry` | 原始标签和 GeoJSON 轮廓；CSV 中保存为 JSON 文本 |

HTML 地图配色为：

| 颜色 | 类别 |
| --- | --- |
| 红色 | 有 OSM `height` 标签（未实地核验） |
| 橙色 | 只有 `building:levels`，高度为估算 |
| 蓝色 | 高度与楼层数都缺失 |

地图还会在有可用数据的建筑中心常驻显示短标签：`H: 11 m` 表示 OSM
提供的高度，`L: 4 层` 表示 OSM 提供的楼层数。缺少两项信息的蓝色建筑不加
常驻文字，以免遮挡校园轮廓；其详情仍可点击查看。

## 用于 GIS 与无线信道地图

在 QGIS 中可直接拖入 `output/buildings_classified.geojson`，以 `category` 分类
着色，并人工补录缺失高度。用于 Sionna 或自建场景时，建议优先使用：

1. `height_m` 不为空的 OSM 标注高度。
2. `estimated_height` 作为明确标记的替代高度，并在实验记录中说明估算规则。
3. `missing_height_and_levels` 建筑人工调查或外部数据补充后再用于精细仿真。

OpenStreetMap 是协作式数据库，标签数量会随社区编辑变化；重新运行得到的是当时
的最新 OSM 快照，并不等于经过实地测绘核验的完整建筑高度数据库。

## 测试

测试使用本地模拟 OSM 数据，不消耗 Overpass 请求次数：

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q
```

测试覆盖高度解析、分类优先级、way/relation 轮廓、area/bbox 查询策略、请求
重试、CSV/GeoJSON 导出以及 HTML 地图生成。

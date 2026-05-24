# 北京交通大学 OSM 建筑高度数据工具设计

## 目标

在现有 Geo2SigMap / Sionna 工程旁新增独立子项目 `bjtu_osm_buildings/`，
从 OpenStreetMap 的 Overpass API 获取北京交通大学主校区与东校区范围内的
全部建筑对象，分类整理高度信息，并产出可供无线信道地图、Sionna 和 GIS 工具
继续处理的 CSV、GeoJSON 与交互地图。

本工具只整理 OSM 明确提供的数据及可识别的楼层数估算。没有 `height` 的建筑
不会被写成具有真实建筑高度。

## 子项目布局

新增子项目位于 `bjtu_osm_buildings/`，不覆盖根目录已有仿真工程的
`README.md` 与 `requirements.txt`。

```text
bjtu_osm_buildings/
  main.py
  requirements.txt
  README.md
  tests/
  output/
    buildings_with_height.csv
    buildings_with_levels_only.csv
    buildings_without_height_or_levels.csv
    all_buildings_classified.csv
    buildings_classified.geojson
    bjtu_building_height_map.html
```

`output/` 是默认导出位置，并允许通过命令行参数修改。

## 校园范围与查询策略

默认目标范围为两块 OSM 校园边界的合并区域：

| 校区 | OSM 对象 | OSM ID |
| --- | --- | --- |
| 北京交通大学主校区 | way | `266512538` |
| 北京交通大学东校区 | way | `266297360` |

查询顺序如下：

1. 首先尝试题目要求的名称 area 查询：

   ```overpass
   area["name"="北京交通大学"]->.searchArea;
   ```

2. 若名称 area 没有返回建筑，则将已确认的两条校园边界 way 使用
   `map_to_area` 转为校园 area，并在合并范围内查询建筑。
3. 若用户显式传入 `--bbox south,west,north,east`，则直接使用 bbox 查询，
   便于 OSM 边界变化或人工研究范围调整时覆盖默认策略。

所有查询均包含：

```overpass
way["building"](...);
relation["building"](...);
```

请求输出使用 `out body geom;`，以便保存建筑轮廓并计算中心点。程序提供可设置的
Overpass 端点、超时和重试逻辑，并在查询失败时给出可操作的错误提示。

设计验证时的实时查询结果为：两校区联合区域当前包含 `83` 个 building way，
`0` 个 building relation。程序仍处理 relation，以适应后续 OSM 更新或 bbox
查询得到的复合建筑。

## 提取字段

每栋建筑形成一条记录，包含：

| 字段 | 含义 |
| --- | --- |
| `osm_id` | OSM 对象 ID |
| `osm_type` | `way` 或 `relation` |
| `name` | 建筑名称；缺失时使用 `未命名建筑_<osm_id>` |
| `building` | 原始 `building` 标签 |
| `height` | 原始 `height` 标签 |
| `height_m` | 可安全解析时的米制高度 |
| `building:levels` | 原始楼层标签 |
| `estimated_height` | 仅楼层类建筑可解析楼层时的 `levels * 3.0` 米估算值 |
| `height_source` | `osm_height`、`estimated_from_levels_3m_per_floor` 或 `missing` |
| `category` | 三分类字段 |
| `centroid_lat` / `centroid_lon` | 建筑 geometry 计算得到的中心点 |
| `tags` | 原始 OSM tags 的 JSON 文本/对象 |
| `geometry` | CSV 中为 GeoJSON 文本，GeoJSON 文件中为正式 geometry |
| `geometry_status` | `ok` 或说明轮廓不可完整构造的诊断状态 |

## 解析与分类规则

分类采用严格的互斥优先级：

1. **`has_height`**：只要存在非空 `height` 标签，就进入第一类，无论是否同时
   具有 `building:levels`。保留原始 `height`；支持将 `18`、`18 m`、`18米`
   及小数形式解析为 `height_m`。无法可靠解析的原始高度仍属于本类，但
   `height_m` 留空，避免编造数值。
2. **`levels_only`**：从没有 `height` 的剩余建筑中，存在非空
   `building:levels` 标签者进入第二类。可解析楼层数时写入
   `estimated_height = building:levels * 3.0`，并用 `height_source`
   明确标注该值为楼层估算，而不是 OSM 真实高度。不能可靠解析时估算值留空。
3. **`missing_height_and_levels`**：既无 `height` 也无 `building:levels`
   的剩余建筑进入第三类，状态显示“缺少高度和楼层数信息”。

建筑只出现在一个分类中，不会重复计入多个分类文件。

## Geometry 与中心点

- `way` 的 `geometry` 节点序列转换为 GeoJSON `Polygon`；需要时自动闭合外环。
- `relation` 使用 `outer` 与 `inner` 成员轮廓构造 `Polygon` 或
  `MultiPolygon`，保留洞结构；若成员无法拼成完整面，建筑记录仍被保留，
  但 geometry 与中心点留空，并在 `geometry_status` 中记录诊断提示。
- `centroid_lat` 和 `centroid_lon` 从最终 GeoJSON 面 geometry 计算，供
  CSV 快速定位和地图弹窗使用。

## 输出产物

默认输出目录 `bjtu_osm_buildings/output/` 下生成：

| 文件 | 内容 |
| --- | --- |
| `buildings_with_height.csv` | `has_height` 建筑 |
| `buildings_with_levels_only.csv` | `levels_only` 建筑 |
| `buildings_without_height_or_levels.csv` | `missing_height_and_levels` 建筑 |
| `all_buildings_classified.csv` | 全部建筑，含 `category` |
| `buildings_classified.geojson` | 全部建筑 feature 与轮廓 geometry |
| `bjtu_building_height_map.html` | folium 交互式分类地图 |

地图以三种颜色显示三类建筑，尽量绘制建筑轮廓；点击弹窗显示名称、OSM 类型及
ID、原始高度、解析后高度、楼层数、估算高度及分类。

## 命令行与文档

`main.py` 提供以下核心使用方式：

```powershell
python .\main.py
python .\main.py --bbox south,west,north,east
python .\main.py --output-dir .\output --overpass-url https://overpass-api.de/api/interpreter
```

子项目 `README.md` 以适合初学者的方式说明安装、运行、输出字段、三分类含义、
bbox 用法、OSM 数据完整性限制，以及如何在 QGIS / 后续信道建模中使用 GeoJSON。

## 验证策略

自动化测试使用固定本地 Overpass 样本，不依赖网络，至少验证：

- `height` 的米制解析及无法解析时不捏造数值；
- `height` 优先、楼层其次、两者缺失最后的互斥分类；
- `building:levels * 3.0` 的估算规则和估算标记；
- 未命名建筑的回退名称；
- `way` 与 `relation` 的 geometry/centroid 转换；
- 四个 CSV、一个 GeoJSON 与 folium HTML 文件能够导出。

在测试通过后执行一次实时 Overpass 抓取，检查导出文件存在、总分类条数与
GeoJSON feature 数一致，并记录实时数据的三分类统计。实时结果是 OSM 当前快照，
后续重新运行时数量可能随社区编辑而变化。

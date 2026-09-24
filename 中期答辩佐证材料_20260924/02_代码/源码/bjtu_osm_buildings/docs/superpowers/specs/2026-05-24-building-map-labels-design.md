# 北京交通大学建筑地图常驻数据标签设计

## 目标

增强 `bjtu_osm_buildings/output/bjtu_building_height_map.html`：在已有建筑轮廓
和点击弹窗之外，把可用的高度或楼层数直接标在建筑中心，便于快速浏览信道建模
所需的建筑数据。

## 已确认显示规则

- `has_height` 建筑在中心显示短标签 `H: <高度> m`。
  - 若 `height_m` 已解析，使用米制数值，例如 `H: 11 m`。
  - 若 OSM 存在 `height` 但无法解析，使用原始标签文字，不推测数值。
- `levels_only` 建筑在中心显示短标签 `L: <building:levels> 层`，例如
  `L: 4 层`。标签表达的是楼层数，而非真实高度。
- `missing_height_and_levels` 建筑不显示常驻文字，避免大量缺失提示遮挡地图；
  仍保留原有轮廓颜色与点击弹窗。

## 实现方式

在 `bjtu_osm_buildings/main.py` 内新增纯函数生成标签文字。`make_map()` 绘制每个
有效建筑轮廓后，对标签文字非空的记录，按其 `centroid_lat`、`centroid_lon`
增加 `folium.Marker` 与 `DivIcon`。标签采用紧凑、半透明白底与深色边框，
不改变原有红/橙/蓝三分类轮廓和弹窗字段。

仅 HTML 地图会因本次增强而改变；CSV 与 GeoJSON 的数据结构和内容保持不变。

## 验证

- 自动化测试检查 `has_height` 得到高度标签、`levels_only` 得到楼层标签、
  缺失类不产生标签。
- 导出 HTML 后检查地图中包含高度和楼层文本，且缺失建筑名称仍可通过原有弹窗
  存在。
- 用当前 Overpass 数据重新生成已交付 HTML，并再次运行完整测试套件。

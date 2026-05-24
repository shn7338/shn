"""Download and classify BJTU building height data from OpenStreetMap."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any

import folium
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping, shape
from shapely.ops import polygonize, unary_union


HEIGHT_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres|米)?\s*$",
    re.IGNORECASE,
)
LEVELS_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")
MAIN_CAMPUS_WAY_ID = 266512538
EAST_CAMPUS_WAY_ID = 266297360
REQUIRED_OUTPUT_FILENAMES = (
    "buildings_with_height.csv",
    "buildings_with_levels_only.csv",
    "buildings_without_height_or_levels.csv",
    "all_buildings_classified.csv",
    "buildings_classified.geojson",
    "bjtu_building_height_map.html",
)
CATEGORY_COLORS = {
    "has_height": "#d73027",
    "levels_only": "#fc8d59",
    "missing_height_and_levels": "#4575b4",
}
CSV_FIELDS = (
    "osm_id",
    "osm_type",
    "name",
    "building",
    "height",
    "height_m",
    "building:levels",
    "estimated_height",
    "height_source",
    "height_note",
    "category",
    "centroid_lat",
    "centroid_lon",
    "geometry_status",
    "tags",
    "geometry",
)


def parse_height_m(value: Any) -> float | None:
    """Parse a simple OSM height value expressed in metres."""
    if value is None:
        return None
    match = HEIGHT_PATTERN.match(str(value))
    return float(match.group(1)) if match else None


def parse_levels(value: Any) -> float | None:
    """Parse one numeric building:levels value without guessing ranges."""
    if value is None:
        return None
    match = LEVELS_PATTERN.match(str(value))
    return float(match.group(1)) if match else None


def classify_element(element: dict[str, Any]) -> dict[str, Any]:
    """Extract attributes and place one OSM building in exactly one category."""
    tags = element.get("tags", {})
    height = tags.get("height")
    levels = tags.get("building:levels")

    if height not in (None, ""):
        category = "has_height"
        estimated_height = None
        height_source = "osm_height"
        height_note = "OSM 提供 height 字段"
    elif levels not in (None, ""):
        numeric_levels = parse_levels(levels)
        category = "levels_only"
        estimated_height = numeric_levels * 3.0 if numeric_levels is not None else None
        height_source = "estimated_from_levels_3m_per_floor"
        height_note = "估算高度（building:levels × 3 米），不是 OSM 真实 height"
    else:
        category = "missing_height_and_levels"
        estimated_height = None
        height_source = "missing"
        height_note = "缺少高度和楼层数信息"

    osm_id = element["id"]
    return {
        "osm_id": osm_id,
        "osm_type": element["type"],
        "name": tags.get("name") or f"未命名建筑_{osm_id}",
        "building": tags.get("building", ""),
        "height": height,
        "height_m": parse_height_m(height),
        "building:levels": levels,
        "estimated_height": estimated_height,
        "height_source": height_source,
        "height_note": height_note,
        "category": category,
    }


def build_name_area_query() -> str:
    """Query the requested school-name OSM area if it exists."""
    return """[out:json][timeout:60];
area["name"="北京交通大学"]->.searchArea;
(
  way["building"](area.searchArea);
  relation["building"](area.searchArea);
);
out body geom;"""


def build_campus_ways_query() -> str:
    """Query the verified main-campus and east-campus boundary ways."""
    return f"""[out:json][timeout:60];
(way({MAIN_CAMPUS_WAY_ID});way({EAST_CAMPUS_WAY_ID});)->.campuses;
.campuses map_to_area->.searchArea;
(
  way["building"](area.searchArea);
  relation["building"](area.searchArea);
);
out body geom;"""


def build_bbox_query(bbox: tuple[float, float, float, float]) -> str:
    """Query a manually supplied south, west, north, east rectangle."""
    south, west, north, east = bbox
    return f"""[out:json][timeout:60];
(
  way["building"]({south},{west},{north},{east});
  relation["building"]({south},{west},{north},{east});
);
out body geom;"""


def _point_coordinates(points: list[dict[str, Any]]) -> list[tuple[float, float]]:
    coordinates = []
    for point in points:
        if "lon" not in point or "lat" not in point:
            continue
        coordinates.append((float(point["lon"]), float(point["lat"])))
    return coordinates


def _way_polygon(points: list[dict[str, Any]]) -> Polygon | None:
    coordinates = _point_coordinates(points)
    if len(set(coordinates)) < 3:
        return None
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    polygon = Polygon(coordinates)
    if polygon.is_empty or not polygon.is_valid or polygon.area == 0:
        return None
    return polygon


def _member_polygons(members: list[dict[str, Any]], role: str) -> list[Polygon]:
    lines = []
    for member in members:
        member_role = member.get("role", "")
        is_requested_role = member_role == role or (role == "outer" and member_role == "")
        if not is_requested_role:
            continue
        coordinates = _point_coordinates(member.get("geometry", []))
        if len(coordinates) >= 2:
            lines.append(LineString(coordinates))
    if not lines:
        return []
    return [polygon for polygon in polygonize(unary_union(lines)) if polygon.area > 0]


def _relation_polygon(element: dict[str, Any]) -> Polygon | MultiPolygon | None:
    members = element.get("members", [])
    outer_polygons = _member_polygons(members, "outer")
    if not outer_polygons:
        return None
    inner_polygons = _member_polygons(members, "inner")
    shells_with_holes = []
    for outer in outer_polygons:
        inner_rings = [
            list(inner.exterior.coords)
            for inner in inner_polygons
            if outer.contains(inner.representative_point())
        ]
        shells_with_holes.append(Polygon(outer.exterior.coords, inner_rings))
    geometry = (
        shells_with_holes[0]
        if len(shells_with_holes) == 1
        else MultiPolygon(shells_with_holes)
    )
    if geometry.is_empty or not geometry.is_valid:
        return None
    return geometry


def geometry_from_element(element: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Convert an Overpass building element to a GeoJSON polygon geometry."""
    if element.get("type") == "way":
        geometry = _way_polygon(element.get("geometry", []))
        status = "ok" if geometry else "invalid_way_geometry"
    elif element.get("type") == "relation":
        geometry = _relation_polygon(element)
        status = "ok" if geometry else "relation_without_valid_outer_geometry"
    else:
        geometry = None
        status = "unsupported_osm_type"
    return (mapping(geometry) if geometry else None), status


def elements_to_records(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create export-ready records from Overpass building elements."""
    records = []
    for element in elements:
        record = classify_element(element)
        geometry, geometry_status = geometry_from_element(element)
        centroid = shape(geometry).centroid if geometry else None
        record.update(
            {
                "centroid_lat": centroid.y if centroid else None,
                "centroid_lon": centroid.x if centroid else None,
                "geometry_status": geometry_status,
                "tags": element.get("tags", {}),
                "geometry": geometry,
            }
        )
        records.append(record)
    return records


def _csv_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {field: record.get(field) for field in CSV_FIELDS}
    result["tags"] = json.dumps(record.get("tags", {}), ensure_ascii=False, sort_keys=True)
    geometry = record.get("geometry")
    result["geometry"] = json.dumps(geometry, ensure_ascii=False) if geometry else ""
    return result


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write CSV records with JSON text fields for tags and geometry."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_record(record) for record in records)


def _feature_properties(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "geometry"}


def _popup_html(record: dict[str, Any]) -> str:
    rows = (
        ("建筑名称", record["name"]),
        ("OSM ID", f'{record["osm_type"]}/{record["osm_id"]}'),
        ("height 原值", record.get("height")),
        ("height_m", record.get("height_m")),
        ("building:levels", record.get("building:levels")),
        ("estimated_height", record.get("estimated_height")),
        ("分类", record["category"]),
        ("说明", record["height_note"]),
    )
    return "<br>".join(
        f"<strong>{html.escape(label)}:</strong> {html.escape(str(value if value is not None else ''))}"
        for label, value in rows
    )


def make_map(records: list[dict[str, Any]]) -> folium.Map:
    """Render classified building polygons as an interactive folium map."""
    valid_records = [record for record in records if record.get("geometry")]
    if valid_records:
        center_lat = sum(record["centroid_lat"] for record in valid_records) / len(valid_records)
        center_lon = sum(record["centroid_lon"] for record in valid_records) / len(valid_records)
    else:
        center_lat, center_lon = 39.9504404, 116.3360690
    map_object = folium.Map(location=[center_lat, center_lon], zoom_start=16)
    bounds = []
    for record in valid_records:
        polygon = shape(record["geometry"])
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        bounds.extend([[min_lat, min_lon], [max_lat, max_lon]])
        color = CATEGORY_COLORS[record["category"]]
        layer = folium.GeoJson(
            {"type": "Feature", "geometry": record["geometry"], "properties": {}},
            style_function=lambda _feature, fill_color=color: {
                "color": fill_color,
                "fillColor": fill_color,
                "fillOpacity": 0.55,
                "weight": 1.5,
            },
            tooltip=record["name"],
        )
        folium.Popup(_popup_html(record), max_width=360).add_to(layer)
        layer.add_to(map_object)
    if bounds:
        map_object.fit_bounds(bounds)
    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
    background: white; padding: 10px; border: 1px solid #888; font-size: 13px;">
    <b>建筑高度分类</b><br>
    <span style="color:#d73027;">&#9632;</span> 有 height<br>
    <span style="color:#fc8d59;">&#9632;</span> 仅有 building:levels<br>
    <span style="color:#4575b4;">&#9632;</span> 缺少高度和楼层数信息
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend))
    return map_object


def export_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Export classified CSV files, GeoJSON, and an interactive HTML map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "buildings_with_height.csv",
        [record for record in records if record["category"] == "has_height"],
    )
    write_csv(
        output_dir / "buildings_with_levels_only.csv",
        [record for record in records if record["category"] == "levels_only"],
    )
    write_csv(
        output_dir / "buildings_without_height_or_levels.csv",
        [
            record
            for record in records
            if record["category"] == "missing_height_and_levels"
        ],
    )
    write_csv(output_dir / "all_buildings_classified.csv", records)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": record["geometry"],
                "properties": _feature_properties(record),
            }
            for record in records
        ],
    }
    (output_dir / "buildings_classified.geojson").write_text(
        json.dumps(feature_collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_map(records).save(str(output_dir / "bjtu_building_height_map.html"))

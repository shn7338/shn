from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import main as building_app
from main import (
    REQUIRED_OUTPUT_FILENAMES,
    build_bbox_query,
    build_campus_ways_query,
    build_name_area_query,
    classify_element,
    elements_to_records,
    export_outputs,
    fetch_building_elements,
    parse_bbox,
    parse_height_m,
    parse_levels,
    request_overpass,
)


def way(osm_id: int, tags: dict[str, str]) -> dict:
    return {"type": "way", "id": osm_id, "tags": {"building": "yes", **tags}}


def square(lon: float = 116.3300, lat: float = 39.9500, size: float = 0.001) -> list[dict]:
    return [
        {"lon": lon, "lat": lat},
        {"lon": lon + size, "lat": lat},
        {"lon": lon + size, "lat": lat + size},
        {"lon": lon, "lat": lat + size},
        {"lon": lon, "lat": lat},
    ]


def polygon_way(osm_id: int, tags: dict[str, str] | None = None) -> dict:
    element = way(osm_id, tags or {})
    element["geometry"] = square(lon=116.3300 + osm_id * 0.002)
    return element


def polygon_relation() -> dict:
    return {
        "type": "relation",
        "id": 99,
        "tags": {"building": "university", "name": "关系建筑"},
        "members": [
            {
                "type": "way",
                "role": "outer",
                "geometry": [
                    {"lon": 116.3400, "lat": 39.9500},
                    {"lon": 116.3420, "lat": 39.9500},
                    {"lon": 116.3420, "lat": 39.9520},
                ],
            },
            {
                "type": "way",
                "role": "outer",
                "geometry": [
                    {"lon": 116.3420, "lat": 39.9520},
                    {"lon": 116.3400, "lat": 39.9520},
                    {"lon": 116.3400, "lat": 39.9500},
                ],
            },
            {
                "type": "way",
                "role": "inner",
                "geometry": square(lon=116.3405, lat=39.9505, size=0.0004),
            },
        ],
    }


def test_parse_height_m_accepts_common_meter_formats() -> None:
    assert parse_height_m("18") == 18.0
    assert parse_height_m("18 m") == 18.0
    assert parse_height_m("18米") == 18.0
    assert parse_height_m("18.5 米") == 18.5


def test_parse_height_m_does_not_invent_unparseable_value() -> None:
    assert parse_height_m("about eighteen metres") is None
    assert parse_height_m("12;18") is None


def test_parse_levels_accepts_numeric_floors_only() -> None:
    assert parse_levels("4") == 4.0
    assert parse_levels("2.5") == 2.5
    assert parse_levels("3;4") is None


def test_height_tag_has_priority_over_levels() -> None:
    record = classify_element(way(1, {"height": "18 m", "building:levels": "4"}))

    assert record["category"] == "has_height"
    assert record["height"] == "18 m"
    assert record["height_m"] == 18.0
    assert record["building:levels"] == "4"
    assert record["estimated_height"] is None
    assert record["height_source"] == "osm_height"
    assert record["height_note"] == "OSM 提供 height 字段"


def test_unparseable_height_still_remains_in_height_category() -> None:
    record = classify_element(way(2, {"height": "roof varies", "building:levels": "6"}))

    assert record["category"] == "has_height"
    assert record["height_m"] is None
    assert record["estimated_height"] is None


def test_levels_without_height_is_estimated_and_marked_as_estimate() -> None:
    record = classify_element(way(3, {"building:levels": "4"}))

    assert record["category"] == "levels_only"
    assert record["estimated_height"] == 12.0
    assert record["height_source"] == "estimated_from_levels_3m_per_floor"
    assert record["height_note"] == "估算高度（building:levels × 3 米），不是 OSM 真实 height"


def test_missing_height_and_levels_remains_unestimated() -> None:
    record = classify_element(way(4, {}))

    assert record["category"] == "missing_height_and_levels"
    assert record["estimated_height"] is None
    assert record["height_source"] == "missing"
    assert record["height_note"] == "缺少高度和楼层数信息"


def test_unnamed_building_uses_osm_id_in_name() -> None:
    record = classify_element(way(987654, {}))

    assert record["name"] == "未命名建筑_987654"


def test_queries_include_both_building_object_types_and_geometry() -> None:
    queries = (
        build_name_area_query(),
        build_campus_ways_query(),
        build_bbox_query((39.94, 116.33, 39.96, 116.35)),
    )

    for query in queries:
        assert 'way["building"]' in query
        assert 'relation["building"]' in query
        assert "out body geom;" in query
    assert "266512538" in queries[1]
    assert "266297360" in queries[1]
    assert "39.94,116.33,39.96,116.35" in queries[2]


def test_way_geometry_has_polygon_and_centroid() -> None:
    record = elements_to_records([polygon_way(10)])[0]

    assert record["geometry"]["type"] == "Polygon"
    assert record["geometry_status"] == "ok"
    assert record["centroid_lat"] == pytest.approx(39.9505)
    assert record["centroid_lon"] == pytest.approx(116.3505)


def test_relation_fragments_are_assembled_and_keep_inner_hole() -> None:
    record = elements_to_records([polygon_relation()])[0]

    assert record["osm_type"] == "relation"
    assert record["geometry_status"] == "ok"
    assert record["geometry"]["type"] == "Polygon"
    assert len(record["geometry"]["coordinates"][0]) >= 4
    assert len(record["geometry"]["coordinates"]) == 2


def test_invalid_relation_is_retained_with_geometry_diagnostic() -> None:
    element = {
        "type": "relation",
        "id": 100,
        "tags": {"building": "yes"},
        "members": [],
    }

    record = elements_to_records([element])[0]

    assert record["name"] == "未命名建筑_100"
    assert record["geometry"] is None
    assert record["centroid_lat"] is None
    assert record["geometry_status"] == "relation_without_valid_outer_geometry"


def test_export_outputs_writes_required_files_and_category_rows(tmp_path: Path) -> None:
    records = elements_to_records(
        [
            polygon_way(1, {"name": "有高度楼", "height": "18 m"}),
            polygon_way(2, {"name": "有层数楼", "building:levels": "4"}),
            polygon_way(3, {"name": "缺信息楼"}),
        ]
    )

    export_outputs(records, tmp_path)

    for filename in REQUIRED_OUTPUT_FILENAMES:
        assert (tmp_path / filename).exists()
    with (tmp_path / "all_buildings_classified.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["category"] for row in rows] == [
        "has_height",
        "levels_only",
        "missing_height_and_levels",
    ]
    geojson = json.loads((tmp_path / "buildings_classified.geojson").read_text(encoding="utf-8"))
    assert len(geojson["features"]) == 3
    assert geojson["features"][1]["properties"]["estimated_height"] == 12.0
    html = (tmp_path / "bjtu_building_height_map.html").read_text(encoding="utf-8")
    assert "有高度楼" in html
    assert "有层数楼" in html


def test_parse_bbox_accepts_four_ordered_coordinates() -> None:
    assert parse_bbox("39.94,116.33,39.96,116.35") == (39.94, 116.33, 39.96, 116.35)


def test_parse_bbox_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="south"):
        parse_bbox("39.96,116.33,39.94,116.35")


def test_auto_fetch_uses_name_area_when_it_contains_buildings() -> None:
    queries: list[str] = []

    def request_json(query: str) -> dict:
        queries.append(query)
        return {"elements": [polygon_way(11)]}

    elements, method = fetch_building_elements(request_json=request_json)

    assert method == "name_area"
    assert len(elements) == 1
    assert len(queries) == 1


def test_auto_fetch_falls_back_to_two_campus_boundaries_when_area_is_empty() -> None:
    queries: list[str] = []

    def request_json(query: str) -> dict:
        queries.append(query)
        if len(queries) == 1:
            return {"elements": []}
        return {"elements": [polygon_way(12)]}

    elements, method = fetch_building_elements(request_json=request_json)

    assert method == "campus_boundary_ways"
    assert len(elements) == 1
    assert len(queries) == 2
    assert "266512538" in queries[1]
    assert "266297360" in queries[1]


def test_auto_fetch_falls_back_when_name_area_request_fails() -> None:
    calls = 0

    def request_json(query: str) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("name area is unavailable")
        return {"elements": [polygon_way(14)]}

    elements, method = fetch_building_elements(request_json=request_json)

    assert method == "campus_boundary_ways"
    assert len(elements) == 1
    assert calls == 2


def test_bbox_fetch_bypasses_area_queries() -> None:
    queries: list[str] = []

    elements, method = fetch_building_elements(
        bbox=(39.94, 116.33, 39.96, 116.35),
        request_json=lambda query: queries.append(query) or {"elements": [polygon_way(13)]},
    )

    assert method == "bbox"
    assert len(elements) == 1
    assert len(queries) == 1
    assert "39.94,116.33,39.96,116.35" in queries[0]


def test_request_overpass_retries_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"elements": []}

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise building_app.requests.Timeout("temporary timeout")
        return Response()

    monkeypatch.setattr(building_app.requests, "post", fake_post)
    monkeypatch.setattr(building_app.time, "sleep", lambda seconds: None)

    assert request_overpass("query", "https://example.test", timeout=1, retries=2) == {
        "elements": []
    }
    assert len(calls) == 2

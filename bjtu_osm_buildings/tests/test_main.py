from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from main import classify_element, parse_height_m, parse_levels


def way(osm_id: int, tags: dict[str, str]) -> dict:
    return {"type": "way", "id": osm_id, "tags": {"building": "yes", **tags}}


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


def test_missing_height_and_levels_remains_unestimated() -> None:
    record = classify_element(way(4, {}))

    assert record["category"] == "missing_height_and_levels"
    assert record["estimated_height"] is None
    assert record["height_source"] == "missing"


def test_unnamed_building_uses_osm_id_in_name() -> None:
    record = classify_element(way(987654, {}))

    assert record["name"] == "未命名建筑_987654"

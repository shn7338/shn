"""Download and classify BJTU building height data from OpenStreetMap."""

from __future__ import annotations

import re
from typing import Any


HEIGHT_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres|米)?\s*$",
    re.IGNORECASE,
)
LEVELS_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


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
    elif levels not in (None, ""):
        numeric_levels = parse_levels(levels)
        category = "levels_only"
        estimated_height = numeric_levels * 3.0 if numeric_levels is not None else None
        height_source = "estimated_from_levels_3m_per_floor"
    else:
        category = "missing_height_and_levels"
        estimated_height = None
        height_source = "missing"

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
        "category": category,
    }

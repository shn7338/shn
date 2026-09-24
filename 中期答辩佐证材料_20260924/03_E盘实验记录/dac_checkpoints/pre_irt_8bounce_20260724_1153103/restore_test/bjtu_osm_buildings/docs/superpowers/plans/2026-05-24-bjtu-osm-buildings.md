# BJTU OSM Building Height Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python subproject that downloads all OSM buildings in Beijing Jiaotong University's main and east campuses, classifies available height information, and exports tabular, GIS, and interactive-map outputs.

**Architecture:** `bjtu_osm_buildings/main.py` is a small command-line data pipeline with testable functions for query selection, record transformation, geometry conversion, export, and mapping. It first tries the requested name-area query, then the two verified campus boundary ways converted to areas, while an explicit bbox bypasses automatic selection. Local fixture-driven tests verify behavior without relying on live OSM; a final live run produces the current OSM snapshot.

**Tech Stack:** Python 3.10+, `requests`, `shapely`, `folium`, standard-library `argparse`/`csv`/`json`, `pytest`

---

### Task 1: Create Project Skeleton and Pure Classification Behavior

**Files:**
- Create: `bjtu_osm_buildings/requirements.txt`
- Create: `bjtu_osm_buildings/tests/test_main.py`
- Create: `bjtu_osm_buildings/main.py`

- [x] **Step 1: Write failing parser and classification tests**

Create `bjtu_osm_buildings/tests/test_main.py` with import setup and tests that specify:

```python
from main import classify_element, parse_height_m, parse_levels

def test_parse_height_m_accepts_meter_spellings():
    assert parse_height_m("18") == 18.0
    assert parse_height_m("18 m") == 18.0
    assert parse_height_m("18米") == 18.0

def test_height_tag_has_priority_over_levels():
    record = classify_element(way(1, {"height": "18 m", "building:levels": "4"}))
    assert record["category"] == "has_height"
    assert record["height_m"] == 18.0
    assert record["estimated_height"] is None

def test_levels_without_height_is_estimated_only():
    record = classify_element(way(2, {"building:levels": "4"}))
    assert record["category"] == "levels_only"
    assert record["estimated_height"] == 12.0
    assert record["height_source"] == "estimated_from_levels_3m_per_floor"

def test_missing_tags_remains_unestimated():
    record = classify_element(way(3, {}))
    assert record["category"] == "missing_height_and_levels"
    assert record["height_source"] == "missing"
```

- [x] **Step 2: Run test to verify missing implementation fails**

Run: `python -m pytest bjtu_osm_buildings/tests/test_main.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'main'`.

- [x] **Step 3: Implement pure parsing and record classification**

Create `bjtu_osm_buildings/main.py` with constants and the minimal callable API:

```python
HEIGHT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:m|米|米高)?\s*$", re.I)
LEVELS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")

def parse_height_m(value):
    match = HEIGHT_RE.match(str(value)) if value is not None else None
    return float(match.group(1)) if match else None

def parse_levels(value):
    match = LEVELS_RE.match(str(value)) if value is not None else None
    return float(match.group(1)) if match else None

def classify_element(element):
    tags = element.get("tags", {})
    height = tags.get("height")
    levels = tags.get("building:levels")
    if height not in (None, ""):
        category, source, estimated = "has_height", "osm_height", None
    elif levels not in (None, ""):
        parsed = parse_levels(levels)
        category = "levels_only"
        source = "estimated_from_levels_3m_per_floor"
        estimated = parsed * 3.0 if parsed is not None else None
    else:
        category, source, estimated = "missing_height_and_levels", "missing", None
    return {
        "osm_id": element["id"],
        "osm_type": element["type"],
        "name": tags.get("name") or f"未命名建筑_{element['id']}",
        "height": height,
        "height_m": parse_height_m(height),
        "building:levels": levels,
        "estimated_height": estimated,
        "height_source": source,
        "category": category,
    }
```

- [x] **Step 4: Run pure classification tests green**

Run: `python -m pytest bjtu_osm_buildings/tests/test_main.py -q`

Expected: parser and classification tests pass.

### Task 2: Add Queries, Geometry, and Export Artifacts

**Files:**
- Modify: `bjtu_osm_buildings/main.py`
- Modify: `bjtu_osm_buildings/tests/test_main.py`

- [x] **Step 1: Write failing tests for query forms, geometry, and artifact output**

Extend tests with fixture elements for a polygon way and multipolygon relation:

```python
from main import (
    build_bbox_query, build_campus_ways_query, build_name_area_query,
    elements_to_records, export_outputs,
)

def test_query_variants_include_way_and_relation_buildings():
    for query in (build_name_area_query(), build_campus_ways_query(), build_bbox_query((1, 2, 3, 4))):
        assert 'way["building"]' in query
        assert 'relation["building"]' in query
        assert "out body geom;" in query

def test_geometry_and_centroid_are_kept_for_way_and_relation():
    records = elements_to_records([polygon_way(), polygon_relation()])
    assert records[0]["geometry"]["type"] == "Polygon"
    assert records[1]["geometry"]["type"] in ("Polygon", "MultiPolygon")
    assert records[0]["centroid_lat"] is not None

def test_export_outputs_writes_all_required_files(tmp_path):
    records = elements_to_records([height_way(), levels_way(), missing_way()])
    export_outputs(records, tmp_path)
    for name in REQUIRED_OUTPUT_FILENAMES:
        assert (tmp_path / name).exists()
```

- [x] **Step 2: Run tests to verify the missing behaviors fail**

Run: `python -m pytest bjtu_osm_buildings/tests/test_main.py -q`

Expected: import failure for newly specified query/export functions.

- [x] **Step 3: Implement query builders, geometry conversion, and output writers**

Extend `main.py` with:

```python
def build_name_area_query():
    return """[out:json][timeout:60];
area["name"="北京交通大学"]->.searchArea;
(way["building"](area.searchArea);relation["building"](area.searchArea););
out body geom;"""

def build_campus_ways_query():
    return """[out:json][timeout:60];
(way(266512538);way(266297360);)->.campuses;
.campuses map_to_area->.searchArea;
(way["building"](area.searchArea);relation["building"](area.searchArea););
out body geom;"""

def build_bbox_query(bbox):
    south, west, north, east = bbox
    return f"""[out:json][timeout:60];
(way["building"]({south},{west},{north},{east});
relation["building"]({south},{west},{north},{east}););
out body geom;"""

def ring_from_points(points):
    ring = [(point["lon"], point["lat"]) for point in points]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring if len(ring) >= 4 else None

def geometry_from_element(element):
    if element["type"] == "way":
        ring = ring_from_points(element.get("geometry", []))
        return (mapping(Polygon(ring)), "ok") if ring else (None, "invalid_way_geometry")
    outer_rings = []
    inner_rings = []
    for member in element.get("members", []):
        ring = ring_from_points(member.get("geometry", []))
        if ring and member.get("role") == "outer":
            outer_rings.append(ring)
        elif ring and member.get("role") == "inner":
            inner_rings.append(ring)
    if not outer_rings:
        return None, "relation_without_valid_outer_ring"
    polygons = []
    for outer in outer_rings:
        shell = Polygon(outer)
        holes = [inner for inner in inner_rings if shell.contains(Polygon(inner).representative_point())]
        polygons.append(Polygon(outer, holes))
    polygon = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return mapping(polygon), "ok"

def elements_to_records(elements):
    records = []
    for element in elements:
        record = classify_element(element)
        geometry, status = geometry_from_element(element)
        centroid = shape(geometry).centroid if geometry else None
        record.update({
            "tags": element.get("tags", {}),
            "geometry": geometry,
            "geometry_status": status,
            "centroid_lat": centroid.y if centroid else None,
            "centroid_lon": centroid.x if centroid else None,
        })
        records.append(record)
    return records

def export_outputs(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "buildings_with_height.csv", [r for r in records if r["category"] == "has_height"])
    write_csv(output_dir / "buildings_with_levels_only.csv", [r for r in records if r["category"] == "levels_only"])
    write_csv(output_dir / "buildings_without_height_or_levels.csv", [r for r in records if r["category"] == "missing_height_and_levels"])
    write_csv(output_dir / "all_buildings_classified.csv", records)
    features = [
        {"type": "Feature", "geometry": r["geometry"], "properties": feature_properties(r)}
        for r in records
    ]
    (output_dir / "buildings_classified.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_map(records).save(output_dir / "bjtu_building_height_map.html")
```

Implement `write_csv()` to serialize `tags` and `geometry` with `json.dumps(value, ensure_ascii=False)`, `feature_properties()` to omit only the geometry payload, and `make_map()` to add each non-null GeoJSON geometry to a folium map using `#d73027` for `has_height`, `#fc8d59` for `levels_only`, and `#4575b4` for `missing_height_and_levels`, with the required popup fields.

- [x] **Step 4: Run all fixture tests green**

Run: `python -m pytest bjtu_osm_buildings/tests/test_main.py -q`

Expected: all parser, query, geometry, and export tests pass.

### Task 3: Implement Reliable Overpass CLI and Beginner-Friendly Documentation

**Files:**
- Modify: `bjtu_osm_buildings/main.py`
- Modify: `bjtu_osm_buildings/tests/test_main.py`
- Create: `bjtu_osm_buildings/README.md`

- [x] **Step 1: Write failing selection and CLI argument tests**

Add request injection tests:

```python
from main import fetch_building_elements, parse_bbox

def test_parse_bbox_accepts_four_ordered_coordinates():
    assert parse_bbox("39.94,116.33,39.96,116.35") == (39.94, 116.33, 39.96, 116.35)

def test_auto_fetch_falls_back_when_name_area_returns_empty():
    queries = []
    def request(query):
        queries.append(query)
        return {"elements": []} if len(queries) == 1 else {"elements": [polygon_way()]}
    elements, method = fetch_building_elements(request_json=request)
    assert method == "campus_boundary_ways"
    assert len(elements) == 1

def test_bbox_fetch_bypasses_auto_area_queries():
    queries = []
    elements, method = fetch_building_elements(
        bbox=(39.94, 116.33, 39.96, 116.35),
        request_json=lambda query: queries.append(query) or {"elements": [polygon_way()]},
    )
    assert method == "bbox"
    assert len(queries) == 1
```

- [x] **Step 2: Run tests to verify CLI/fetch tests fail**

Run: `python -m pytest bjtu_osm_buildings/tests/test_main.py -q`

Expected: import failures for `fetch_building_elements` and `parse_bbox`.

- [x] **Step 3: Implement network retries, CLI, and usage documentation**

Extend `main.py` with:

```python
def request_overpass(query, url, timeout, retries):
    for attempt in range(retries):
        try:
            response = requests.post(url, data={"data": query}, timeout=timeout, headers=HEADERS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"Overpass 请求失败: {error}") from error
            time.sleep(2 ** attempt)

def fetch_building_elements(bbox=None, request_json=None):
    request_json = request_json or default_request_callback
    if bbox:
        return request_json(build_bbox_query(bbox)).get("elements", []), "bbox"
    first = request_json(build_name_area_query()).get("elements", [])
    if first:
        return first, "name_area"
    return request_json(build_campus_ways_query()).get("elements", []), "campus_boundary_ways"

def main():
    args = parse_args()
    callback = lambda query: request_overpass(
        query, args.overpass_url, args.request_timeout, args.retries
    )
    elements, method = fetch_building_elements(args.bbox, request_json=callback)
    records = elements_to_records(elements)
    export_outputs(records, args.output_dir)
    print_summary(records, method, args.output_dir)
```

Create `README.md` explaining the Overpass source, both campus boundaries, install/run commands, output files, category meanings, bbox usage, GeoJSON use, and OSM data limitations. Set `requirements.txt` to include pinned-or-bounded `requests`, `shapely`, `folium`, and `pytest`.

- [x] **Step 4: Run full automated test suite**

Run: `python -m pytest bjtu_osm_buildings/tests -q`

Expected: all tests pass.

### Task 4: Generate and Verify the Live OSM Snapshot

**Files:**
- Create: `bjtu_osm_buildings/output/buildings_with_height.csv`
- Create: `bjtu_osm_buildings/output/buildings_with_levels_only.csv`
- Create: `bjtu_osm_buildings/output/buildings_without_height_or_levels.csv`
- Create: `bjtu_osm_buildings/output/all_buildings_classified.csv`
- Create: `bjtu_osm_buildings/output/buildings_classified.geojson`
- Create: `bjtu_osm_buildings/output/bjtu_building_height_map.html`

- [x] **Step 1: Install subproject requirements in the existing virtual environment**

Run: `.\.venv\Scripts\python.exe -m pip install -r .\bjtu_osm_buildings\requirements.txt`

Expected: dependencies install or are reported already satisfied.

- [x] **Step 2: Run the live Overpass export**

Run: `.\.venv\Scripts\python.exe .\bjtu_osm_buildings\main.py`

Expected: the program reports query method `campus_boundary_ways`, nonzero total buildings, and output paths.

- [x] **Step 3: Validate classification totals and generated artifacts**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import csv,json,pathlib; p=pathlib.Path('bjtu_osm_buildings/output'); rows=list(csv.DictReader((p/'all_buildings_classified.csv').open(encoding='utf-8-sig'))); geo=json.loads((p/'buildings_classified.geojson').read_text(encoding='utf-8')); print(len(rows), len(geo['features']), {c: sum(r['category']==c for r in rows) for c in ['has_height','levels_only','missing_height_and_levels']}); assert len(rows)==len(geo['features']) and len(rows)>0"
```

Expected: CSV and GeoJSON totals match and all three classification totals add to the total.

- [x] **Step 4: Re-run all automated tests after live generation**

Run: `.\.venv\Scripts\python.exe -m pytest .\bjtu_osm_buildings\tests -q`

Expected: all tests pass.

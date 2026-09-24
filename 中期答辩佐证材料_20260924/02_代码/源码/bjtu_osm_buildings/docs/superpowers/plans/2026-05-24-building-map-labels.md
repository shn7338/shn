# BJTU Building Map Permanent Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show always-visible height or level-count text centered on BJTU buildings that have usable OSM height metadata.

**Architecture:** Add a pure label-formatting function to the existing `main.py` folium pipeline, then place a `DivIcon` marker at each labeled building centroid during `make_map()`. Preserve existing CSV, GeoJSON, polygon coloring, and popup behavior.

**Tech Stack:** Python 3.12, folium, pytest

---

### Task 1: Add Permanent Building Data Labels

**Files:**
- Modify: `bjtu_osm_buildings/main.py`
- Modify: `bjtu_osm_buildings/tests/test_main.py`
- Modify: `bjtu_osm_buildings/README.md`
- Regenerate: `bjtu_osm_buildings/output/bjtu_building_height_map.html`

- [x] **Step 1: Write failing label-format and HTML assertions**

Add tests specifying:

```python
from main import map_label_text

def test_map_label_uses_height_before_levels():
    record = classify_element(way(1, {"height": "18 m", "building:levels": "4"}))
    assert map_label_text(record) == "H: 18 m"

def test_map_label_shows_levels_only_as_floors():
    record = classify_element(way(2, {"building:levels": "4"}))
    assert map_label_text(record) == "L: 4 层"

def test_map_label_omits_missing_information():
    assert map_label_text(classify_element(way(3, {}))) is None
```

Extend export HTML assertions to check `H: 18 m` and `L: 4 层` are rendered.

- [x] **Step 2: Run the tests and observe the missing label function failure**

Run: `.\bjtu_osm_buildings\.venv\Scripts\python.exe -m pytest .\bjtu_osm_buildings\tests -q`

Expected: collection fails because `map_label_text` is not yet defined.

- [x] **Step 3: Implement label text and centroid `DivIcon` markers**

Implement:

```python
def map_label_text(record):
    if record["category"] == "has_height":
        value = record["height_m"] if record["height_m"] is not None else record["height"]
        return f"H: {value:g} m" if isinstance(value, float) else f"H: {value}"
    if record["category"] == "levels_only":
        return f'L: {record["building:levels"]} 层'
    return None
```

In `make_map()`, for each valid geometry whose label is not `None`, add:

```python
folium.Marker(
    location=[record["centroid_lat"], record["centroid_lon"]],
    icon=folium.DivIcon(html=label_html),
).add_to(map_object)
```

Use escaped text and compact CSS so the centered text remains readable over building fills.

- [x] **Step 4: Document and regenerate the map**

Update `README.md` to explain permanent center labels, then run:

```powershell
$env:PYTHONUTF8='1'
.\bjtu_osm_buildings\.venv\Scripts\python.exe .\bjtu_osm_buildings\main.py
```

- [x] **Step 5: Verify the complete change**

Run:

```powershell
.\bjtu_osm_buildings\.venv\Scripts\python.exe -m pytest .\bjtu_osm_buildings\tests -q
.\bjtu_osm_buildings\.venv\Scripts\python.exe -m py_compile .\bjtu_osm_buildings\main.py
```

Inspect the generated HTML text for `H: 11 m` and at least one `L:` label; verify no
trailing-whitespace regressions are introduced.

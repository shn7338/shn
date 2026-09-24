from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import shapefile
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.validation import make_valid


INPUT_BASE = Path(r"D:\桌面\dac\datasets\wallman_ready\beijing_25d_wallman_dedup_utm50n")
OUTPUT_ROOT = Path(r"D:\桌面\dac\512mdata")
TILE_SIZE_M = 512.0
MIN_BUILDING_COVERAGE = 0.20
MIN_INTERSECTION_AREA_M2 = 0.01
ANALYZE_PROGRESS_EVERY = 100_000
WRITE_PROGRESS_EVERY = 100


def log(message: str) -> None:
    print(message, flush=True)


def clean_output_root(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    tile_dir_pattern = re.compile(r"^tile_\d{6}$")
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and tile_dir_pattern.match(child.name):
            shutil.rmtree(child)
            removed += 1
    tmp_root = root / "_union_filtered_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    for name in ("tile_index.csv", "tiling_report.json"):
        path = root / name
        if path.exists():
            path.unlink()
    return removed


def source_path(ext: str) -> Path:
    return INPUT_BASE.with_suffix(ext)


def tile_path(tile_name: str, ext: str) -> Path:
    return OUTPUT_ROOT / tile_name / f"{tile_name}{ext}"


def add_source_fields(writer: shapefile.Writer, fields) -> None:
    for field in fields:
        try:
            name = field.name
            field_type = field.field_type.value if hasattr(field.field_type, "value") else field.field_type
            size = field.size
            decimal = field.decimal
        except AttributeError:
            name, field_type, size, decimal = field[:4]
        writer.field(name, field_type, size, decimal)


def shape_to_geometry(shape: shapefile.Shape):
    starts = list(shape.parts) + [len(shape.points)]
    polygons = []
    for start, end in zip(starts, starts[1:]):
        ring = shape.points[start:end]
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        poly = Polygon(ring)
        if not poly.is_empty and poly.area > 0:
            polygons.append(poly)
    if not polygons:
        return None
    geom = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.is_empty:
        return None
    return geom


def tile_key_range_for_bounds(bounds: tuple[float, float, float, float], origin_x: float, origin_y: float):
    minx, miny, maxx, maxy = bounds
    col0 = math.floor((minx - origin_x) / TILE_SIZE_M)
    col1 = math.floor((maxx - origin_x) / TILE_SIZE_M)
    row0 = math.floor((miny - origin_y) / TILE_SIZE_M)
    row1 = math.floor((maxy - origin_y) / TILE_SIZE_M)
    for row in range(int(row0), int(row1) + 1):
        for col in range(int(col0), int(col1) + 1):
            yield row, col


def tile_bounds(row: int, col: int, origin_x: float, origin_y: float) -> tuple[float, float, float, float]:
    minx = origin_x + col * TILE_SIZE_M
    miny = origin_y + row * TILE_SIZE_M
    return minx, miny, minx + TILE_SIZE_M, miny + TILE_SIZE_M


def tile_geometry(row: int, col: int, origin_x: float, origin_y: float) -> Polygon:
    return box(*tile_bounds(row, col, origin_x, origin_y))


def polygon_parts(poly: Polygon) -> list[list[tuple[float, float]]]:
    parts: list[list[tuple[float, float]]] = []
    exterior = [(round(float(x), 3), round(float(y), 3)) for x, y in poly.exterior.coords]
    if len(exterior) >= 4:
        parts.append(exterior)
    for interior in poly.interiors:
        ring = [(round(float(x), 3), round(float(y), 3)) for x, y in interior.coords]
        if len(ring) >= 4:
            parts.append(ring)
    return parts


def geometry_to_parts(geom) -> list[list[tuple[float, float]]]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return polygon_parts(geom)
    if isinstance(geom, MultiPolygon):
        parts: list[list[tuple[float, float]]] = []
        for poly in geom.geoms:
            parts.extend(polygon_parts(poly))
        return parts
    if isinstance(geom, GeometryCollection):
        parts: list[list[tuple[float, float]]] = []
        for item in geom.geoms:
            if isinstance(item, (Polygon, MultiPolygon)):
                parts.extend(geometry_to_parts(item))
        return parts
    return []


def record_values(record, clipped_area_m2: float) -> list:
    values = list(record)
    # Source fields are BLDG_ID, HEIGHT_M, MAT_ID, MAT_NAME, ROOF_TYPE, SRC, AREA_M2.
    # AREA_M2 should describe the clipped footprint area inside this 512m tile.
    values[-1] = round(float(clipped_area_m2), 2)
    return values


def main() -> int:
    input_shp = source_path(".shp")
    if not input_shp.exists():
        raise FileNotFoundError(input_shp)

    start_time = time.time()
    removed_dirs = clean_output_root(OUTPUT_ROOT)
    log(f"CLEANED old tile dirs: {removed_dirs}")

    reader = shapefile.Reader(str(input_shp), encoding="utf-8")
    fields = reader.fields[1:]
    source_bbox = tuple(float(v) for v in reader.bbox)
    origin_x = math.floor(source_bbox[0] / TILE_SIZE_M) * TILE_SIZE_M
    origin_y = math.floor(source_bbox[1] / TILE_SIZE_M) * TILE_SIZE_M
    tile_area_m2 = TILE_SIZE_M * TILE_SIZE_M
    min_coverage_area_m2 = tile_area_m2 * MIN_BUILDING_COVERAGE

    log(f"INPUT records={len(reader):,} bbox={source_bbox}")
    log(
        f"TILE origin=({origin_x}, {origin_y}) size={TILE_SIZE_M}m "
        f"coverage_threshold={MIN_BUILDING_COVERAGE:.0%}"
    )

    coverage_area_by_tile: dict[tuple[int, int], float] = defaultdict(float)
    source_indices_by_tile: dict[tuple[int, int], list[int]] = defaultdict(list)
    tile_box_cache: dict[tuple[int, int], Polygon] = {}
    bad_geometry_count = 0
    intersection_count = 0

    for idx, shape in enumerate(reader.iterShapes()):
        geom = shape_to_geometry(shape)
        if geom is None:
            bad_geometry_count += 1
            continue

        for key in tile_key_range_for_bounds(tuple(float(v) for v in geom.bounds), origin_x, origin_y):
            tile_box = tile_box_cache.get(key)
            if tile_box is None:
                tile_box = tile_geometry(key[0], key[1], origin_x, origin_y)
                tile_box_cache[key] = tile_box
            if not geom.intersects(tile_box):
                continue
            clipped = geom.intersection(tile_box)
            clipped_area = float(clipped.area)
            if clipped_area <= MIN_INTERSECTION_AREA_M2:
                continue
            coverage_area_by_tile[key] += clipped_area
            source_indices_by_tile[key].append(idx)
            intersection_count += 1

        if (idx + 1) % ANALYZE_PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            above_threshold = sum(1 for area in coverage_area_by_tile.values() if area >= min_coverage_area_m2)
            log(
                f"ANALYZE input={idx + 1:,} non_empty_tiles={len(coverage_area_by_tile):,} "
                f"candidate_tiles_ge_threshold={above_threshold:,} intersections={intersection_count:,} "
                f"elapsed_s={elapsed:.1f}"
            )

    selected_keys = sorted(key for key, area in coverage_area_by_tile.items() if area >= min_coverage_area_m2)
    log(
        f"ANALYZE complete: non_empty_tiles={len(coverage_area_by_tile):,} "
        f"selected_tiles_ge_threshold={len(selected_keys):,}"
    )

    index_rows = []
    for tile_number, key in enumerate(selected_keys, start=1):
        tile_name = f"tile_{tile_number:06d}"
        tile_dir = OUTPUT_ROOT / tile_name
        tile_dir.mkdir(parents=True, exist_ok=True)

        base = tile_dir / tile_name
        writer = shapefile.Writer(str(base), shapeType=reader.shapeType)
        writer.autoBalance = 1
        add_source_fields(writer, fields)

        tile_box = tile_box_cache[key]
        written_features = 0
        for record_index in source_indices_by_tile[key]:
            sr = reader.shapeRecord(record_index)
            geom = shape_to_geometry(sr.shape)
            if geom is None or not geom.intersects(tile_box):
                continue
            clipped = geom.intersection(tile_box)
            clipped_area = float(clipped.area)
            if clipped_area <= MIN_INTERSECTION_AREA_M2:
                continue
            parts = geometry_to_parts(clipped)
            if not parts:
                continue
            writer.poly(parts)
            writer.record(*record_values(sr.record, clipped_area))
            written_features += 1
        writer.close()

        for ext in (".prj", ".cpg"):
            src = source_path(ext)
            dst = tile_path(tile_name, ext)
            if src.exists():
                shutil.copyfile(src, dst)

        minx, miny, maxx, maxy = tile_bounds(key[0], key[1], origin_x, origin_y)
        coverage_area = coverage_area_by_tile[key]
        index_rows.append(
            {
                "tile_name": tile_name,
                "row": key[0],
                "col": key[1],
                "tile_minx": f"{minx:.3f}",
                "tile_miny": f"{miny:.3f}",
                "tile_maxx": f"{maxx:.3f}",
                "tile_maxy": f"{maxy:.3f}",
                "building_area_m2": f"{coverage_area:.3f}",
                "building_coverage": f"{coverage_area / tile_area_m2:.6f}",
                "feature_count": written_features,
                "folder": str(tile_dir),
                "shp": str(tile_path(tile_name, ".shp")),
            }
        )

        if tile_number % WRITE_PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            log(
                f"WRITE tile={tile_number:,}/{len(selected_keys):,} "
                f"name={tile_name} features={written_features:,} elapsed_s={elapsed:.1f}"
            )

    index_path = OUTPUT_ROOT / "tile_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_name",
                "row",
                "col",
                "tile_minx",
                "tile_miny",
                "tile_maxx",
                "tile_maxy",
                "building_area_m2",
                "building_coverage",
                "feature_count",
                "folder",
                "shp",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    report = {
        "input_shp": str(input_shp),
        "output_root": str(OUTPUT_ROOT),
        "tile_size_m": TILE_SIZE_M,
        "tile_area_m2": tile_area_m2,
        "minimum_building_coverage": MIN_BUILDING_COVERAGE,
        "minimum_building_area_m2": min_coverage_area_m2,
        "crs": "EPSG:32650",
        "assignment_rule": (
            "Tiles are strict 512m x 512m grid cells. Building footprints are clipped to tile "
            "boundaries. Only candidate tiles whose clipped building area is at least the configured "
            "coverage threshold are written before strict union filtering."
        ),
        "source_record_count": len(reader),
        "bad_geometry_count": bad_geometry_count,
        "non_empty_tile_count_before_filter": len(coverage_area_by_tile),
        "selected_tile_count": len(selected_keys),
        "removed_old_tile_dirs": removed_dirs,
        "source_bbox": source_bbox,
        "tile_origin": [origin_x, origin_y],
        "tile_index_csv": str(index_path),
        "elapsed_s": round(time.time() - start_time, 2),
    }
    (OUTPUT_ROOT / "tiling_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("DONE")
    log(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

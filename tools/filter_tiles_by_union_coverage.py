from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import shapefile
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid


ROOT = Path(r"D:\桌面\dac\01_data\512mdata")
TMP_ROOT = ROOT / "_union_filtered_tmp"
TILE_AREA_M2 = 512.0 * 512.0
MIN_COVERAGE = 0.20
EXPECTED_EXTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


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
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if not polys:
            return None
        geom = unary_union(polys)
    return None if geom.is_empty else geom


def union_coverage(tile_dir: Path) -> tuple[float, float, int]:
    shp = tile_dir / f"{tile_dir.name}.shp"
    reader = shapefile.Reader(str(shp), encoding="utf-8")
    try:
        geometries = []
        for shape in reader.iterShapes():
            geom = shape_to_geometry(shape)
            if geom is not None and not geom.is_empty:
                geometries.append(geom)
        record_count = len(reader)
    finally:
        reader.close()
    if not geometries:
        return 0.0, 0.0, 0
    union_geom = unary_union(geometries)
    area = float(union_geom.area)
    return area / TILE_AREA_M2, area, record_count


def copy_tile_with_new_name(src_dir: Path, dst_name: str) -> Path:
    dst_dir = TMP_ROOT / dst_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    for ext in EXPECTED_EXTS:
        shutil.copyfile(src_dir / f"{src_dir.name}{ext}", dst_dir / f"{dst_name}{ext}")
    return dst_dir


def main() -> int:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    tile_pattern = re.compile(r"^tile_\d{6}$")
    old_tiles = sorted([p for p in ROOT.iterdir() if p.is_dir() and tile_pattern.match(p.name)])

    kept_rows = []
    dropped_rows = []
    for tile_dir in old_tiles:
        coverage, area_m2, feature_count = union_coverage(tile_dir)
        row = {
            "old_tile_name": tile_dir.name,
            "union_building_area_m2": f"{area_m2:.3f}",
            "union_building_coverage": f"{coverage:.6f}",
            "feature_count": feature_count,
        }
        if coverage >= MIN_COVERAGE:
            new_name = f"tile_{len(kept_rows) + 1:06d}"
            copy_tile_with_new_name(tile_dir, new_name)
            row["tile_name"] = new_name
            kept_rows.append(row)
        else:
            dropped_rows.append(row)

    for tile_dir in old_tiles:
        shutil.rmtree(tile_dir)
    for new_dir in sorted(TMP_ROOT.iterdir()):
        shutil.move(str(new_dir), str(ROOT / new_dir.name))
    shutil.rmtree(TMP_ROOT)

    index_path = ROOT / "tile_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "tile_name",
            "old_tile_name",
            "union_building_area_m2",
            "union_building_coverage",
            "feature_count",
            "folder",
            "shp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept_rows:
            tile_name = row["tile_name"]
            row["folder"] = str(ROOT / tile_name)
            row["shp"] = str(ROOT / tile_name / f"{tile_name}.shp")
            writer.writerow(row)

    report = {
        "root": str(ROOT),
        "tile_area_m2": TILE_AREA_M2,
        "minimum_union_building_coverage": MIN_COVERAGE,
        "input_tile_count": len(old_tiles),
        "kept_tile_count": len(kept_rows),
        "dropped_tile_count": len(dropped_rows),
        "min_kept_union_coverage": min(float(r["union_building_coverage"]) for r in kept_rows) if kept_rows else None,
        "max_kept_union_coverage": max(float(r["union_building_coverage"]) for r in kept_rows) if kept_rows else None,
        "tile_index_csv": str(index_path),
        "dropped_tiles": dropped_rows,
    }
    (ROOT / "tiling_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

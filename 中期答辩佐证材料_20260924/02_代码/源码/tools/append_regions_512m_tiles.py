from __future__ import annotations

import bisect
import csv
import json
import math
import re
import shutil
import time
from array import array
from collections import defaultdict
from pathlib import Path

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid


DATASETS_ROOT = Path(r"D:\桌面\dac\datasets")
OUTPUT_ROOT = Path(r"D:\桌面\dac\512mdata")
TMP_ROOT = OUTPUT_ROOT / "_append_tiles_tmp"

REGIONS = [
    {
        "name": "Shanghai",
        "code": "sh",
        "input_dir": DATASETS_ROOT / "Shanghai_上海市",
        "id_offset": 200_000_000,
    },
    {
        "name": "Tianjin",
        "code": "tj",
        "input_dir": DATASETS_ROOT / "Tianjin_天津市",
        "id_offset": 300_000_000,
    },
    {
        "name": "Guangdong",
        "code": "gd",
        "input_dir": DATASETS_ROOT / "Guangdong_广东省",
        "id_offset": 400_000_000,
    },
]

TARGET_CRS = CRS.from_epsg(32650)
TILE_SIZE_M = 512.0
MIN_UNION_COVERAGE = 0.20
MIN_INTERSECTION_AREA_M2 = 0.01
DEDUP_OVERLAP_AREA_M2 = 1.0
DEDUP_OVERLAP_RATIO = 0.90
DEDUP_CELL_SIZE_M = 80.0

ANALYZE_PROGRESS_EVERY = 200_000
WRITE_PROGRESS_EVERY = 100


def log(message: str) -> None:
    print(message, flush=True)


def natural_part_number(path: Path) -> int:
    match = re.search(r"_part(\d+)\.shp$", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def existing_tile_numbers(root: Path) -> list[int]:
    nums = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"tile_(\d{6})", child.name)
        if match:
            nums.append(int(match.group(1)))
    return sorted(nums)


def source_crs_from_prj(shp_path: Path) -> CRS:
    prj_path = shp_path.with_suffix(".prj")
    text = prj_path.read_text(encoding="utf-8", errors="ignore") if prj_path.exists() else ""
    if "China_Geodetic_Coordinate_System_2000" in text or "CGCS2000" in text or "China_2000" in text:
        return CRS.from_epsg(4490)
    return CRS.from_epsg(4326)


def tile_bounds(row: int, col: int, origin_x: float, origin_y: float) -> tuple[float, float, float, float]:
    minx = origin_x + col * TILE_SIZE_M
    miny = origin_y + row * TILE_SIZE_M
    return minx, miny, minx + TILE_SIZE_M, miny + TILE_SIZE_M


def tile_geometry(row: int, col: int, origin_x: float, origin_y: float) -> Polygon:
    return box(*tile_bounds(row, col, origin_x, origin_y))


def tile_key_range_for_bounds(bounds: tuple[float, float, float, float], origin_x: float, origin_y: float):
    minx, miny, maxx, maxy = bounds
    col0 = math.floor((minx - origin_x) / TILE_SIZE_M)
    col1 = math.floor((maxx - origin_x) / TILE_SIZE_M)
    row0 = math.floor((miny - origin_y) / TILE_SIZE_M)
    row1 = math.floor((maxy - origin_y) / TILE_SIZE_M)
    for row in range(int(row0), int(row1) + 1):
        for col in range(int(col0), int(col1) + 1):
            yield row, col


def shape_to_projected_geometry(shape: shapefile.Shape, transformer: Transformer):
    starts = list(shape.parts) + [len(shape.points)]
    polygons = []
    for start, end in zip(starts, starts[1:]):
        ring = shape.points[start:end]
        if len(ring) < 4:
            continue
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        tx, ty = transformer.transform(xs, ys)
        points = [(float(x), float(y)) for x, y in zip(tx, ty) if math.isfinite(x) and math.isfinite(y)]
        if len(points) < 4:
            continue
        if points[0] != points[-1]:
            points.append(points[0])
        poly = Polygon(points)
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
    if geom is None or geom.is_empty:
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


def parse_height(value: object) -> float | None:
    try:
        height = float(str(value).strip())
    except Exception:
        return None
    if not math.isfinite(height) or height <= 0 or height > 300:
        return None
    return height


def clean_roof(value: object) -> str:
    roof = str(value or "").strip()
    return roof[:20] if roof else "unknown"


def tile_cells_for_bbox(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = bounds
    ix0 = math.floor(minx / DEDUP_CELL_SIZE_M)
    ix1 = math.floor(maxx / DEDUP_CELL_SIZE_M)
    iy0 = math.floor(miny / DEDUP_CELL_SIZE_M)
    iy1 = math.floor(maxy / DEDUP_CELL_SIZE_M)
    return [(ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1)]


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


class TileDeduper:
    def __init__(self) -> None:
        self.geoms = []
        self.bounds: list[tuple[float, float, float, float]] = []
        self.areas: list[float] = []
        self.grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    def accept(self, geom) -> bool:
        bounds = tuple(float(v) for v in geom.bounds)
        area = float(geom.area)
        candidates: set[int] = set()
        for cell in tile_cells_for_bbox(bounds):
            candidates.update(self.grid.get(cell, ()))
        for idx in candidates:
            if not bbox_intersects(bounds, self.bounds[idx]):
                continue
            other = self.geoms[idx]
            if not geom.intersects(other):
                continue
            inter_area = float(geom.intersection(other).area)
            if inter_area < DEDUP_OVERLAP_AREA_M2:
                continue
            ratio = inter_area / max(min(area, self.areas[idx]), 1e-9)
            if ratio >= DEDUP_OVERLAP_RATIO:
                return False
        idx = len(self.geoms)
        self.geoms.append(geom)
        self.bounds.append(bounds)
        self.areas.append(area)
        for cell in tile_cells_for_bbox(bounds):
            self.grid[cell].append(idx)
        return True


def add_output_fields(writer: shapefile.Writer) -> None:
    writer.field("BLDG_ID", "N", 12, 0)
    writer.field("HEIGHT_M", "N", 8, 2)
    writer.field("MAT_ID", "N", 4, 0)
    writer.field("MAT_NAME", "C", 20)
    writer.field("ROOF_TYPE", "C", 20)
    writer.field("SRC", "C", 8)
    writer.field("AREA_M2", "N", 14, 2)


def write_prj_cpg(base: Path) -> None:
    base.with_suffix(".prj").write_text(TARGET_CRS.to_wkt("WKT1_ESRI"), encoding="utf-8")
    base.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def load_existing_index(root: Path) -> list[dict[str, str]]:
    index_path = root / "tile_index.csv"
    if not index_path.exists():
        return []
    backup = root / "tile_index_beijing_before_append.csv"
    if not backup.exists():
        shutil.copyfile(index_path, backup)
    with index_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    normalized = []
    for row in rows:
        tile_name = row.get("tile_name", "")
        normalized.append(
            {
                "tile_name": tile_name,
                "region": row.get("region") or "Beijing",
                "source_key": row.get("source_key") or row.get("old_tile_name", ""),
                "union_building_area_m2": row.get("union_building_area_m2", ""),
                "union_building_coverage": row.get("union_building_coverage", ""),
                "feature_count": row.get("feature_count", ""),
                "folder": str(root / tile_name) if tile_name else row.get("folder", ""),
                "shp": str(root / tile_name / f"{tile_name}.shp") if tile_name else row.get("shp", ""),
            }
        )
    return normalized


def write_combined_index(root: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "tile_name",
        "region",
        "source_key",
        "union_building_area_m2",
        "union_building_coverage",
        "feature_count",
        "folder",
        "shp",
    ]
    for filename in ("tile_index.csv", "tile_index_all.csv"):
        with (root / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def prepare_region_files(region: dict) -> list[dict]:
    shp_paths = sorted(region["input_dir"].glob("*.shp"), key=natural_part_number)
    if not shp_paths:
        raise FileNotFoundError(f"No shapefiles found in {region['input_dir']}")
    files = []
    offset = 0
    for file_id, shp_path in enumerate(shp_paths):
        reader = shapefile.Reader(str(shp_path), encoding="utf-8")
        count = len(reader)
        reader.close()
        part_no = natural_part_number(shp_path)
        src = f"{region['code']}_p{part_no:02d}"[:8]
        files.append(
            {
                "file_id": file_id,
                "path": shp_path,
                "crs": source_crs_from_prj(shp_path),
                "count": count,
                "offset": offset,
                "src": src,
                "part_no": part_no,
            }
        )
        offset += count
    return files


def locate_file(global_index: int, offsets: list[int]) -> tuple[int, int]:
    file_id = bisect.bisect_right(offsets, global_index) - 1
    return file_id, global_index - offsets[file_id]


def process_region(region: dict, next_tile_number: int, tmp_root: Path) -> tuple[int, list[dict], dict]:
    start_time = time.time()
    files = prepare_region_files(region)
    total_records = sum(f["count"] for f in files)
    log(f"\nREGION {region['name']} START files={len(files)} records={total_records:,}")

    # Project source bbox corners to a shared meter grid origin for this region.
    projected_bounds = []
    for info in files:
        reader = shapefile.Reader(str(info["path"]), encoding="utf-8")
        bbox = reader.bbox
        reader.close()
        transformer = Transformer.from_crs(info["crs"], TARGET_CRS, always_xy=True)
        xs = [bbox[0], bbox[2]]
        ys = [bbox[1], bbox[3]]
        tx, ty = transformer.transform(xs, ys)
        projected_bounds.append((min(tx), min(ty), max(tx), max(ty)))
    minx = min(b[0] for b in projected_bounds)
    miny = min(b[1] for b in projected_bounds)
    origin_x = math.floor(minx / TILE_SIZE_M) * TILE_SIZE_M
    origin_y = math.floor(miny / TILE_SIZE_M) * TILE_SIZE_M
    tile_area = TILE_SIZE_M * TILE_SIZE_M
    min_area = tile_area * MIN_UNION_COVERAGE

    coverage_by_tile: dict[tuple[int, int], float] = defaultdict(float)
    indices_by_tile: dict[tuple[int, int], array] = defaultdict(lambda: array("I"))
    tile_box_cache: dict[tuple[int, int], Polygon] = {}
    transformers = {info["file_id"]: Transformer.from_crs(info["crs"], TARGET_CRS, always_xy=True) for info in files}

    bad_height = bad_geometry = intersections = 0
    global_base = 0
    for info in files:
        reader = shapefile.Reader(str(info["path"]), encoding="utf-8")
        transformer = transformers[info["file_id"]]
        log(
            f"REGION {region['name']} ANALYZE {info['path'].name} "
            f"records={info['count']:,} crs={info['crs'].to_string()}"
        )
        for local_index, sr in enumerate(reader.iterShapeRecords()):
            global_index = global_base + local_index
            if parse_height(sr.record.as_dict().get("height")) is None:
                bad_height += 1
                continue
            geom = shape_to_projected_geometry(sr.shape, transformer)
            if geom is None:
                bad_geometry += 1
                continue
            for key in tile_key_range_for_bounds(tuple(float(v) for v in geom.bounds), origin_x, origin_y):
                tile_box = tile_box_cache.get(key)
                if tile_box is None:
                    tile_box = tile_geometry(key[0], key[1], origin_x, origin_y)
                    tile_box_cache[key] = tile_box
                if not geom.intersects(tile_box):
                    continue
                clipped_area = float(geom.intersection(tile_box).area)
                if clipped_area <= MIN_INTERSECTION_AREA_M2:
                    continue
                coverage_by_tile[key] += clipped_area
                indices_by_tile[key].append(global_index)
                intersections += 1
            if (global_index + 1) % ANALYZE_PROGRESS_EVERY == 0:
                selected_now = sum(1 for area in coverage_by_tile.values() if area >= min_area)
                log(
                    f"REGION {region['name']} ANALYZE progress input={global_index + 1:,}/{total_records:,} "
                    f"non_empty_tiles={len(coverage_by_tile):,} candidates={selected_now:,} "
                    f"intersections={intersections:,} elapsed_s={time.time() - start_time:.1f}"
                )
        reader.close()
        global_base += info["count"]

    selected_keys = sorted(key for key, area in coverage_by_tile.items() if area >= min_area)
    log(
        f"REGION {region['name']} ANALYZE complete non_empty_tiles={len(coverage_by_tile):,} "
        f"candidate_tiles={len(selected_keys):,}"
    )

    offsets = [f["offset"] for f in files]
    readers = [shapefile.Reader(str(f["path"]), encoding="utf-8") for f in files]
    appended_rows: list[dict] = []
    kept_tiles = dropped_tiles = skipped_overlap = 0
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        for candidate_number, key in enumerate(selected_keys, start=1):
            tile_box = tile_box_cache[key]
            deduper = TileDeduper()
            features = []
            union_geoms = []
            for global_index in indices_by_tile[key]:
                file_id, local_index = locate_file(int(global_index), offsets)
                info = files[file_id]
                sr = readers[file_id].shapeRecord(local_index)
                rec = sr.record.as_dict()
                height = parse_height(rec.get("height"))
                if height is None:
                    continue
                geom = shape_to_projected_geometry(sr.shape, transformers[file_id])
                if geom is None or not geom.intersects(tile_box):
                    continue
                clipped = geom.intersection(tile_box)
                clipped_area = float(clipped.area)
                if clipped_area <= MIN_INTERSECTION_AREA_M2:
                    continue
                if not deduper.accept(clipped):
                    skipped_overlap += 1
                    continue
                parts = geometry_to_parts(clipped)
                if not parts:
                    continue
                bldg_id = region["id_offset"] + int(global_index) + 1
                record = [
                    bldg_id,
                    round(height, 2),
                    1,
                    "concrete",
                    clean_roof(rec.get("roof_type")),
                    info["src"],
                    round(clipped_area, 2),
                ]
                features.append((parts, record))
                union_geoms.append(clipped)

            if not union_geoms:
                dropped_tiles += 1
                continue
            union_area = float(unary_union(union_geoms).area)
            union_coverage = union_area / tile_area
            if union_coverage < MIN_UNION_COVERAGE:
                dropped_tiles += 1
                continue

            tile_name = f"tile_{next_tile_number:06d}"
            tile_dir = tmp_root / tile_name
            tile_dir.mkdir(parents=True, exist_ok=True)
            base = tile_dir / tile_name
            writer = shapefile.Writer(str(base), shapeType=shapefile.POLYGON)
            writer.autoBalance = 1
            add_output_fields(writer)
            for parts, record in features:
                writer.poly(parts)
                writer.record(*record)
            writer.close()
            write_prj_cpg(base)

            appended_rows.append(
                {
                    "tile_name": tile_name,
                    "region": region["name"],
                    "source_key": f"row={key[0]};col={key[1]}",
                    "union_building_area_m2": f"{union_area:.3f}",
                    "union_building_coverage": f"{union_coverage:.6f}",
                    "feature_count": str(len(features)),
                    "folder": str(OUTPUT_ROOT / tile_name),
                    "shp": str(OUTPUT_ROOT / tile_name / f"{tile_name}.shp"),
                }
            )
            kept_tiles += 1
            next_tile_number += 1

            if candidate_number % WRITE_PROGRESS_EVERY == 0:
                log(
                    f"REGION {region['name']} WRITE candidate={candidate_number:,}/{len(selected_keys):,} "
                    f"kept={kept_tiles:,} dropped={dropped_tiles:,} next_tile={next_tile_number:06d} "
                    f"elapsed_s={time.time() - start_time:.1f}"
                )
    finally:
        for reader in readers:
            reader.close()

    stats = {
        "region": region["name"],
        "input_dir": str(region["input_dir"]),
        "input_files": [
            {
                "path": str(f["path"]),
                "records": f["count"],
                "crs": f["crs"].to_string(),
                "src": f["src"],
            }
            for f in files
        ],
        "source_records": total_records,
        "non_empty_tiles_before_filter": len(coverage_by_tile),
        "candidate_tiles_sum_area_ge_20pct": len(selected_keys),
        "kept_tiles_union_ge_20pct": kept_tiles,
        "dropped_candidate_tiles_union_lt_20pct": dropped_tiles,
        "bad_height": bad_height,
        "bad_geometry": bad_geometry,
        "skipped_near_duplicate_or_severe_overlap_inside_tiles": skipped_overlap,
        "origin": [origin_x, origin_y],
        "elapsed_s": round(time.time() - start_time, 2),
    }
    log(f"REGION {region['name']} DONE {json.dumps(stats, ensure_ascii=False)}")
    return next_tile_number, appended_rows, stats


def main() -> int:
    start_time = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    existing_numbers = existing_tile_numbers(OUTPUT_ROOT)
    if not existing_numbers:
        raise RuntimeError(f"No existing Beijing tiles found in {OUTPUT_ROOT}")
    expected = list(range(1, max(existing_numbers) + 1))
    if existing_numbers != expected:
        raise RuntimeError("Existing tile numbering is not contiguous; refusing to append")

    next_tile_number = max(existing_numbers) + 1
    log(f"APPEND START existing_tiles={len(existing_numbers):,} next_tile=tile_{next_tile_number:06d}")

    combined_rows = load_existing_index(OUTPUT_ROOT)
    all_region_stats = []
    new_rows: list[dict] = []
    try:
        for region in REGIONS:
            next_tile_number, rows, stats = process_region(region, next_tile_number, TMP_ROOT)
            new_rows.extend(rows)
            all_region_stats.append(stats)

        for tile_dir in sorted(TMP_ROOT.iterdir()):
            if tile_dir.is_dir():
                destination = OUTPUT_ROOT / tile_dir.name
                if destination.exists():
                    raise FileExistsError(destination)
                shutil.move(str(tile_dir), str(destination))
        shutil.rmtree(TMP_ROOT)

        combined_rows.extend(new_rows)
        write_combined_index(OUTPUT_ROOT, combined_rows)
    except Exception:
        log("APPEND FAILED; leaving existing Beijing tiles untouched. Temporary files remain for inspection.")
        raise

    report = {
        "output_root": str(OUTPUT_ROOT),
        "target_crs": "EPSG:32650",
        "tile_size_m": TILE_SIZE_M,
        "minimum_union_building_coverage": MIN_UNION_COVERAGE,
        "existing_tiles_before_append": len(existing_numbers),
        "first_appended_tile": f"tile_{max(existing_numbers) + 1:06d}",
        "new_tiles_appended": len(new_rows),
        "final_tile_count": len(combined_rows),
        "regions": all_region_stats,
        "tile_index_csv": str(OUTPUT_ROOT / "tile_index.csv"),
        "tile_index_all_csv": str(OUTPUT_ROOT / "tile_index_all.csv"),
        "elapsed_s": round(time.time() - start_time, 2),
    }
    (OUTPUT_ROOT / "append_regions_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("APPEND DONE")
    log(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

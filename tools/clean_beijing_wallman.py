from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import shapefile
from pyproj import CRS, Transformer


INPUT_DIR = Path(r"D:\桌面\dac\01_data\raw_datasets\Beijing_北京市")
OSM_REFERENCE = Path(r"D:\桌面\dac\01_data\raw_datasets\buildings.shp")
OUT_DIR = Path(r"D:\桌面\dac\01_data\raw_datasets\wallman_ready")

FULL_NAME = "beijing_25d_wallman_utm50n"
SAMPLE_NAME = "beijing_25d_wallman_sample_utm50n"

TARGET_CRS = CRS.from_epsg(32650)  # WGS 84 / UTM zone 50N, meters
PART_CRS = {
    "part1": CRS.from_epsg(4490),  # CGCS2000 geographic
    "part2": CRS.from_epsg(4326),  # WGS84 geographic
}

MAX_HEIGHT_M = 300.0
MIN_AREA_M2 = 2.0
PROGRESS_EVERY = 100_000


def log(message: str) -> None:
    print(message, flush=True)


def remove_existing_shapefile(base: Path) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        path = base.with_suffix(suffix)
        if path.exists():
            path.unlink()


def write_sidecars(base: Path) -> None:
    base.with_suffix(".prj").write_text(TARGET_CRS.to_wkt("WKT1_ESRI"), encoding="utf-8")
    base.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def init_writer(base: Path) -> shapefile.Writer:
    writer = shapefile.Writer(str(base), shapeType=shapefile.POLYGON)
    writer.autoBalance = 1
    writer.field("BLDG_ID", "N", 12, 0)
    writer.field("HEIGHT_M", "N", 8, 2)
    writer.field("MAT_ID", "N", 4, 0)
    writer.field("MAT_NAME", "C", 20)
    writer.field("ROOF_TYPE", "C", 20)
    writer.field("SRC", "C", 8)
    writer.field("AREA_M2", "N", 14, 2)
    return writer


def parse_height(value: object) -> float | None:
    try:
        height = float(str(value).strip())
    except Exception:
        return None
    if not math.isfinite(height):
        return None
    if height <= 0 or height > MAX_HEIGHT_M:
        return None
    return height


def clean_roof(value: object) -> str:
    roof = str(value or "").strip()
    return roof[:20] if roof else "unknown"


def shape_digest(shape: shapefile.Shape, height: float, roof: str) -> bytes:
    digest = hashlib.blake2b(digest_size=12)
    digest.update(struct.pack("<i", len(shape.parts)))
    digest.update(struct.pack("<i", int(round(height * 100))))
    digest.update(roof.encode("utf-8", "ignore"))
    for part in shape.parts:
        digest.update(struct.pack("<i", int(part)))
    for x, y in shape.points:
        digest.update(struct.pack("<ii", int(round(x * 1e7)), int(round(y * 1e7))))
    return digest.digest()


def ring_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 4:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        area += x1 * y2 - x2 * y1
    return area * 0.5


def transform_parts(
    shape: shapefile.Shape, transformer: Transformer
) -> tuple[list[list[tuple[float, float]]], float, tuple[float, float, float, float]] | None:
    parts: list[list[tuple[float, float]]] = []
    total_abs_area = 0.0
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    part_starts = list(shape.parts) + [len(shape.points)]
    for start, end in zip(part_starts, part_starts[1:]):
        raw_ring = shape.points[start:end]
        if len(raw_ring) < 4:
            continue

        xs = [pt[0] for pt in raw_ring]
        ys = [pt[1] for pt in raw_ring]
        tx, ty = transformer.transform(xs, ys)

        ring: list[tuple[float, float]] = []
        last: tuple[float, float] | None = None
        for x, y in zip(tx, ty):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            point = (round(float(x), 3), round(float(y), 3))
            if point == last:
                continue
            ring.append(point)
            last = point

        if len(ring) < 3:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            continue

        area = ring_area(ring)
        if abs(area) < MIN_AREA_M2:
            continue

        for x, y in ring:
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
        total_abs_area += abs(area)
        parts.append(ring)

    if not parts or total_abs_area < MIN_AREA_M2:
        return None
    return parts, total_abs_area, (minx, miny, maxx, maxy)


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def reference_sample_bbox() -> tuple[float, float, float, float]:
    if OSM_REFERENCE.exists():
        reader = shapefile.Reader(str(OSM_REFERENCE), encoding="utf-8")
        minx, miny, maxx, maxy = reader.bbox
        buffer_m = 500.0
        return (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)
    return (440_000.0, 4_420_000.0, 446_000.0, 4_424_000.0)


def source_crs_for(path: Path) -> CRS:
    name = path.stem.lower()
    if "part1" in name:
        return PART_CRS["part1"]
    if "part2" in name:
        return PART_CRS["part2"]
    return CRS.from_epsg(4326)


def main() -> int:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_base = OUT_DIR / FULL_NAME
    sample_base = OUT_DIR / SAMPLE_NAME
    for base in (full_base, sample_base):
        remove_existing_shapefile(base)

    full_writer = init_writer(full_base)
    sample_writer = init_writer(sample_base)
    sample_bbox = reference_sample_bbox()

    seen: set[bytes] = set()
    stats = {
        "input_features": 0,
        "written_full": 0,
        "written_sample": 0,
        "skipped_bad_height": 0,
        "skipped_bad_geometry": 0,
        "skipped_duplicate": 0,
        "min_height": None,
        "max_height": None,
        "sources": {},
        "sample_bbox_utm50n": sample_bbox,
    }

    next_id = 1
    start_time = time.time()

    shp_paths = sorted(INPUT_DIR.glob("*.shp"))
    if not shp_paths:
        raise FileNotFoundError(f"No .shp files found in {INPUT_DIR}")

    try:
        for shp_path in shp_paths:
            source_crs = source_crs_for(shp_path)
            transformer = Transformer.from_crs(source_crs, TARGET_CRS, always_xy=True)
            reader = shapefile.Reader(str(shp_path), encoding="utf-8")
            source_name = "part1" if "part1" in shp_path.stem.lower() else "part2"
            stats["sources"][source_name] = {
                "path": str(shp_path),
                "input_crs": source_crs.to_string(),
                "records": len(reader),
            }
            log(f"START {source_name}: {shp_path.name}, records={len(reader):,}, crs={source_crs.to_string()}")

            for sr in reader.iterShapeRecords():
                stats["input_features"] += 1
                rec = sr.record.as_dict()
                height = parse_height(rec.get("height"))
                if height is None:
                    stats["skipped_bad_height"] += 1
                    continue
                roof = clean_roof(rec.get("roof_type"))

                key = shape_digest(sr.shape, height, roof)
                if key in seen:
                    stats["skipped_duplicate"] += 1
                    continue
                seen.add(key)

                transformed = transform_parts(sr.shape, transformer)
                if transformed is None:
                    stats["skipped_bad_geometry"] += 1
                    continue
                parts, area_m2, bbox = transformed

                record = [
                    next_id,
                    round(height, 2),
                    1,
                    "concrete",
                    roof,
                    source_name,
                    round(area_m2, 2),
                ]
                full_writer.poly(parts)
                full_writer.record(*record)
                stats["written_full"] += 1

                if bbox_intersects(bbox, sample_bbox):
                    sample_writer.poly(parts)
                    sample_writer.record(*record)
                    stats["written_sample"] += 1

                if stats["min_height"] is None or height < stats["min_height"]:
                    stats["min_height"] = height
                if stats["max_height"] is None or height > stats["max_height"]:
                    stats["max_height"] = height

                next_id += 1

                if stats["input_features"] % PROGRESS_EVERY == 0:
                    elapsed = time.time() - start_time
                    log(
                        "PROGRESS "
                        f"input={stats['input_features']:,} "
                        f"written={stats['written_full']:,} "
                        f"sample={stats['written_sample']:,} "
                        f"dup={stats['skipped_duplicate']:,} "
                        f"bad_geom={stats['skipped_bad_geometry']:,} "
                        f"elapsed_s={elapsed:.1f}"
                    )
    finally:
        full_writer.close()
        sample_writer.close()
        write_sidecars(full_base)
        write_sidecars(sample_base)

    stats["elapsed_s"] = round(time.time() - start_time, 2)
    stats["target_crs"] = "EPSG:32650"
    stats["output_schema"] = {
        "BLDG_ID": "sequential numeric building id",
        "HEIGHT_M": "building height in meters",
        "MAT_ID": "default numeric material id, 1=concrete",
        "MAT_NAME": "default material name for WallMan material assignment",
        "ROOF_TYPE": "source roof_type, truncated to 20 chars",
        "SRC": "source shapefile part",
        "AREA_M2": "projected footprint area in square meters",
    }
    stats_path = OUT_DIR / "beijing_25d_wallman_cleaning_report.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = OUT_DIR / "README_WallMan_import.txt"
    readme.write_text(
        "\n".join(
            [
                "Beijing 2.5D building vector data for Altair WinProp WallMan",
                "",
                "Recommended import file:",
                f"  {full_base.with_suffix('.shp')}",
                "",
                "Small import-test file:",
                f"  {sample_base.with_suffix('.shp')}",
                "",
                "CRS:",
                "  EPSG:32650, WGS 84 / UTM zone 50N, unit: meter",
                "",
                "Fields:",
                "  BLDG_ID   numeric id",
                "  HEIGHT_M  building height in meters; map this as WallMan building height",
                "  MAT_ID    default numeric material id, 1=concrete",
                "  MAT_NAME  default material name; map or replace in WallMan as needed",
                "  ROOF_TYPE source roof type",
                "  SRC       original source part",
                "  AREA_M2   projected footprint area",
                "",
                "Notes:",
                "  The original Beijing source was split into part1 and part2.",
                "  part1 was treated as CGCS2000 geographic (EPSG:4490).",
                "  part2 was treated as WGS84 geographic (EPSG:4326).",
                "  Both were reprojected to UTM 50N for meter-based propagation work.",
                "  Very tiny polygons below 2 square meters were removed.",
                "  Exact duplicate source geometries were removed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    log("DONE")
    log(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

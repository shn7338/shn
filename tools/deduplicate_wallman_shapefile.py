from __future__ import annotations

import collections
import json
import math
import shutil
import sys
import time
from pathlib import Path

import shapefile
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid


OUT_DIR = Path(r"D:\桌面\dac\01_data\raw_datasets\wallman_ready")
INPUT_BASE = OUT_DIR / "beijing_25d_wallman_utm50n"
OUTPUT_BASE = OUT_DIR / "beijing_25d_wallman_dedup_utm50n"
SAMPLE_OUTPUT_BASE = OUT_DIR / "beijing_25d_wallman_dedup_sample_utm50n"

CELL_SIZE_M = 80.0
PROGRESS_EVERY = 100_000
OVERLAP_AREA_THRESHOLD_M2 = 1.0
OVERLAP_RATIO_THRESHOLD = 0.90


def log(message: str) -> None:
    print(message, flush=True)


def remove_existing_shapefile(base: Path) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        path = base.with_suffix(suffix)
        if path.exists():
            path.unlink()


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


def copy_sidecars(src_base: Path, dst_base: Path) -> None:
    for suffix in (".prj", ".cpg"):
        src = src_base.with_suffix(suffix)
        dst = dst_base.with_suffix(suffix)
        if src.exists():
            shutil.copyfile(src, dst)


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def cells_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = bbox
    ix0 = math.floor(minx / CELL_SIZE_M)
    iy0 = math.floor(miny / CELL_SIZE_M)
    ix1 = math.floor(maxx / CELL_SIZE_M)
    iy1 = math.floor(maxy / CELL_SIZE_M)
    return [(ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1)]


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


def reference_sample_bbox() -> tuple[float, float, float, float]:
    sample = OUT_DIR / "beijing_25d_wallman_sample_utm50n.shp"
    if sample.exists():
        reader = shapefile.Reader(str(sample), encoding="utf-8")
        minx, miny, maxx, maxy = reader.bbox
        return (float(minx), float(miny), float(maxx), float(maxy))
    return (440_000.0, 4_420_000.0, 446_000.0, 4_424_000.0)


def write_record(writer: shapefile.Writer, shape: shapefile.Shape, record: dict) -> None:
    writer.shape(shape)
    writer.record(
        int(record["BLDG_ID"]),
        float(record["HEIGHT_M"]),
        int(record["MAT_ID"]),
        str(record["MAT_NAME"]),
        str(record["ROOF_TYPE"]),
        str(record["SRC"]),
        float(record["AREA_M2"]),
    )


def main() -> int:
    input_shp = INPUT_BASE.with_suffix(".shp")
    if not input_shp.exists():
        raise FileNotFoundError(input_shp)

    for base in (OUTPUT_BASE, SAMPLE_OUTPUT_BASE):
        remove_existing_shapefile(base)

    reader = shapefile.Reader(str(input_shp), encoding="utf-8")
    full_writer = init_writer(OUTPUT_BASE)
    sample_writer = init_writer(SAMPLE_OUTPUT_BASE)
    sample_bbox = reference_sample_bbox()

    grid: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    accepted_geoms = []
    accepted_bboxes: list[tuple[float, float, float, float]] = []
    accepted_areas: list[float] = []
    accepted_ids: list[int] = []

    stats = {
        "input_path": str(input_shp),
        "input_features": len(reader),
        "written_full": 0,
        "written_sample": 0,
        "skipped_bad_geometry": 0,
        "skipped_near_duplicate_or_severe_overlap": 0,
        "near_duplicate_rule": {
            "intersection_area_m2_at_least": OVERLAP_AREA_THRESHOLD_M2,
            "intersection_ratio_of_smaller_building_at_least": OVERLAP_RATIO_THRESHOLD,
            "cell_size_m": CELL_SIZE_M,
        },
        "sample_bbox_utm50n": sample_bbox,
        "first_skipped_pairs": [],
    }

    start = time.time()
    try:
        for input_index, sr in enumerate(reader.iterShapeRecords(), start=1):
            rec = sr.record.as_dict()
            geom = shape_to_geometry(sr.shape)
            if geom is None:
                stats["skipped_bad_geometry"] += 1
                continue

            bbox_tuple = tuple(float(v) for v in geom.bounds)
            area = float(geom.area)
            candidate_indices: set[int] = set()
            for cell in cells_for_bbox(bbox_tuple):
                candidate_indices.update(grid.get(cell, ()))

            duplicate_of = None
            duplicate_overlap_area = 0.0
            duplicate_overlap_ratio = 0.0
            for idx in candidate_indices:
                if not bbox_intersects(bbox_tuple, accepted_bboxes[idx]):
                    continue
                other = accepted_geoms[idx]
                if not geom.intersects(other):
                    continue
                overlap_area = float(geom.intersection(other).area)
                if overlap_area < OVERLAP_AREA_THRESHOLD_M2:
                    continue
                overlap_ratio = overlap_area / max(min(area, accepted_areas[idx]), 1e-9)
                if overlap_ratio >= OVERLAP_RATIO_THRESHOLD:
                    duplicate_of = idx
                    duplicate_overlap_area = overlap_area
                    duplicate_overlap_ratio = overlap_ratio
                    break

            if duplicate_of is not None:
                stats["skipped_near_duplicate_or_severe_overlap"] += 1
                if len(stats["first_skipped_pairs"]) < 50:
                    stats["first_skipped_pairs"].append(
                        {
                            "skipped_bldg_id": int(rec["BLDG_ID"]),
                            "kept_bldg_id": int(accepted_ids[duplicate_of]),
                            "overlap_m2": round(duplicate_overlap_area, 2),
                            "overlap_ratio_min_area": round(duplicate_overlap_ratio, 4),
                        }
                    )
                continue

            accepted_idx = len(accepted_geoms)
            accepted_geoms.append(geom)
            accepted_bboxes.append(bbox_tuple)
            accepted_areas.append(area)
            accepted_ids.append(int(rec["BLDG_ID"]))
            for cell in cells_for_bbox(bbox_tuple):
                grid[cell].append(accepted_idx)

            write_record(full_writer, sr.shape, rec)
            stats["written_full"] += 1
            if bbox_intersects(bbox_tuple, sample_bbox):
                write_record(sample_writer, sr.shape, rec)
                stats["written_sample"] += 1

            if input_index % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                log(
                    "PROGRESS "
                    f"input={input_index:,} "
                    f"written={stats['written_full']:,} "
                    f"sample={stats['written_sample']:,} "
                    f"skipped_overlap={stats['skipped_near_duplicate_or_severe_overlap']:,} "
                    f"bad_geom={stats['skipped_bad_geometry']:,} "
                    f"elapsed_s={elapsed:.1f}"
                )
    finally:
        full_writer.close()
        sample_writer.close()
        copy_sidecars(INPUT_BASE, OUTPUT_BASE)
        copy_sidecars(INPUT_BASE, SAMPLE_OUTPUT_BASE)

    stats["elapsed_s"] = round(time.time() - start, 2)
    stats_path = OUT_DIR / "beijing_25d_wallman_dedup_report.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = OUT_DIR / "README_WallMan_import_dedup.txt"
    readme.write_text(
        "\n".join(
            [
                "Deduplicated Beijing 2.5D building vector data for Altair WinProp WallMan",
                "",
                "Recommended import file:",
                str(OUTPUT_BASE.with_suffix(".shp")),
                "",
                "Small import-test file:",
                str(SAMPLE_OUTPUT_BASE.with_suffix(".shp")),
                "",
                "CRS:",
                "  EPSG:32650, WGS 84 / UTM zone 50N, unit: meter",
                "",
                "Fields:",
                "  BLDG_ID   unique numeric building id inherited from the cleaned full file",
                "  HEIGHT_M  building height in meters; map this as WallMan building height",
                "  MAT_ID    default numeric material id, 1=concrete",
                "  MAT_NAME  default material name; map or replace in WallMan as needed",
                "  ROOF_TYPE source roof type",
                "  SRC       original source part",
                "  AREA_M2   projected footprint area",
                "",
                "Deduplication rule:",
                "  If a new building overlaps an already-kept building by at least 1 m2 and",
                "  the overlap covers at least 90% of the smaller footprint, the new building",
                "  is treated as a near duplicate / severe overlap and is skipped.",
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

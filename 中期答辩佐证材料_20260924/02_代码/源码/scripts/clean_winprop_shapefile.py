"""Create a WallMan-friendly building shapefile.

The source shapefile is never modified. Invalid polygonal parts are repaired,
duplicate precision noise is removed, and optional topology-preserving
simplification removes tiny segments that WallMan treats as invalid. Overlaps can
either be preserved (the native WinProp urban model supports them) or resolved
deterministically: taller buildings win and, for equal height, the smaller
footprint wins.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import shapefile
from shapely import make_valid, set_precision
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree
from shapely.ops import unary_union

warnings.filterwarnings(
    "ignore",
    message=r"Possible issue encountered when converting Shape .* to GeoJSON.*",
)


@dataclass
class CleaningStats:
    source_features: int = 0
    source_polygon_parts: int = 0
    source_invalid_features: int = 0
    source_overlap_pairs: int = 0
    output_features: int = 0
    clipped_features: int = 0
    dropped_non_polygonal: int = 0
    dropped_small_parts: int = 0
    dropped_hole_parts: int = 0
    clearance_repaired_features: int = 0
    convex_hull_fallbacks: int = 0
    output_invalid_features: int = 0
    output_overlap_pairs: int = 0


@dataclass
class PolygonItem:
    geometry: Polygon
    record: list[Any]
    source_index: int
    height: float
    area: float


def polygon_parts(geometry: Any) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def finite_height(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def count_overlap_pairs(geometries: list[Polygon], area_tolerance: float) -> int:
    if not geometries:
        return 0
    tree = STRtree(geometries)
    count = 0
    for index, geometry in enumerate(geometries):
        for other_index in tree.query(geometry, predicate="intersects"):
            other = int(other_index)
            if other <= index:
                continue
            intersection = geometry.intersection(geometries[other])
            if not intersection.is_empty and intersection.area > area_tolerance:
                count += 1
    return count


def copy_projection_sidecars(source: Path, destination: Path) -> None:
    for suffix in (".prj", ".cpg"):
        source_sidecar = source.with_suffix(suffix)
        if source_sidecar.exists():
            shutil.copy2(source_sidecar, destination.with_suffix(suffix))


def stabilize_polygon(
    polygon: Polygon,
    minimum_clearance: float,
    precision: float,
    stats: CleaningStats,
) -> Any:
    """Remove near self-touches that WallMan rejects despite GEOS validity."""
    if minimum_clearance <= 0 or polygon.minimum_clearance >= minimum_clearance:
        return polygon

    stats.clearance_repaired_features += 1
    repaired = polygon.buffer(
        minimum_clearance, join_style=2
    ).buffer(-minimum_clearance, join_style=2)
    repaired = set_precision(make_valid(repaired), precision)

    repaired_parts = list(polygon_parts(repaired))
    if (
        not repaired_parts
        or any(
            part.minimum_clearance < minimum_clearance
            for part in repaired_parts
        )
    ):
        # This fallback is limited to pathological sub-centimetre folds. It
        # guarantees a consistently ordered ring while leaving normal concave
        # buildings untouched.
        stats.convex_hull_fallbacks += 1
        repaired = set_precision(polygon.convex_hull, precision)
    return repaired


def clean_shapefile(
    source_shp: Path,
    output_shp: Path,
    height_field: str,
    gap: float,
    min_area: float,
    precision: float,
    simplify_tolerance: float,
    minimum_clearance: float,
    preserve_overlaps: bool,
) -> CleaningStats:
    source_shp = source_shp.resolve()
    output_shp = output_shp.resolve()
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    stats = CleaningStats()

    reader = shapefile.Reader(str(source_shp))
    field_defs = list(reader.fields[1:])
    field_names = [field[0] for field in field_defs]
    try:
        height_index = next(
            index for index, name in enumerate(field_names) if name.upper() == height_field.upper()
        )
    except StopIteration as exc:
        raise ValueError(f"Height field {height_field!r} was not found in {source_shp}") from exc

    items: list[PolygonItem] = []
    for source_index, shape_record in enumerate(reader.iterShapeRecords()):
        stats.source_features += 1
        source_geometry = shape(shape_record.shape.__geo_interface__)
        if not source_geometry.is_valid:
            stats.source_invalid_features += 1
        valid_geometry = make_valid(source_geometry)
        valid_geometry = set_precision(valid_geometry, precision)
        if simplify_tolerance > 0:
            valid_geometry = valid_geometry.simplify(
                simplify_tolerance, preserve_topology=True
            )
            valid_geometry = make_valid(valid_geometry)
            valid_geometry = set_precision(valid_geometry, precision)
        parts = list(polygon_parts(valid_geometry))
        if not parts:
            stats.dropped_non_polygonal += 1
            continue
        record = list(shape_record.record)
        height = finite_height(record[height_index])
        for part in parts:
            if part.is_empty or part.area < min_area:
                stats.dropped_small_parts += 1
                continue
            stats.source_polygon_parts += 1
            items.append(
                PolygonItem(
                    geometry=part,
                    record=record,
                    source_index=source_index,
                    height=height,
                    area=float(part.area),
                )
            )
    reader.close()

    # Taller footprints win their overlap. For equal heights, detailed/smaller
    # footprints win over broad outlines such as OSM building shells.
    items.sort(key=lambda item: (-item.height, item.area, item.source_index))
    originals = [item.geometry for item in items]
    stats.source_overlap_pairs = count_overlap_pairs(originals, precision * precision)
    source_tree = STRtree(originals) if originals else None

    output_items: list[PolygonItem] = []
    for rank, item in enumerate(items):
        result = item.geometry
        if not preserve_overlaps and source_tree is not None and rank > 0:
            search_geometry = item.geometry.buffer(gap)
            blocker_indices = [
                int(candidate)
                for candidate in source_tree.query(search_geometry, predicate="intersects")
                if int(candidate) < rank
            ]
            if blocker_indices:
                blockers = unary_union([originals[index] for index in blocker_indices])
                result = item.geometry.difference(blockers.buffer(gap, join_style=2))
                if not result.equals(item.geometry):
                    stats.clipped_features += 1

        result = make_valid(result)
        result = set_precision(result, precision)
        if simplify_tolerance > 0:
            result = result.simplify(simplify_tolerance, preserve_topology=True)
            result = make_valid(result)
            result = set_precision(result, precision)
        for part in polygon_parts(result):
            stable_geometry = stabilize_polygon(
                part, minimum_clearance, precision, stats
            )
            for stable_part in polygon_parts(stable_geometry):
                if stable_part.is_empty or stable_part.area < min_area:
                    stats.dropped_small_parts += 1
                    continue
                # WinProp's ODA writer uses one outer ring per building. Retaining
                # a hole would silently fill it during conversion.
                if len(stable_part.interiors) > 0:
                    stats.dropped_hole_parts += 1
                    continue
                output_items.append(
                    PolygonItem(
                        geometry=stable_part,
                        record=item.record,
                        source_index=item.source_index,
                        height=item.height,
                        area=float(stable_part.area),
                    )
                )

    output_geometries = [item.geometry for item in output_items]
    stats.output_invalid_features = sum(not geometry.is_valid for geometry in output_geometries)
    stats.output_overlap_pairs = count_overlap_pairs(
        output_geometries, precision * precision
    )
    if stats.output_invalid_features or (
        not preserve_overlaps and stats.output_overlap_pairs
    ):
        raise RuntimeError(
            "Cleaner validation failed: "
            f"invalid={stats.output_invalid_features}, "
            f"overlap_pairs={stats.output_overlap_pairs}"
        )

    writer = shapefile.Writer(
        str(output_shp.with_suffix("")),
        shapeType=shapefile.POLYGON,
        encoding=getattr(reader, "encoding", "utf-8"),
    )
    for field in field_defs:
        writer.field(*field)
    for item in output_items:
        polygon = orient(item.geometry, sign=-1.0)
        coordinates = list(polygon.exterior.coords)
        writer.poly([coordinates])
        writer.record(*item.record)
    writer.close()
    copy_projection_sidecars(source_shp, output_shp)

    stats.output_features = len(output_items)
    stats_path = output_shp.with_suffix(".clean_stats.json")
    stats_path.write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-shp", type=Path, required=True)
    parser.add_argument("--output-shp", type=Path, required=True)
    parser.add_argument("--height-field", default="HEIGHT_M")
    parser.add_argument("--gap", type=float, default=0.02)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--precision", type=float, default=0.001)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--minimum-clearance", type=float, default=0.0)
    parser.add_argument("--preserve-overlaps", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = clean_shapefile(
        source_shp=args.source_shp,
        output_shp=args.output_shp,
        height_field=args.height_field,
        gap=args.gap,
        min_area=args.min_area,
        precision=args.precision,
        simplify_tolerance=args.simplify_tolerance,
        minimum_clearance=args.minimum_clearance,
        preserve_overlaps=args.preserve_overlaps,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a local-coordinate watertight Sionna mesh from a tile shapefile.

The input footprints remain untouched.  Buildings are clipped to the exact
prepared 512 m grid, translated to a local [0, 512] x [0, 512] coordinate
system, extruded from z=0 to HEIGHT_M, and exported as a triangle PLY mesh.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import mapbox_earcut
import numpy as np
import shapefile
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.geometry.polygon import orient
from shapely.validation import make_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def polygons_from_geometry(geometry: Any) -> Iterable[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from polygons_from_geometry(part)


def clean_ring(coords: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for x, y in coords:
        point = (float(x), float(y))
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def append_extruded_polygon(
    polygon: Polygon,
    height: float,
    origin_x: float,
    origin_y: float,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> bool:
    polygon = orient(polygon, sign=1.0)
    rings = [clean_ring(polygon.exterior.coords)]
    rings.extend(clean_ring(ring.coords) for ring in polygon.interiors)
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        return False

    local_rings = [
        [(x - origin_x, y - origin_y) for x, y in ring]
        for ring in rings
    ]
    points_2d = np.asarray(
        [point for ring in local_rings for point in ring], dtype=np.float64
    )
    ring_ends = np.cumsum([len(ring) for ring in local_rings], dtype=np.uint32)
    triangles = mapbox_earcut.triangulate_float64(points_2d, ring_ends)
    if triangles.size == 0:
        return False

    base = len(vertices)
    count = len(points_2d)
    vertices.extend((float(x), float(y), 0.0) for x, y in points_2d)
    vertices.extend((float(x), float(y), float(height)) for x, y in points_2d)

    for index in range(0, len(triangles), 3):
        a, b, c = (int(triangles[index + offset]) for offset in range(3))
        pa, pb, pc = points_2d[[a, b, c]]
        edge_b = pb - pa
        edge_c = pc - pa
        signed_area = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]
        if signed_area < 0:
            b, c = c, b
        faces.append((base + count + a, base + count + b, base + count + c))
        faces.append((base + c, base + b, base + a))

    ring_start = 0
    for ring in local_rings:
        ring_count = len(ring)
        for offset in range(ring_count):
            current = ring_start + offset
            following = ring_start + ((offset + 1) % ring_count)
            bottom_current = base + current
            bottom_following = base + following
            top_current = base + count + current
            top_following = base + count + following
            faces.append((bottom_current, bottom_following, top_following))
            faces.append((bottom_current, top_following, top_current))
        ring_start += ring_count
    return True


def ply_text(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> str:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    vertex_lines = [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    face_lines = [f"3 {a} {b} {c}" for a, b, c in faces]
    return "\n".join(header + vertex_lines + face_lines) + "\n"


def ground_ply_text(width: float, height: float) -> str:
    vertices = [
        (0.0, 0.0, 0.0),
        (float(width), 0.0, 0.0),
        (float(width), float(height), 0.0),
        (0.0, float(height), 0.0),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    return ply_text(vertices, faces)


def build_scene(config_path: Path, overwrite: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata_path = Path(config["prepared_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    grid = metadata["grid"]
    output_root = Path(config["output_root"])
    scene_dir = output_root / "scene"
    building_ply = scene_dir / "buildings.ply"
    ground_ply = scene_dir / "ground.ply"
    manifest_path = scene_dir / "scene_manifest.json"

    if (
        not overwrite
        and building_ply.exists()
        and ground_ply.exists()
        and manifest_path.exists()
    ):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    xmin = float(grid["xmin"])
    ymin = float(grid["ymin"])
    xmax = float(grid["xmax"])
    ymax = float(grid["ymax"])
    width = xmax - xmin
    height = ymax - ymin
    if not math.isclose(width, 512.0) or not math.isclose(height, 512.0):
        raise ValueError(f"Expected an exact 512 m grid, got {width} x {height}")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    tile_clip = box(xmin, ymin, xmax, ymax)
    source_path = Path(config["source_shapefile"])
    reader = shapefile.Reader(str(source_path))
    field_name = config["scene"]["building_height_field"]
    feature_count = 0
    polygon_count = 0
    skipped_count = 0
    repaired_count = 0
    max_height = 0.0

    for record in reader.iterShapeRecords():
        feature_count += 1
        attributes = record.record.as_dict()
        try:
            building_height = float(attributes[field_name])
        except (KeyError, TypeError, ValueError):
            skipped_count += 1
            continue
        if not math.isfinite(building_height) or building_height <= 0:
            skipped_count += 1
            continue
        geometry = shape(record.shape.__geo_interface__)
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            repaired_count += 1
        geometry = geometry.intersection(tile_clip)
        added_feature = False
        for polygon in polygons_from_geometry(geometry):
            if polygon.area <= 1e-6:
                continue
            if append_extruded_polygon(
                polygon,
                building_height,
                xmin,
                ymin,
                vertices,
                faces,
            ):
                polygon_count += 1
                added_feature = True
                max_height = max(max_height, building_height)
        if not added_feature:
            skipped_count += 1

    if not vertices or not faces:
        raise RuntimeError("No valid building mesh was generated")

    atomic_write_text(building_ply, ply_text(vertices, faces))
    atomic_write_text(ground_ply, ground_ply_text(width, height))
    manifest = {
        "version": 1,
        "tile": config["tile"],
        "source_shapefile": str(source_path),
        "source_crs": config["scene"]["source_crs"],
        "source_grid": grid,
        "local_bounds": [0.0, 0.0, width, height],
        "local_origin_utm": [xmin, ymin],
        "feature_count": feature_count,
        "extruded_polygon_count": polygon_count,
        "skipped_feature_count": skipped_count,
        "repaired_feature_count": repaired_count,
        "max_building_height_m": max_height,
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "building_ply": str(building_ply),
        "ground_ply": str(ground_ply),
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_scene(args.config.resolve(), overwrite=args.overwrite)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

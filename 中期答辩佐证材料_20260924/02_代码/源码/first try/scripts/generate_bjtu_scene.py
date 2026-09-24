"""Generate a Geo2SigMap scene for the Beijing Jiaotong University main campus."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scene_generation.core as scene_core
import scene_generation.utils as scene_utils


DEFAULT_LON = 116.3360690
DEFAULT_LAT = 39.9504404
DEFAULT_SIZE_M = 512
DEFAULT_SEED = 20260524


def building_height_from_osm(building: dict, polygon) -> float:
    """Use height or floor-count tags before the upstream random fallback."""
    for tag in ("building:height", "height"):
        if tag in building and scene_utils.is_float(building[tag]):
            return float(building[tag])
    if "building:levels" in building and scene_utils.is_float(
        building["building:levels"]
    ):
        return float(building["building:levels"]) * 3.5
    return scene_utils.random_building_height(building, polygon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--size-m", type=int, default=DEFAULT_SIZE_M)
    parser.add_argument("--resolution-m", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scenes/BJTU_Main_512"),
        help="Directory for scene.xml, meshes, and building-height maps.",
    )
    parser.add_argument(
        "--osm-server",
        default="https://overpass-api.de/api/interpreter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # The upstream helper currently ignores building:levels unless a "level"
    # tag also exists. This local override preserves available OSM floor counts.
    scene_core.random_building_height = building_height_from_osm

    points = scene_utils.rect_from_point_and_size(
        args.lon, args.lat, "center", args.size_m, args.size_m
    )
    height_map = scene_core.Scene()(
        points=points,
        data_dir=str(args.output),
        hag_tiff_path=None,
        osm_server_addr=args.osm_server,
        lidar_calibration=False,
        generate_building_map=True,
        ground_material_type="mat-itu_wet_ground",
        rooftop_material_type="mat-itu_metal",
        wall_material_type="mat-itu_concrete",
    )

    step = args.resolution_m
    height_map_reduced = height_map[::step, ::step]
    np.save(args.output / f"2D_Building_Height_Map_{step}m.npy", height_map_reduced)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(height_map_reduced, origin="upper", cmap="viridis")
    ax.set_title(f"BJTU Main Campus Building Height Map ({step} m grid)")
    ax.set_xlabel("East-West cell")
    ax.set_ylabel("North-South cell")
    fig.colorbar(image, ax=ax, label="Height (m)")
    fig.savefig(args.output / f"2D_Building_Height_Map_{step}m.png", dpi=180)
    plt.close(fig)

    print(f"Scene XML: {args.output / 'scene.xml'}")
    print(f"Building map: {height_map_reduced.shape}, max height={height_map_reduced.max():.1f} m")
    print(f"Fallback seed: {args.seed}")


if __name__ == "__main__":
    main()

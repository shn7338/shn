"""Create 512 x 512 Geo2SigMap-style figures for the BJTU data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


DEFAULT_LON = 116.3360690
DEFAULT_LAT = 39.9504404
DEFAULT_SIZE_M = 512.0
FIG_DPI = 100
FIG_SIZE_IN = 5.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root directory containing scenes/, cache/, and outputs/.",
    )
    parser.add_argument(
        "--scene-name",
        default="BJTU_Main_512",
        help="Scene/output folder name under scenes/ and outputs/.",
    )
    parser.add_argument(
        "--radio-map",
        choices=("iso", "directional_north"),
        default="iso",
        help="Radio-map output to use for the path-gain figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the 512 x 512 figures.",
    )
    return parser.parse_args()


def meters_from_lon_lat(lon: float, lat: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(DEFAULT_LAT))
    x = (lon - DEFAULT_LON) * meters_per_deg_lon
    y = (lat - DEFAULT_LAT) * meters_per_deg_lat
    return x, y


def load_osm_polygons(cache_dir: Path) -> list[np.ndarray]:
    cache_files = sorted(cache_dir.glob("*.json"))
    if not cache_files:
        raise FileNotFoundError(f"No OSM cache JSON found in {cache_dir}")

    with cache_files[0].open(encoding="utf-8") as f:
        data = json.load(f)

    nodes: dict[int, tuple[float, float]] = {}
    for element in data["elements"]:
        if element.get("type") == "node":
            nodes[int(element["id"])] = meters_from_lon_lat(
                float(element["lon"]), float(element["lat"])
            )

    half = DEFAULT_SIZE_M / 2.0
    polygons: list[np.ndarray] = []
    for element in data["elements"]:
        tags = element.get("tags", {})
        if element.get("type") != "way" or "building" not in tags:
            continue
        points = [nodes[node_id] for node_id in element["nodes"] if node_id in nodes]
        if len(points) < 3:
            continue
        polygon = np.asarray(points, dtype=float)
        if (
            polygon[:, 0].max() < -half
            or polygon[:, 0].min() > half
            or polygon[:, 1].max() < -half
            or polygon[:, 1].min() > half
        ):
            continue
        polygons.append(polygon)
    return polygons


def add_boundary(ax: plt.Axes) -> None:
    half = DEFAULT_SIZE_M / 2.0
    ax.add_patch(
        Rectangle(
            (-half, -half),
            DEFAULT_SIZE_M,
            DEFAULT_SIZE_M,
            fill=False,
            edgecolor="#f97316",
            linewidth=2.2,
            zorder=10,
        )
    )


def save_square(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_geographic_map(polygons: list[np.ndarray], output: Path) -> None:
    fig = plt.figure(figsize=(FIG_SIZE_IN, FIG_SIZE_IN), dpi=FIG_DPI)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.set_facecolor("#dce8d2")
    half = DEFAULT_SIZE_M / 2.0

    # Soft map-like land texture from deterministic sinusoidal bands.
    grid = np.linspace(-half, half, 512)
    xx, yy = np.meshgrid(grid, grid)
    texture = (
        0.55
        + 0.20 * np.sin((xx + 1.7 * yy) / 58.0)
        + 0.12 * np.cos((1.2 * xx - yy) / 41.0)
    )
    ax.imshow(
        texture,
        extent=[-half, half, -half, half],
        origin="lower",
        cmap="Greens",
        alpha=0.22,
        zorder=0,
    )

    if polygons:
        building_collection = PolyCollection(
            polygons,
            facecolors="#d7ddd8",
            edgecolors="#64756f",
            linewidths=0.55,
            zorder=5,
        )
        ax.add_collection(building_collection)

    add_boundary(ax)
    ax.scatter([0], [0], marker="*", s=80, c="#d7191c", edgecolors="white", linewidths=0.8, zorder=15)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")
    save_square(fig, output)


def render_3d_with_pyvista(scene_dir: Path, output: Path) -> bool:
    try:
        import pyvista as pv
    except ImportError:
        return False

    mesh_files = sorted((scene_dir / "mesh").glob("*.ply"))
    if not mesh_files:
        return False

    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
    plotter.set_background("#f3f4f1")

    half = DEFAULT_SIZE_M / 2.0
    plane = pv.Plane(
        center=(0, 0, -0.15),
        direction=(0, 0, 1),
        i_size=DEFAULT_SIZE_M,
        j_size=DEFAULT_SIZE_M,
        i_resolution=1,
        j_resolution=1,
    )
    plotter.add_mesh(plane, color="#aeb3b0", smooth_shading=False)

    for mesh_file in mesh_files:
        if mesh_file.name == "ground.ply":
            continue
        mesh = pv.read(mesh_file)
        inside_window = []
        for cell_id in range(mesh.n_cells):
            points = mesh.get_cell(cell_id).points
            inside_window.append(
                bool(
                    np.all(points[:, 0] >= -half)
                    and np.all(points[:, 0] <= half)
                    and np.all(points[:, 1] >= -half)
                    and np.all(points[:, 1] <= half)
                )
            )
        inside_window = np.asarray(inside_window, dtype=bool)
        if not np.any(inside_window):
            continue
        mesh = mesh.extract_cells(inside_window)
        if "rooftop" in mesh_file.stem:
            plotter.add_mesh(mesh, color="#cfe6e3", smooth_shading=False, ambient=0.35, diffuse=0.65)
        else:
            plotter.add_mesh(mesh, color="#8fa19d", smooth_shading=False, ambient=0.28, diffuse=0.72)

    outline_points = np.array(
        [
            [-half, -half, 0.08],
            [half, -half, 0.08],
            [half, half, 0.08],
            [-half, half, 0.08],
            [-half, -half, 0.08],
        ]
    )
    outline = pv.PolyData(outline_points)
    outline.lines = np.hstack([[5, 0, 1, 2, 3, 4]])
    plotter.add_mesh(outline.tube(radius=1.6), color="#f97316", smooth_shading=True)

    tx = np.array([0.0, 0.0, 50.0])
    plotter.add_mesh(pv.Sphere(radius=9, center=tx), color="#d7191c", smooth_shading=True)
    ray_targets = [
        (-half, -half, 1),
        (half, -half, 1),
        (half, half, 1),
        (-half, half, 1),
        (-176, -half, 1),
        (0, -half, 1),
        (174, -half, 1),
        (-half, -86, 1),
        (half, -46, 1),
        (-146, half, 1),
        (84, half, 1),
    ]
    for target in ray_targets:
        line = pv.Line(tx, target, resolution=1)
        plotter.add_mesh(line, color="#2d3436", opacity=0.22, line_width=1)

    plotter.camera_position = [(180, -620, 520), (0, 0, 8), (0, 0, 1)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 332
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(str(output))
    plotter.close()
    return True


def render_3d_with_matplotlib(height_map: np.ndarray, output: Path) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(FIG_SIZE_IN, FIG_SIZE_IN), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f3f4f1")
    ax.set_facecolor("#f3f4f1")

    step = 8
    y_idx, x_idx = np.where(height_map[::step, ::step] > 0)
    heights = height_map[::step, ::step][y_idx, x_idx].astype(float)
    xs = x_idx * step
    ys = y_idx * step
    ax.bar3d(xs, ys, np.zeros_like(xs), step, step, heights, color="#9eb8b4", shade=True, linewidth=0)

    half = DEFAULT_SIZE_M
    boundary = [[(0, 0, 0), (half, 0, 0), (half, half, 0), (0, half, 0)]]
    ax.add_collection3d(Poly3DCollection(boundary, facecolor="#aeb3b0", edgecolor="#f97316", linewidth=2.2, alpha=0.45))
    ax.scatter([256], [256], [50], marker="*", s=110, c="#d7191c")
    ax.view_init(elev=32, azim=-65)
    ax.set_xlim(0, half)
    ax.set_ylim(0, half)
    ax.set_zlim(0, 70)
    ax.axis("off")
    save_square(fig, output)


def draw_path_gain(height_map_4m: np.ndarray, path_gain_db: np.ndarray, output: Path) -> None:
    fig = plt.figure(figsize=(FIG_SIZE_IN, FIG_SIZE_IN), dpi=FIG_DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.06, 0.06, 0.76, 0.88])
    cax = fig.add_axes([0.86, 0.15, 0.045, 0.70])

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(path_gain_db)
    image = ax.imshow(
        masked,
        cmap=cmap,
        norm=Normalize(vmin=-160, vmax=-75),
        origin="upper",
        extent=[-256, 256, -256, 256],
        interpolation="bilinear",
        zorder=1,
    )

    buildings = np.ma.masked_where(height_map_4m <= 0, height_map_4m)
    building_cmap = plt.get_cmap("Greys").copy()
    building_cmap.set_bad(alpha=0.0)
    ax.imshow(
        buildings,
        cmap=building_cmap,
        vmin=0,
        vmax=45,
        origin="upper",
        extent=[-256, 256, -256, 256],
        alpha=0.22,
        interpolation="nearest",
        zorder=3,
    )

    add_boundary(ax)
    ax.scatter([0], [0], marker="*", s=135, c="#d7191c", edgecolors="white", linewidths=1.0, zorder=15)
    ax.set_xlim(-256, 256)
    ax.set_ylim(-256, 256)
    ax.set_aspect("equal")
    ax.axis("off")

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=8)
    save_square(fig, output)


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    scene_dir = project_root / "scenes" / args.scene_name
    output_dir = args.output or project_root / "outputs" / args.scene_name / "figures_512"
    output_dir.mkdir(parents=True, exist_ok=True)

    polygons = load_osm_polygons(project_root / "cache")
    height_map = np.load(scene_dir / "2D_Building_Height_Map.npy")
    height_map_4m = np.load(scene_dir / "2D_Building_Height_Map_4m.npy")
    path_gain_db = np.load(project_root / "outputs" / args.scene_name / args.radio_map / "path_gain_outdoor_db.npy")

    draw_geographic_map(polygons, output_dir / "bjtu_geographic_map_512.png")
    if not render_3d_with_pyvista(scene_dir, output_dir / "bjtu_3d_model_512.png"):
        render_3d_with_matplotlib(height_map, output_dir / "bjtu_3d_model_512.png")
    draw_path_gain(height_map_4m, path_gain_db, output_dir / "bjtu_path_gain_512.png")

    print(f"Saved 512 x 512 figures to {output_dir}")


if __name__ == "__main__":
    main()

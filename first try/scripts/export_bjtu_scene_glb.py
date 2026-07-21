"""Export the BJTU 512 m window as a single GLB model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


DEFAULT_SIZE_M = 512.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root directory containing scenes/ and outputs/.",
    )
    parser.add_argument(
        "--scene-name",
        default="BJTU_Main_512",
        help="Scene folder name under scenes/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GLB path.",
    )
    return parser.parse_args()


def color_mesh(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual.face_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def pv_to_trimesh(mesh: pv.DataSet) -> trimesh.Trimesh | None:
    surface = mesh.extract_surface().triangulate()
    if surface.n_points == 0 or surface.n_cells == 0:
        return None
    faces = surface.faces.reshape((-1, 4))[:, 1:]
    return trimesh.Trimesh(vertices=np.asarray(surface.points), faces=faces, process=False)


def keep_cells_inside_window(mesh: pv.DataSet, half: float) -> pv.DataSet | None:
    inside_window: list[bool] = []
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
    mask = np.asarray(inside_window, dtype=bool)
    if not np.any(mask):
        return None
    return mesh.extract_cells(mask)


def cylinder_between(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    rgba: tuple[int, int, int, int],
    sections: int = 16,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, segment=[start, end], sections=sections)
    return color_mesh(mesh, rgba)


def build_scene(mesh_dir: Path) -> trimesh.Scene:
    half = DEFAULT_SIZE_M / 2.0
    scene = trimesh.Scene()

    ground = trimesh.creation.box(extents=(DEFAULT_SIZE_M, DEFAULT_SIZE_M, 0.2))
    ground.apply_translation((0, 0, -0.1))
    scene.add_geometry(color_mesh(ground, (150, 159, 151, 255)), node_name="ground_512m")

    wall_parts: list[trimesh.Trimesh] = []
    rooftop_parts: list[trimesh.Trimesh] = []
    for mesh_file in sorted(mesh_dir.glob("*.ply")):
        if mesh_file.name == "ground.ply":
            continue
        clipped = keep_cells_inside_window(pv.read(mesh_file), half)
        if clipped is None:
            continue
        mesh = pv_to_trimesh(clipped)
        if mesh is None:
            continue
        if "rooftop" in mesh_file.stem:
            rooftop_parts.append(mesh)
        else:
            wall_parts.append(mesh)

    if wall_parts:
        scene.add_geometry(
            color_mesh(trimesh.util.concatenate(wall_parts), (126, 145, 141, 255)),
            node_name="building_walls",
        )
    if rooftop_parts:
        scene.add_geometry(
            color_mesh(trimesh.util.concatenate(rooftop_parts), (206, 230, 226, 255)),
            node_name="building_rooftops",
        )

    border_points = [
        (-half, -half, 0.35),
        (half, -half, 0.35),
        (half, half, 0.35),
        (-half, half, 0.35),
    ]
    for index, start in enumerate(border_points):
        end = border_points[(index + 1) % len(border_points)]
        scene.add_geometry(
            cylinder_between(start, end, 1.8, (249, 115, 22, 255), sections=18),
            node_name=f"orange_boundary_{index + 1}",
        )

    tx = trimesh.creation.icosphere(subdivisions=3, radius=8.5)
    tx.apply_translation((0, 0, 50))
    scene.add_geometry(color_mesh(tx, (215, 25, 28, 255)), node_name="tx_antenna")

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
    for index, target in enumerate(ray_targets):
        scene.add_geometry(
            cylinder_between((0, 0, 50), target, 0.45, (70, 76, 74, 95), sections=8),
            node_name=f"tx_ray_{index + 1}",
        )

    return scene


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    mesh_dir = project_root / "scenes" / args.scene_name / "mesh"
    output = args.output or project_root / "outputs" / args.scene_name / "models" / "BJTU_Main_512_512m_window.glb"
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = build_scene(mesh_dir)
    scene.export(str(output))
    print(f"Exported {output}")
    print(f"Geometries: {len(scene.geometry)}")


if __name__ == "__main__":
    main()

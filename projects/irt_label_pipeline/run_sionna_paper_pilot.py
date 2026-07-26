#!/usr/bin/env python3
"""Generate Geo2SigMap-paper Sionna radio-map labels for one pilot tile."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import drjit as dr
import matplotlib
import mitsuba as mi
import numpy as np
import sionna.rt as rt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_sionna_scene import build_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override paper samples_per_tx for a smoke/convergence run.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variants: iso,az022,... or 'all'.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def register_geo2sigmap_sector(
    peak_gain_dbi: float,
    horizontal_hpbw_deg: float,
    vertical_hpbw_deg: float,
    attenuation_cap_db: float,
) -> None:
    horizontal_hpbw_rad = math.radians(horizontal_hpbw_deg)
    vertical_hpbw_rad = math.radians(vertical_hpbw_deg)

    def vertical_pattern(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
        wrapped_phi = dr.atan2(dr.sin(phi), dr.cos(phi))
        theta_offset = theta - dr.pi / 2.0
        horizontal_attenuation = dr.minimum(
            12.0 * dr.square(wrapped_phi / horizontal_hpbw_rad),
            attenuation_cap_db,
        )
        vertical_attenuation = dr.minimum(
            12.0 * dr.square(theta_offset / vertical_hpbw_rad),
            attenuation_cap_db,
        )
        total_attenuation = dr.minimum(
            horizontal_attenuation + vertical_attenuation,
            attenuation_cap_db,
        )
        gain_linear = dr.power(
            10.0, (peak_gain_dbi - total_attenuation) / 10.0
        )
        return mi.Complex2f(dr.sqrt(gain_linear), 0.0)

    def pattern_factory(
        *,
        polarization: str,
        polarization_model: str = "tr38901_2",
    ) -> rt.PolarizedAntennaPattern:
        return rt.PolarizedAntennaPattern(
            v_pattern=vertical_pattern,
            polarization=polarization,
            polarization_model=polarization_model,
        )

    try:
        rt.register_antenna_pattern("geo2sigmap_sector", pattern_factory)
    except ValueError as exc:
        if "already" not in str(exc).lower():
            raise


def make_array(pattern: str, polarization: str) -> rt.PlanarArray:
    return rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=pattern,
        polarization=polarization,
        polarization_model="tr38901_2",
    )


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        return np.asarray(tensor.numpy())
    return np.asarray(tensor)


def save_preview(
    output_path: Path,
    path_gain_db: np.ndarray,
    title: str,
) -> None:
    finite = path_gain_db[np.isfinite(path_gain_db)]
    if finite.size:
        vmin = float(np.percentile(finite, 2.0))
        vmax = float(np.percentile(finite, 98.0))
    else:
        vmin, vmax = -160.0, -40.0
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(
        path_gain_db,
        origin="upper",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[0.0, 512.0, 0.0, 512.0],
    )
    axis.scatter([256.0], [256.0], marker="+", s=100, c="red", linewidths=1.5)
    axis.set_title(title)
    axis.set_xlabel("Local east x (m)")
    axis.set_ylabel("Local north y (m)")
    figure.colorbar(image, ax=axis, label="Path gain (dB)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def selected_variants(
    azimuths: list[int],
    requested: str,
) -> list[tuple[str, int | None]]:
    all_variants = [("iso", None)] + [
        (f"az{azimuth:03d}", int(azimuth)) for azimuth in azimuths
    ]
    if requested.strip().lower() == "all":
        return all_variants
    names = {name.strip() for name in requested.split(",") if name.strip()}
    selected = [variant for variant in all_variants if variant[0] in names]
    missing = names - {variant[0] for variant in selected}
    if missing:
        raise ValueError(f"Unknown variants: {sorted(missing)}")
    return selected


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paper = config["paper_parameters"]
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    scene_manifest = build_scene(config_path, overwrite=False)
    samples = int(args.samples or paper["samples_per_tx"])
    if samples <= 0:
        raise ValueError("samples_per_tx must be positive")

    register_geo2sigmap_sector(
        float(paper["directional_boresight_gain_dbi"]),
        float(paper["directional_horizontal_hpbw_deg"]),
        float(paper["directional_vertical_hpbw_deg"]),
        float(paper["directional_attenuation_cap_db"]),
    )

    scene = rt.load_scene()
    building_material = rt.ITURadioMaterial(
        name="pilot-building-concrete",
        itu_type="concrete",
        thickness=float(config["scene"]["building_material_thickness_m"]),
    )
    ground_material = rt.ITURadioMaterial(
        name="pilot-ground-concrete",
        itu_type="concrete",
        thickness=float(config["scene"]["ground_material_thickness_m"]),
    )
    scene.edit(
        add=[
            rt.SceneObject(
                fname=scene_manifest["building_ply"],
                name="buildings",
                radio_material=building_material,
                remove_duplicate_vertices=False,
            ),
            rt.SceneObject(
                fname=scene_manifest["ground_ply"],
                name="ground",
                radio_material=ground_material,
                remove_duplicate_vertices=False,
            ),
        ]
    )
    scene.frequency = float(paper["frequency_hz"])
    scene.rx_array = make_array("iso", str(paper["rx_polarization"]))

    height_map = np.load(config["building_height_map"]).astype(np.float32)
    expected_shape = (int(paper["rows"]), int(paper["cols"]))
    if height_map.shape != expected_shape:
        raise ValueError(
            f"Building map shape {height_map.shape} does not match {expected_shape}"
        )
    outdoor_mask = height_map <= 0.0
    transmitter_height = float(height_map.max()) + 5.0
    tx = rt.Transmitter(
        name="tx",
        position=[
            float(paper["transmitter_xy"][0]),
            float(paper["transmitter_xy"][1]),
            transmitter_height,
        ],
        orientation=[0.0, 0.0, 0.0],
    )
    scene.add(tx)
    solver = rt.RadioMapSolver()

    variants = selected_variants(
        [int(value) for value in paper["directional_azimuths_deg"]],
        args.variants,
    )
    run_started = time.perf_counter()
    summary: dict[str, Any] = {
        "version": 1,
        "status": "running",
        "config": str(config_path),
        "tile": config["tile"],
        "runtime": {
            "python": platform.python_version(),
            "sionna_rt": getattr(rt, "__version__", "unknown"),
            "mitsuba_variant": mi.variant(),
        },
        "parameters": {
            **paper,
            "samples_per_tx": samples,
            "transmitter_height_m": transmitter_height,
        },
        "scene_manifest": scene_manifest,
        "variants": {},
    }
    summary_path = output_root / f"summary_samples_{samples}.json"
    atomic_write_json(summary_path, summary)

    for variant_index, (variant_name, azimuth_deg) in enumerate(variants):
        variant_output = output_root / f"{variant_name}_samples_{samples}.npz"
        preview_output = output_root / f"{variant_name}_samples_{samples}.png"
        if variant_output.exists() and not args.overwrite:
            summary["variants"][variant_name] = {
                "status": "skipped_existing",
                "output": str(variant_output),
            }
            atomic_write_json(summary_path, summary)
            continue

        if variant_name == "iso":
            scene.tx_array = make_array("iso", str(paper["tx_polarization"]))
            tx.orientation = [0.0, 0.0, 0.0]
        else:
            scene.tx_array = make_array(
                "geo2sigmap_sector", str(paper["tx_polarization"])
            )
            tx.orientation = [math.radians(float(azimuth_deg)), 0.0, 0.0]

        variant_started = time.perf_counter()
        print(
            f"START {variant_name} samples={samples} "
            f"max_depth={paper['max_depth']}",
            flush=True,
        )
        radio_map = solver(
            scene=scene,
            center=[
                float(paper["transmitter_xy"][0]),
                float(paper["transmitter_xy"][1]),
                float(paper["receiver_height_m"]),
            ],
            orientation=[0.0, 0.0, 0.0],
            size=[
                float(paper["area_size_m"][0]),
                float(paper["area_size_m"][1]),
            ],
            cell_size=[
                float(paper["cell_size_m"][0]),
                float(paper["cell_size_m"][1]),
            ],
            samples_per_tx=samples,
            max_depth=int(paper["max_depth"]),
            los=bool(paper["los"]),
            specular_reflection=bool(paper["specular_reflection"]),
            diffuse_reflection=bool(paper["diffuse_reflection"]),
            refraction=bool(paper["refraction"]),
            diffraction=bool(paper["diffraction"]),
            edge_diffraction=bool(paper["edge_diffraction"]),
            seed=int(paper["seed"]) + variant_index,
        )
        linear_south_up = tensor_to_numpy(radio_map.path_gain)
        if linear_south_up.ndim == 3:
            linear_south_up = linear_south_up[0]
        if linear_south_up.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected radio-map shape {linear_south_up.shape}; "
                f"expected {expected_shape}"
            )
        linear_north_up = np.flipud(linear_south_up).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            db_north_up = 10.0 * np.log10(linear_north_up)
        finite_hit_mask = np.isfinite(db_north_up) & (linear_north_up > 0.0)
        valid_mask = outdoor_mask & finite_hit_mask
        db_label = db_north_up.astype(np.float32)
        db_label[~valid_mask] = np.nan
        linear_label = linear_north_up.astype(np.float32)
        linear_label[~valid_mask] = np.nan
        elapsed = time.perf_counter() - variant_started

        np.savez_compressed(
            variant_output,
            path_gain_db=db_label,
            path_gain_linear=linear_label,
            valid_mask=valid_mask,
            outdoor_mask=outdoor_mask,
            building_height_m=height_map,
            azimuth_deg=np.asarray(
                np.nan if azimuth_deg is None else azimuth_deg, dtype=np.float32
            ),
            transmitter_xyz_m=np.asarray(
                [
                    paper["transmitter_xy"][0],
                    paper["transmitter_xy"][1],
                    transmitter_height,
                ],
                dtype=np.float32,
            ),
        )
        if bool(config["output"]["save_png"]):
            save_preview(
                preview_output,
                db_label,
                (
                    f"{config['tile']} {variant_name} | "
                    f"Sionna RT {samples:,} rays, depth 8"
                ),
            )
        finite_values = db_label[np.isfinite(db_label)]
        variant_summary = {
            "status": "ok",
            "azimuth_deg": azimuth_deg,
            "seed": int(paper["seed"]) + variant_index,
            "seconds": round(elapsed, 3),
            "shape": list(db_label.shape),
            "valid_outdoor_cells": int(valid_mask.sum()),
            "outdoor_cells": int(outdoor_mask.sum()),
            "coverage_fraction_outdoor": float(
                valid_mask.sum() / max(1, outdoor_mask.sum())
            ),
            "path_gain_db_min": (
                float(finite_values.min()) if finite_values.size else None
            ),
            "path_gain_db_max": (
                float(finite_values.max()) if finite_values.size else None
            ),
            "path_gain_db_mean": (
                float(finite_values.mean()) if finite_values.size else None
            ),
            "output": str(variant_output),
            "preview": str(preview_output),
        }
        summary["variants"][variant_name] = variant_summary
        atomic_write_json(summary_path, summary)
        print(
            f"DONE {variant_name} seconds={elapsed:.3f} "
            f"valid={valid_mask.sum()}/{outdoor_mask.sum()}",
            flush=True,
        )
        del radio_map
        dr.sync_thread()

    summary["status"] = "ok"
    summary["total_seconds"] = round(time.perf_counter() - run_started, 3)
    atomic_write_json(summary_path, summary)
    print(f"SUMMARY {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a Geo2SigMap-style Sionna RT radio map for the BJTU scene."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mitsuba as mi
import sionna
from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("scenes/BJTU_Main_512"))
    parser.add_argument("--output", type=Path, default=Path("outputs/BJTU_Main_512/iso"))
    parser.add_argument("--frequency-hz", type=float, default=3.66e9)
    parser.add_argument("--size-m", type=float, default=512.0)
    parser.add_argument("--cell-size-m", type=float, default=4.0)
    parser.add_argument("--ue-height-m", type=float, default=2.0)
    parser.add_argument("--tx-clearance-m", type=float, default=5.0)
    parser.add_argument("--tx-power-dbm", type=float, default=30.0)
    parser.add_argument(
        "--antenna-pattern",
        choices=("iso", "tr38901"),
        default="iso",
        help="TX pattern: isotropic P_iso or directional 3GPP pattern.",
    )
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        default=0.0,
        help="North-origin clockwise TX azimuth for a directional pattern.",
    )
    parser.add_argument("--samples", type=int, default=7_000_000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def db(array: np.ndarray) -> np.ndarray:
    output = np.full(array.shape, np.nan, dtype=float)
    valid = array > 0.0
    output[valid] = 10.0 * np.log10(array[valid])
    return output


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    height_map_1m = np.load(args.scene_dir / "2D_Building_Height_Map.npy")
    step = int(args.cell_size_m)
    height_map = height_map_1m[::step, ::step]
    tx_height_m = float(np.max(height_map_1m) + args.tx_clearance_m)

    scene = load_scene(str(args.scene_dir / "scene.xml"))
    scene.frequency = args.frequency_hz
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=args.antenna_pattern,
        polarization="VH",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="VH",
    )
    tx_orientation = [0.0, 0.0, 0.0]
    if args.antenna_pattern != "iso":
        # This is the azimuth conversion used in the released Geo2SigMap notebook.
        tx_orientation[0] = float(np.deg2rad(90.0 - args.azimuth_deg))
    scene.add(
        Transmitter(
            name=f"tx_center_{args.antenna_pattern}",
            position=[0.0, 0.0, tx_height_m],
            orientation=tx_orientation,
            power_dbm=args.tx_power_dbm,
        )
    )

    start = time.perf_counter()
    radio_map = RadioMapSolver()(
        scene,
        center=[0.0, 0.0, args.ue_height_m],
        orientation=[0.0, 0.0, 0.0],
        size=[args.size_m, args.size_m],
        cell_size=[args.cell_size_m, args.cell_size_m],
        samples_per_tx=args.samples,
        max_depth=args.max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=True,
        edge_diffraction=False,
        seed=args.seed,
    )
    elapsed_s = time.perf_counter() - start

    path_gain_linear_raw = np.asarray(radio_map.path_gain).squeeze()
    # RadioMap rows run south-to-north; Geo2SigMap's image rows run
    # north-to-south. Align before applying the outdoor building mask.
    path_gain_linear = np.flipud(path_gain_linear_raw)
    path_gain_db = db(path_gain_linear)
    building_mask = height_map > 0
    path_gain_outdoor_db = path_gain_db.copy()
    path_gain_outdoor_db[building_mask] = np.nan
    synthetic_ss_dbm = args.tx_power_dbm + path_gain_outdoor_db

    np.save(args.output / "path_gain_linear_raw.npy", path_gain_linear_raw)
    np.save(args.output / "path_gain_linear_aligned.npy", path_gain_linear)
    np.save(args.output / "path_gain_outdoor_db.npy", path_gain_outdoor_db)
    np.save(args.output / "synthetic_ss_outdoor_dbm.npy", synthetic_ss_dbm)

    metadata = {
        "scene": str(args.scene_dir / "scene.xml"),
        "frequency_hz": args.frequency_hz,
        "map_size_m": args.size_m,
        "cell_size_m": args.cell_size_m,
        "grid_shape": list(path_gain_linear.shape),
        "tx_position_m": [0.0, 0.0, tx_height_m],
        "tx_power_dbm": args.tx_power_dbm,
        "ue_height_m": args.ue_height_m,
        "antenna_pattern": args.antenna_pattern,
        "azimuth_deg_north_clockwise": args.azimuth_deg,
        "sionna_version": sionna.__version__,
        "mitsuba_variant": mi.variant(),
        "max_depth": args.max_depth,
        "samples_per_tx": args.samples,
        "reflection": True,
        "diffraction": True,
        "orientation_alignment": "flipud(raw_radio_map) before building masking",
        "elapsed_seconds": elapsed_s,
        "outdoor_valid_cells": int(np.isfinite(path_gain_outdoor_db).sum()),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    building_image = axes[0].imshow(height_map, cmap="viridis")
    axes[0].set_title("Building height (m)")
    fig.colorbar(building_image, ax=axes[0], shrink=0.85)
    pg_image = axes[1].imshow(path_gain_outdoor_db, cmap="turbo")
    axes[1].set_title("Outdoor path gain (dB)")
    fig.colorbar(pg_image, ax=axes[1], shrink=0.85)
    ss_image = axes[2].imshow(synthetic_ss_dbm, cmap="turbo")
    axes[2].set_title(f"Synthetic SS (dBm), Ptx={args.tx_power_dbm:g} dBm")
    fig.colorbar(ss_image, ax=axes[2], shrink=0.85)
    for ax in axes:
        ax.set_xlabel("East-West cell (4 m)")
        ax.set_ylabel("North-South cell (4 m)")
    fig.suptitle(
        f"Geo2SigMap / Sionna RT - BJTU Main Campus - TX {args.antenna_pattern}"
    )
    fig.savefig(args.output / "bjtu_radio_map.png", dpi=180)
    plt.close(fig)

    print(f"Output: {args.output / 'bjtu_radio_map.png'}")
    print(f"Grid: {path_gain_linear.shape}; TX height: {tx_height_m:.1f} m")
    print(f"Elapsed: {elapsed_s:.2f} s; valid outdoor cells: {metadata['outdoor_valid_cells']}")


if __name__ == "__main__":
    main()

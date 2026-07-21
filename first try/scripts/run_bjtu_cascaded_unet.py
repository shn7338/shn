"""Run Geo2SigMap's released cascaded U-Net weights on the BJTU synthetic map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene_generation.empirical_pathloss_model import pathloss_38901
from scene_generation.unet.unet_model_rt import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("scenes/BJTU_Main_512"))
    parser.add_argument(
        "--radio-dir",
        type=Path,
        default=Path("outputs/BJTU_Main_512/directional_north"),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("models/geo2sigmap_pretrained_weights"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/BJTU_Main_512/cascaded_unet"),
    )
    parser.add_argument("--frequency-ghz", type=float, default=3.66)
    parser.add_argument("--resolution-m", type=int, default=4)
    parser.add_argument("--ue-height-m", type=float, default=2.0)
    parser.add_argument("--tx-clearance-m", type=float, default=5.0)
    parser.add_argument("--sparse-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def run_model(model: UNet, state_path: Path, channels: np.ndarray) -> np.ndarray:
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    tensor = torch.as_tensor(channels[None], dtype=torch.float32)
    with torch.no_grad():
        return model(tensor).squeeze().numpy()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    height_1m = np.load(args.scene_dir / "2D_Building_Height_Map.npy")
    height = height_1m[:: args.resolution_m, :: args.resolution_m].astype(float)
    synthetic_ss = np.load(args.radio_dir / "synthetic_ss_outdoor_dbm.npy")
    outdoor = (height == 0) & np.isfinite(synthetic_ss)
    tx_height_m = float(height_1m.max() + args.tx_clearance_m)

    rows, cols = np.indices(height.shape)
    tx_row, tx_col = np.array(height.shape) // 2
    tx_channel = np.zeros(height.shape, dtype=float)
    tx_channel[tx_row, tx_col] = tx_height_m
    distance_m = np.hypot(
        (rows - tx_row) * args.resolution_m, (cols - tx_col) * args.resolution_m
    )
    distance_m = np.maximum(distance_m, 10.0)
    uma_loss_db, _ = pathloss_38901(
        distance_m, args.frequency_ghz, h_bs=tx_height_m, h_ut=args.ue_height_m
    )
    uma_gain_db = -uma_loss_db

    unet_iso = run_model(
        UNet(n_channels=2, n_classes=1, bilinear=False, pathloss=True),
        args.weights_dir / "1st_checkpoint_epoch58.pth",
        np.stack([height, tx_channel, uma_gain_db]),
    )

    rng = np.random.default_rng(args.seed)
    candidates = np.argwhere(outdoor)
    selected = candidates[
        rng.choice(len(candidates), min(args.sparse_points, len(candidates)), replace=False)
    ]
    sparse_ss = np.full(height.shape, -160.0, dtype=float)
    sparse_ss[selected[:, 0], selected[:, 1]] = synthetic_ss[
        selected[:, 0], selected[:, 1]
    ]

    unet_dir = run_model(
        UNet(n_channels=3, n_classes=1, bilinear=False, pathloss=False),
        args.weights_dir / "2nd_checkpoint_epoch110.pth",
        np.stack([height, sparse_ss, unet_iso]),
    )
    unet_dir_outdoor = unet_dir.copy()
    unet_dir_outdoor[~outdoor] = np.nan

    np.save(args.output / "uma_path_gain_db.npy", uma_gain_db)
    np.save(args.output / "unet_iso_path_gain_db.npy", unet_iso)
    np.save(args.output / "sparse_synthetic_ss_dbm.npy", sparse_ss)
    np.save(args.output / "unet_dir_synthetic_ss_outdoor_dbm.npy", unet_dir_outdoor)
    np.save(args.output / "selected_sparse_cells.npy", selected)

    diagnostic_rmse = float(
        np.sqrt(np.mean((unet_dir_outdoor[outdoor] - synthetic_ss[outdoor]) ** 2))
    )
    metadata = {
        "weights_dir": str(args.weights_dir),
        "frequency_ghz": args.frequency_ghz,
        "resolution_m": args.resolution_m,
        "grid_shape": list(height.shape),
        "tx_height_m": tx_height_m,
        "sparse_points": int(len(selected)),
        "sparse_source": "Sionna synthetic SS, not field measurements",
        "diagnostic_rmse_to_sionna_ss_db": diagnostic_rmse,
        "interpretation": (
            "Demonstration of the released cascaded U-Net pipeline. "
            "This is not a field-calibrated BJTU RSRP prediction."
        ),
        "torch_version": torch.__version__,
        "torch_device": "cpu",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    sparse_plot = sparse_ss.copy()
    sparse_plot[sparse_plot == -160.0] = np.nan
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    maps = (
        (height, "Building height (m)", "viridis"),
        (uma_gain_db, "3GPP UMa path gain (dB)", "turbo"),
        (unet_iso, "U-Net-Iso output (dB)", "turbo"),
        (synthetic_ss, "Sionna synthetic SS (dBm)", "turbo"),
        (sparse_plot, f"Sparse SS input ({len(selected)} points)", "turbo"),
        (unet_dir_outdoor, "U-Net-Dir synthetic prediction (dBm)", "turbo"),
    )
    for ax, (values, title, cmap) in zip(axes.flat, maps):
        image = ax.imshow(values, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("East-West cell (4 m)")
        ax.set_ylabel("North-South cell (4 m)")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Geo2SigMap Cascaded U-Net Demo - Beijing Jiaotong University")
    fig.savefig(args.output / "bjtu_cascaded_unet_map.png", dpi=180)
    plt.close(fig)

    print(f"Output: {args.output / 'bjtu_cascaded_unet_map.png'}")
    print(f"Sparse samples: {len(selected)}; diagnostic RMSE to Sionna SS: {diagnostic_rmse:.2f} dB")


if __name__ == "__main__":
    main()

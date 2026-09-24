"""Benchmark pure GPU inference and end-to-end per-tile prediction latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import building_edge_map
from model import DPMUNet, parameter_count


def load_input(
    root: Path,
    tile: str,
    input_mode: str,
    add_edge: bool,
    edge_radius: int,
) -> np.ndarray:
    tile_dir = root / tile
    all_arrays = {
        "height": np.load(tile_dir / "building_height_norm.npy"),
        "tx": np.load(tile_dir / "tx_position_height_norm.npy"),
        "distance": np.load(tile_dir / "tx_distance_norm.npy"),
    }
    arrays = {
        "height": [all_arrays["height"]],
        "height_tx": [all_arrays["height"], all_arrays["tx"]],
        "height_tx_distance": [all_arrays["height"], all_arrays["tx"], all_arrays["distance"]],
    }[input_mode]
    if add_edge:
        arrays.append(building_edge_map(arrays[0], edge_radius))
    return np.stack(arrays, axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,16,32")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--end-to-end-tiles", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the project benchmark")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = checkpoint.get("args", {})
    add_edge = bool(saved.get("building_edge_channel", False))
    input_mode = saved.get("input_mode", "height_tx_distance")
    edge_radius = int(saved.get("edge_channel_radius_pixels", 1))
    base_input_channels = {"height": 1, "height_tx": 2, "height_tx_distance": 3}[input_mode]
    input_channels = base_input_channels + int(add_edge)
    model = DPMUNet(input_channels, int(saved.get("base_channels", 32))).to(
        device, memory_format=torch.channels_last
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.backends.cudnn.benchmark = True

    test_tiles = [
        line.strip()
        for line in (args.data_root / "test_tiles.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample = load_input(args.data_root, test_tiles[0], input_mode, add_edge, edge_radius)
    batch_results: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for batch_size in [int(item) for item in args.batch_sizes.split(",")]:
            batch = torch.from_numpy(np.repeat(sample[None], batch_size, axis=0)).to(
                device, memory_format=torch.channels_last
            )
            for _ in range(args.warmup):
                with torch.autocast("cuda", dtype=torch.float16):
                    model(batch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                with torch.autocast("cuda", dtype=torch.float16):
                    model(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            batch_ms = elapsed / args.iterations * 1000
            batch_results[str(batch_size)] = {
                "batch_latency_ms": batch_ms,
                "latency_per_tile_ms": batch_ms / batch_size,
                "tiles_per_second": batch_size * args.iterations / elapsed,
            }

        end_to_end_times: list[float] = []
        for tile in test_tiles[: args.end_to_end_tiles]:
            started = time.perf_counter()
            array = load_input(args.data_root, tile, input_mode, add_edge, edge_radius)
            tensor = torch.from_numpy(array[None]).to(device, memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                model(tensor)
            torch.cuda.synchronize()
            end_to_end_times.append((time.perf_counter() - started) * 1000)

    timings = np.asarray(end_to_end_times, dtype=np.float64)
    result = {
        "checkpoint": str(args.checkpoint),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameters": parameter_count(model),
        "input_channels": input_channels,
        "input_mode": input_mode,
        "tile_shape": list(sample.shape[-2:]),
        "pure_gpu": batch_results,
        "end_to_end_batch1": {
            "tiles": len(timings),
            "mean_ms": float(timings.mean()),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "tiles_per_second_from_mean": float(1000 / timings.mean()),
            "includes": "three NPY reads, NumPy stack, CPU-to-GPU copy, and synchronized inference",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

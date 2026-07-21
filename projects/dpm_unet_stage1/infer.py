"""Run Stage-1 DPM inference on one unlabeled normalized tile."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import load_model, predict_tile, resolve_model_files, save_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--mask", type=Path, help="Optional externally generated outdoor-valid mask NPY")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--palette", choices=("color", "gray"), default="color")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint, normalization = resolve_model_files(
        args.model_dir, args.checkpoint, args.normalization
    )
    loaded = load_model(checkpoint, normalization, args.device)
    prediction_norm, prediction_db, metadata = predict_tile(args.tile_dir, loaded, args.mask)
    outputs = save_outputs(
        args.output_dir,
        prediction_norm,
        prediction_db,
        metadata,
        save_png=True,
        palette_name=args.palette,
    )
    print(f"device={loaded.device} input_mode={loaded.input_mode}")
    print(f"inference_ms={metadata['inference_ms']:.3f}")
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

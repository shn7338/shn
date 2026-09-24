"""Render building-overlaid grayscale predictions for every tile in a dataset split."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from inference import (
    load_model,
    predict_tile,
    render_prediction_with_buildings,
    resolve_model_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="Only render the first N tiles (for testing)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_tiles(data_root: Path, split: str, limit: int | None) -> list[str]:
    split_path = data_root / f"{split}_tiles.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split list: {split_path}")
    tiles = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        tiles = tiles[:limit]
    if not tiles:
        raise ValueError(f"No tiles selected from {split_path}")
    return tiles


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ("tile", "status", "inference_ms", "png", "error")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    checkpoint, normalization = resolve_model_files(
        args.model_dir, args.checkpoint, args.normalization
    )
    loaded = load_model(checkpoint, normalization, args.device)
    tiles = read_tiles(args.data_root, args.split, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    print(f"split={args.split} tiles={len(tiles)} device={loaded.device}")

    for index, tile in enumerate(tiles, 1):
        tile_dir = args.data_root / tile
        output_path = args.output_dir / f"{tile}_path_gain_prediction_with_buildings.png"
        if output_path.is_file() and not args.overwrite:
            rows.append({"tile": tile, "status": "skipped", "png": str(output_path)})
        else:
            try:
                _, prediction_db, metadata = predict_tile(tile_dir, loaded)
                render_prediction_with_buildings(prediction_db, tile_dir, output_path)
                rows.append(
                    {
                        "tile": tile,
                        "status": "ok",
                        "inference_ms": round(float(metadata["inference_ms"]), 4),
                        "png": str(output_path),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "tile": tile,
                        "status": "error",
                        "png": str(output_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if index == 1 or index % 100 == 0 or index == len(tiles):
            ok = sum(row["status"] == "ok" for row in rows)
            skipped = sum(row["status"] == "skipped" for row in rows)
            errors = sum(row["status"] == "error" for row in rows)
            print(f"[{index}/{len(tiles)}] ok={ok} skipped={skipped} error={errors}")

    summary_path = args.output_dir / f"{args.split}_visualization_summary.csv"
    write_summary(summary_path, rows)
    elapsed = time.perf_counter() - started
    ok = sum(row["status"] == "ok" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    print(f"complete ok={ok} skipped={skipped} error={errors} elapsed_s={elapsed:.2f}")
    print(f"summary={summary_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

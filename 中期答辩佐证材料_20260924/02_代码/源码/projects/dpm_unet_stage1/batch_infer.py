"""Run Stage-1 DPM inference on many unlabeled normalized tile directories."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from inference import load_model, predict_tile, resolve_model_files, save_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--tiles-file", type=Path, help="Optional text file containing one tile name per line")
    parser.add_argument("--mask-filename", help="Optional mask filename expected inside every tile")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--palette", choices=("color", "gray"), default="color")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tile_names(input_root: Path, tiles_file: Path | None, limit: int | None) -> list[str]:
    if tiles_file is not None:
        names = [
            line.strip()
            for line in tiles_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        names = sorted(path.name for path in input_root.glob("tile_*") if path.is_dir())
    if limit is not None:
        names = names[:limit]
    if not names:
        raise ValueError("No tile directories selected")
    return names


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("tile", "status", "inference_ms", "output_dir", "error")
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
    names = tile_names(args.input_root, args.tiles_file, args.limit)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    print(f"tiles={len(names)} device={loaded.device} input_mode={loaded.input_mode}")

    for index, name in enumerate(names, 1):
        tile_dir = args.input_root / name
        output_dir = args.output_root / name
        db_output = output_dir / "path_gain_pred_db.npy"
        if db_output.is_file() and not args.overwrite:
            rows.append({"tile": name, "status": "skipped", "output_dir": str(output_dir)})
            continue
        mask_path = tile_dir / args.mask_filename if args.mask_filename else None
        try:
            prediction_norm, prediction_db, metadata = predict_tile(tile_dir, loaded, mask_path)
            save_outputs(
                output_dir,
                prediction_norm,
                prediction_db,
                metadata,
                save_png=args.save_png,
                palette_name=args.palette,
            )
            rows.append(
                {
                    "tile": name,
                    "status": "ok",
                    "inference_ms": round(float(metadata["inference_ms"]), 4),
                    "output_dir": str(output_dir),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "tile": name,
                    "status": "error",
                    "output_dir": str(output_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if index == 1 or index % 100 == 0 or index == len(names):
            ok = sum(row["status"] == "ok" for row in rows)
            errors = sum(row["status"] == "error" for row in rows)
            print(f"[{index}/{len(names)}] ok={ok} error={errors}")

    summary_path = args.output_root / "batch_inference_summary.csv"
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

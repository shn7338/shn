#!/usr/bin/env python3
"""Copy verified compact IRT labels between compatible dataset roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from run_winprop_direct_pilot import directional_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main() -> int:
    args = parse_args()
    source_config = load_json(args.source_config)
    target_config = load_json(args.target_config)
    for key in ("grid", "physics", "transmitter", "directional_antenna"):
        if source_config[key] != target_config[key]:
            raise ValueError(f"source and target {key} settings differ")

    source_root = Path(source_config["output_root"])
    target_root = Path(target_config["output_root"])
    target_tiles = {
        line.strip()
        for line in Path(target_config["tiles_file"])
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    }
    source_completion_root = source_root / "completion"
    reused: list[dict[str, Any]] = []
    for source_completion_path in sorted(
        source_completion_root.glob("tile_*.json")
    ):
        tile = source_completion_path.stem
        if tile not in target_tiles:
            continue
        expected_variants = [
            "iso",
            *directional_variants(target_config, tile),
        ]
        source_npz = source_root / "compact_tiles" / f"{tile}.npz"
        source_metadata = source_npz.with_suffix(".json")
        for path in (source_completion_path, source_npz, source_metadata):
            if not path.is_file():
                raise FileNotFoundError(path)
        completion = load_json(source_completion_path)
        metadata = load_json(source_metadata)
        if completion.get("status") != "ok":
            raise RuntimeError(f"source completion is not ok for {tile}")
        if metadata.get("variants") != expected_variants:
            raise ValueError(f"compact variants differ for {tile}")
        if sha256_file(source_npz) != metadata["output_sha256"]:
            raise ValueError(f"source compact hash mismatch for {tile}")
        grid = metadata["grid"]
        if min(
            float(grid["xmin"]),
            float(grid["ymin"]),
            float(grid["xmax"]),
            float(grid["ymax"]),
        ) < 0:
            raise ValueError(f"refusing to reuse negative-coordinate tile {tile}")

        target_npz = target_root / "compact_tiles" / f"{tile}.npz"
        target_metadata = target_npz.with_suffix(".json")
        target_completion = target_root / "completion" / f"{tile}.json"
        for path in (target_npz, target_metadata, target_completion):
            if path.exists():
                raise FileExistsError(path)
        atomic_copy(source_npz, target_npz)
        atomic_copy(source_metadata, target_metadata)
        record = {
            "version": target_config["version"],
            "status": "ok",
            "tile": tile,
            "variants": expected_variants,
            "mode": "verified_dataset_reuse",
            "source_completion": str(source_completion_path),
            "source_compact": str(source_npz),
            "source_compact_sha256": sha256_file(source_npz),
            "target_compact": str(target_npz),
            "target_compact_sha256": sha256_file(target_npz),
        }
        atomic_write_json(target_completion, record)
        reused.append(record)
        print(f"REUSED {tile}", flush=True)

    report = {
        "status": "ok",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "target_tiles": len(target_tiles),
        "reused_tiles": len(reused),
        "tiles": [item["tile"] for item in reused],
    }
    atomic_write_json(target_root / "dataset_reuse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

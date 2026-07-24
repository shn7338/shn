#!/usr/bin/env python3
"""Run a resumable direct-WinProp IRT pilot, one tile at a time."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from compact_irt_result import find_path_loss_text, parse_winprop_ascii


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tile", action="append", default=[])
    parser.add_argument(
        "--tiles-file",
        type=Path,
        help="UTF-8 text file containing one tile name per line.",
    )
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--preprocess-threads", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_progress(
    config: dict[str, Any],
    *,
    status: str,
    phase: str,
    tile: str | None,
    index: int,
    total: int,
    completed: int,
    error: str | None = None,
) -> None:
    payload = {
        "version": config["version"],
        "status": status,
        "phase": phase,
        "tile": tile,
        "index": index,
        "total": total,
        "completed": completed,
        "remaining": total - completed,
        "updated_unix": time.time(),
        "error": error,
    }
    atomic_write_json(Path(config["output_root"]) / "progress.json", payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tile(tile: str) -> None:
    if re.fullmatch(r"tile_\d{6}", tile) is None:
        raise ValueError(f"invalid tile name: {tile!r}")


def select_tiles(config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.tile:
        tiles = args.tile
    elif args.tiles_file is not None or config.get("tiles_file"):
        tiles_file = args.tiles_file or Path(config["tiles_file"])
        if not tiles_file.is_file():
            raise FileNotFoundError(tiles_file)
        tiles = [
            line.strip()
            for line in tiles_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if args.limit is not None:
            tiles = tiles[: args.limit]
    else:
        ready_manifest = Path(config["ready_manifest"])
        with ready_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            tiles = [
                row["Tile"]
                for row in rows
                if row.get("Status", "").strip().upper() == "READY"
            ]
        run_config = config.get("run", config.get("pilot", {}))
        limit = args.limit or int(run_config["tile_count"])
        tiles = tiles[:limit]
    if not tiles:
        raise ValueError("no tiles selected")
    if len(tiles) != len(set(tiles)):
        raise ValueError("selected tile list contains duplicates")
    for tile in tiles:
        validate_tile(tile)
    return tiles


def directional_variants(config: dict[str, Any], tile: str) -> list[str]:
    count = int(config["directional_antenna"]["labels_per_tile"])
    seed = int(config["directional_antenna"]["azimuth_seed"])
    tile_number = int(tile.split("_")[1])
    generator = np.random.default_rng(np.random.SeedSequence([seed, tile_number]))
    azimuths = sorted(
        int(value)
        for value in generator.choice(360, size=count, replace=False).tolist()
    )
    return [f"az{azimuth:03d}" for azimuth in azimuths]


def run_logged(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, (
        stderr_path.open("w", encoding="utf-8", errors="replace")
    ) as stderr:
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def resolve_oib_base(
    config: dict[str, Any],
    config_path: Path,
    tile: str,
    logs_dir: Path,
    preprocess_threads: int,
    timeout_seconds: int,
) -> tuple[Path, dict[str, Any]]:
    reused = config.get("oib_reuse", {}).get(tile)
    if reused:
        base = Path(reused)
        oib = base.with_suffix(".oib")
        if not oib.is_file() or oib.stat().st_size == 0:
            raise FileNotFoundError(f"configured reusable OIB is missing: {oib}")
        return base, {
            "mode": "reused",
            "path": str(oib),
            "bytes": oib.stat().st_size,
            "sha256": sha256_file(oib),
        }

    base_directory = (
        Path(config["output_root"])
        / "oib_keep"
        / tile
    )
    legacy_base = base_directory / f"{tile}_irt3d_r4_h2_v2"
    checked_base = base_directory / f"{tile}_irt3d_r4_h2_v2_checked"
    original_base = base_directory / f"{tile}_irt3d_r4_h2_v2_original"
    for mode, base in (
        ("resume_legacy_checked", legacy_base),
        ("resume_checked", checked_base),
        ("resume_original_fallback", original_base),
    ):
        oib = base.with_suffix(".oib")
        if oib.is_file() and oib.stat().st_size > 0:
            return base, {
                "mode": mode,
                "path": str(oib),
                "bytes": oib.stat().st_size,
                "sha256": sha256_file(oib),
            }

    script = Path(__file__).with_name("preprocess_irt.py")
    source_directory = Path(config["source_root"]) / tile
    attempts = [
        (
            "checked",
            source_directory / f"{tile}_checked.odb",
            checked_base,
        ),
        (
            "original_fallback",
            source_directory / f"{tile}.odb",
            original_base,
        ),
    ]
    failures: list[dict[str, str]] = []
    for mode, source_odb, base in attempts:
        oib = base.with_suffix(".oib")
        if oib.exists():
            failures.append(
                {
                    "mode": mode,
                    "error": f"invalid or partial OIB already exists: {oib}",
                }
            )
            continue
        if not source_odb.is_file():
            failures.append(
                {
                    "mode": mode,
                    "error": f"source ODB is missing: {source_odb}",
                }
            )
            continue
        command = [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--tile",
            tile,
            "--source-odb",
            str(source_odb),
            "--output-base",
            str(base),
            "--threads",
            str(preprocess_threads),
        ]
        try:
            run = run_logged(
                command,
                logs_dir / f"preprocess_{mode}.stdout.log",
                logs_dir / f"preprocess_{mode}.stderr.log",
                timeout_seconds,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            failures.append({"mode": mode, "error": str(error)})
            continue
        if not oib.is_file() or oib.stat().st_size == 0:
            failures.append(
                {
                    "mode": mode,
                    "error": f"preprocessing did not create a valid OIB: {oib}",
                }
            )
            continue
        return base, {
            "mode": f"generated_{mode}",
            "path": str(oib),
            "bytes": oib.stat().st_size,
            "sha256": sha256_file(oib),
            "run": run,
            "previous_failures": failures,
        }
    raise RuntimeError(
        f"all OIB preprocessing attempts failed for {tile}: "
        + json.dumps(failures, ensure_ascii=False)
    )


def prepare_projects(
    config: dict[str, Any],
    config_path: Path,
    tile: str,
    oib_base: Path,
    variants: list[str],
    logs_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    output_root = Path(config["output_root"])
    manifest_path = output_root / "projects" / tile / "tile_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        actual = [item["name"] for item in manifest["variants"]]
        if actual != variants:
            raise RuntimeError(
                f"existing project variants differ for {tile}: "
                f"actual={actual}, planned={variants}"
            )
        for item in manifest["variants"]:
            for key in ("net", "project_base"):
                if not Path(item[key]).with_suffix(".net").is_file():
                    raise FileNotFoundError(
                        f"incomplete existing project for {tile}: {item[key]}"
                    )
        return {"mode": "resume", "manifest": manifest}

    pattern_base = Path(config["directional_antenna"]["pattern_base"])
    for suffix in (".apa", ".apb"):
        pattern_path = pattern_base.with_suffix(suffix)
        if not pattern_path.is_file():
            raise FileNotFoundError(pattern_path)
    script = Path(__file__).with_name("prepare_irt_projects.py")
    command = [
        sys.executable,
        str(script),
        "--config",
        str(config_path),
        "--tile",
        tile,
        "--oib-base",
        str(oib_base),
        "--variants",
        *variants,
        "--pattern-base",
        str(pattern_base),
    ]
    run = run_logged(
        command,
        logs_dir / "prepare.stdout.log",
        logs_dir / "prepare.stderr.log",
        timeout_seconds,
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"project preparation did not create {manifest_path}")
    return {"mode": "generated", "manifest": load_json(manifest_path), "run": run}


def validate_result(
    config: dict[str, Any],
    tile: str,
    result_dir: Path,
) -> dict[str, Any]:
    metadata_path = Path(config["prepared_root"]) / tile / "metadata.json"
    grid = load_json(metadata_path)["grid"]
    result_path = find_path_loss_text(result_dir)
    array, source = parse_winprop_ascii(result_path, grid)
    if not np.count_nonzero(np.isfinite(array)):
        raise RuntimeError(f"WinProp result has no finite pixels: {result_path}")
    return {
        **source,
        "sha256": sha256_file(result_path),
    }


def run_variant(
    config: dict[str, Any],
    item: dict[str, Any],
    tile: str,
    logs_dir: Path,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    variant = item["name"]
    result_dir = Path(item["result_dir"])
    try:
        source = validate_result(config, tile, result_dir)
        return variant, {"mode": "resume", "source": source}
    except FileNotFoundError:
        pass
    net = Path(item["net"])
    cli = Path(config["winprop"]["cli"])
    command = [
        str(cli),
        "-F",
        str(net),
        "-P",
        "--multi-threading",
        str(int(config["winprop"]["multi_threading"])),
        "--disable-project-update",
    ]
    run = run_logged(
        command,
        logs_dir / f"{variant}.stdout.log",
        logs_dir / f"{variant}.stderr.log",
        timeout_seconds,
    )
    source = validate_result(config, tile, result_dir)
    return variant, {"mode": "generated", "run": run, "source": source}


def compact_tile(
    config: dict[str, Any],
    config_path: Path,
    tile: str,
    variants: list[str],
    logs_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    output = Path(config["output_root"]) / "compact_tiles" / f"{tile}.npz"
    metadata_path = output.with_suffix(".json")
    if output.is_file() and metadata_path.is_file():
        metadata = load_json(metadata_path)
        if metadata.get("variants") != variants:
            raise RuntimeError(
                f"existing compact variants differ for {tile}: "
                f"{metadata.get('variants')} != {variants}"
            )
        return {"mode": "resume", "metadata": metadata}
    if output.exists() or metadata_path.exists():
        raise RuntimeError(f"partial compact output blocks safe resume for {tile}")
    script = Path(__file__).with_name("compact_irt_result.py")
    command = [
        sys.executable,
        str(script),
        "--config",
        str(config_path),
        "--tile",
        tile,
        "--variants",
        *variants,
        "--quantity",
        "path_loss",
    ]
    run = run_logged(
        command,
        logs_dir / "compact.stdout.log",
        logs_dir / "compact.stderr.log",
        timeout_seconds,
    )
    if not output.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"compaction did not finish for {tile}")
    return {"mode": "generated", "metadata": load_json(metadata_path), "run": run}


def cleanup_generated_oib(
    config: dict[str, Any],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    if bool(config["storage"].get("retain_oib_for_validation_tiles", True)):
        return {"deleted": False, "reason": "retention enabled by configuration"}
    path = Path(preprocessing["path"]).resolve()
    allowed_root = (
        Path(config["output_root"]).resolve() / "oib_keep"
    )
    try:
        path.relative_to(allowed_root)
    except ValueError:
        return {
            "deleted": False,
            "reason": "OIB is outside the configured generated-oib root",
            "path": str(path),
        }
    if not path.is_file():
        return {
            "deleted": False,
            "reason": "generated OIB is already absent",
            "path": str(path),
        }
    bytes_before = path.stat().st_size
    path.unlink()
    return {
        "deleted": True,
        "reason": "compact tile was verified and OIB retention is disabled",
        "path": str(path),
        "bytes_released": bytes_before,
    }


def process_tile(
    config: dict[str, Any],
    config_path: Path,
    tile: str,
    index: int,
    total: int,
    max_parallel: int,
    preprocess_threads: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    output_root = Path(config["output_root"])
    completion_path = output_root / "completion" / f"{tile}.json"
    if completion_path.is_file():
        completion = load_json(completion_path)
        if completion.get("status") == "ok":
            print(f"[{index}/{total}] SKIP completed {tile}", flush=True)
            return completion

    started = time.perf_counter()
    variants = ["iso", *directional_variants(config, tile)]
    logs_dir = output_root / "logs" / tile
    write_progress(
        config,
        status="running",
        phase="preprocess",
        tile=tile,
        index=index,
        total=total,
        completed=index - 1,
    )
    print(f"[{index}/{total}] PREPROCESS {tile}", flush=True)
    oib_base, preprocessing = resolve_oib_base(
        config,
        config_path,
        tile,
        logs_dir,
        preprocess_threads,
        timeout_seconds,
    )
    write_progress(
        config,
        status="running",
        phase="prepare",
        tile=tile,
        index=index,
        total=total,
        completed=index - 1,
    )
    print(f"[{index}/{total}] PREPARE {tile} {' '.join(variants)}", flush=True)
    preparation = prepare_projects(
        config,
        config_path,
        tile,
        oib_base,
        variants,
        logs_dir,
        timeout_seconds,
    )
    items = preparation["manifest"]["variants"]
    propagation: dict[str, Any] = {}
    write_progress(
        config,
        status="running",
        phase="propagate",
        tile=tile,
        index=index,
        total=total,
        completed=index - 1,
    )
    print(
        f"[{index}/{total}] PROPAGATE {tile} "
        f"({min(max_parallel, len(items))} parallel)",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                run_variant,
                config,
                item,
                tile,
                logs_dir,
                timeout_seconds,
            ): item["name"]
            for item in items
        }
        for future in as_completed(futures):
            variant, result = future.result()
            propagation[variant] = result
            print(f"[{index}/{total}] OK {tile} {variant}", flush=True)
    write_progress(
        config,
        status="running",
        phase="compact",
        tile=tile,
        index=index,
        total=total,
        completed=index - 1,
    )
    print(f"[{index}/{total}] COMPACT {tile}", flush=True)
    compact = compact_tile(
        config,
        config_path,
        tile,
        variants,
        logs_dir,
        timeout_seconds,
    )
    oib_cleanup = cleanup_generated_oib(config, preprocessing)
    completion = {
        "version": config["version"],
        "status": "ok",
        "tile": tile,
        "index": index,
        "total": total,
        "variants": variants,
        "preprocessing": preprocessing,
        "preparation": preparation,
        "propagation": propagation,
        "compact": compact,
        "oib_cleanup": oib_cleanup,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(completion_path, completion)
    write_progress(
        config,
        status="running" if index < total else "completed",
        phase="tile_complete",
        tile=tile,
        index=index,
        total=total,
        completed=index,
    )
    print(
        f"[{index}/{total}] DONE {tile} "
        f"{completion['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return completion


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    tiles = select_tiles(config, args)
    run_config = config.get("run", config.get("pilot", {}))
    max_parallel = args.max_parallel or int(
        run_config["max_parallel_propagations"]
    )
    preprocess_threads = args.preprocess_threads or int(
        run_config["preprocess_threads"]
    )
    timeout_seconds = args.timeout_seconds or int(
        run_config["propagation_timeout_seconds"]
    )
    if min(max_parallel, preprocess_threads, timeout_seconds) < 1:
        raise ValueError("parallelism, thread count and timeout must be positive")
    manifest = {
        "version": config["version"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output_root": config["output_root"],
        "tiles": [
            {
                "tile": tile,
                "variants": ["iso", *directional_variants(config, tile)],
            }
            for tile in tiles
        ],
        "max_parallel_propagations": max_parallel,
        "preprocess_threads": preprocess_threads,
        "timeout_seconds": timeout_seconds,
        "dry_run": args.dry_run,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    output_root = Path(config["output_root"])
    manifest_filename = config.get("run_manifest_filename", "pilot_manifest.json")
    manifest_path = output_root / manifest_filename
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing != manifest:
            raise RuntimeError(
                f"existing pilot manifest differs; refusing to overwrite "
                f"{manifest_path}"
            )
    else:
        atomic_write_json(manifest_path, manifest)
    started = time.perf_counter()
    completions = []
    for index, tile in enumerate(tiles, start=1):
        try:
            completions.append(
                process_tile(
                    config,
                    config_path,
                    tile,
                    index,
                    len(tiles),
                    max_parallel,
                    preprocess_threads,
                    timeout_seconds,
                )
            )
        except Exception as error:
            write_progress(
                config,
                status="failed",
                phase="error",
                tile=tile,
                index=index,
                total=len(tiles),
                completed=index - 1,
                error=f"{type(error).__name__}: {error}",
            )
            raise
    summary = {
        "version": config["version"],
        "status": "ok",
        "tiles": len(tiles),
        "completed": sum(item.get("status") == "ok" for item in completions),
        "elapsed_seconds": time.perf_counter() - started,
        "completion_files": [
            str(output_root / "completion" / f"{tile}.json") for tile in tiles
        ],
    }
    summary_filename = config.get("run_summary_filename", "pilot_summary.json")
    atomic_write_json(output_root / summary_filename, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

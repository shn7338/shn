#!/usr/bin/env python3
"""Generate a resumable multi-tile Sionna radio-map dataset.

Each tile is generated in an isolated child process, validated, compacted to
one float16 NPZ, and marked complete before the next tile is accepted.  A
stopped or failed run can therefore be resumed without regenerating completed
tiles or holding the full dataset in memory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TILE_RUNNER = SCRIPT_DIR / "run_sionna_paper_pilot.py"
TILE_VALIDATOR = SCRIPT_DIR / "validate_sionna_paper_pilot.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the stratified pilot subset configured in pilot_split_counts.",
    )
    parser.add_argument(
        "--tile",
        action="append",
        default=[],
        help="Generate only this tile. May be repeated.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_selection(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tile", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid selection CSV: {path}")
    names = [row["tile"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("selection CSV contains duplicate tiles")
    next_split_index = {"train": 0, "val": 0, "test": 0}
    used_split_indices = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"invalid split for {row['tile']}: {row['split']}")
        split = row["split"]
        value = row.get("split_index", "").strip()
        if value:
            split_index = int(value)
            if split_index < 0:
                raise ValueError(
                    f"negative split_index for {row['tile']}: {split_index}"
                )
        else:
            split_index = next_split_index[split]
            while split_index in used_split_indices[split]:
                split_index += 1
        if split_index in used_split_indices[split]:
            raise ValueError(
                f"duplicate {split} split_index: {split_index}"
            )
        row["split_index"] = str(split_index)
        used_split_indices[split].add(split_index)
        next_split_index[split] = max(
            next_split_index[split], split_index + 1
        )
    return rows


def deterministic_azimuths(config: dict[str, Any], tile: str) -> list[int]:
    seed = int(config["randomness"]["azimuth_seed"])
    count = int(config["radio_map"]["directional_labels_per_tile"])
    tile_number = int(tile.split("_")[1])
    generator = np.random.default_rng(np.random.SeedSequence([seed, tile_number]))
    return sorted(
        int(value)
        for value in generator.choice(360, size=count, replace=False).tolist()
    )


def deterministic_ray_seed(config: dict[str, Any], tile: str) -> int:
    seed = int(config["randomness"]["ray_seed"])
    tile_number = int(tile.split("_")[1])
    state = np.random.SeedSequence([seed, tile_number]).generate_state(1)
    return int(state[0])


def build_manifest(
    config: dict[str, Any],
    config_path: Path,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        azimuths = deterministic_azimuths(config, row["tile"])
        entries.append(
            {
                **row,
                "variants": ["iso"] + [f"az{value:03d}" for value in azimuths],
                "azimuths_deg": azimuths,
                "ray_seed": deterministic_ray_seed(config, row["tile"]),
            }
        )
    signature_payload = {
        "selection_sha256": sha256_file(Path(config["selection_csv"])),
        "source_root": config["source_root"],
        "prepared_root": config["prepared_root"],
        "radio_map": config["radio_map"],
        "scene": config["scene"],
        "randomness": config["randomness"],
        "entries": entries,
    }
    return {
        "version": int(config["version"]),
        "method": config["storage"]["dataset_method"],
        "name": config["name"],
        "config": str(config_path),
        "selection_csv": config["selection_csv"],
        "selection_sha256": signature_payload["selection_sha256"],
        "dataset_signature_sha256": sha256_json(signature_payload),
        "tiles": entries,
    }


def ensure_manifest(output_root: Path, expected: dict[str, Any]) -> Path:
    path = output_root / "run_manifest.json"
    if path.exists():
        existing = load_json(path)
        if (
            existing.get("dataset_signature_sha256")
            != expected["dataset_signature_sha256"]
        ):
            raise RuntimeError(
                "existing run_manifest.json belongs to different physics, "
                "inputs, or azimuths; use a new output_root"
            )
    else:
        atomic_write_json(path, expected)
    return path


def compact_paths(output_root: Path, tile: str) -> tuple[Path, Path, Path]:
    compact = output_root / "compact_tiles" / f"{tile}.npz"
    metadata = compact.with_suffix(".json")
    completion = output_root / "completion" / f"{tile}.json"
    return compact, metadata, completion


def is_complete(output_root: Path, tile: str, verify_hash: bool = False) -> bool:
    compact, metadata_path, completion_path = compact_paths(output_root, tile)
    if not (compact.is_file() and metadata_path.is_file() and completion_path.is_file()):
        return False
    try:
        completion = load_json(completion_path)
        metadata = load_json(metadata_path)
        expected_hash = metadata["output_sha256"]
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    metadata_matches = (
        completion.get("status") == "ok"
        and completion.get("compact_sha256") == expected_hash
        and compact.stat().st_size == int(metadata.get("output_bytes", -1))
    )
    return metadata_matches and (
        not verify_hash or sha256_file(compact) == expected_hash
    )


def select_entries(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    by_name = {entry["tile"]: entry for entry in entries}
    if args.tile:
        missing = [tile for tile in args.tile if tile not in by_name]
        if missing:
            raise ValueError(f"tile(s) not in selection: {missing}")
        selected = [by_name[tile] for tile in args.tile]
    elif args.pilot:
        selected_names: set[str] = set()
        for split in ("train", "val", "test"):
            split_entries = [entry for entry in entries if entry["split"] == split]
            count = int(config["pilot_split_counts"][split])
            if not 0 <= count <= len(split_entries):
                raise ValueError(f"invalid pilot count for {split}: {count}")
            selected_names.update(entry["tile"] for entry in split_entries[:count])
        selected = [entry for entry in entries if entry["tile"] in selected_names]
    else:
        selected = list(entries)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]
    return selected


def tile_config(
    dataset_config: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    tile = entry["tile"]
    source_root = Path(dataset_config["source_root"])
    prepared_root = Path(dataset_config["prepared_root"])
    radio_map = {
        key: value
        for key, value in dataset_config["radio_map"].items()
        if key != "directional_labels_per_tile"
    }
    radio_map["directional_azimuths_deg"] = entry["azimuths_deg"]
    radio_map["seed"] = entry["ray_seed"]
    return {
        "version": int(dataset_config["version"]),
        "name": f"{dataset_config['name']}_{tile}",
        "tile": tile,
        "source_shapefile": str(source_root / tile / f"{tile}.shp"),
        "prepared_metadata": str(prepared_root / tile / "metadata.json"),
        "building_height_map": str(
            prepared_root / tile / "building_height_4m.npy"
        ),
        "output_root": str(output_root / "work_tiles" / tile),
        "environment_python": dataset_config["environment_python"],
        "paper_parameters": radio_map,
        "scene": dataset_config["scene"],
        "output": {
            "array_convention": "array[row,col]; row 0 is north and col 0 is west",
            "quantity": "path_gain_db",
            "indoor_pixels": "NaN",
            "unhit_outdoor_pixels": "NaN",
            "save_linear_path_gain": True,
            "save_png": bool(dataset_config["execution"]["save_png"]),
        },
        "provenance": {
            "dataset": dataset_config["name"],
            "dataset_signature_sha256": dataset_config[
                "dataset_signature_sha256"
            ],
            "runtime_sionna_version": dataset_config["runtime_sionna_version"],
            "frequency_note": "3.5 GHz selected to align the project frequency.",
        },
    }


def validate_and_compact(
    dataset_config: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    tile = entry["tile"]
    samples = int(dataset_config["radio_map"]["samples_per_tx"])
    work_root = output_root / "work_tiles" / tile
    arrays: dict[str, np.ndarray] = {}
    sources: dict[str, Any] = {}
    expected_shape = (
        int(dataset_config["radio_map"]["rows"]),
        int(dataset_config["radio_map"]["cols"]),
    )
    for variant in entry["variants"]:
        path = work_root / f"{variant}_samples_{samples}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            path_gain = data["path_gain_db"].astype(np.float32)
            valid_mask = data["valid_mask"].astype(bool)
            outdoor_mask = data["outdoor_mask"].astype(bool)
        if path_gain.shape != expected_shape:
            raise ValueError(f"{tile} {variant}: invalid shape {path_gain.shape}")
        if np.any(valid_mask & ~outdoor_mask):
            raise ValueError(f"{tile} {variant}: indoor pixels marked valid")
        if not np.array_equal(np.isfinite(path_gain), valid_mask):
            raise ValueError(f"{tile} {variant}: finite mask mismatch")
        if not np.any(valid_mask):
            raise ValueError(f"{tile} {variant}: no valid pixels")
        arrays[variant] = path_gain.astype(np.float16)
        sources[variant] = {
            "source": str(path),
            "source_sha256": sha256_file(path),
            "valid_pixels": int(valid_mask.sum()),
            "minimum_db": float(np.min(path_gain[valid_mask])),
            "maximum_db": float(np.max(path_gain[valid_mask])),
        }
    iso = arrays["iso"].astype(np.float32)
    for variant in entry["variants"][1:]:
        direction = arrays[variant].astype(np.float32)
        common = np.isfinite(iso) & np.isfinite(direction)
        if not np.any(common):
            raise ValueError(f"{tile} {variant}: no common pixels with iso")
        mae = float(np.mean(np.abs(direction[common] - iso[common])))
        if mae < 0.1:
            raise ValueError(f"{tile} {variant}: directional map is not distinct")

    compact, metadata_path, _ = compact_paths(output_root, tile)
    compact.parent.mkdir(parents=True, exist_ok=True)
    if compact.exists():
        compact.unlink()
    temporary = compact.with_suffix(".npz.tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, compact)
    metadata = {
        "version": int(dataset_config["version"]),
        "method": dataset_config["storage"]["dataset_method"],
        "tile": tile,
        "split": entry["split"],
        "array_convention": "array[row, col]; row=0 north, col=0 west",
        "dtype": "float16",
        "quantity": "path_gain_db",
        "variants": entry["variants"],
        "azimuths_deg": entry["azimuths_deg"],
        "sources": sources,
        "output": str(compact),
        "output_bytes": compact.stat().st_size,
        "output_sha256": sha256_file(compact),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def cleanup_work_files(
    dataset_config: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> list[str]:
    execution = dataset_config["execution"]
    work_root = output_root / "work_tiles" / entry["tile"]
    samples = int(dataset_config["radio_map"]["samples_per_tx"])
    removed: list[str] = []
    if not bool(execution["retain_raw_variant_npz"]):
        for variant in entry["variants"]:
            path = work_root / f"{variant}_samples_{samples}.npz"
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    if not bool(execution["save_png"]):
        for variant in entry["variants"]:
            path = work_root / f"{variant}_samples_{samples}.png"
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    if not bool(execution["retain_scene_meshes"]):
        for name in ("buildings.ply", "ground.ply"):
            path = work_root / "scene" / name
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def run_logged(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    env: dict[str, str],
) -> tuple[int, str | None]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, (
            stderr_path.open("w", encoding="utf-8", newline="\n")
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=SCRIPT_DIR,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                env=env,
                check=False,
            )
        return completed.returncode, None
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout_seconds} seconds"


def process_tile(
    dataset_config: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    tile = entry["tile"]
    started = time.perf_counter()
    if is_complete(output_root, tile):
        return {"tile": tile, "status": "skipped_complete", "seconds": 0.0}
    tile_cfg = tile_config(dataset_config, entry, output_root)
    tile_config_path = output_root / "tile_configs" / f"{tile}.json"
    atomic_write_json(tile_config_path, tile_cfg)
    python = str(dataset_config["environment_python"])
    samples = int(dataset_config["radio_map"]["samples_per_tx"])
    timeout_seconds = int(dataset_config["execution"]["tile_timeout_seconds"])
    max_attempts = int(dataset_config["execution"]["max_attempts"])
    logs = output_root / "logs" / tile
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["MPLCONFIGDIR"] = str(output_root / "mplconfig" / tile)
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        run_command = [
            python,
            str(TILE_RUNNER),
            "--config",
            str(tile_config_path),
            "--samples",
            str(samples),
            "--variants",
            "all",
            "--overwrite",
        ]
        returncode, timeout_error = run_logged(
            run_command,
            logs / f"attempt_{attempt}.stdout.log",
            logs / f"attempt_{attempt}.stderr.log",
            timeout_seconds,
            env,
        )
        if timeout_error is not None:
            errors.append(timeout_error)
            continue
        if returncode != 0:
            errors.append(f"generator returned {returncode}")
            continue
        validation_command = [
            python,
            str(TILE_VALIDATOR),
            "--config",
            str(tile_config_path),
            "--samples",
            str(samples),
        ]
        validation_code, validation_timeout = run_logged(
            validation_command,
            logs / f"attempt_{attempt}.validation.stdout.log",
            logs / f"attempt_{attempt}.validation.stderr.log",
            timeout_seconds,
            env,
        )
        if validation_timeout is not None:
            errors.append(f"validation {validation_timeout}")
            continue
        if validation_code != 0:
            errors.append(f"validator returned {validation_code}")
            continue
        try:
            compact_metadata = validate_and_compact(
                dataset_config, entry, output_root
            )
            removed = cleanup_work_files(dataset_config, entry, output_root)
            completion = {
                "version": int(dataset_config["version"]),
                "status": "ok",
                "tile": tile,
                "split": entry["split"],
                "variants": entry["variants"],
                "azimuths_deg": entry["azimuths_deg"],
                "ray_seed": entry["ray_seed"],
                "attempt": attempt,
                "elapsed_seconds": time.perf_counter() - started,
                "compact": compact_metadata["output"],
                "compact_sha256": compact_metadata["output_sha256"],
                "removed_generated_work_files": len(removed),
            }
            _, _, completion_path = compact_paths(output_root, tile)
            atomic_write_json(completion_path, completion)
            failure_path = output_root / "failures" / f"{tile}.json"
            if failure_path.exists():
                failure_path.unlink()
            return completion
        except Exception as exc:  # continue retrying a corrupt/partial tile
            errors.append(f"postprocess failed: {type(exc).__name__}: {exc}")
    failure = {
        "version": int(dataset_config["version"]),
        "status": "failed",
        "tile": tile,
        "split": entry["split"],
        "attempts": max_attempts,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_root / "failures" / f"{tile}.json", failure)
    return failure


def status_report(
    output_root: Path,
    entries: list[dict[str, Any]],
    selected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_entries = selected or entries
    complete_all = {entry["tile"] for entry in entries if is_complete(output_root, entry["tile"])}
    selected_names = {entry["tile"] for entry in selected_entries}
    failed = sorted(
        path.stem
        for path in (output_root / "failures").glob("tile_*.json")
        if path.stem in selected_names
    ) if (output_root / "failures").exists() else []
    split_complete = {
        split: sum(
            entry["tile"] in complete_all
            for entry in entries
            if entry["split"] == split
        )
        for split in ("train", "val", "test")
    }
    return {
        "status": "ok",
        "output_root": str(output_root),
        "tiles_planned": len(entries),
        "tiles_complete": len(complete_all),
        "tiles_remaining": len(entries) - len(complete_all),
        "split_complete": split_complete,
        "selected_tiles": len(selected_entries),
        "selected_complete": len(selected_names & complete_all),
        "selected_failed": len(failed),
        "failed_first": failed[:20],
        "stop_requested": (output_root / "STOP").exists(),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    selection_path = Path(config["selection_csv"])
    rows = read_selection(selection_path)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    expected_manifest = build_manifest(config, config_path, rows)
    ensure_manifest(output_root, expected_manifest)
    config = {
        **config,
        "dataset_signature_sha256": expected_manifest[
            "dataset_signature_sha256"
        ],
    }
    entries = expected_manifest["tiles"]
    selected = select_entries(entries, config, args)

    if args.status:
        print(
            json.dumps(
                status_report(output_root, entries, selected),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not selected:
        raise RuntimeError("no tiles selected")
    workers = args.workers or int(config["execution"]["max_parallel_tiles"])
    hard_limit = int(config["execution"]["max_parallel_tiles_hard_limit"])
    if not 1 <= workers <= hard_limit:
        raise ValueError(f"workers must be in [1, {hard_limit}]")
    if args.dry_run:
        payload = {
            "status": "dry_run",
            "config": str(config_path),
            "output_root": str(output_root),
            "tiles_selected": len(selected),
            "workers": workers,
            "first_tiles": [entry["tile"] for entry in selected[:10]],
            "last_tiles": [entry["tile"] for entry in selected[-10:]],
            "dataset_signature_sha256": config["dataset_signature_sha256"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    stop_path = output_root / "STOP"
    if stop_path.exists():
        raise RuntimeError(f"stop flag exists; remove it before resuming: {stop_path}")

    pending = deque(
        entry for entry in selected if not is_complete(output_root, entry["tile"])
    )
    initial_report = status_report(output_root, entries, selected)
    print(json.dumps(initial_report, ensure_ascii=False), flush=True)
    if not pending:
        return 0

    active: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    interrupted = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        while pending or active:
            while pending and len(active) < workers and not stop_path.exists():
                entry = pending.popleft()
                future = executor.submit(process_tile, config, entry, output_root)
                active[future] = entry
                print(f"START {entry['tile']} split={entry['split']}", flush=True)
            if not active:
                break
            done, _ = concurrent.futures.wait(
                active,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                entry = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # protect the remaining queue
                    result = {
                        "status": "failed",
                        "tile": entry["tile"],
                        "errors": [f"worker crashed: {type(exc).__name__}: {exc}"],
                    }
                    atomic_write_json(
                        output_root / "failures" / f"{entry['tile']}.json",
                        result,
                    )
                results.append(result)
                print(
                    f"DONE {entry['tile']} status={result['status']} "
                    f"seconds={float(result.get('elapsed_seconds', result.get('seconds', 0.0))):.2f}",
                    flush=True,
                )
                progress = status_report(output_root, entries, selected)
                progress["phase"] = "generation"
                progress["last_tile"] = entry["tile"]
                progress["last_result"] = result["status"]
                atomic_write_json(output_root / "progress.json", progress)
            if stop_path.exists() and not active:
                break
    except KeyboardInterrupt:
        interrupted = True
        atomic_write_json(
            output_root / "interrupt.json",
            {
                "status": "interrupted",
                "time_unix": time.time(),
                "message": "Completed tiles are safe; rerun the same command to resume.",
            },
        )
        print("INTERRUPTED: waiting for active tile processes to exit", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    final_report = status_report(output_root, entries, selected)
    final_report["status"] = (
        "interrupted"
        if interrupted
        else "stopped"
        if stop_path.exists()
        else "complete"
        if final_report["selected_complete"] == final_report["selected_tiles"]
        else "completed_with_failures"
    )
    atomic_write_json(output_root / "progress.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)
    return 0 if final_report["status"] in {"complete", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

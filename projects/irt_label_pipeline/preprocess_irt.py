#!/usr/bin/env python3
"""Create a WinProp 2020 urban 3D-IRT OIB without opening WallMan.

The script calls the WinProp API shipped with the local Altair installation.
It is deliberately fail-closed: existing OIB files are never replaced.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


PREDMODEL_IRT = 2
PP_MODE_IRT_3D = 0
PP_MODE_IRT_ZONE_OFF = 0


class CoordPoint(ct.Structure):
    _fields_ = [
        ("x", ct.c_double),
        ("y", ct.c_double),
        ("z", ct.c_double),
    ]


class WinPropPreProUrban(ct.Structure):
    _fields_ = [
        ("Model", ct.c_int),
        ("Mode", ct.c_int),
        ("NrCornersPolygon", ct.c_int),
        ("NrCornersIRTPolygon", ct.c_int),
        ("CornersPolygon", ct.POINTER(CoordPoint)),
        ("CornersIRTPolygon", ct.POINTER(CoordPoint)),
        ("lowerLeft", CoordPoint),
        ("upperRight", CoordPoint),
        ("Resolution", ct.c_double),
        ("Height", ct.c_double),
        ("HeightAbsolute", ct.c_int),
        ("MultipleInteractions", ct.c_int),
        ("IRTPointMode", ct.c_int),
        ("TileVertical", ct.c_double),
        ("TileHorizontal", ct.c_double),
        ("SegmentVertical", ct.c_double),
        ("SegmentHorizontal", ct.c_double),
        ("AdaptiveResolution", ct.c_int),
        ("SphericMode", ct.c_int),
        ("SphericRadiusConstant", ct.c_double),
        ("SphericRadiusConstantHH", ct.c_double),
        ("SphericRadiusBasic", ct.c_double),
        ("SphericRadiusIncrement", ct.c_double),
        ("ConsiderIndoorPixels", ct.c_int),
        ("ConsiderTopography", ct.c_int),
        ("AbsoluteBuildingHeights", ct.c_int),
        ("FilenameBuildings", ct.c_char * 300),
        ("FilenameTopography", ct.c_char * 300),
        ("MultiThreading", ct.c_int),
    ]


if os.name == "nt":
    PercentageCallback = ct.WINFUNCTYPE(ct.c_int, ct.c_int, ct.c_char_p)
    MessageCallback = ct.WINFUNCTYPE(ct.c_int, ct.c_char_p)
    ErrorCallback = ct.WINFUNCTYPE(ct.c_int, ct.c_char_p, ct.c_int)
else:  # pragma: no cover - this pipeline targets WinProp on Windows
    PercentageCallback = ct.CFUNCTYPE(ct.c_int, ct.c_int, ct.c_char_p)
    MessageCallback = ct.CFUNCTYPE(ct.c_int, ct.c_char_p)
    ErrorCallback = ct.CFUNCTYPE(ct.c_int, ct.c_char_p, ct.c_int)


class WinPropCallback(ct.Structure):
    _fields_ = [
        ("Percentage", PercentageCallback),
        ("Message", MessageCallback),
        ("Error", ErrorCallback),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--source-odb", type=Path)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def native_bytes(value: str) -> bytes:
    return value.encode("mbcs", errors="strict")


def decode_native(value: bytes | None) -> str:
    return "" if value is None else value.decode("mbcs", errors="replace")


def validate_tile_name(tile: str) -> None:
    if re.fullmatch(r"tile_\d{6}", tile) is None:
        raise ValueError(f"invalid tile name: {tile!r}")


def tile_metadata(config: dict[str, Any], tile: str) -> tuple[Path, dict[str, Any]]:
    path = Path(config["prepared_root"]) / tile / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"prepared-grid metadata is missing: {path}")
    metadata = load_json(path)
    grid = metadata["grid"]
    expected = config["grid"]
    checks = {
        "width": float(grid["xmax"]) - float(grid["xmin"]),
        "height": float(grid["ymax"]) - float(grid["ymin"]),
        "resolution": float(grid["resolution"]),
        "rows": int(grid["rows"]),
        "cols": int(grid["cols"]),
    }
    expected_values = {
        "width": float(expected["width_m"]),
        "height": float(expected["height_m"]),
        "resolution": float(expected["resolution_m"]),
        "rows": int(expected["rows"]),
        "cols": int(expected["cols"]),
    }
    if checks != expected_values:
        raise ValueError(
            f"{tile} grid does not match formal configuration: "
            f"actual={checks}, expected={expected_values}"
        )
    return path, metadata


def resolve_paths(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[Path, Path, Path]:
    tile = args.tile
    source = args.source_odb or (
        Path(config["source_root"]) / tile / f"{tile}_checked.odb"
    )
    if source.suffix.lower() != ".odb":
        raise ValueError(f"source database must be an .odb file: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"source database is missing: {source}")
    output_base = args.output_base or (
        Path(config["output_root"])
        / "oib_keep"
        / tile
        / f"{tile}_irt3d_r4_h2_v2"
    )
    if output_base.suffix:
        raise ValueError("--output-base must not include a file extension")
    output_oib = output_base.with_suffix(".oib")
    log_path = output_base.with_suffix(".preprocess.log")
    return source.resolve(), output_base.resolve(), log_path.resolve()


def main() -> int:
    args = parse_args()
    validate_tile_name(args.tile)
    config = load_json(args.config)
    metadata_path, metadata = tile_metadata(config, args.tile)
    source_odb, output_base, log_path = resolve_paths(args, config)
    output_oib = output_base.with_suffix(".oib")
    if output_oib.exists():
        raise FileExistsError(f"refusing to replace existing OIB: {output_oib}")

    grid = metadata["grid"]
    pre_cfg = config["preprocessing"]
    threads = args.threads or int(config["winprop"]["multi_threading"])
    if threads < 1:
        raise ValueError("thread count must be positive")

    run_manifest = {
        "version": config["version"],
        "tile": args.tile,
        "source_odb": str(source_odb),
        "source_metadata": str(metadata_path),
        "output_base": str(output_base),
        "output_oib": str(output_oib),
        "grid": grid,
        "preprocessing": pre_cfg,
        "threads": threads,
        "winprop_prepro_struct_size": ct.sizeof(WinPropPreProUrban),
        "status": "dry_run" if args.dry_run else "running",
    }
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    output_base.parent.mkdir(parents=True, exist_ok=True)
    api_bin = Path(config["winprop"]["api_bin"]).resolve()
    engine_path = api_bin / "Engine.dll"
    if not engine_path.is_file():
        raise FileNotFoundError(f"WinProp Engine.dll is missing: {engine_path}")
    os.add_dll_directory(str(api_bin))
    os.environ["PATH"] = str(api_bin) + os.pathsep + os.environ.get("PATH", "")
    engine = ct.WinDLL(str(engine_path))

    init = engine.WinProp_Structure_Init_PreProUrban
    init.argtypes = [ct.POINTER(WinPropPreProUrban)]
    init.restype = None
    compute = engine.OutdoorPlugIn_ComputePrePro
    compute.argtypes = [
        ct.POINTER(WinPropPreProUrban),
        ct.c_char_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(WinPropCallback),
        ct.c_void_p,
    ]
    compute.restype = ct.c_int

    parameter = WinPropPreProUrban()
    init(ct.byref(parameter))
    parameter.Model = PREDMODEL_IRT
    parameter.Mode = PP_MODE_IRT_3D
    parameter.NrCornersPolygon = 0
    parameter.NrCornersIRTPolygon = 0
    parameter.CornersPolygon = None
    parameter.CornersIRTPolygon = None
    parameter.lowerLeft = CoordPoint(
        float(grid["xmin"]), float(grid["ymin"]), float(config["grid"]["receiver_height_m"])
    )
    parameter.upperRight = CoordPoint(
        float(grid["xmax"]), float(grid["ymax"]), float(config["grid"]["receiver_height_m"])
    )
    parameter.Resolution = float(config["grid"]["resolution_m"])
    parameter.Height = float(config["grid"]["receiver_height_m"])
    parameter.HeightAbsolute = 0
    parameter.MultipleInteractions = int(bool(pre_cfg["multiple_interactions"]))
    parameter.IRTPointMode = 0
    parameter.TileVertical = float(pre_cfg["tile_vertical_m"])
    parameter.TileHorizontal = float(pre_cfg["tile_horizontal_m"])
    parameter.SegmentVertical = float(pre_cfg["segment_vertical_m"])
    parameter.SegmentHorizontal = float(pre_cfg["segment_horizontal_m"])
    parameter.AdaptiveResolution = int(pre_cfg["adaptive_resolution"])
    parameter.SphericMode = PP_MODE_IRT_ZONE_OFF
    parameter.ConsiderIndoorPixels = int(bool(pre_cfg["consider_indoor_pixels"]))
    parameter.ConsiderTopography = int(bool(pre_cfg["consider_topography"]))
    parameter.AbsoluteBuildingHeights = 0
    parameter.FilenameBuildings = native_bytes(str(source_odb.with_suffix("")))
    parameter.FilenameTopography = b""
    parameter.MultiThreading = threads

    messages: list[str] = []
    last_progress = {"value": -1}

    def on_percentage(value: int, text: bytes | None) -> int:
        decoded = decode_native(text)
        if value != last_progress["value"]:
            line = f"PROGRESS {value:3d}% {decoded}".rstrip()
            print(line, flush=True)
            messages.append(line)
            last_progress["value"] = value
        return 0

    def on_message(text: bytes | None) -> int:
        line = decode_native(text).strip()
        if line:
            print(line, flush=True)
            messages.append(line)
        return 0

    def on_error(text: bytes | None, code: int) -> int:
        line = f"ERROR {code}: {decode_native(text).strip()}"
        print(line, file=sys.stderr, flush=True)
        messages.append(line)
        return 0

    percentage_callback = PercentageCallback(on_percentage)
    message_callback = MessageCallback(on_message)
    error_callback = ErrorCallback(on_error)
    callbacks = WinPropCallback(
        percentage_callback,
        message_callback,
        error_callback,
    )

    started = time.perf_counter()
    error = compute(
        ct.byref(parameter),
        native_bytes(str(output_base)),
        None,
        None,
        ct.byref(callbacks),
        None,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text("\n".join(messages) + "\n", encoding="utf-8")
    run_manifest.update(
        {
            "status": "ok" if error == 0 else "failed",
            "winprop_error_code": int(error),
            "seconds": elapsed,
            "output_exists": output_oib.is_file(),
            "output_bytes": output_oib.stat().st_size if output_oib.is_file() else 0,
            "log": str(log_path),
        }
    )
    manifest_path = output_base.with_suffix(".preprocess.json")
    atomic_write_json(manifest_path, run_manifest)
    if error != 0:
        raise RuntimeError(f"WinProp preprocessing failed with error code {error}")
    if not output_oib.is_file() or output_oib.stat().st_size == 0:
        raise RuntimeError(f"WinProp reported success but did not create {output_oib}")
    print(
        f"OK {output_oib} ({output_oib.stat().st_size:,} bytes, {elapsed:.1f} s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

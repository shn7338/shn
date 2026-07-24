#!/usr/bin/env python3
"""Generate complete, non-overwriting WinProp IRT projects for one tile."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--oib-base", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["iso", "az000", "az090"],
        help="iso and/or azDDD variants",
    )
    parser.add_argument("--pattern-base", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, content: str, encoding: str = "mbcs") -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding=encoding, newline="")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def replace_required(
    text: str, pattern: str, replacement: str, description: str, count: int = 1
) -> str:
    updated, replacements = re.subn(
        pattern,
        lambda _match: replacement,
        text,
        count=count,
        flags=re.MULTILINE,
    )
    if replacements != count:
        raise ValueError(
            f"expected {count} replacement(s) for {description}, found {replacements}"
        )
    return updated


def dbf_max_numeric(path: Path, field_name: str) -> float:
    data = path.read_bytes()
    if len(data) < 33:
        raise ValueError(f"invalid DBF header: {path}")
    record_count = struct.unpack_from("<I", data, 4)[0]
    header_length = struct.unpack_from("<H", data, 8)[0]
    record_length = struct.unpack_from("<H", data, 10)[0]
    field_offset = 32
    record_offset = 1
    target_start: int | None = None
    target_length: int | None = None
    while field_offset + 32 <= len(data) and data[field_offset] != 0x0D:
        name = data[field_offset : field_offset + 11].split(b"\0", 1)[0].decode(
            "ascii", errors="strict"
        )
        length = int(data[field_offset + 16])
        if name.casefold() == field_name.casefold():
            target_start = record_offset
            target_length = length
        record_offset += length
        field_offset += 32
    if target_start is None or target_length is None:
        raise KeyError(f"DBF field {field_name!r} is missing in {path}")
    maximum = -math.inf
    for index in range(record_count):
        row_start = header_length + index * record_length
        if row_start + record_length > len(data):
            break
        if data[row_start] == ord("*"):
            continue
        raw = data[
            row_start + target_start : row_start + target_start + target_length
        ].decode("ascii", errors="strict").strip()
        if raw:
            maximum = max(maximum, float(raw))
    if not math.isfinite(maximum):
        raise ValueError(f"no numeric values found for {field_name!r} in {path}")
    return maximum


def parse_variant(name: str) -> tuple[str, float | None]:
    if name == "iso":
        return "iso", None
    match = re.fullmatch(r"az(\d{3})", name)
    if match is None:
        raise ValueError(f"invalid variant {name!r}; expected iso or azDDD")
    azimuth = float(int(match.group(1)))
    if not 0 <= azimuth < 360:
        raise ValueError(f"azimuth outside [0, 360): {azimuth}")
    return "directional", azimuth


def directional_block(
    pattern_base: Path, azimuth: float, config: dict[str, Any]
) -> str:
    antenna = config["directional_antenna"]
    project_gain_dbi = float(
        antenna.get(
            "project_antenna_gain_dbi",
            antenna["boresight_gain_dbi"],
        )
    )
    pattern_without_extension = str(pattern_base.with_suffix(""))
    return "\n".join(
        [
            "ANTENNA 1 TYPE SECTOR",
            f'ANTENNA 1 PATTERN_VERTICAL "{pattern_without_extension}"',
            "ANTENNA 1 PATTERN_TYPE 1",
            (
                "ANTENNA 1 ANTENNA_GAIN_VERTICAL "
                f"{project_gain_dbi:.3f}"
            ),
            f"ANTENNA 1 HORIZONTAL {azimuth:.2f} N",
            (
                "ANTENNA 1 VERTICAL "
                f"{float(antenna['mechanical_tilt_in_project_deg']):.2f}"
            ),
            (
                "ANTENNA 1 ELECTRICAL_DOWNTILT "
                f"{float(antenna['project_electrical_downtilt_deg']):.2f}"
            ),
        ]
    )


def configure_net(
    template: str,
    tile: str,
    project_base: Path,
    result_dir: Path,
    checked_odb: Path,
    grid: dict[str, Any],
    tx_height: float,
    config: dict[str, Any],
    variant_kind: str,
    azimuth: float | None,
    pattern_base: Path | None,
) -> str:
    xmin, ymin = float(grid["xmin"]), float(grid["ymin"])
    xmax, ymax = float(grid["xmax"]), float(grid["ymax"])
    center_x, center_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    text = template
    text = replace_required(
        text,
        r'^OUTPUT_PROPAGATION_FILES\s+".*"[ \t\r]*$',
        f'OUTPUT_PROPAGATION_FILES "{result_dir}\\"',
        "OUTPUT_PROPAGATION_FILES",
    )
    text = replace_required(
        text,
        r'^PROJECT_FILE\s+".*"[ \t\r]*$',
        f'PROJECT_FILE "{project_base}"',
        "PROJECT_FILE",
    )
    text = replace_required(
        text,
        r"^COORDINATES_AREA\s+.+$",
        f"COORDINATES_AREA {xmin:.10f} {ymin:.10f} {xmax:.10f} {ymax:.10f}",
        "project coordinates",
    )
    for antenna_id in ("0", "-1"):
        text = replace_required(
            text,
            rf"^ANTENNA {re.escape(antenna_id)} PREDICTION_AREA COORDINATES_AREA\s+.+$",
            (
                f"ANTENNA {antenna_id} PREDICTION_AREA COORDINATES_AREA "
                f"{xmin:.10f} {ymin:.10f} {xmax:.10f} {ymax:.10f}"
            ),
            f"template antenna {antenna_id} prediction area",
        )
    text = replace_required(
        text,
        r"^SITE 1 SITE_LOCATION\s+.+$",
        f"SITE 1 SITE_LOCATION {center_x:.10f} {center_y:.10f} 0.0000000000",
        "site location",
    )
    text = replace_required(
        text,
        r"^ANTENNA 1 POSITION\s+.+$",
        f"ANTENNA 1 POSITION {center_x:.10f}, {center_y:.10f}, {tx_height:.10f} ",
        "transmitter position",
    )
    text = replace_required(
        text,
        r"^ANTENNA 1 POWER\s+.+$",
        f"ANTENNA 1 POWER {float(config['transmitter']['power_dbm']):.5f}",
        "transmitter power",
    )
    text = replace_required(
        text,
        r"^ANTENNA 1 UNIT\s+.+$",
        "ANTENNA 1 UNIT DBM",
        "transmitter power unit",
    )
    text = replace_required(
        text,
        r"^ANTENNA 1 FREQUENCY\s+.+$",
        f"ANTENNA 1 FREQUENCY {float(config['physics']['frequency_mhz']):.2f}",
        "transmitter frequency",
    )
    text = replace_required(
        text,
        r"^ANTENNA 1 POLARIZATION\s+.+$",
        f"ANTENNA 1 POLARIZATION {int(config['transmitter']['polarization'])}",
        "transmitter polarization",
    )
    text = replace_required(
        text,
        r'^CLUTTER_DATABASE_TRAFFIC\s+".*"[ \t\r]*$',
        f'CLUTTER_DATABASE_TRAFFIC "{checked_odb}"',
        "traffic database",
    )
    text = replace_required(
        text,
        r"^ASCII_OUTPUT_PROP\s+.+$",
        "ASCII_OUTPUT_PROP y",
        "ASCII propagation output",
    )
    export_ascii_ray_paths = bool(
        config["physics"].get("export_ascii_ray_paths", False)
    )
    text = replace_required(
        text,
        r"^PROPAGATION_PATHS\s+.+$",
        f"PROPAGATION_PATHS {'y' if export_ascii_ray_paths else 'n'}",
        "ASCII propagation-path output",
    )
    # The binary .ray file duplicates the much easier to audit .str content and
    # is not retained by this pipeline.
    text = replace_required(
        text,
        r"^RAY_PATHS\s+.+$",
        "RAY_PATHS n",
        "binary propagation-path output",
    )
    if variant_kind == "iso":
        text = replace_required(
            text,
            r"^ANTENNA 1 ANTENNA_GAIN\s+.+$",
            "ANTENNA 1 ANTENNA_GAIN 0.000",
            "isotropic gain",
        )
    else:
        if pattern_base is None or azimuth is None:
            raise ValueError("directional variant requires a pattern and azimuth")
        if not pattern_base.with_suffix(".apa").is_file():
            raise FileNotFoundError(
                f"directional APA pattern is missing: {pattern_base.with_suffix('.apa')}"
            )
        block = directional_block(pattern_base, azimuth, config)
        text = replace_required(
            text,
            r"^ANTENNA 1 TYPE ISO\r?\n(?:\*+[^\r\n]*\r?\n)?ANTENNA 1 ANTENNA_GAIN\s+.+$",
            block,
            "isotropic antenna block",
        )
    return text


def configure_nup(
    template: str,
    oib_base: Path,
    project_dir: Path,
    config: dict[str, Any],
) -> str:
    physics = config["physics"]
    ray_number_limit_enabled = bool(
        physics.get("ray_number_limit_enabled", True)
    )
    max_saved_rays_per_pixel = int(
        physics.get("max_saved_rays_per_pixel", 100)
    )
    if max_saved_rays_per_pixel < 1:
        raise ValueError("max_saved_rays_per_pixel must be at least 1")
    # WinProp 2020 treats DATABASE_FILE as project-relative even when an
    # absolute Windows path is supplied. Supplying an absolute path therefore
    # produces "<project-dir>\\E:\\..." and fails while opening the OIB.
    database_file = Path(os.path.relpath(oib_base, start=project_dir))
    text = template
    replacements = [
        (
            r'^DATABASE_FILE\s+".*"[ \t\r]*$',
            f'DATABASE_FILE "{database_file}"',
            "database",
        ),
        (r"^DATABASE_MODE\s+.+$", "DATABASE_MODE 3", "database mode"),
        (r"^PREDICTION_MODEL_ID\s+.+$", "PREDICTION_MODEL_ID 1", "IRT model"),
        (r"^MATERIAL_PROPERTIES\s+.+$", "MATERIAL_PROPERTIES y", "materials"),
        (
            r"^SUPERPOSITION\s+.+$",
            f"SUPERPOSITION {physics['superposition']}",
            "superposition",
        ),
        (
            r"^DIFFRACTION_MODEL\s+.+$",
            f"DIFFRACTION_MODEL {physics['diffraction_model']}",
            "diffraction model",
        ),
        (
            r"^MAX_REFLECTIONS\s+.+$",
            f"MAX_REFLECTIONS {int(physics['max_reflections'])}",
            "maximum reflections",
        ),
        (
            r"^MAX_DIFFRACTIONS\s+.+$",
            f"MAX_DIFFRACTIONS {int(physics['max_diffractions'])}",
            "maximum diffractions",
        ),
        (
            r"^MAX_SCATTERINGS\s+.+$",
            f"MAX_SCATTERINGS {int(physics['max_scatterings'])}",
            "maximum scatterings",
        ),
        (
            r"^MAX_REFLECTIONS_AND_DIFFRACTIONS\s+.+$",
            (
                "MAX_REFLECTIONS_AND_DIFFRACTIONS "
                f"{int(physics['max_reflections_and_diffractions'])}"
            ),
            "maximum combined interactions",
        ),
        (
            r"^RAYS_MAX_NUMBER\s+.+$",
            (
                "RAYS_MAX_NUMBER "
                f"{'y' if ray_number_limit_enabled else 'n'} "
                f"{max_saved_rays_per_pixel}"
            ),
            "ray cap",
        ),
        (
            r"^POST_PROCESSING\s+.+$",
            f"POST_PROCESSING {'y' if physics['post_processing'] else 'n'}",
            "post processing",
        ),
        (
            r"^POST_TRANSITION\s+.+$",
            (
                "POST_TRANSITION "
                f"{'y' if physics['transition_post_processing'] else 'n'}"
            ),
            "transition post processing",
        ),
        (
            r"^KE_POST_OMLY_AREA_IRT\s+.+$",
            f"KE_POST_OMLY_AREA_IRT {'y' if physics['knife_edge_fill'] else 'n'}",
            "knife-edge fill",
        ),
    ]
    for pattern, replacement, description in replacements:
        text = replace_required(text, pattern, replacement, description)
    return text


def main() -> int:
    args = parse_args()
    if re.fullmatch(r"tile_\d{6}", args.tile) is None:
        raise ValueError(f"invalid tile name: {args.tile!r}")
    config = load_json(args.config)
    tile_dir = Path(config["source_root"]) / args.tile
    checked_odb = tile_dir / f"{args.tile}_checked.odb"
    dbf = tile_dir / f"{args.tile}.dbf"
    template_base = tile_dir / (
        f"{args.tile}_{config['winprop']['template_project_suffix']}"
    )
    for path in (
        checked_odb,
        dbf,
        template_base.with_suffix(".net"),
        template_base.with_suffix(".nup"),
        template_base.with_suffix(".wpi"),
        template_base.with_suffix(".mic"),
        args.oib_base.with_suffix(".oib"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata_path = Path(config["prepared_root"]) / args.tile / "metadata.json"
    metadata = load_json(metadata_path)
    grid = metadata["grid"]
    max_height = dbf_max_numeric(dbf, "HEIGHT_M")
    tx_height = max_height + float(
        config["transmitter"]["height_above_max_building_m"]
    )
    output_root = Path(config["output_root"])
    pattern_base = args.pattern_base or (
        output_root / "patterns" / "sector_h65_v8_g6p3dBi_az000"
    )
    template_net = template_base.with_suffix(".net").read_text(encoding="mbcs")
    template_nup = template_base.with_suffix(".nup").read_text(encoding="mbcs")
    template_wpi = template_base.with_suffix(".wpi").read_text(encoding="mbcs")
    template_mic = template_base.with_suffix(".mic").read_text(encoding="mbcs")
    manifest: dict[str, Any] = {
        "version": config["version"],
        "tile": args.tile,
        "oib_base": str(args.oib_base),
        "grid": grid,
        "max_building_height_m": max_height,
        "transmitter_height_m": tx_height,
        "transmitter_xy_m": [
            (float(grid["xmin"]) + float(grid["xmax"])) / 2,
            (float(grid["ymin"]) + float(grid["ymax"])) / 2,
        ],
        "variants": [],
    }
    for variant_name in args.variants:
        kind, azimuth = parse_variant(variant_name)
        project_dir = output_root / "projects" / args.tile / variant_name
        result_dir = output_root / "raw_results" / args.tile / variant_name
        interaction_limit = int(
            config["physics"]["max_reflections_and_diffractions"]
        )
        project_base = (
            project_dir
            / f"{args.tile}_irt{interaction_limit}_{variant_name}"
        )
        target_files = [
            project_base.with_suffix(extension)
            for extension in (".net", ".nup", ".wpi", ".mic")
        ]
        existing = [path for path in target_files if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to replace existing project file(s): "
                + ", ".join(str(path) for path in existing)
            )
        net_text = configure_net(
            template_net,
            args.tile,
            project_base,
            result_dir,
            checked_odb,
            grid,
            tx_height,
            config,
            kind,
            azimuth,
            pattern_base,
        )
        nup_text = configure_nup(
            template_nup,
            args.oib_base,
            project_dir,
            config,
        )
        wpi_text = template_wpi.replace(str(template_base), str(project_base))
        mic_text = template_mic.replace(str(template_base), str(project_base))
        variant_payload = {
            "name": variant_name,
            "kind": kind,
            "azimuth_deg": azimuth,
            "pattern_base": str(pattern_base) if kind == "directional" else None,
            "project_base": str(project_base),
            "net": str(project_base.with_suffix(".net")),
            "result_dir": str(result_dir),
        }
        manifest["variants"].append(variant_payload)
        if not args.dry_run:
            project_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(project_base.with_suffix(".net"), net_text)
            atomic_write(project_base.with_suffix(".nup"), nup_text)
            atomic_write(project_base.with_suffix(".wpi"), wpi_text)
            atomic_write(project_base.with_suffix(".mic"), mic_text)
            atomic_write_json(project_dir / "project_manifest.json", variant_payload)
    if not args.dry_run:
        manifest_path = output_root / "projects" / args.tile / "tile_manifest.json"
        atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

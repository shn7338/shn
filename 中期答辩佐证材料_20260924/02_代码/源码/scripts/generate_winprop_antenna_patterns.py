#!/usr/bin/env python3
"""Generate WinProp 3D antenna-pattern ASCII files (.apa).

The generated sector pattern follows the common 3GPP-style element model:

    A_h(phi)   = min(12 * (delta_phi / HPBW_h)^2, A_max)
    A_v(theta) = min(12 * (delta_theta / HPBW_v)^2, A_max)
    G          = G_max - min(A_h + A_v, A_max)

WinProp APA columns are theta (vertical angle), phi (horizontal angle), and
gain relative to an isotropic radiator in dB.  In WinProp's pattern coordinate
system, the unrotated boresight is theta=90 degrees, phi=0 degrees.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PatternParameters:
    pattern: str
    azimuth_deg: float
    downtilt_deg: float
    horizontal_hpbw_deg: float
    vertical_hpbw_deg: float
    max_gain_dbi: float
    max_attenuation_db: float
    resolution_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate WinProp .apa files for an isotropic or directional "
            "sector antenna, including reproducible random azimuth batches."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "winprop_antenna_patterns",
        help="Destination directory (default: ./winprop_antenna_patterns).",
    )
    parser.add_argument(
        "--pattern",
        choices=("sector", "isotropic"),
        default="sector",
        help="Pattern type (default: sector).",
    )
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        default=0.0,
        help="Sector boresight phi angle in degrees (default: 0).",
    )
    parser.add_argument(
        "--downtilt-deg",
        type=float,
        default=0.0,
        help="Positive electrical downtilt in degrees (default: 0).",
    )
    parser.add_argument(
        "--horizontal-hpbw-deg",
        type=float,
        default=65.0,
        help="Horizontal half-power beamwidth in degrees (default: 65).",
    )
    parser.add_argument(
        "--vertical-hpbw-deg",
        type=float,
        default=8.0,
        help="Vertical half-power beamwidth in degrees (default: 8).",
    )
    parser.add_argument(
        "--max-gain-dbi",
        type=float,
        default=6.3,
        help="Boresight gain relative to isotropic in dBi (default: 6.3).",
    )
    parser.add_argument(
        "--max-attenuation-db",
        type=float,
        default=30.0,
        help="Front-to-back / sidelobe attenuation cap in dB (default: 30).",
    )
    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=1.0,
        help="Theta/phi sampling interval in degrees (default: 1).",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=0,
        help=(
            "Generate this many sector files with distinct random azimuths "
            "on the pattern grid. A CSV manifest is also written."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260721,
        help="Random seed used with --random-count (default: 20260721).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files that already exist.",
    )
    return parser.parse_args()


def normalize_azimuth(angle_deg: float) -> float:
    normalized = angle_deg % 360.0
    return 0.0 if math.isclose(normalized, 360.0, abs_tol=1e-10) else normalized


def angle_grid(stop_deg: float, resolution_deg: float) -> list[float]:
    count = int(round(stop_deg / resolution_deg))
    return [index * resolution_deg for index in range(count + 1)]


def validate_arguments(args: argparse.Namespace) -> None:
    finite_values = {
        "azimuth": args.azimuth_deg,
        "downtilt": args.downtilt_deg,
        "horizontal HPBW": args.horizontal_hpbw_deg,
        "vertical HPBW": args.vertical_hpbw_deg,
        "maximum gain": args.max_gain_dbi,
        "maximum attenuation": args.max_attenuation_db,
        "resolution": args.resolution_deg,
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    if args.horizontal_hpbw_deg <= 0 or args.vertical_hpbw_deg <= 0:
        raise ValueError("HPBW values must be positive")
    if args.max_attenuation_db < 0:
        raise ValueError("maximum attenuation must be non-negative")
    if args.resolution_deg <= 0:
        raise ValueError("resolution must be positive")
    if not math.isclose(180.0 / args.resolution_deg, round(180.0 / args.resolution_deg), abs_tol=1e-9):
        raise ValueError("resolution must divide 180 degrees exactly")
    if not math.isclose(360.0 / args.resolution_deg, round(360.0 / args.resolution_deg), abs_tol=1e-9):
        raise ValueError("resolution must divide 360 degrees exactly")
    if not -90.0 <= args.downtilt_deg <= 90.0:
        raise ValueError("downtilt must be between -90 and +90 degrees")
    if args.random_count < 0:
        raise ValueError("random-count cannot be negative")
    if args.pattern == "isotropic" and args.random_count:
        raise ValueError("random directions have no effect on an isotropic antenna")

    available_azimuths = int(round(360.0 / args.resolution_deg))
    if args.random_count > available_azimuths:
        raise ValueError(
            f"random-count cannot exceed {available_azimuths} distinct azimuth bins "
            f"at {args.resolution_deg:g}-degree resolution"
        )


def wrapped_angle_difference(angle_deg: float, reference_deg: float) -> float:
    return (angle_deg - reference_deg + 180.0) % 360.0 - 180.0


def gain_dbi(theta_deg: float, phi_deg: float, params: PatternParameters) -> float:
    if params.pattern == "isotropic":
        return 0.0

    phi_offset = wrapped_angle_difference(phi_deg, params.azimuth_deg)
    theta_boresight = 90.0 + params.downtilt_deg
    theta_offset = theta_deg - theta_boresight

    horizontal_attenuation = min(
        12.0 * (phi_offset / params.horizontal_hpbw_deg) ** 2,
        params.max_attenuation_db,
    )
    vertical_attenuation = min(
        12.0 * (theta_offset / params.vertical_hpbw_deg) ** 2,
        params.max_attenuation_db,
    )
    total_attenuation = min(
        horizontal_attenuation + vertical_attenuation,
        params.max_attenuation_db,
    )
    return params.max_gain_dbi - total_attenuation


def format_number_for_filename(value: float, width: int = 0) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        integer = int(round(value))
        return f"{integer:0{width}d}" if width else str(integer)
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def default_filename(params: PatternParameters, prefix: str) -> str:
    if params.pattern == "isotropic":
        base = "ISO"
    else:
        horizontal = format_number_for_filename(params.horizontal_hpbw_deg)
        vertical = format_number_for_filename(params.vertical_hpbw_deg)
        gain = format_number_for_filename(params.max_gain_dbi)
        azimuth = format_number_for_filename(params.azimuth_deg, width=3)
        base = f"sector_h{horizontal}_v{vertical}_g{gain}dBi_az{azimuth}"
    return f"{prefix}{base}.apa"


def apa_lines(params: PatternParameters) -> Iterable[str]:
    theta_values = angle_grid(180.0, params.resolution_deg)
    phi_values = angle_grid(360.0, params.resolution_deg)
    yield "# WinProp 3D antenna pattern ASCII (.apa)\n"
    yield "# Theta(deg) Phi(deg) Gain_relative_to_isotropic(dBi)\n"
    if params.pattern == "isotropic":
        yield f"# Pattern=isotropic Resolution={params.resolution_deg:g} deg\n"
    else:
        yield (
            "# Pattern=3GPP-style-sector "
            f"BoresightTheta={90.0 + params.downtilt_deg:g} "
            f"BoresightPhi={params.azimuth_deg:g} "
            f"HPBW_H={params.horizontal_hpbw_deg:g} "
            f"HPBW_V={params.vertical_hpbw_deg:g} "
            f"Gmax={params.max_gain_dbi:g}dBi "
            f"Amax={params.max_attenuation_db:g}dB "
            f"Resolution={params.resolution_deg:g}deg\n"
        )
    for theta_deg in theta_values:
        for phi_deg in phi_values:
            gain = gain_dbi(theta_deg, phi_deg, params)
            yield f"{theta_deg:.4f} {phi_deg:.4f} {gain:.4f}\n"


def atomic_write_lines(path: Path, lines: Iterable[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def validate_apa(path: Path, resolution_deg: float) -> dict[str, float | int]:
    expected_count = (int(round(180.0 / resolution_deg)) + 1) * (
        int(round(360.0 / resolution_deg)) + 1
    )
    count = 0
    minimum_gain = math.inf
    maximum_gain = -math.inf
    seam_zero: dict[float, float] = {}
    seam_360: dict[float, float] = {}

    with path.open("r", encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "*")):
                continue
            parts = line.split()
            if len(parts) not in (3, 4):
                raise ValueError(f"{path}:{line_number}: expected 3 or 4 columns")
            theta_deg, phi_deg, gain = map(float, parts[:3])
            if not (0.0 <= theta_deg <= 180.0 and 0.0 <= phi_deg <= 360.0):
                raise ValueError(f"{path}:{line_number}: angle outside WinProp range")
            if not math.isfinite(gain):
                raise ValueError(f"{path}:{line_number}: gain is not finite")
            count += 1
            minimum_gain = min(minimum_gain, gain)
            maximum_gain = max(maximum_gain, gain)
            if math.isclose(phi_deg, 0.0, abs_tol=1e-9):
                seam_zero[theta_deg] = gain
            elif math.isclose(phi_deg, 360.0, abs_tol=1e-9):
                seam_360[theta_deg] = gain

    if count != expected_count:
        raise ValueError(f"{path}: expected {expected_count} samples, found {count}")
    if seam_zero.keys() != seam_360.keys():
        raise ValueError(f"{path}: incomplete phi=0/360 seam")
    for theta_deg in seam_zero:
        if not math.isclose(seam_zero[theta_deg], seam_360[theta_deg], abs_tol=1e-6):
            raise ValueError(f"{path}: phi seam mismatch at theta={theta_deg:g}")
    return {
        "samples": count,
        "minimum_gain_dbi": minimum_gain,
        "maximum_gain_dbi": maximum_gain,
    }


def random_azimuths(count: int, seed: int, resolution_deg: float) -> list[float]:
    if count == 0:
        return []
    bins = int(round(360.0 / resolution_deg))
    generator = random.Random(seed)
    return sorted(index * resolution_deg for index in generator.sample(range(bins), count))


def write_manifest(
    path: Path,
    generated: list[tuple[Path, PatternParameters]],
    seed: int,
    overwrite: bool,
) -> None:
    rows = [
        [
            "index",
            "filename",
            "azimuth_deg",
            "downtilt_deg",
            "horizontal_hpbw_deg",
            "vertical_hpbw_deg",
            "max_gain_dbi",
            "max_attenuation_db",
            "resolution_deg",
            "seed",
        ]
    ]
    for index, (file_path, params) in enumerate(generated, start=1):
        rows.append(
            [
                index,
                file_path.name,
                f"{params.azimuth_deg:g}",
                f"{params.downtilt_deg:g}",
                f"{params.horizontal_hpbw_deg:g}",
                f"{params.vertical_hpbw_deg:g}",
                f"{params.max_gain_dbi:g}",
                f"{params.max_attenuation_db:g}",
                f"{params.resolution_deg:g}",
                seed,
            ]
        )

    def serialized_rows() -> Iterable[str]:
        for row in rows:
            buffer: list[str] = []
            # The generated fields contain no commas; use csv for correct quoting
            # while retaining the atomic line writer used for APA output.
            from io import StringIO

            stream = StringIO(newline="")
            csv.writer(stream, lineterminator="\n").writerow(row)
            buffer.append(stream.getvalue())
            yield "".join(buffer)

    atomic_write_lines(path, serialized_rows(), overwrite)


def main() -> int:
    args = parse_args()
    validate_arguments(args)
    output_dir = args.output_dir.expanduser().resolve()

    if args.pattern == "isotropic":
        azimuths = [0.0]
    elif args.random_count:
        azimuths = random_azimuths(args.random_count, args.seed, args.resolution_deg)
    else:
        azimuths = [normalize_azimuth(args.azimuth_deg)]

    generated: list[tuple[Path, PatternParameters]] = []
    for azimuth_deg in azimuths:
        params = PatternParameters(
            pattern=args.pattern,
            azimuth_deg=azimuth_deg,
            downtilt_deg=args.downtilt_deg,
            horizontal_hpbw_deg=args.horizontal_hpbw_deg,
            vertical_hpbw_deg=args.vertical_hpbw_deg,
            max_gain_dbi=args.max_gain_dbi,
            max_attenuation_db=args.max_attenuation_db,
            resolution_deg=args.resolution_deg,
        )
        destination = output_dir / default_filename(params, args.prefix)
        atomic_write_lines(destination, apa_lines(params), args.overwrite)
        stats = validate_apa(destination, args.resolution_deg)
        generated.append((destination, params))
        print(
            f"generated: {destination} | azimuth={azimuth_deg:g} deg | "
            f"samples={stats['samples']} | gain="
            f"[{stats['minimum_gain_dbi']:.3f}, {stats['maximum_gain_dbi']:.3f}] dBi"
        )

    if args.random_count:
        manifest = output_dir / f"{args.prefix}directions_seed{args.seed}.csv"
        write_manifest(manifest, generated, args.seed, args.overwrite)
        print(f"manifest:  {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

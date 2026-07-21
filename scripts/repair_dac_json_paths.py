from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def replace_prefix(value: Any, old_prefix: str, new_prefix: str) -> tuple[Any, int]:
    if isinstance(value, dict):
        changed = 0
        result = {}
        for key, item in value.items():
            result[key], item_changed = replace_prefix(item, old_prefix, new_prefix)
            changed += item_changed
        return result, changed
    if isinstance(value, list):
        changed = 0
        result = []
        for item in value:
            fixed, item_changed = replace_prefix(item, old_prefix, new_prefix)
            result.append(fixed)
            changed += item_changed
        return result, changed
    if isinstance(value, str) and value.casefold().startswith(old_prefix.casefold()):
        return new_prefix + value[len(old_prefix) :], 1
    return value, 0


def collect_absolute_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            paths.extend(collect_absolute_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(collect_absolute_paths(item))
    elif isinstance(value, str) and len(value) >= 3 and value[1:3] == ":\\":
        paths.append(value)
    return paths


def validate_metadata(dac_root: Path) -> tuple[int, list[str]]:
    metadata_files = sorted((dac_root / "prepared_4m").glob("tile_*/metadata.json"))
    errors: list[str] = []
    for metadata_path in metadata_files:
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{metadata_path}: invalid JSON: {exc}")
            continue

        for field in ("input_shapefile", "input_winprop_path_loss"):
            referenced_path = metadata.get(field)
            if not isinstance(referenced_path, str) or not os.path.exists(referenced_path):
                errors.append(f"{metadata_path}: missing {field}: {referenced_path!r}")
    return len(metadata_files), errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair stale absolute paths in DAC JSON reports and validate tile metadata."
    )
    parser.add_argument("--dac-root", type=Path, default=Path(r"D:\桌面\dac"))
    args = parser.parse_args()

    dac_root = args.dac_root.resolve()
    report_path = dac_root / "512mdata" / "append_regions_report.json"
    backup_path = report_path.with_name("append_regions_report.before_path_repair.json.bak")
    old_prefix = str(dac_root / "datasets")
    new_prefix = str(dac_root / "raw_datasets")

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    fixed_report, replacement_count = replace_prefix(report, old_prefix, new_prefix)
    if replacement_count:
        missing_after_repair = [
            path for path in collect_absolute_paths(fixed_report) if not os.path.exists(path)
        ]
        if missing_after_repair:
            raise FileNotFoundError(
                "Refusing to write because repaired paths are still missing:\n"
                + "\n".join(missing_after_repair)
            )

        if not backup_path.exists():
            shutil.copy2(report_path, backup_path)

        temporary_path = report_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(fixed_report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, report_path)

    with report_path.open("r", encoding="utf-8") as handle:
        verified_report = json.load(handle)
    report_paths = collect_absolute_paths(verified_report)
    missing_report_paths = [path for path in report_paths if not os.path.exists(path)]

    metadata_count, metadata_errors = validate_metadata(dac_root)
    if missing_report_paths or metadata_errors:
        problems = missing_report_paths + metadata_errors
        raise RuntimeError("Validation failed:\n" + "\n".join(problems[:100]))

    print(f"Repaired report paths: {replacement_count}")
    print(f"Validated report paths: {len(report_paths)}")
    print(f"Validated metadata files: {metadata_count}")
    print(f"Backup: {backup_path if backup_path.exists() else 'not needed'}")


if __name__ == "__main__":
    main()

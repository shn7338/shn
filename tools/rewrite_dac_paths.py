"""Rewrite project source/configuration paths after the 2026-09-24 dac merge.

The script changes active source files only. Historical run manifests, model
outputs, and the original evidence provenance ledger are intentionally kept.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = {name: "01_data" for name in (
    "512mdata", "antenna_patterns", "normalized_4m_v1", "prepared_4m", "raw_datasets"
)}
INVERSION = {name: "02_inversion" for name in (
    "bs_inversion_pilot_v1", "bs_inversion_scale4_v1"
)}
RUNS = {name: "03_runs" for name in (
    "experiments", "inference_demo", "models", "stage1_dpm_unet_runs"
)}
LEGACY = {name: "05_legacy_code" for name in ("code", "github code")}
REFERENCE = {name: "06_reference" for name in ("开题", "参考文献")}
D_GROUPS = DATA | INVERSION | RUNS | LEGACY | REFERENCE


def e_group(name: str) -> str:
    if name == "dac_bs_inversion_scale4_v1_work":
        return "02_inversion/E_work"
    if name == "dac_checkpoints":
        return "07_backup"
    if name == "dac_sionna_rt_env":
        return "08_runtime"
    if name == "dac_tools":
        return "05_legacy_code"
    if name.startswith("dac_irt_"):
        return "04_simulation/irt"
    if name.startswith("dac_sionna_"):
        return "04_simulation/sionna"
    if name.startswith("dac_winprop_"):
        return "04_simulation/winprop"
    if name.startswith("dac_stage"):
        return "03_runs/E_stage"
    raise ValueError(name)


def replace_path_forms(value: str, old: str, new: str) -> str:
    # JSON uses doubled backslashes; Python/PowerShell files and Markdown may
    # use single backslashes or forward slashes.
    for before, after in (
        (old.replace("\\", "\\\\"), new.replace("\\", "\\\\")),
        (old, new),
        (old.replace("\\", "/"), new.replace("\\", "/")),
    ):
        value = value.replace(before, after)
    return value


def rewrite(value: str, e_names: list[str]) -> str:
    d = r"D:\桌面\dac"
    for name, group in D_GROUPS.items():
        value = replace_path_forms(value, f"{d}\\{name}", f"{d}\\{group.replace('/', chr(92))}\\{name}")
    for name in e_names:
        group = e_group(name).replace("/", "\\")
        value = replace_path_forms(value, f"E:\\{name}", f"{d}\\{group}\\{name}")
    value = replace_path_forms(value, r"C:\Users\pc\Documents\大创", d)
    return value


def main() -> None:
    if ROOT.resolve() != Path(r"D:\桌面\dac").resolve():
        raise RuntimeError(f"Unexpected repo root: {ROOT}")
    e_names = sorted(path.name for path in Path("E:/").iterdir() if path.is_dir() and path.name.startswith("dac_"))
    if len(e_names) != 33:
        raise RuntimeError(f"Expected 33 E directories, found {len(e_names)}")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    excluded = {
        "prediction_ready_summary.json",
        "docs/目录迁移记录_20260924.md",
        "tools/rewrite_dac_paths.py",
    }
    files = [ROOT / name for name in tracked if name and name not in excluded and not name.startswith("docs/superpowers/")]
    files.append(ROOT / "tools/build_midterm_evidence.py")
    for legacy_root in (ROOT / "05_legacy_code/code", ROOT / "05_legacy_code/github code"):
        if legacy_root.is_dir():
            files.extend(path for path in legacy_root.rglob("*") if path.is_file() and ".git" not in path.parts)
    suffixes = {".py", ".ps1", ".sh", ".bat", ".cmd", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".ini"}
    changed = []
    for path in sorted(set(files)):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        original = path.read_bytes()
        try:
            before = original.decode("utf-8")
        except UnicodeDecodeError:
            continue
        after = rewrite(before, e_names)
        if after == before:
            continue
        if path.suffix.lower() == ".json":
            json.loads(after)
        path.write_bytes(after.encode("utf-8"))
        changed.append(str(path.relative_to(ROOT)))
    print(f"Changed {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()

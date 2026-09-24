"""Build a compact, traceable midterm-defense evidence folder.

Original datasets and model weights stay at their source paths. Every copied
file is byte-for-byte identical to its source, and the manifest records SHA256.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "中期答辩佐证材料_20260924"
E_NAMES = [
    "dac_bs_inversion_scale4_v1_work", "dac_checkpoints", "dac_irt_v2_8bounce",
    "dac_irt_v2_winprop2020_max6", "dac_sionna_35ghz_depth8_3200_v1",
    "dac_sionna_35ghz_depth8_27360_v1", "dac_sionna_35ghz_depth8_27360_v2",
    "dac_sionna_35ghz_depth8_27360_v3", "dac_sionna_35ghz_pilot",
    "dac_sionna_paper_pilot", "dac_sionna_paper8_v1", "dac_sionna_rt_env",
    "dac_sionna_winprop_irt2_alignment", "dac_stage1_dpm_predictions_positive3200_v2",
    "dac_stage2a_pilot32_smoke_v1", "dac_stage2a_sionna35_depth8_paper_smoke_20260805_205643",
    "dac_stage2b_pilot32_smoke_v1", "dac_stage2b_sionna35_depth8_paper_smoke_20260806_175904",
    "dac_tools", "dac_winprop_irt2_direct_3200_positive_v2",
    "dac_winprop_irt2_direct_3200_v1", "dac_winprop_irt2_direct_pilot32_v1",
    "dac_winprop_irt2_direct_v1", "dac_winprop_irt2_no_ray_cap_diagnostic",
    "dac_winprop_irt2_pattern_gain0_diagnostic", "dac_winprop_irt2_ray_diagnostic_v1",
    "dac_winprop_irt2_ray100_diagnostic_v1", "dac_winprop_irt2_ray500_diagnostic_v1",
    "dac_winprop_irt2_ray500_diagnostic_v2",
    "dac_winprop_irt2_roundtrip_pattern_validation_v3",
    "dac_winprop_irt2_verified_pattern_diagnostic_v1",
    "dac_winprop_irt2_verified_pattern_validation_v1",
    "dac_winprop_irt6_direct_pilot_positive_v3",
]
records: list[dict[str, str | int]] = []
seen_targets: set[str] = set()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, relative: Path, category: str) -> None:
    if not source.is_file():
        return
    key = relative.as_posix()
    if key in seen_targets:
        return
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if target.exists():
        if sha256(target) != source_hash:
            raise RuntimeError(f"Refusing to overwrite changed file: {target}")
    else:
        shutil.copy2(source, target)
    stat = source.stat()
    records.append({
        "类别": category,
        "佐证相对路径": relative.as_posix(),
        "来源绝对路径": str(source),
        "字节数": stat.st_size,
        "来源修改时间": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "SHA256": source_hash,
    })
    seen_targets.add(key)


def copy_if(root: Path, names: list[str], base: Path, category: str) -> None:
    for name in names:
        copy(root / name, base / name, category)


def e_inventory_and_samples() -> list[dict[str, str | int]]:
    overview = []
    ebase = Path("03_E盘实验记录")
    record_exts = {".json", ".csv", ".md", ".txt", ".log", ".xml", ".ini"}
    for name in E_NAMES:
        root = Path("E:/") / name
        count = 0
        size = 0
        extensions: Counter[str] = Counter()
        candidates: list[tuple[int, Path]] = []
        if not root.is_dir():
            overview.append({"目录": str(root), "状态": "不存在", "文件数": 0, "总字节数": 0, "已收录文件数": 0, "类型统计": ""})
            continue
        start = len(records)
        for directory, subdirs, files in os.walk(root):
            for filename in files:
                path = Path(directory) / filename
                try:
                    stat = path.stat()
                except OSError:
                    continue
                count += 1
                size += stat.st_size
                extensions[path.suffix.lower() or "(无后缀)"] += 1
                rel = path.relative_to(root)
                if len(rel.parts) == 1:
                    if (path.suffix.lower() in record_exts and stat.st_size <= 20_000_000
                            and (path.suffix.lower() != ".log" or stat.st_size <= 1_000_000)):
                        copy(path, ebase / name / rel, "E盘根目录记录")
                    if name == "dac_checkpoints" and filename.endswith(".bundle"):
                        copy(path, ebase / name / rel, "Git历史备份")
                    if name == "dac_sionna_rt_env" and filename == "pyvenv.cfg":
                        copy(path, ebase / name / rel, "环境配置")
                    continue
                if name == "dac_sionna_rt_env" or any(part in {".git", "__pycache__"} for part in rel.parts):
                    continue
                if name == "dac_tools" and path.suffix.lower() in {".py", ".md", ".txt"} and stat.st_size <= 200_000:
                    copy(path, Path("02_代码") / "E盘工具源码" / rel, "E盘工具源码（需核对授权）")
                if path.suffix.lower() not in record_exts | {".npz", ".npy"} or stat.st_size > 2_000_000:
                    continue
                priority = 0
                lower = filename.lower()
                if "manifest" in lower or "validation" in lower or "comparison" in lower:
                    priority += 20
                if "summary" in lower or "report" in lower or "metric" in lower:
                    priority += 15
                if "tile_000001" in str(rel).lower() or "tile_002865" in str(rel).lower():
                    priority += 10
                if path.suffix.lower() == ".npz":
                    priority += 4
                if "compact_tiles" in str(rel).lower():
                    priority += 8
                if "smoke" in str(rel).lower():
                    priority -= 2
                candidates.append((priority, path))
        # A compact, deterministic set of nested originals from every run.
        candidates.sort(key=lambda item: (-item[0], str(item[1]).lower()))
        picked = 0
        npz_picked = False
        for _, path in candidates:
            if picked >= 8:
                break
            if path.suffix.lower() in {".npz", ".npy"}:
                if npz_picked:
                    continue
                npz_picked = True
            copy(path, ebase / name / path.relative_to(root), "E盘嵌套记录样本")
            picked += 1
        compact = root / "compact_tiles"
        if compact.is_dir():
            example = next(iter(sorted(compact.glob("*.json"))), None)
            if example is not None:
                copy(example, ebase / name / "compact_tiles" / example.name, "E盘compact tile元数据样本")
                matching_npz = example.with_suffix(".npz")
                if matching_npz.exists():
                    copy(matching_npz, ebase / name / "compact_tiles" / matching_npz.name, "E盘compact tile数据样本")
        if name == "dac_stage1_dpm_predictions_positive3200_v2":
            first_tile = next(iter(sorted(root.glob("tile_*"))), None)
            if first_tile is not None:
                for array in sorted(first_tile.glob("*.npy"))[:2]:
                    copy(array, ebase / name / first_tile.name / array.name, "Stage1预测数组样本")
        overview.append({
            "目录": str(root), "状态": "空目录" if count == 0 else "存在",
            "文件数": count, "总字节数": size, "已收录文件数": len(records) - start,
            "类型统计": "; ".join(f"{ext}:{n}" for ext, n in extensions.most_common(8)),
        })
        print(f"E: {name}: {count} files, {size / 2**30:.2f} GiB, {len(records)-start} copied", flush=True)
    return overview


def d_records() -> None:
    phone = Path("01_数据记录/01_校园手机试采")
    copy(Path("D:/桌面/signal-2026-08-28.csv"), phone / "signal-2026-08-28.csv", "手机试采原始CSV")
    wx = Path("D:/微信/存储文件/xwechat_files/wxid_6gfzrt1x2m1m22_6183/msg/file/2026-09")
    copy_if(wx, ["signal-2026-09-01.csv", "Cellular-Z 20260831 224742.683 SLOT1.CSV"], phone, "手机试采原始CSV")
    copy(REPO / "prediction_ready_summary.json", Path("01_数据记录/02_WinProp与预处理/prediction_ready_summary.json"), "数据集汇总")
    d = Path("D:/桌面/dac")
    win = Path("01_数据记录/02_WinProp与预处理")
    copy(d / "512mdata/tile_000001/tile_000001_result1/Site  1 Antenna 1 Path Loss.txt", win / "原始样例/Site  1 Antenna 1 Path Loss.txt", "WinProp原始结果样本")
    for name in ["metadata.json", "building_height_4m.npy", "path_gain_4m.npy"]:
        copy(d / "prepared_4m/tile_000001" / name, win / "预处理样例" / name, "预处理样本")
    for name in ["normalization_summary.csv", "normalization.json", "splits.csv", "tx_feature_summary.csv", "train_tiles.txt", "val_tiles.txt", "test_tiles.txt"]:
        copy(d / "normalized_4m_v1" / name, win / "标准化与划分" / name, "预处理记录")
    for name in ["bs_inversion_pilot_v1", "bs_inversion_scale4_v1"]:
        src = d / name
        base = Path("01_数据记录/04_随机基站数据") / name
        copy_if(src, ["FINAL_REPORT.json", "VALIDATION_REPORT.json", "DATASET_README.md", "normalization.json", "site_index.csv", "run_manifest.json"], base, "基站反演数据集记录")
    sample = d / "bs_inversion_scale4_v1/samples/test"
    for suffix in ["json", "npz"]:
        copy(sample / f"tile_000011_site00.{suffix}", Path("01_数据记录/04_随机基站数据/样本") / f"tile_000011_site00.{suffix}", "随机基站测试样本")
    model_dirs = [
        "stage1_dpm_unet_runs/baseline_v1",
        "models/stage2a_iso_refine_sionna35_depth8_paper_v1",
        "models/stage2b_directional_ss_sionna35_depth8_paper_v1",
        "models/bs_parameter_estimator_scale4_v2",
        "models/stage2b_position_encoding_scale4_v3",
        "models/stage2b_latent_fusion_scale4_v4",
        "models/stage2b_multiscale_adapter_scale4_v5",
        "models/joint_estimator_stage2b_scale4_v6",
    ]
    for item in model_dirs:
        src = d / item
        base = Path("01_数据记录/05_训练评估记录") / Path(item).name
        copy_if(src, ["run_config.json", "history.csv", "test_metrics.json", "benchmark.json", "training_summary.json", "REPORT.md"], base, "模型训练评估记录")
    copy(d / "stage1_results.md", Path("01_数据记录/05_训练评估记录/stage1_results.md"), "第一阶段报告")
    experiments = d / "experiments"
    for path in experiments.glob("*.json"):
        if any(tag in path.name for tag in ("stage2b_v3_test_bins", "latent_fusion_v4_test_bins", "multiscale_adapter_v5_test_bins", "joint_estimator_stage2b_v6_test_bins", "latent_fusion_v4_gate", "multiscale_adapter_v5_gate", "joint_estimator_stage2b_v6_gate")):
            copy(path, Path("01_数据记录/05_训练评估记录/版本对比") / path.name, "V3-V6评估或门槛记录")


def code_snapshot() -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=REPO).decode("utf-8").splitlines()
    prefixes = ("projects/", "scripts/", "tools/", "bjtu_osm_buildings/", "first try/scripts/")
    selected = [name for name in files if name.startswith(prefixes) and Path(name).suffix.lower() in {".py", ".ps1", ".sh", ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini", ".csv"}]
    archive = OUT / "02_代码/源码快照.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in selected:
                data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
                zf.writestr(name, data)
    with zipfile.ZipFile(archive) as zf:
        if sorted(zf.namelist()) != sorted(selected):
            raise RuntimeError("Existing source archive does not match committed file list")
        for name in selected:
            target = OUT / "02_代码/源码" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(name)
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError(f"Refusing to overwrite changed code snapshot: {target}")
            else:
                target.write_bytes(data)
            records.append({"类别": "Git基线源码", "佐证相对路径": target.relative_to(OUT).as_posix(),
                            "来源绝对路径": f"git:{commit}:{name}", "字节数": len(data),
                            "来源修改时间": "Git提交快照", "SHA256": hashlib.sha256(data).hexdigest()})
    records.append({"类别": "Git基线源码压缩包", "佐证相对路径": archive.relative_to(OUT).as_posix(),
                    "来源绝对路径": f"git:{commit}", "字节数": archive.stat().st_size,
                    "来源修改时间": "Git提交快照", "SHA256": sha256(archive)})
    return commit


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8-sig") != content:
        raise RuntimeError(f"Refusing to overwrite changed document: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d_records()
    e_overview = e_inventory_and_samples()
    commit = code_snapshot()
    copy(Path(__file__), Path("02_代码/佐证材料构建脚本.py"), "佐证材料构建脚本")
    with (OUT / "来源与校验清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["类别", "佐证相对路径", "来源绝对路径", "字节数", "来源修改时间", "SHA256"])
        writer.writeheader(); writer.writerows(records)
    with (OUT / "03_E盘实验记录/E盘目录总览.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["目录", "状态", "文件数", "总字节数", "已收录文件数", "类型统计"])
        writer.writeheader(); writer.writerows(e_overview)
    write_new(OUT / "00_先看这里.md", f"""# 中期答辩佐证材料索引

整理日期：2026-09-24。代码基线：`{commit}`。

## 从哪里看

1. `01_数据记录/01_校园手机试采`：三份原始手机 CSV，属于前期试采；测试条件和时间的已知范围见同目录说明。
2. `01_数据记录/02_WinProp与预处理`：原始仿真结果、预处理样本、标准化及划分记录。
3. `01_数据记录/03_Sionna合成数据`：各轮进度说明；原始进度、清单和 tile 样本收录在 `03_E盘实验记录`。
4. `01_数据记录/04_随机基站数据`：pilot 和 scale4 的最终/验证报告、清单及测试样本。
5. `01_数据记录/05_训练评估记录`：Stage1、Stage2、参数估计及 V3–V6 的配置、历史、结果和门槛记录。
6. `02_代码/源码` 可直接浏览；`源码快照.zip` 便于整体交付。`02_代码/E盘工具源码` 是 E 盘工具文件，使用/分发前核对其原有授权。
7. `03_E盘实验记录`：E 盘列出的 33 个目录逐一建档，实验阶段见 `过程总览.md`；全量体量及原路径见 `E盘目录总览.csv`。

`来源与校验清单.csv` 记录每份实际收录文件的来源、原修改时间及 SHA256。大规模仿真数据、权重和 Python 环境仍在原路径，没有当作小型佐证包复制。目录总览中的“已收录文件数”用于核对：空目录为 0；非空目录有原始记录或样本。移动原数据后，清单中的绝对路径需更新。

本目录收录的是已有材料，不等于所有实验均达到目标。V5 是当前计划展示的版本；其原始测试 RMSE 为 6.897 dB，但没有通过预设的严格升级门槛。手机试采不是模型正式实测验证。
""")
    write_new(OUT / "01_数据记录/01_校园手机试采/数据说明.md", """# 校园手机信号试采

- `signal-2026-08-28.csv`：105 条 NR-SA 原始记录，8 个小区组；最大一组 73 条是记录行数，不能表述为 73 个不同空间点。文件无逐行时间戳。
- `signal-2026-09-01.csv`：37 条 LTE 记录，5 个小区组；文件无逐行时间戳。
- `Cellular-Z 20260831 224742.683 SLOT1.CSV`：155 条数据，其中 154 条 LTE、1 条 NONE；NR 字段为空。文件名含 2026-08-31 22:47:42.683。

这些是前期手机试采。设备、SIM、软件设置、每条记录的地点及测试条件没有完整原始说明，不能补造；文件系统修改时间见校验清单，也不能当成逐行采样时间。正式现场测试与模型真实数据验证仍需另备明确条件和时间的记录。
""")
    write_new(OUT / "01_数据记录/03_Sionna合成数据/数据说明.md", """# Sionna 生成数据与试验轮次

原始记录和代表性 tile 在 `../../03_E盘实验记录/` 下按 E 盘原目录名保存；全量数据仍在 E 盘。每个收录文件的来源和 SHA256 见根目录的 `来源与校验清单.csv`。

| 目录 | `progress.json` 记录 | 答辩口径 |
| --- | --- | --- |
| `dac_sionna_35ghz_depth8_3200_v1` | `status=complete`，计划 3200，完成 3 | 属于早期小规模/试运行记录，不能仅凭状态字段称已生成 3200 个 tile。 |
| `dac_sionna_35ghz_depth8_27360_v1` | `completed_with_failures`，计划 27360，完成 31 | 留存失败轮次。 |
| `dac_sionna_35ghz_depth8_27360_v2` | `stopped`，计划 27360，完成 169 | 留存中止轮次。 |
| `dac_sionna_35ghz_depth8_27360_v3` | `complete`，计划/完成均为 27360 | 当前完整合成数据轮次；有 `run_manifest.json`、`shard_manifest.csv`、`shard_metadata.json` 和 compact tile 样本。 |

这里列的是各轮 `progress.json` 的记录，不把试验轮次合并成一次成功。其他 Sionna pilot、paper 和 WinProp 对齐记录也逐目录列在 `03_E盘实验记录/E盘目录总览.csv`。
""")
    write_new(OUT / "03_E盘实验记录/过程总览.md", """# E 盘实验过程总览

33 个用户指定目录均已检查；逐目录文件数、体量、收录数和原路径见 `E盘目录总览.csv`。`dac_bs_inversion_scale4_v1_work` 当前为空目录，仍保留登记。

- `dac_checkpoints`：历史 Git bundle 和恢复记录。
- `dac_irt_v2_8bounce`、`dac_irt_v2_winprop2020_max6`：早期 IRT/WinProp 项目、日志及结果样本。
- `dac_sionna_*`：Sionna pilot、paper、3200 和 27360 轮次；`27360_v1` 留有失败记录，`v2` 停止，`v3` 的进度清单显示 27360/27360 完成。详情见 `../01_数据记录/03_Sionna合成数据/数据说明.md`。
- `dac_stage1_dpm_predictions_positive3200_v2`：第一阶段预测清单及数组样本。
- `dac_stage2a_*`、`dac_stage2b_*`：小规模 smoke 训练的配置、历史和测试指标；大权重在原目录。
- `dac_winprop_irt2_direct_*`、`dac_winprop_irt2_*_diagnostic*`、`dac_winprop_irt2_*_validation*`：直接仿真、诊断和验证的多轮记录，包含停止或效果不佳的尝试，不能统称为通过验证。
- `dac_winprop_irt6_direct_pilot_positive_v3`：IRT6 pilot 清单、日志和项目样本。
- `dac_tools`：工具源码和相关说明；`dac_sionna_rt_env`：Python 环境，仅收录根配置，环境本体仍在 E 盘。

这里的每个目录均按原名保留，便于从答辩材料追溯到原实验。压缩包和源码快照位于 `../02_代码`。
""")
    write_new(OUT / "02_代码/代码索引.md", f"""# 代码索引

基线提交：`{commit}`。`源码` 与 `源码快照.zip` 由该提交生成；可按路径直接打开。

- `projects/irt_label_pipeline`：Sionna/WinProp 标签生成与评估。
- `projects/dpm_unet_stage1`：第一阶段 DPM U-Net 训练及推断。
- `scripts`、`tools`：数据准备、标准化、实验、诊断与辅助工具。
- `bjtu_osm_buildings`、`first try/scripts`：早期场景构建及试验代码。
- `E盘工具源码`：E 盘 `dac_tools` 中的另存工具源码；来源和校验见总清单，注意其中可能含外部项目代码。

模型权重未打包；本机原位置见训练配置与 E 盘目录总览。运行前按各项目 README 和配置准备依赖及数据路径。
""")
    print(f"Built {OUT} with {len(records)} copied files, source commit {commit}", flush=True)


if __name__ == "__main__":
    main()

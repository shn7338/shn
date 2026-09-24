"""Regression coverage for Pilot constants leaking into Scale-4 documentation."""

import json
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))

from run_bs_inversion_pilot import render_dataset_readme


def test_readme_uses_manifest_counts_and_configured_sampling():
    entries = [
        {"tile": "tile_a", "split": "train", "azimuths_deg": [0, 90, 180, 270]},
        {"tile": "tile_a", "split": "train", "azimuths_deg": [1, 91, 181, 271]},
        {"tile": "tile_b", "split": "test", "azimuths_deg": [2, 92, 182, 272]},
    ]
    normalization = {"directional_signal_strength_db": {"mean": -60.5, "std": 20.5}}
    for name, sampling in [
        ("pilot", "四象限"),
        ("scale4", "径向区间"),
    ]:
        config = json.loads(
            (PIPELINE / f"config_bs_inversion_{name}_v1.json").read_text(encoding="utf-8")
        )
        text = render_dataset_readme(config, entries, normalization)
        assert config["name"] in text
        assert "建筑场景（瓦片）：2（train/val/test = 1/0/1）" in text
        assert "基站站点：3（train/val/test = 2/0/1）" in text
        assert "传播图：15" in text
        assert "方向训练样本：12（train/val/test = 8/0/4）" in text
        assert sampling in text
        assert "(signal_db - (-60.500000000000)) / 20.500000000000" in text
        assert "显式覆盖数据集默认值" in text
        assert "建筑：384" not in text

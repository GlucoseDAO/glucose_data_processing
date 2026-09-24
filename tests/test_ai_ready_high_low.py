#!/usr/bin/env python3
"""
AI-READI Dexcom records whose glucose value is the string "High" or "Low" must map to the
same bounds the Dexcom CSV converter uses, not be dropped.

Runs against a locally downloaded AI-READI release; skipped when none is present.
"""

import json
import os
import zipfile
from pathlib import Path

import pytest

from formats.ai_ready.ai_ready_database_converter import AIReadyDatabaseConverter, _AIReadyZipLayout
from formats.glucose_bounds import dexcom_style_bounds

# An extracted AI-READI release (the folder holding participants.tsv). Point
# AI_READI_DATASET_DIR at it, or place/symlink it at DATA/ai_ready_dataset.
AI_READI_DATASET = Path(
    os.environ.get("AI_READI_DATASET_DIR", str(Path(__file__).parent.parent / "DATA" / "ai_ready_dataset"))
)
DEX_GLOB = "wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/*/*_DEX.json"
CONFIG = {"dexcom": {"high_glucose_value": 401, "low_glucose_value": 39}}


def _first_participant_with_high_and_low() -> Path | None:
    for dex in sorted(AI_READI_DATASET.glob(DEX_GLOB)):
        text = dex.read_text(encoding="utf-8")
        if '"value": "High"' in text and '"value": "Low"' in text:
            return dex
    return None


@pytest.fixture(scope="module")
def dex_json() -> Path:
    if not AI_READI_DATASET.exists():
        pytest.skip(f"AI-READI dataset not available at {AI_READI_DATASET}")
    dex = _first_participant_with_high_and_low()
    if dex is None:
        pytest.skip("No AI-READI participant with both High and Low readings")
    return dex


def test_high_low_mapped_to_dexcom_bounds(tmp_path: Path, dex_json: Path) -> None:
    user_id = dex_json.parent.name
    records = json.loads(dex_json.read_text(encoding="utf-8"))["body"]["cgm"]
    values = [r.get("blood_glucose", {}).get("value") for r in records]
    n_high = sum(v == "High" for v in values)
    n_low = sum(v == "Low" for v in values)
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    low, high = dexcom_style_bounds(CONFIG)

    zip_path = tmp_path / "ai_readi.zip"
    member = _AIReadyZipLayout(dataset_root="dataset/").dexcom_cgm_json(user_id)
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(dex_json, member)

    converter = AIReadyDatabaseConverter(CONFIG, database_type="ai_ready")
    with zipfile.ZipFile(zip_path) as z:
        df = converter._extract_cgm_df(z, _AIReadyZipLayout(dataset_root="dataset/"), user_id)

    assert df is not None
    glucose = df["glucose_value_mgdl"]
    assert df.height == len(numeric) + n_high + n_low
    assert (glucose == high).sum() == n_high + sum(v == high for v in numeric)
    assert (glucose == low).sum() == n_low + sum(v == low for v in numeric)
    assert glucose.min() >= low and glucose.max() <= high

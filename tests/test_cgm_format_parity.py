#!/usr/bin/env python3
"""
Parity between native converters and the cgm_format backend where both read the input.

Each known difference is asserted in its exact form rather than tolerated, so a new
divergence fails here (docs/CGM_FORMAT_PARITY.md, docs/DECISIONS.md D3):

- D1NAMO glucose: cgm_format converts mmol/L with 18.0182, native with 18.0.
- Dexcom: two EGV readings at one wall-clock time (a DST fold) keep different readings.
- BIG IDEAs: only cgm_format reads the food log, so carbs exist only on that side.
"""

import os
from pathlib import Path

import polars as pl
import pytest
from cgm_format import FormatParser
from cgm_format.formats.dexcom_eu import MMOL_TO_MGDL

from formats.cgm_format_input.parity import CorpusParity, compare_corpus
from formats.glucose_bounds import MGDL_PER_MMOL

REPO_ROOT = Path(__file__).parent.parent
DEXCOM_SMALL = REPO_ROOT / "test_data" / "dexcom_small"
D1NAMO_DIABETES = REPO_ROOT / "test_data" / "d1namo_small" / "diabetes"
# The PhysioNet BIG IDEAs release (the folder holding 001..016). Point BIGIDEAS_DIR at it,
# or place/symlink it at DATA/bigideas.
BIGIDEAS = Path(os.environ.get("BIGIDEAS_DIR", str(REPO_ROOT / "DATA" / "bigideas")))
CONFIG = {"dexcom": {"high_glucose_value": 401, "low_glucose_value": 39, "remove_calibration": True}}


def _assert_same_users_and_timestamps(parity: CorpusParity) -> None:
    assert parity.native_users == parity.adapter_users
    for user in parity.users:
        assert (user.only_native, user.only_adapter) == (0, 0), user.user_id


def _assert_sums_equal(parity: CorpusParity, fields: tuple[str, ...]) -> None:
    for user in parity.users:
        for name in fields:
            native, adapter = user.sums[name]
            assert native == pytest.approx(adapter), (user.user_id, name)


def test_dexcom_export_parity() -> None:
    parity = compare_corpus(DEXCOM_SMALL, CONFIG)
    _assert_same_users_and_timestamps(parity)
    _assert_sums_equal(parity, ("fast_acting_insulin_u", "long_acting_insulin_u", "carb_grams"))

    timestamp = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    source = pl.concat([pl.read_csv(f, infer_schema=False) for f in sorted(DEXCOM_SMALL.glob("*.csv"))])
    distinct_egv = source.filter(pl.col("Event Type") == "EGV").unique([timestamp, "Glucose Value (mg/dL)"])
    ambiguous = set(
        distinct_egv.group_by(timestamp).len().filter(pl.col("len") > 1)[timestamp]
        .str.to_datetime("%Y-%m-%dT%H:%M:%S").to_list()
    )
    for user in parity.users:
        # Glucose may differ only where the source holds two different readings at one time.
        assert set(user.glucose_mismatches) <= ambiguous


def test_d1namo_corpus_parity() -> None:
    parity = compare_corpus(D1NAMO_DIABETES, CONFIG)
    _assert_same_users_and_timestamps(parity)
    _assert_sums_equal(parity, ("fast_acting_insulin_u", "long_acting_insulin_u"))
    for user in parity.users:
        assert user.glucose_ratio == pytest.approx(MMOL_TO_MGDL / MGDL_PER_MMOL, rel=1e-5)


@pytest.fixture(scope="module")
def bigideas() -> Path:
    if not (BIGIDEAS / "Demographics.csv").exists():
        pytest.skip(f"BIG IDEAs not available at {BIGIDEAS}")
    return BIGIDEAS


def test_bigideas_corpus_parity(bigideas: Path) -> None:
    parity = compare_corpus(bigideas, CONFIG)
    _assert_same_users_and_timestamps(parity)
    corpus = FormatParser.parse_corpus(bigideas)
    for user in parity.users:
        assert user.glucose_mismatches == ()
        native_carbs, adapter_carbs = user.sums["carb_grams"]
        # Native ignores Food_Log_*.csv; the backend's carbs are exactly cgm_format's.
        assert native_carbs is None
        assert adapter_carbs == pytest.approx(corpus[user.user_id]["carbs"].sum())

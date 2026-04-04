"""Loop converter glucose trimming (Dexcom-style bounds)."""

from __future__ import annotations

import polars as pl
import pytest

from formats.loop.loop_database_converter import LoopDatabaseConverter


@pytest.fixture
def trim_config() -> dict:
    return {"dexcom": {"high_glucose_value": 401, "low_glucose_value": 39}}


def test_apply_glucose_trim_clips_numeric_egv(trim_config: dict) -> None:
    conv = LoopDatabaseConverter(trim_config)
    df = pl.DataFrame(
        {
            "event_type": ["EGV", "EGV", "EGV", "Basal"],
            "glucose_value_mgdl": [700.0, 0.0, 100.0, None],
            "user_id": ["a", "a", "a", "a"],
        }
    )
    out = conv._apply_glucose_trim_to_frame(df)
    vals = out["glucose_value_mgdl"].to_list()
    assert vals[0] == 401.0
    assert vals[1] == 39.0
    assert vals[2] == 100.0
    assert vals[3] is None


def test_apply_glucose_trim_unparseable_egv_becomes_null(trim_config: dict) -> None:
    """Loop uses numeric glucose; stray text parses as null and stays null after clip."""
    conv = LoopDatabaseConverter(trim_config)
    df = pl.DataFrame(
        {
            "event_type": ["EGV"],
            "glucose_value_mgdl": ["not-a-number"],
        }
    )
    out = conv._apply_glucose_trim_to_frame(df)
    assert out["glucose_value_mgdl"].to_list() == [None]


def test_glucose_bounds_helper_respects_config() -> None:
    from formats.glucose_bounds import dexcom_style_bounds

    lo, hi = dexcom_style_bounds({"dexcom": {"low_glucose_value": 50, "high_glucose_value": 420}})
    assert lo == 50.0
    assert hi == 420.0

#!/usr/bin/env python3
"""
Smoke test for Loop dataset processing.

Verifies that:
- CGM and BGM glucose values are correctly converted from mmol/L to mg/dL
- UTCDtTm raw column does not appear in the output
- Output contains physiologically plausible glucose values (39-450 mg/dL)
- All expected columns are present
"""

import pytest
import polars as pl
from pathlib import Path

from formats.loop.loop_database_converter import LoopDatabaseConverter

LOOP_DATA_PATH = Path("DATA/loop")

pytestmark = pytest.mark.skipif(
    not LOOP_DATA_PATH.exists() or not (LOOP_DATA_PATH / "Data Tables").exists(),
    reason="Loop data folder not found"
)


@pytest.fixture(scope="module")
def first_user_df() -> pl.DataFrame:
    """Consolidate data for the first Loop user only."""
    config = {"first_n_users": 1}
    conv = LoopDatabaseConverter(config)
    frames = list(conv.iter_user_event_frames(LOOP_DATA_PATH, interval_minutes=5))
    assert frames, "No user frames produced from Loop data"
    return frames[0]


class TestLoopGlucoseUnits:
    """Verify CGM/BGM values are converted from mmol/L to mg/dL."""

    def test_glucose_column_present(self, first_user_df: pl.DataFrame) -> None:
        assert "glucose_value_mgdl" in first_user_df.columns

    def test_glucose_values_in_mgdl_range(self, first_user_df: pl.DataFrame) -> None:
        """Values should be in mg/dL range (39–450), not mmol/L range (2–25)."""
        egv_rows = first_user_df.filter(pl.col("event_type").is_in(["EGV", "BGM"]))
        vals = egv_rows["glucose_value_mgdl"].drop_nulls().cast(pl.Float64, strict=False).drop_nulls()
        assert len(vals) > 0, "No EGV/BGM glucose values found"
        assert vals.min() >= 30.0, f"Glucose min too low: {vals.min()} — likely still in mmol/L"
        assert vals.max() <= 500.0, f"Glucose max too high: {vals.max()}"
        assert vals.mean() > 70.0, f"Mean glucose too low: {vals.mean()} — likely still in mmol/L"

    def test_utcdttm_not_in_output(self, first_user_df: pl.DataFrame) -> None:
        """Raw UTCDtTm string column must not appear in the output schema."""
        assert "UTCDtTm" not in first_user_df.columns, (
            "UTCDtTm raw column leaked into output — it should be dropped after timestamp parsing"
        )


class TestLoopOutputSchema:
    """Verify output schema completeness."""

    def test_required_columns_present(self, first_user_df: pl.DataFrame) -> None:
        required = {"timestamp", "event_type", "user_id", "glucose_value_mgdl"}
        missing = required - set(first_user_df.columns)
        assert not missing, f"Missing required columns: {missing}"

    def test_timestamp_is_datetime(self, first_user_df: pl.DataFrame) -> None:
        assert first_user_df["timestamp"].dtype == pl.Datetime

    def test_multiple_event_types(self, first_user_df: pl.DataFrame) -> None:
        event_types = set(first_user_df["event_type"].unique().to_list())
        assert "EGV" in event_types, "Should have CGM (EGV) events"

    def test_no_duplicate_timestamps_per_event(self, first_user_df: pl.DataFrame) -> None:
        dups = (
            first_user_df
            .group_by(["user_id", "timestamp", "event_type"])
            .len()
            .filter(pl.col("len") > 1)
        )
        assert len(dups) == 0, f"Found {len(dups)} duplicate (user, timestamp, event_type) groups"

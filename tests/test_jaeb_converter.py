#!/usr/bin/env python3
"""
JAEB comma-separated device tables (``PtID,DeviceDtDaysFromEnroll,DeviceTm,RecordType,Value``).

``test_data/jaeb_small`` is an excerpt of three participants from the Shah healthy
non-diabetic cohort (JAEB public dataset CGMND). Expected values are read from it at runtime.
"""

from pathlib import Path

import polars as pl
import pytest

from formats.database_detector import DatabaseDetector
from formats.jaeb.jaeb_database_converter import JaebDatabaseConverter
from formats.minidose1.minidose1_converter import Minidose1Converter

JAEB_SMALL = Path(__file__).parent.parent / "test_data" / "jaeb_small"


@pytest.fixture(scope="module")
def source_cgm() -> pl.DataFrame:
    return pl.read_csv(JAEB_SMALL / "NonDiabDeviceCGM.csv", infer_schema=False)


@pytest.fixture(scope="module")
def frames() -> list[pl.DataFrame]:
    converter = JaebDatabaseConverter({}, database_type="jaeb")
    return list(converter.iter_user_event_frames(JAEB_SMALL, interval_minutes=5))


def test_detected_as_jaeb() -> None:
    assert DatabaseDetector().detect_database_type(JAEB_SMALL) == "jaeb"


def test_one_frame_per_participant(frames: list[pl.DataFrame], source_cgm: pl.DataFrame) -> None:
    expected = sorted(source_cgm["PtID"].unique().to_list())
    assert [f["user_id"].unique().to_list() for f in frames] == [[pt] for pt in expected]


def test_only_cgm_rows_survive(frames: list[pl.DataFrame], source_cgm: pl.DataFrame) -> None:
    for frame in frames:
        assert frame["event_type"].unique().to_list() == ["EGV"]
        pt = frame["user_id"][0]
        src = source_cgm.filter((pl.col("PtID") == pt) & (pl.col("RecordType") == "CGM"))
        # One output row per distinct device timestamp; duplicates keep the first value.
        src = src.unique(["DeviceDtDaysFromEnroll", "DeviceTm"], keep="first", maintain_order=True)
        assert frame.height == src.height
        assert sorted(frame["glucose_value_mgdl"].to_list()) == sorted(src["Value"].cast(pl.Float64).to_list())


def test_timestamps_follow_enrollment_offset(frames: list[pl.DataFrame], source_cgm: pl.DataFrame) -> None:
    reference = Minidose1Converter.REFERENCE_ENROLLMENT_DATE
    for frame in frames:
        pt = frame["user_id"][0]
        days = source_cgm.filter((pl.col("PtID") == pt) & (pl.col("RecordType") == "CGM"))[
            "DeviceDtDaysFromEnroll"
        ].cast(pl.Int64)
        span_days = (frame["timestamp"].max() - frame["timestamp"].min()).days
        assert span_days <= days.max() - days.min() + 1
        assert frame["timestamp"].min().date().toordinal() - reference.date().toordinal() == days.min()

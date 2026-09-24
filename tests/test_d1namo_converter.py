#!/usr/bin/env python3
"""
D1NAMO converter (``glucose.csv`` in mmol/L, ``insulin.csv``; one folder per subject).

``test_data/d1namo_small`` holds subjects 001 and 002 of the diabetes subset and subject 004
of the healthy subset, copied unchanged. Expected values are read from them at runtime.
"""

from pathlib import Path

import polars as pl
import pytest

from formats.d1namo.d1namo_database_converter import D1namoDatabaseConverter
from formats.database_detector import DatabaseDetector
from formats.glucose_bounds import MGDL_PER_MMOL
from glucose_ml_preprocessor import GlucoseMLPreprocessor

D1NAMO_SMALL = Path(__file__).parent.parent / "test_data" / "d1namo_small"
DIABETES = D1NAMO_SMALL / "diabetes"
HEALTHY = D1NAMO_SMALL / "healthy"


def _read_source(subject: Path, name: str) -> pl.DataFrame:
    df = pl.read_csv(subject / name, infer_schema=False)
    time = pl.col("time")
    return df.with_columns(
        (pl.col("date") + "T" + pl.when(time.str.count_matches(":") == 1).then(time + ":00").otherwise(time))
        .str.to_datetime("%Y-%m-%dT%H:%M:%S")
        .alias("ts")
    )


@pytest.fixture(scope="module")
def frames() -> dict[str, pl.DataFrame]:
    converter = D1namoDatabaseConverter({}, database_type="d1namo")
    return {f["user_id"][0]: f for f in converter.iter_user_event_frames(DIABETES, interval_minutes=5)}


@pytest.mark.parametrize("root", [DIABETES, HEALTHY])
def test_detected_as_d1namo(root: Path) -> None:
    assert DatabaseDetector().detect_database_type(root) == "d1namo"


def test_one_user_per_subject_folder(frames: dict[str, pl.DataFrame]) -> None:
    assert list(frames) == sorted(p.name for p in DIABETES.iterdir() if p.is_dir())


def test_cgm_glucose_converted_to_mgdl(frames: dict[str, pl.DataFrame]) -> None:
    for user_id, frame in frames.items():
        src = _read_source(DIABETES / user_id, "glucose.csv").filter(pl.col("type") == "cgm")
        src = src.unique("ts", keep="first", maintain_order=True)
        got = frame.filter(pl.col("glucose_value_mgdl").is_not_null())
        assert set(got["timestamp"].to_list()) == set(src["ts"].to_list())
        expected = sorted(v * MGDL_PER_MMOL for v in src["glucose"].cast(pl.Float64).to_list())
        assert sorted(got["glucose_value_mgdl"].to_list()) == pytest.approx(expected)


def test_fingersticks_dropped_insulin_kept(frames: dict[str, pl.DataFrame]) -> None:
    for user_id, frame in frames.items():
        glucose = _read_source(DIABETES / user_id, "glucose.csv")
        fingerstick_only = set(glucose.filter(pl.col("type") != "cgm")["ts"].to_list()) - set(
            glucose.filter(pl.col("type") == "cgm")["ts"].to_list()
        )
        insulin = _read_source(DIABETES / user_id, "insulin.csv").unique("ts", keep="first")
        # A fingerstick timestamp may survive only as an insulin row, never with glucose.
        leaked = frame.filter(pl.col("timestamp").is_in(list(fingerstick_only)) & pl.col("glucose_value_mgdl").is_not_null())
        assert leaked.height == 0
        for src_col, out_col in [("fast_insulin", "fast_acting_insulin_u"), ("slow_insulin", "long_acting_insulin_u")]:
            assert frame[out_col].sum() == pytest.approx(insulin[src_col].cast(pl.Float64).sum())


def test_fingerstick_only_subset_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no rows left after d1namo-specific filtering"):
        GlucoseMLPreprocessor().process(HEALTHY, tmp_path / "out.csv", database_type="d1namo")

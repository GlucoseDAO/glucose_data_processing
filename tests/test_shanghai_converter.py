#!/usr/bin/env python3
"""
ShanghaiT1DM/T2DM workbooks (``<patient>_<period>_<YYYYMMDD>.xls[x]``).

``test_data/shanghai_small`` holds, unchanged: patient 1002 (three recording periods, .xls),
patient 1001 (.xlsx), patient 2045 (T2DM, CGM column labelled ``CGM `` without a unit) and
the T1DM summary workbook, which is not a patient sheet.
Expected values are read from the workbooks at runtime.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest

from formats.database_detector import DatabaseDetector
from formats.shanghai.shanghai_database_converter import (
    SHANGHAI_WORKBOOK_PATTERN,
    ShanghaiDatabaseConverter,
    find_shanghai_workbooks,
    shanghai_cgm_column,
)

SHANGHAI_SMALL = Path(__file__).parent.parent / "test_data" / "shanghai_small"


def _source_by_patient() -> dict[str, pl.DataFrame]:
    out: dict[str, list[pl.DataFrame]] = {}
    for workbook in find_shanghai_workbooks(SHANGHAI_SMALL):
        sheet = pl.read_excel(workbook)
        cgm = shanghai_cgm_column(sheet.columns)
        assert cgm is not None, workbook
        patient = SHANGHAI_WORKBOOK_PATTERN.match(workbook.name).group("patient")
        out.setdefault(patient, []).append(
            sheet.select(pl.col("Date").alias("ts"), pl.col(cgm).cast(pl.Float64).alias("g")).drop_nulls()
        )
    return {p: pl.concat(frames).unique("ts", keep="first", maintain_order=True) for p, frames in out.items()}


@pytest.fixture(scope="module")
def frames() -> dict[str, pl.DataFrame]:
    converter = ShanghaiDatabaseConverter({}, database_type="shanghai")
    return {f["user_id"][0]: f for f in converter.iter_user_event_frames(SHANGHAI_SMALL, interval_minutes=15)}


@pytest.mark.parametrize("root", [SHANGHAI_SMALL, SHANGHAI_SMALL / "Shanghai_T2DM"])
def test_detected_as_shanghai(root: Path) -> None:
    assert DatabaseDetector().detect_database_type(root) == "shanghai"


def test_one_user_per_patient_across_periods(frames: dict[str, pl.DataFrame]) -> None:
    patients = {SHANGHAI_WORKBOOK_PATTERN.match(p.name).group("patient") for p in find_shanghai_workbooks(SHANGHAI_SMALL)}
    assert list(frames) == sorted(patients)


def test_cgm_values_and_timestamps_preserved(frames: dict[str, pl.DataFrame]) -> None:
    source = _source_by_patient()
    for patient, frame in frames.items():
        src = source[patient]
        assert frame["event_type"].unique().to_list() == ["EGV"]
        assert frame["timestamp"].is_sorted()
        assert set(frame["timestamp"].to_list()) == set(src["ts"].to_list())
        assert sorted(frame["glucose_value_mgdl"].to_list()) == sorted(src["g"].to_list())


def test_workbook_without_cgm_column_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    shutil.copytree(SHANGHAI_SMALL, root)
    # The dataset's own summary workbook has no CGM column; named like a patient sheet it
    # must be reported, not turned into a user.
    shutil.copy(SHANGHAI_SMALL / "Shanghai_T1DM_Summary.xlsx", root / "9999_0_20200101.xlsx")
    converter = ShanghaiDatabaseConverter({}, database_type="shanghai")
    users = [f["user_id"][0] for f in converter.iter_user_event_frames(root, interval_minutes=15)]
    assert "9999" not in users
    assert converter.file_report["no converter recognised the header"] == 1
    assert converter.file_report["matched a converter"] == len(find_shanghai_workbooks(SHANGHAI_SMALL))

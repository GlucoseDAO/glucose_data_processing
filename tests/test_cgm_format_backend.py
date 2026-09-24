#!/usr/bin/env python3
"""
The cgm_format backend on an input no native converter reads (CGMacros).

Runs against a local CGMacros release; skipped when none is present. Point CGMACROS_DIR at
the folder holding CGMacros-0xx/, or place/symlink it at DATA/cgmacros.
"""

import os
from pathlib import Path

import pytest
from cgm_format import FormatParser

from formats.cgm_format_input.cgm_format_database_converter import (
    CGM_FORMAT_DATABASE_TYPE,
    CgmFormatDatabaseConverter,
)
from formats.database_detector import DatabaseDetector

CGMACROS = Path(os.environ.get("CGMACROS_DIR", str(Path(__file__).parent.parent / "DATA" / "cgmacros")))


@pytest.fixture(scope="module")
def cgmacros() -> Path:
    if not any(CGMACROS.glob("CGMacros-*")):
        pytest.skip(f"CGMacros not available at {CGMACROS}")
    return CGMACROS


def test_cgmacros_routes_to_backend(cgmacros: Path) -> None:
    assert DatabaseDetector().detect_database_type(cgmacros) == CGM_FORMAT_DATABASE_TYPE


def test_multi_track_corpus_requires_explicit_track(cgmacros: Path) -> None:
    converter = CgmFormatDatabaseConverter({}, database_type=CGM_FORMAT_DATABASE_TYPE)
    with pytest.raises(ValueError, match=r"cgm_format\.track"):
        next(iter(converter.iter_user_event_frames(cgmacros, interval_minutes=5)))


def test_unknown_track_is_refused(cgmacros: Path) -> None:
    converter = CgmFormatDatabaseConverter({"cgm_format": {"track": "guardian"}}, database_type=CGM_FORMAT_DATABASE_TYPE)
    with pytest.raises(ValueError, match="guardian"):
        next(iter(converter.iter_user_event_frames(cgmacros, interval_minutes=5)))


def test_one_user_per_subject_on_selected_track(cgmacros: Path) -> None:
    subjects = [entry.subject_id for entry in FormatParser.list_subjects(cgmacros)]
    first_two = subjects[:2]
    converter = CgmFormatDatabaseConverter({"cgm_format": {"track": "libre"}}, database_type=CGM_FORMAT_DATABASE_TYPE)
    corpus = FormatParser.parse_corpus(cgmacros, track="libre", subjects=first_two)
    frames = {
        f["user_id"][0]: f
        for f in converter.iter_user_event_frames(cgmacros, interval_minutes=5)
        if f["user_id"][0] in first_two
    }
    assert list(frames) == first_two
    for subject in first_two:
        unified = corpus[f"{subject}/libre"]
        egv = unified.filter(unified["event_type"] == "EGV_READ")
        assert frames[subject]["glucose_value_mgdl"].drop_nulls().len() == egv["datetime"].n_unique()
        assert frames[subject]["carb_grams"].sum() == pytest.approx(unified["carbs"].sum())

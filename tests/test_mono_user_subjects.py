#!/usr/bin/env python3
"""
Subject separation and refusal behaviour of mono-user (folder-based) database converters.

Built from the committed ``test_data/dexcom_small`` export: copying it into two subject
folders gives a multi-subject root whose expected per-user output is, by construction,
the single-folder output of the original export.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest

from formats.database_detector import DatabaseDetector
from formats.dexcom.dexcom_database_converter import DexcomDatabaseConverter
from glucose_ml_preprocessor import GlucoseMLPreprocessor

REPO_ROOT = Path(__file__).parent.parent
DEXCOM_SMALL = REPO_ROOT / "test_data" / "dexcom_small"
DATASETS_CSV = REPO_ROOT / "docs" / "datasets.csv"  # a real CSV that holds no glucose data


def _frames(folder: Path) -> list[pl.DataFrame]:
    converter = DexcomDatabaseConverter({}, database_type="dexcom")
    return list(converter.iter_user_event_frames(folder, interval_minutes=5))


def _copy_export(dest: Path) -> Path:
    shutil.copytree(DEXCOM_SMALL, dest)
    return dest


def test_flat_folder_is_one_user_named_after_folder(tmp_path: Path) -> None:
    folder = _copy_export(tmp_path / "subject_x")
    frames = _frames(folder)
    assert [f["user_id"].unique().to_list() for f in frames] == [["subject_x"]]


def test_subject_subfolders_are_separate_users(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _copy_export(root / "A")
    _copy_export(root / "B")

    single = _frames(DEXCOM_SMALL)[0].drop("user_id")
    frames = _frames(root)

    assert [f["user_id"].unique().to_list() for f in frames] == [["A"], ["B"]]
    # Identical exports must not be merged or de-duplicated against each other.
    for frame in frames:
        assert frame.drop("user_id").equals(single)


def test_root_level_non_glucose_csv_does_not_create_a_user(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    _copy_export(root / "A")
    shutil.copy(DATASETS_CSV, root / "Demographics.csv")

    converter = DexcomDatabaseConverter({}, database_type="dexcom")
    users = [f["user_id"][0] for f in converter.iter_user_event_frames(root, interval_minutes=5)]
    assert users == ["A"]
    n_export_files = len(list(DEXCOM_SMALL.glob("*.csv")))
    assert converter.file_report["matched a converter"] == n_export_files
    assert converter.file_report["no converter recognised the header"] == 1


@pytest.mark.parametrize("name", ["NonDiabDeviceCGM.csv", "CGMacros-001.csv", "g6_export.csv"])
def test_cgm_in_filename_is_not_dexcom(tmp_path: Path, name: str) -> None:
    folder = tmp_path / "cohort"
    folder.mkdir()
    shutil.copy(DATASETS_CSV, folder / name)
    assert DatabaseDetector().detect_database_type(folder) == "unknown"


def test_dexcom_header_detected_regardless_of_filename(tmp_path: Path) -> None:
    folder = tmp_path / "cohort"
    folder.mkdir()
    for i, src in enumerate(sorted(DEXCOM_SMALL.glob("*.csv"))):
        shutil.copy(src, folder / f"cgm_{i}.csv")
    assert DatabaseDetector().detect_database_type(folder) == "dexcom"


def test_no_matching_file_refuses_with_explicit_error(tmp_path: Path) -> None:
    folder = tmp_path / "cohort"
    folder.mkdir()
    shutil.copy(DATASETS_CSV, folder / "NonDiabDeviceCGM.csv")

    preprocessor = GlucoseMLPreprocessor()
    with pytest.raises(ValueError, match=r"0 of 1 data files matched a converter"):
        preprocessor.process(folder, tmp_path / "out.csv", database_type="dexcom")


@pytest.mark.parametrize("workers", [1, 2])
def test_worker_count_does_not_change_output(tmp_path: Path, workers: int) -> None:
    root = tmp_path / "cohort"
    _copy_export(root / "A")
    _copy_export(root / "B")
    reference = tmp_path / "reference.csv"
    GlucoseMLPreprocessor().process(root, reference, database_type="dexcom")

    out = tmp_path / f"workers_{workers}.csv"
    preprocessor = GlucoseMLPreprocessor(max_workers=workers)
    assert preprocessor.max_workers == workers
    preprocessor.process(root, out, database_type="dexcom")
    assert pl.read_csv(out, infer_schema=False).equals(pl.read_csv(reference, infer_schema=False))


def test_worker_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        GlucoseMLPreprocessor(max_workers=0)

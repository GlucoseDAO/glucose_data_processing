#!/usr/bin/env python3
"""
cgm_format input backend.

Reads sources through ``cgm_format.FormatParser`` and maps its unified frames onto this
pipeline's standard fields. Two input shapes:

- a registered **corpus** root (``FormatParser.detect_path_format`` succeeds: BIG IDEAs,
  CGMacros, D1NAMO), parsed with ``parse_corpus``, one user per corpus subject;
- otherwise a folder of single-subject **exports** (Dexcom, Libre, Medtronic, Nightscout and
  their EU variants), grouped into subjects by first-level subfolder exactly as the native
  folder converters do, each file read with ``parse_file``.

The detector routes here only when no native converter recognises the input, so native
output is unchanged; docs/CGM_FORMAT_PARITY.md records how the two agree where both apply.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import polars as pl
from cgm_format import (
    CGMACROS_MEAN_TRACK,
    CGMACROS_TRACKS,
    CGM_SCHEMA,
    FormatParser,
    MalformedDataError,
    Quality,
    SupportedCGMFormat,
    UnknownFormatError,
    ZeroValidInputError,
)
from cgm_format.formats.dexcom import DEXCOM_HIGH_GLUCOSE_DEFAULT, DEXCOM_LOW_GLUCOSE_DEFAULT
from loguru import logger

from formats.database_converters import (
    FILE_MATCHED,
    FILE_NO_CONVERTER,
    FILE_READ_ERROR,
    DatabaseConverter,
    group_files_by_subject,
    merge_same_timestamp_rows,
)
from formats.glucose_bounds import dexcom_style_bounds

CGM_FORMAT_DATABASE_TYPE = "cgm_format"
UNIFIED_SERVICE_COLUMNS: frozenset[str] = frozenset(CGM_SCHEMA.get_column_names(data_only=False)) - frozenset(
    CGM_SCHEMA.get_column_names(data_only=True)
)
CGM_FORMAT_CONFIG_KEY = "cgm_format"
CGM_FORMAT_TRACK_KEY = "track"
MULTI_TRACK_FORMATS: frozenset[SupportedCGMFormat] = frozenset({SupportedCGMFormat.CGMACROS})
VALID_TRACKS: tuple[str, ...] = (*CGMACROS_TRACKS, CGMACROS_MEAN_TRACK)
# Exports cgm_format reads as single files; Nightscout also arrives as JSON.
EXPORT_SUFFIXES: frozenset[str] = frozenset({".csv", ".txt", ".json"})

# Unified column -> this pipeline's standard field name.
UNIFIED_TO_STANDARD: Dict[str, str] = {
    "datetime": "timestamp",
    "glucose": "glucose_value_mgdl",
    "carbs": "carb_grams",
    "insulin_fast": "fast_acting_insulin_u",
    "insulin_slow": "long_acting_insulin_u",
    "heart_rate": "heart_rate",
    "steps": "step_count",
}

# Unified event codes -> the event_type the native Dexcom converter writes for the same row.
UNIFIED_TO_NATIVE_EVENT: Dict[str, str] = {
    "EGV_READ": "EGV",
    "CALIBRAT": "Calibration",
    "CARBS_IN": "Carbs",
    "INS_FAST": "Insulin",
    "INS_SLOW": "Insulin",
    "XRCS_LTE": "Exercise",
    "XRCS_MED": "Exercise",
    "XRCS_HVY": "Exercise",
}
NATIVE_TRACE_EVENT = "EGV"
NATIVE_OTHER_EVENT = "Other"


def cgm_format_can_read(path: Path) -> bool:
    """True if cgm_format recognises ``path`` as a corpus root or holds a file it can parse."""
    if not path.is_dir():
        return False
    try:
        FormatParser.detect_path_format(path)
        return True
    except UnknownFormatError:
        pass
    return any(
        FormatParser.format_supported(f.read_bytes())
        for _, files in group_files_by_subject(path, EXPORT_SUFFIXES)
        for f in files
    )


class CgmFormatDatabaseConverter(DatabaseConverter):
    """Database converter backed by ``cgm_format.FormatParser``."""

    def get_database_name(self) -> str:
        return "cgm_format input"

    def _track(self) -> Optional[str]:
        track = (self.config.get(CGM_FORMAT_CONFIG_KEY) or {}).get(CGM_FORMAT_TRACK_KEY)
        if track is not None and track not in VALID_TRACKS:
            raise ValueError(f"{CGM_FORMAT_CONFIG_KEY}.{CGM_FORMAT_TRACK_KEY}={track!r}; expected one of {VALID_TRACKS}")
        return track

    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        frames = list(self.iter_user_event_frames(data_folder, interval_minutes=5))
        if not frames:
            raise ValueError(f"No valid data found in {data_folder}: {self.describe_file_report()}")
        df = pl.concat(frames, how="diagonal_relaxed")
        if output_file:
            logger.info(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
        return df

    def iter_user_event_frames(self, data_folder: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        root = Path(data_folder)
        if not root.is_dir():
            raise ValueError(f"Input must be a directory, got: {data_folder}")
        try:
            corpus_format = FormatParser.detect_path_format(root)
        except UnknownFormatError:
            corpus_format = None

        subjects = self._corpus_frames(root, corpus_format) if corpus_format else self._export_frames(root)
        for user_id, unified in subjects.items():
            frame = self._to_standard(unified, user_id)
            if frame.height == 0:
                self.subjects_without_trace_rows += 1
                continue
            logger.info(f"Consolidated {frame.height:,} records for user {user_id}")
            yield self._enforce_output_schema(frame)

    def _corpus_frames(self, root: Path, corpus_format: SupportedCGMFormat) -> Dict[str, pl.DataFrame]:
        track = self._track()
        if corpus_format in MULTI_TRACK_FORMATS and track is None:
            raise ValueError(
                f"{root} is a {corpus_format.value} corpus with {len(CGMACROS_TRACKS)} concurrent sensors. "
                f"Set {CGM_FORMAT_CONFIG_KEY}.{CGM_FORMAT_TRACK_KEY} in the config to one of {VALID_TRACKS}; "
                "processing both would give every subject two traces."
            )
        corpus_track = track if corpus_format in MULTI_TRACK_FORMATS else None
        logger.info(f"Reading {root} as a {corpus_format.value} corpus via cgm_format")
        corpus = FormatParser.parse_corpus(root, track=corpus_track)
        self.file_report[FILE_MATCHED] += len(corpus)
        # Multi-track keys are "<subject>/<track>"; one track is selected, so the subject is the user.
        return {key.split("/", 1)[0]: frame for key, frame in corpus.items()}

    def _export_frames(self, root: Path) -> Dict[str, pl.DataFrame]:
        subjects: Dict[str, pl.DataFrame] = {}
        for subject, files in group_files_by_subject(root, EXPORT_SUFFIXES):
            parsed: List[pl.DataFrame] = []
            for f in files:
                frame = self._parse_export(f)
                if frame is not None:
                    parsed.append(frame)
            if parsed:
                # Files are parsed one at a time, so a record repeated in two overlapping
                # exports arrives twice. Identical data columns and event type make a true
                # duplicate (cgm_format's primary key); service columns such as sequence_id
                # are assigned per file and must not keep the copies apart.
                frame = pl.concat(parsed, how="diagonal_relaxed")
                key = [c for c in frame.columns if c not in UNIFIED_SERVICE_COLUMNS or c == "event_type"]
                subjects[subject] = frame.unique(subset=key, keep="first", maintain_order=True)
        return subjects

    def _parse_export(self, file_path: Path) -> Optional[pl.DataFrame]:
        raw = file_path.read_bytes()
        if not FormatParser.format_supported(raw):
            self.file_report[FILE_NO_CONVERTER] += 1
            return None
        try:
            frame = FormatParser.parse_from_bytes(raw)
        except (MalformedDataError, ZeroValidInputError) as e:
            logger.info(f"cgm_format could not parse {file_path}: {e}")
            self.file_report[FILE_READ_ERROR] += 1
            return None
        self.file_report[FILE_MATCHED] += 1
        return frame

    def _to_standard(self, unified: pl.DataFrame, user_id: str) -> pl.DataFrame:
        """Map one unified frame to standard fields, one row per timestamp, trace rows only."""
        low_value, high_value = dexcom_style_bounds(self.config)
        out_of_range = (pl.col("quality") & Quality.OUT_OF_RANGE.value) != 0
        glucose = (
            pl.when(out_of_range & (pl.col("glucose") == DEXCOM_HIGH_GLUCOSE_DEFAULT))
            .then(pl.lit(high_value))
            .when(out_of_range & (pl.col("glucose") == DEXCOM_LOW_GLUCOSE_DEFAULT))
            .then(pl.lit(low_value))
            .otherwise(pl.col("glucose"))
        )
        present = [c for c in UNIFIED_TO_STANDARD if c in unified.columns]
        df = unified.select(
            *[
                (glucose if c == "glucose" else pl.col(c)).alias(UNIFIED_TO_STANDARD[c])
                for c in present
            ],
            pl.col("event_type").replace_strict(UNIFIED_TO_NATIVE_EVENT, default=NATIVE_OTHER_EVENT).alias("event_type"),
            pl.lit(user_id).alias("user_id"),
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

        # Fingersticks (CALIBRAT) never reach the trace (docs/DECISIONS.md D2); non-glucose
        # event rows are kept for their insulin/carbs.
        is_non_trace_glucose = pl.col("glucose_value_mgdl").is_not_null() & (pl.col("event_type") != NATIVE_TRACE_EVENT)
        n_dropped = df.filter(is_non_trace_glucose).height
        if n_dropped:
            logger.info(f"  User {user_id}: dropped {n_dropped:,} non-CGM glucose rows")
        # The unified sort puts non-glucose events before the reading at the same instant;
        # put the reading first so a merged row carrying glucose is labelled EGV.
        df = df.filter(~is_non_trace_glucose).sort(
            ["timestamp", pl.col("event_type") != NATIVE_TRACE_EVENT], maintain_order=True
        )
        # cgm_format already de-duplicated on its own primary key, which includes columns
        # dropped above (food names, calories); two identical items eaten together would
        # otherwise collapse into one here.
        return merge_same_timestamp_rows(df, drop_exact_duplicates=False)

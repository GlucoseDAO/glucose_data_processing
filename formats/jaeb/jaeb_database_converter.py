#!/usr/bin/env python3
"""
JAEB database converter.

Participants share one file and are told apart by ``PtID``, which the row converter
stores as ``user_id``; the mono-user base class then yields one frame per participant.
"""

import polars as pl
from loguru import logger

from formats.database_converters import MonoUserDatabaseConverter

JAEB_TRACE_EVENT_TYPE = "EGV"


class JaebDatabaseConverter(MonoUserDatabaseConverter):
    """Converter for JAEB comma-separated device tables."""

    def _drop_non_trace_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Keep CGM rows only. The pipeline resamples every glucose value into the trace
        regardless of event_type, so calibration and meter rows are dropped here
        (docs/DECISIONS.md D2).
        """
        dropped = (
            df.filter(pl.col("event_type") != JAEB_TRACE_EVENT_TYPE)
            .group_by("event_type")
            .agg(pl.len(), pl.col("user_id").n_unique().alias("users"))
            .sort("event_type")
        )
        if dropped.height > 0:
            summary = ", ".join(
                f"{r['len']:,} {r['event_type']} rows from {r['users']} users" for r in dropped.iter_rows(named=True)
            )
            logger.info(f"Dropped non-CGM rows: {summary}")
        return df.filter(pl.col("event_type") == JAEB_TRACE_EVENT_TYPE)

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("glucose_value_mgdl").cast(pl.Float64, strict=False))

    def get_database_name(self) -> str:
        return "JAEB Device Export Database"

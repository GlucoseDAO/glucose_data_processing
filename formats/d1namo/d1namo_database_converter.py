#!/usr/bin/env python3
"""
D1NAMO database converter.

One subject per folder; the mono-user base class yields one user per first-level
subfolder, named after it (``001`` ... ``009``).
"""

import polars as pl
from loguru import logger

from formats.d1namo.d1namo_converter import D1NAMO_EVENT_CGM
from formats.database_converters import MonoUserDatabaseConverter


class D1namoDatabaseConverter(MonoUserDatabaseConverter):
    """Converter for the D1NAMO dataset."""

    def _drop_non_trace_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Drop fingerstick glucose rows, which the pipeline would otherwise resample into the
        sensor trace (docs/DECISIONS.md D2). Insulin rows carry no glucose and are kept.
        """
        is_fingerstick = pl.col("glucose_value_mgdl").is_not_null() & (pl.col("event_type") != D1NAMO_EVENT_CGM)
        dropped = df.filter(is_fingerstick)
        if dropped.height:
            logger.info(
                f"Dropped {dropped.height:,} fingerstick glucose rows from {dropped['user_id'].n_unique()} users"
            )
        return df.filter(~is_fingerstick)

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("glucose_value_mgdl").cast(pl.Float64, strict=False),
            pl.col("fast_acting_insulin_u").cast(pl.Float64, strict=False),
            pl.col("long_acting_insulin_u").cast(pl.Float64, strict=False),
        )

    def get_database_name(self) -> str:
        return "D1NAMO Database"

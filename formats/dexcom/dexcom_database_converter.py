#!/usr/bin/env python3
"""
Dexcom database converter.

This module provides the converter for Dexcom G6 databases (mono-user).
Handles High/Low value replacement and calibration removal specific to Dexcom.
"""

from typing import Any, Dict

import polars as pl
from loguru import logger

from formats.database_converters import MonoUserDatabaseConverter
from formats.glucose_bounds import dexcom_style_bounds

DEXCOM_HIGH = "High"
DEXCOM_LOW = "Low"
DEXCOM_CALIBRATION_EVENT = "Calibration"


class DexcomDatabaseConverter(MonoUserDatabaseConverter):
    """Converter for Dexcom G6 databases."""

    def _drop_non_trace_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove calibration events (fingersticks entered into the receiver) when
        ``dexcom.remove_calibration`` is on. Runs before per-timestamp de-duplication so a
        calibration in the same second as a sensor reading cannot replace it.
        """
        dexcom_config: Dict[str, Any] = self.config.get("dexcom", {})
        if not dexcom_config.get("remove_calibration", True):
            return df
        is_calibration = pl.col("event_type") == DEXCOM_CALIBRATION_EVENT
        n_calibration = df.filter(is_calibration).height
        if n_calibration:
            logger.info(f"Removing {n_calibration} Dexcom calibration events")
        return df.filter(~is_calibration)

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Replace the out-of-range strings ``High``/``Low`` with the configured bounds and make
        glucose numeric. Before this ran on standard field names, the check looked for the
        display name ``Glucose Value (mg/dL)``, never matched, and High/Low were nulled.
        """
        low_value, high_value = dexcom_style_bounds(self.config)
        glucose = pl.col("glucose_value_mgdl")
        high_count = df.filter(glucose == DEXCOM_HIGH).height
        low_count = df.filter(glucose == DEXCOM_LOW).height
        if high_count or low_count:
            logger.info(
                f"  User {df['user_id'][0]}: replaced {high_count} '{DEXCOM_HIGH}' with {high_value} "
                f"and {low_count} '{DEXCOM_LOW}' with {low_value}"
            )
        return df.with_columns(
            pl.when(glucose == DEXCOM_HIGH)
            .then(pl.lit(high_value))
            .when(glucose == DEXCOM_LOW)
            .then(pl.lit(low_value))
            .otherwise(glucose.cast(pl.Float64, strict=False))
            .alias("glucose_value_mgdl")
        )

    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Dexcom G6 Database"

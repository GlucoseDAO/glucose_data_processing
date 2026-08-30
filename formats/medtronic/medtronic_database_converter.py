#!/usr/bin/env python3
"""
Medtronic database converter.

This module provides the converter for Medtronic databases (mono-user).
Handles calibration event removal and sensor data processing.
"""

from typing import Any, Dict

import polars as pl
from loguru import logger

from formats.database_converters import MonoUserDatabaseConverter


class MedtronicDatabaseConverter(MonoUserDatabaseConverter):
    """Converter for Medtronic Guardian Connect databases."""

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Medtronic-specific processing.

        Medtronic data may have calibration events that can affect CGM accuracy.
        This method handles:
        1. Logging of data statistics
        2. Optional calibration-based filtering (if configured)

        Args:
            df: DataFrame to process

        Returns:
            Processed DataFrame
        """
        # Get Medtronic-specific config
        medtronic_config: Dict[str, Any] = self.config.get("medtronic", {})

        # Log statistics about the data
        if "glucose_value_mgdl" in df.columns:
            # Count non-empty glucose readings
            glucose_col = df["glucose_value_mgdl"]

            # Handle string column - count non-empty, non-null values
            if glucose_col.dtype in [pl.Utf8, pl.String]:
                valid_glucose = df.filter(
                    (pl.col("glucose_value_mgdl").is_not_null())
                    & (pl.col("glucose_value_mgdl") != "")
                )
            else:
                valid_glucose = df.filter(pl.col("glucose_value_mgdl").is_not_null())

            glucose_count = len(valid_glucose)
            logger.info(f"Medtronic data: {glucose_count:,} glucose readings found")

            # Convert glucose to numeric for statistics if it's a string
            if glucose_col.dtype in [pl.Utf8, pl.String]:
                try:
                    numeric_glucose = valid_glucose.with_columns(
                        pl.col("glucose_value_mgdl").cast(pl.Float64, strict=False).alias("_glucose_numeric")
                    ).filter(pl.col("_glucose_numeric").is_not_null())

                    if len(numeric_glucose) > 0:
                        min_val = numeric_glucose["_glucose_numeric"].min()
                        max_val = numeric_glucose["_glucose_numeric"].max()
                        mean_val = numeric_glucose["_glucose_numeric"].mean()
                        logger.info(f"  Glucose range: {min_val:.0f} - {max_val:.0f} mg/dL (mean: {mean_val:.1f})")
                except Exception:
                    pass  # Statistics are optional, don't fail on errors

        # Log insulin and carb data if present
        if "fast_acting_insulin_u" in df.columns:
            insulin_col = df["fast_acting_insulin_u"]
            if insulin_col.dtype in [pl.Utf8, pl.String]:
                insulin_events = df.filter(
                    (pl.col("fast_acting_insulin_u").is_not_null())
                    & (pl.col("fast_acting_insulin_u") != "")
                )
            else:
                insulin_events = df.filter(pl.col("fast_acting_insulin_u").is_not_null())
            logger.info(f"  Insulin events: {len(insulin_events):,}")

        if "carb_grams" in df.columns:
            carb_col = df["carb_grams"]
            if carb_col.dtype in [pl.Utf8, pl.String]:
                carb_events = df.filter(
                    (pl.col("carb_grams").is_not_null())
                    & (pl.col("carb_grams") != "")
                )
            else:
                carb_events = df.filter(pl.col("carb_grams").is_not_null())
            logger.info(f"  Carb events: {len(carb_events):,}")

        return df

    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Medtronic Guardian Connect Database"

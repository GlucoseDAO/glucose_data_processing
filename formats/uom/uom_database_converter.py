#!/usr/bin/env python3
"""
University of Manchester (UoM) database converter.

This module provides the converter for UoM T1D databases (multi-user).
"""

from pathlib import Path
from typing import Optional

import polars as pl

from formats.database_converters import MultiUserDatabaseConverter


class UoMDatabaseConverter(MultiUserDatabaseConverter):
    """Converter for University of Manchester T1D databases."""

    def _extract_user_id_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Extract user ID from UoM filename format.

        Args:
            file_path: Path to the CSV file

        Returns:
            User ID string or None if not found
        """
        filename = file_path.stem  # Get filename without extension

        # UoM format: UoMGlucose2301.csv -> participant ID: 2301
        # Also handles UoM2301sleeptime.csv -> participant ID: 2301
        if filename.startswith("UoM"):
            # Extract the ID part (digits)
            import re
            match = re.search(r"(\d+)", filename)
            if match:
                return match.group(1)

        return None

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply UoM specific data cleaning and processing."""
        import polars as pl
        
        # 1. Clean Sentinels
        if "stress_level" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("stress_level").cast(pl.Float64, strict=False) < 0)
                .then(None)
                .otherwise(pl.col("stress_level"))
                .alias("stress_level")
            )
            
        if "heart_rate" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("heart_rate").cast(pl.Float64, strict=False) <= 0)
                .then(None)
                .otherwise(pl.col("heart_rate"))
                .alias("heart_rate")
            )
            
        # 2. Fix Step Count
        # Activity records are 15-min window summaries with multiple activity types
        # (RUNNING, WALKING, SEDENTARY) at the same timestamp that must not be summed.
        # Sleep records store cumulative period totals, not per-5-min counts → null them.
        if "step_count" in df.columns and "event_type" in df.columns:
            df = df.with_columns(
                pl.col("step_count").cast(pl.Float64, strict=False).alias("_step_raw")
            )

            # Only Activity events carry valid per-bin step counts.
            # Divide by 3 since Activity records cover 15-minute windows.
            # All other event types (Sleep period totals, etc.) → null.
            df = df.with_columns(
                pl.when(pl.col("event_type") == "Activity")
                .then(pl.col("_step_raw") / 3.0)
                .otherwise(pl.lit(None))
                .alias("_step_adj")
            )

            # For Activity events at the same [user_id, timestamp], keep only the max
            # activity type so the downstream occasional_sum doesn't double-count.
            df = df.with_columns(
                pl.col("_step_adj")
                .fill_null(-1.0)
                .rank("ordinal", descending=True)
                .over(["user_id", "timestamp", "event_type"])
                .alias("_step_rank")
            )
            df = df.with_columns(
                pl.when((pl.col("event_type") == "Activity") & (pl.col("_step_rank") > 1))
                .then(0.0)
                .otherwise(pl.col("_step_adj"))
                .alias("_step_final")
            )

            # Restore nulls: original null _step_raw or explicitly nulled non-Activity rows.
            df = df.with_columns(
                pl.when(pl.col("_step_raw").is_null() | pl.col("_step_adj").is_null())
                .then(pl.lit(None))
                .otherwise(pl.col("_step_final"))
                .cast(pl.Utf8)
                .alias("step_count")
            )

            df = df.drop(["_step_raw", "_step_adj", "_step_rank", "_step_final"])
            
        # 3. Add Core Masks
        if "glucose_value_mgdl" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("glucose_value_mgdl").is_not_null())
                .then(1.0)
                .otherwise(0.0)
                .alias("y_observed")
            )
            
        if "step_count" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("step_count").is_not_null())
                .then(1.0)
                .otherwise(0.0)
                .alias("steps_observed")
            )
            
        if "heart_rate" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("heart_rate").is_not_null())
                .then(1.0)
                .otherwise(0.0)
                .alias("hr_observed")
            )
            
        return df

    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "University of Manchester T1D Database"



#!/usr/bin/env python3
"""
Loop database converter.

This module provides the converter for Loop multi-user dataset using Polars-native 
processing to handle very large datasets efficiently.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Union
import polars as pl
from loguru import logger

from formats.database_converters import DatabaseConverter
from formats.base_converter import CSVFormatConverter


class LoopDatabaseConverter(DatabaseConverter):
    """Converter for Loop database using Polars for high-performance consolidation."""

    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Loop Database"

    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        """
        Consolidate Loop data by iterating per-user frames.
        
        This method is provided for backwards compatibility and tests.
        For large datasets, the preprocessor uses iter_user_event_frames directly.
        """
        interval_minutes = int(self.config.get("expected_interval_minutes", 5))
        
        frames: List[pl.DataFrame] = []
        for user_df in self.iter_user_event_frames(data_folder, interval_minutes=interval_minutes):
            frames.append(user_df)

        if not frames:
            raise ValueError(f"No Loop records produced from {data_folder}")

        df = pl.concat(frames, how="diagonal").sort(["user_id", "timestamp"])
        
        if output_file:
            logger.info(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
            
        return df

    def iter_user_event_frames(self, data_folder: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        """
        Yield per-user DataFrames from Loop dataset.
        
        This is the streaming-friendly entry point used by the preprocessor.
        Each yielded DataFrame contains all events for a single user.
        """
        data_path = Path(data_folder)
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        # Find all relevant files
        all_files = list(data_path.glob("**/*.txt"))
        
        cgm_files = [f for f in all_files if "cgm" in f.name.lower()]
        bgm_files = [f for f in all_files if "bgm" in f.name.lower()]
        basal_files = [f for f in all_files if "basal" in f.name.lower()]
        bolus_files = [f for f in all_files if "bolus" in f.name.lower()]
        food_files = [f for f in all_files if "food" in f.name.lower()]

        logger.info(f"Found {len(cgm_files)} CGM files, {len(bgm_files)} BGM, {len(basal_files)} Basal, {len(bolus_files)} Bolus, {len(food_files)} Food")

        # Build the combined lazy frame with all modalities
        processed_lazy_frames: List[pl.LazyFrame] = []

        if cgm_files:
            cgm_ldf = self._read_loop_files_lazy(cgm_files, {
                "PtID": pl.Utf8,
                "UTCDtTm": pl.Utf8,
                "CGMVal": pl.Utf8
            })
            cgm_ldf = cgm_ldf.select([
                pl.col("PtID").alias("user_id"),
                pl.col("UTCDtTm"),
                pl.lit("EGV").alias("event_type"),
                pl.col("CGMVal").alias("glucose_value_mgdl"),
            ])
            processed_lazy_frames.append(cgm_ldf)

        if bgm_files:
            bgm_ldf = self._read_loop_files_lazy(bgm_files, {
                "PtID": pl.Utf8,
                "UTCDtTm": pl.Utf8,
                "BGMVal": pl.Utf8
            })
            bgm_ldf = bgm_ldf.select([
                pl.col("PtID").alias("user_id"),
                pl.col("UTCDtTm"),
                pl.lit("BGM").alias("event_type"),
                pl.col("BGMVal").alias("glucose_value_mgdl"),
            ])
            processed_lazy_frames.append(bgm_ldf)

        if basal_files:
            basal_ldf = self._read_loop_files_lazy(basal_files, {
                "PtID": pl.Utf8,
                "UTCDtTm": pl.Utf8,
                "Rate": pl.Utf8
            })
            basal_ldf = basal_ldf.select([
                pl.col("PtID").alias("user_id"),
                pl.col("UTCDtTm"),
                pl.lit("Basal").alias("event_type"),
                pl.col("Rate").alias("basal_rate"),
            ])
            processed_lazy_frames.append(basal_ldf)

        if bolus_files:
            bolus_ldf = self._read_loop_files_lazy(bolus_files, {
                "PtID": pl.Utf8,
                "UTCDtTm": pl.Utf8,
                "Normal": pl.Utf8
            })
            bolus_ldf = bolus_ldf.select([
                pl.col("PtID").alias("user_id"),
                pl.col("UTCDtTm"),
                pl.lit("Bolus").alias("event_type"),
                pl.col("Normal").alias("fast_acting_insulin_u"),
            ])
            processed_lazy_frames.append(bolus_ldf)

        if food_files:
            food_ldf = self._read_loop_files_lazy(food_files, {
                "PtID": pl.Utf8,
                "UTCDtTm": pl.Utf8,
                "CarbsNet": pl.Utf8
            })
            food_ldf = food_ldf.select([
                pl.col("PtID").alias("user_id"),
                pl.col("UTCDtTm"),
                pl.lit("Meal").alias("event_type"),
                pl.col("CarbsNet").alias("carb_grams"),
            ])
            processed_lazy_frames.append(food_ldf)

        if not processed_lazy_frames:
            raise ValueError("No valid Loop data found in the provided directory.")

        # Combine all modalities lazily
        combined_ldf = pl.concat(processed_lazy_frames, how="diagonal")

        # Parse timestamps
        combined_ldf = combined_ldf.with_columns([
            pl.col("UTCDtTm").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("timestamp")
        ]).filter(pl.col("timestamp").is_not_null())

        # Get unique user IDs (small collect - just distinct user_id strings)
        logger.info("Identifying unique users...")
        user_ids_series = combined_ldf.select("user_id").unique().sort("user_id").collect()["user_id"]
        user_ids = user_ids_series.to_list()
        
        # Apply start_with_user_id skipping if specified
        start_user_id = self._get_start_with_user_id()
        if start_user_id:
            start_index = 0
            found = False
            for i, uid in enumerate(user_ids):
                if uid == start_user_id:
                    start_index = i
                    found = True
                    break
            if found:
                logger.info(f"Skipping Loop users before {start_user_id} (found at index {start_index})")
                user_ids = user_ids[start_index:]
            else:
                logger.info(f"Warning: start_with_user_id '{start_user_id}' not found in Loop database.")

        # Apply first_n_users filtering if specified
        first_n_users = self.config.get("first_n_users")
        if first_n_users and int(first_n_users) > 0:
            user_ids = user_ids[: int(first_n_users)]

        logger.info(f"Processing {len(user_ids)} Loop users...")

        # Yield per-user frames
        for user_id in user_ids:
            user_ldf = combined_ldf.filter(pl.col("user_id") == user_id)
            
            # Deduplicate per user
            user_ldf = user_ldf.unique(subset=["user_id", "timestamp", "event_type"], keep="first")
            
            # Enforce output schema
            user_ldf = self._enforce_output_schema(user_ldf)
            
            # Sort and collect just this user's data
            user_df = user_ldf.sort("timestamp").collect()
            
            if len(user_df) > 0:
                logger.info(f"  User {user_id}: {len(user_df)} records")
                yield user_df

    def _read_loop_files_lazy(self, files: List[Path], schema: Dict[str, Any]) -> pl.LazyFrame:
        """Helper to read multiple pipe-separated files as a single LazyFrame."""
        if not files:
            return pl.LazyFrame()

        ldf = pl.scan_csv(
            files,
            separator="|",
            schema_overrides=schema,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )

        # Select only the columns we need from the schema
        # Use collect_schema() to avoid performance warning
        ldf_schema = ldf.collect_schema()
        available_cols = [col for col in schema.keys() if col in ldf_schema.names()]
        if not available_cols:
            return pl.LazyFrame()

        return ldf.select(available_cols)

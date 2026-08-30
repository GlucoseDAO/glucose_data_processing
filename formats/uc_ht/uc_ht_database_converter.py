#!/usr/bin/env python3
"""
UC_HT database converter.

This module provides the converter for the UC_HT dataset (multi-user, Excel-based).
"""

from pathlib import Path
from typing import Dict, List, Optional, Iterable, Union
import polars as pl
from loguru import logger

from formats.database_converters import DatabaseConverter
from formats.base_converter import CSVFormatConverter


class UCHTDatabaseConverter(DatabaseConverter):
    """Converter for UC_HT dataset (multi-user, Excel-based)."""

    def get_database_name(self) -> str:
        return "UC_HT Database"

    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        """
        Consolidate data from the UC_HT database folder.
        """
        data_path = Path(data_folder)
        if not data_path.exists():
            raise FileNotFoundError(f"UC_HT data folder not found: {data_folder}")

        interval_minutes = int(self.config.get("expected_interval_minutes", 5))
        
        frames: List[pl.DataFrame] = []
        for user_df in self.iter_user_event_frames(data_path, interval_minutes=interval_minutes):
            frames.append(user_df)

        if not frames:
            raise ValueError(f"No UC_HT records produced from {data_folder}")

        # Concatenate all user frames
        df = pl.concat(frames)
        
        # Enforce output schema
        df = self._enforce_output_schema(df)
        
        # Sort by user_id and timestamp
        df = df.sort(["user_id", "timestamp"])

        if output_file:
            logger.info(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
            
        return df

    def iter_user_event_frames(self, data_path: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        """
        Iterate through users and yield resampled event frames.
        """
        data_path = Path(data_path)
        # Check if this folder itself is a user directory (contains modality files)
        modality_filenames = ['Glucose.xlsx', 'Heart Rate.xlsx', 'Steps.xlsx', 'Carbohidrates.xlsx', 'Insulin.xlsx', 'IGAR.xlsx']
        is_user_dir = any((data_path / f).exists() for f in modality_filenames)
        
        if is_user_dir:
            user_dirs = [data_path]
        else:
            # Identify users (folders), excluding hidden ones like .git
            user_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
        
        user_ids = [d.name for d in user_dirs]

        # Apply start_with_user_id skipping if specified
        start_user_id = self._get_start_with_user_id()
        if start_user_id:
            if start_user_id in user_ids:
                start_index = user_ids.index(start_user_id)
                logger.info(f"Skipping UC_HT users before {start_user_id}")
                user_dirs = user_dirs[start_index:]
            else:
                logger.info(f"Warning: start_with_user_id '{start_user_id}' not found in UC_HT database.")

        # Apply first_n_users filtering if specified
        first_n_users = self.config.get("first_n_users")
        if first_n_users and int(first_n_users) > 0:
            user_dirs = user_dirs[: int(first_n_users)]

        logger.info(f"Processing {len(user_dirs)} UC_HT users...")

        for user_dir in user_dirs:
            user_id = user_dir.name
            df = self._process_user_dir(user_dir, user_id)
            if df is not None and len(df) > 0:
                yield df

    def _process_user_dir(self, user_dir: Path, user_id: str) -> Optional[pl.DataFrame]:
        """
        Process all Excel files in a user directory and merge them.
        """
        logger.info(f"  Processing user: {user_id}")
        
        modality_files = {
            "Glucose": "Glucose.xlsx",
            "Heart Rate": "Heart Rate.xlsx",
            "Steps": "Steps.xlsx",
            "Carbohydrates": "Carbohidrates.xlsx",
            "Insulin": "Insulin.xlsx",
            "IGAR": "IGAR.xlsx"
        }

        frames = []
        
        # Load schema for field mappings
        schema = CSVFormatConverter._load_schema("uc_ht")
        converters = schema.get("converters", {})

        for modality, filename in modality_files.items():
            file_path = user_dir / filename
            if not file_path.exists():
                continue
            
            try:
                # Read Excel file
                df_mod = pl.read_excel(file_path)
                if df_mod.is_empty():
                    continue

                # Get converter for this modality
                conv_key = modality.lower().replace(" ", "_")
                if conv_key not in converters:
                    continue
                
                mapping = converters[conv_key].get("field_mappings", {})
                
                # Rename columns based on mapping
                rename_map = {}
                for source, target in mapping.items():
                    if source in df_mod.columns:
                        rename_map[source] = target
                
                if not rename_map:
                    continue
                
                df_mod = df_mod.rename(rename_map)
                
                # Keep only mapped columns
                df_mod = df_mod.select(list(rename_map.values()))
                
                # Parse timestamp if it's not already datetime
                dtype = df_mod["timestamp"].dtype
                if not isinstance(dtype, pl.Datetime):
                    # Use native Polars expressions - faster than map_elements
                    df_mod = df_mod.with_columns(
                        pl.coalesce(
                            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
                            pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False),
                            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
                        ).alias("timestamp")
                    )
                
                # Ensure microsecond precision to avoid join issues later
                df_mod = df_mod.with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
                
                # Drop rows with null timestamps
                df_mod = df_mod.filter(pl.col("timestamp").is_not_null())
                
                if df_mod.is_empty():
                    continue
                
                frames.append(df_mod)
                
            except Exception as e:
                logger.error(f"    Error processing {filename} for user {user_id}: {e}")

        if not frames:
            return None

        # Merge all modalities for this user
        user_df = frames[0]
        for next_df in frames[1:]:
            # Outer join on timestamp
            user_df = user_df.join(next_df, on="timestamp", how="full", coalesce=True)
        
        # Add user_id and event_type
        user_df = user_df.with_columns([
            pl.lit(user_id).alias("user_id"),
            pl.lit("UC_HT").alias("event_type")
        ])
        
        return user_df

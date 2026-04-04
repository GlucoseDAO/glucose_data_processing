"""Data cleaning logic for removing covariates far from glucose values."""

import polars as pl
from typing import Tuple, Dict, Any, Optional
from processing.core.fields import StandardFieldNames
from loguru import logger

class DataCleaner:
    """
    Removes covariate data that is far from glucose values.
    Keeps data within small gaps (<= small_gap_max_minutes) and data where glucose is present.
    """
    
    def __init__(self, small_gap_max_minutes: int) -> None:
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        
    def clean_remote_data(
        self, 
        df: pl.DataFrame,
        field_categories_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Removes covariate data in large gaps between glucose values.
        
        Args:
            df: Input DataFrame
            field_categories_dict: Optional dictionary of field categories
            
        Returns:
            Tuple of (cleaned DataFrame, statistics)
        """
        if df.is_empty():
            return df, {"removed_records": 0}
            
        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        user_id_col = StandardFieldNames.USER_ID
        
        original_count = len(df)
        
        # If glucose column is missing, it means there is no glucose data at all for this set.
        # In this case, we remove all records as they are "far from glucose".
        if glucose_col not in df.columns:
            logger.debug(f"Glucose column {glucose_col} not found. Removing all {original_count} records.")
            return df.filter(pl.lit(False)), {"removed_records": original_count}
            
        # Check if there are any non-null glucose values
        if df.select(pl.col(glucose_col).is_not_null().any()).item() is False:
            logger.debug(f"No non-null glucose values found in {glucose_col}. Removing all {original_count} records.")
            return df.filter(pl.lit(False)), {"removed_records": original_count}

        logger.debug(f"Cleaning covariate data in gaps > {self.small_gap_max_seconds / 60} minutes...")
        
        # Sort by user and timestamp if multi-user
        if user_id_col in df.columns:
            df = df.sort([user_id_col, ts_col])
        else:
            df = df.sort(ts_col)
            
        # Per-user cleaning logic
        if user_id_col in df.columns:
            # Multi-user processing using Polars over() if possible, 
            # but cleaning logic involving fills might be safer with group_by if we want to ensure no leakage between users.
            # Actually forward_fill().over() works fine.
            df = self._apply_cleaning_logic(df, ts_col, glucose_col, user_id_col)
        else:
            df = self._apply_cleaning_logic(df, ts_col, glucose_col)
            
        removed_count = original_count - len(df)
        logger.debug(f"Removed {removed_count} records located in large glucose gaps")
        
        return df, {"removed_records": removed_count}

    def _apply_cleaning_logic(self, df: pl.DataFrame, ts_col: str, glucose_col: str, group_col: Optional[str] = None) -> pl.DataFrame:
        """
        Internal logic to identify and remove remote data.
        """
        # Mark timestamps where glucose is present
        expr_ts_at_glucose = pl.when(pl.col(glucose_col).is_not_null()).then(pl.col(ts_col)).otherwise(None)
        
        if group_col:
            # Use window functions for multi-user safety
            prev_glucose_ts = expr_ts_at_glucose.forward_fill().over(group_col).alias("_prev_glucose_ts")
            next_glucose_ts = expr_ts_at_glucose.backward_fill().over(group_col).alias("_next_glucose_ts")
        else:
            prev_glucose_ts = expr_ts_at_glucose.forward_fill().alias("_prev_glucose_ts")
            next_glucose_ts = expr_ts_at_glucose.backward_fill().alias("_next_glucose_ts")
            
        df = df.with_columns([
            prev_glucose_ts,
            next_glucose_ts
        ])
        
        # Calculate gap between the surrounding glucose values
        df = df.with_columns(
            (pl.col("_next_glucose_ts") - pl.col("_prev_glucose_ts")).dt.total_seconds().alias("_gap_sec")
        )
        
        # Rows to keep:
        # 1. Glucose is not null (this row is a glucose measurement)
        # 2. Row is in a small gap: both neighbors exist and distance between them <= small_gap
        #    (This means glucose was measured recently and will be measured soon)
        
        df_cleaned = df.filter(
            pl.col(glucose_col).is_not_null() |
            (
                pl.col("_prev_glucose_ts").is_not_null() & 
                pl.col("_next_glucose_ts").is_not_null() & 
                (pl.col("_gap_sec") <= self.small_gap_max_seconds)
            )
        )
        
        return df_cleaned.drop(["_prev_glucose_ts", "_next_glucose_ts", "_gap_sec"])

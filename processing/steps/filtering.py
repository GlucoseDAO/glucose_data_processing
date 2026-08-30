"""Filtering logic for sequences and glucose-only data."""

import polars as pl
from typing import Tuple, Dict, Any, List, Optional
from loguru import logger
from processing.core.fields import StandardFieldNames

class SequenceFilter:
    """
    Filters sequences by length and provides glucose-only filtering.
    """
    
    def __init__(self, min_sequence_len: int, glucose_only: bool = False) -> None:
        self.min_sequence_len = min_sequence_len
        self.glucose_only = glucose_only

    def filter_sequences_by_length(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Filters out sequences shorter than min_sequence_len.
        """
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        if df.height == 0 or seq_id_col not in df.columns:
            logger.debug("Empty dataframe or missing sequence_id - skipping sequence filtering")
            return df, {
                'original_sequences': 0,
                'filtered_sequences': 0,
                'removed_sequences': 0,
                'original_records': 0,
                'filtered_records': 0,
                'removed_records': 0
            }

        logger.debug(f"Filtering sequences with length < {self.min_sequence_len}...")
        
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        sequence_counts = df.group_by(seq_id_col).len().sort(seq_id_col)
        sequences_to_keep = sequence_counts.filter(pl.col('len') >= self.min_sequence_len)
        
        filtering_stats = {
            'original_sequences': sequence_counts.height,
            'filtered_sequences': sequences_to_keep.height,
            'removed_sequences': sequence_counts.height - sequences_to_keep.height,
            'original_records': len(df),
            'filtered_records': 0,
            'removed_records': 0
        }
        
        if len(sequences_to_keep) == 0:
            logger.debug("Warning: No sequences meet the minimum length requirement!")
            return df.clear(), filtering_stats
        
        # Optimized: Use window function for filtering instead of to_list() and is_in()
        filtered_df = df.with_columns(
            pl.col(seq_id_col).count().over(seq_id_col).alias("_seq_len")
        ).filter(pl.col("_seq_len") >= self.min_sequence_len).drop("_seq_len")
        
        filtering_stats['filtered_records'] = len(filtered_df)
        filtering_stats['removed_records'] = len(df) - len(filtered_df)
        
        logger.debug(f"Original sequences: {filtering_stats['original_sequences']}")
        logger.debug(f"Sequences after filtering: {filtering_stats['filtered_sequences']}")
        logger.debug(f"Sequences removed: {filtering_stats['removed_sequences']}")
        logger.debug(f"Original records: {filtering_stats['original_records']:,}")
        logger.debug(f"Records after filtering: {filtering_stats['filtered_records']:,}")
        logger.debug(f"Records removed: {filtering_stats['removed_records']:,}")
        
        if filtering_stats['removed_sequences'] > 0:
            removed_sequences = sequence_counts.filter(pl.col('len') < self.min_sequence_len)
            if len(removed_sequences) > 0:
                min_len_removed = removed_sequences['len'].min()
                max_len_removed = removed_sequences['len'].max()
                avg_len_removed = removed_sequences['len'].mean()
                logger.debug(f"Removed sequence lengths - Min: {min_len_removed}, Max: {max_len_removed}, Avg: {avg_len_removed:.1f}")
        
        return filtered_df, filtering_stats

    def filter_glucose_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Keep only rows with non-null glucose values and remove other event fields.
        """
        if df.height == 0:
            logger.debug("Empty dataframe - skipping glucose-only filtering")
            return df, {
                'original_records': 0,
                'glucose_only_enabled': self.glucose_only,
                'records_after_filtering': 0,
                'records_removed': 0,
                'fields_removed': []
            }

        logger.debug("Filtering to glucose-only data...")
        
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        event_type_col = StandardFieldNames.EVENT_TYPE
        fast_acting_col = StandardFieldNames.FAST_ACTING_INSULIN
        long_acting_col = StandardFieldNames.LONG_ACTING_INSULIN
        carb_col = StandardFieldNames.CARB_VALUE

        filtering_stats = {
            'original_records': len(df),
            'glucose_only_enabled': self.glucose_only,
            'records_after_filtering': 0,
            'records_removed': 0,
            'fields_removed': []
        }
        
        if not self.glucose_only:
            logger.debug("Glucose-only filtering is disabled - keeping all data")
            filtering_stats['records_after_filtering'] = len(df)
            return df, filtering_stats
        
        df_filtered = df.filter(pl.col(glucose_col).is_not_null())
        
        fields_to_remove = [event_type_col, fast_acting_col, long_acting_col, carb_col]
        existing_fields_to_remove = [field for field in fields_to_remove if field in df_filtered.columns]
        
        if existing_fields_to_remove:
            df_filtered = df_filtered.drop(existing_fields_to_remove)
            filtering_stats['fields_removed'] = existing_fields_to_remove
        
        filtering_stats['records_after_filtering'] = len(df_filtered)
        filtering_stats['records_removed'] = len(df) - len(df_filtered)
        
        logger.debug(f"Original records: {filtering_stats['original_records']:,}")
        logger.debug(f"Records with glucose values: {filtering_stats['records_after_filtering']:,}")
        logger.debug(f"Records removed (no glucose): {filtering_stats['records_removed']:,}")
        if filtering_stats['fields_removed']:
            logger.debug(f"Fields removed: {', '.join(filtering_stats['fields_removed'])}")
        logger.debug("OK: Glucose-only filtering complete")
        
        return df_filtered, filtering_stats


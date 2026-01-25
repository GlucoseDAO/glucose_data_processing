"""Gap detection and sequence creation logic."""

import polars as pl
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from processing.core.fields import StandardFieldNames

class GapDetector:
    """
    Detects time gaps and creates sequence IDs, marking calibration periods 
    and sequences for removal.
    """
    
    def __init__(
        self,
        small_gap_max_minutes: int,
        calibration_period_minutes: int,
        remove_after_calibration_hours: int
    ) -> None:
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        self.calibration_period_seconds = calibration_period_minutes * 60
        self.remove_after_calibration_hours = remove_after_calibration_hours
        
    def detect_gaps_and_sequences(
        self, 
        df: pl.DataFrame, 
        last_sequence_id: int = 0, 
        field_categories_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """
        Main entry point for gap detection and sequence creation.
        """
        if df.height == 0:
            logger.info("Empty dataframe - skipping gap detection")
            return df, {
                'total_sequences': 0,
                'gap_positions': 0,
                'total_gaps': 0,
                'sequence_lengths': {},
                'calibration_period_analysis': {
                    'calibration_periods_detected': 0,
                    'sequences_marked_for_removal': 0,
                    'total_records_marked_for_removal': 0
                }
            }, last_sequence_id

        logger.info("Detecting gaps and creating sequences...")
        
        ts_col = StandardFieldNames.TIMESTAMP
        user_id_col = StandardFieldNames.USER_ID
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        calibration_stats = {
            'calibration_periods_detected': 0,
            'sequences_marked_for_removal': 0,
            'total_records_marked_for_removal': 0
        }
        
        current_last_sequence_id = last_sequence_id
        
        if user_id_col in df.columns:
            logger.info("Processing multi-user data - creating sequences per user...")
            all_sequences: List[pl.DataFrame] = []
            
            for user_id in sorted(df[user_id_col].unique().to_list()):
                user_data = df.filter(pl.col(user_id_col) == user_id).sort(ts_col)
                user_sequences, user_calib_stats, current_last_sequence_id = self._create_sequences_for_user(
                    user_data, current_last_sequence_id, user_id, field_categories_dict
                )
                all_sequences.append(user_sequences)
                
                calibration_stats['calibration_periods_detected'] += int(user_calib_stats['calibration_periods_detected'])
                calibration_stats['total_records_marked_for_removal'] += int(user_calib_stats['total_records_marked_for_removal'])
            
            if all_sequences:
                df = pl.concat(all_sequences).sort([user_id_col, seq_id_col, ts_col])
            else:
                df = df.clear()
        else:
            df = df.sort(ts_col)
            df, calibration_stats, current_last_sequence_id = self._create_sequences_for_user(
                df, current_last_sequence_id, None, field_categories_dict
            )
        
        if len(df) > 0 and seq_id_col in df.columns:
            if user_id_col in df.columns:
                sequence_counts = df.group_by([user_id_col, seq_id_col]).len().sort([user_id_col, seq_id_col])
            else:
                sequence_counts = df.group_by([seq_id_col]).len().sort(seq_id_col)
            
            stats = {
                'total_sequences': df[seq_id_col].n_unique(),
                'gap_positions': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
                'total_gaps': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
                'sequence_lengths': dict(zip(sequence_counts[seq_id_col].to_list(), sequence_counts['len'].to_list())) if len(sequence_counts) > 0 else {},
                'calibration_period_analysis': calibration_stats
            }
        else:
            stats = {
                'total_sequences': 0,
                'gap_positions': 0,
                'total_gaps': 0,
                'sequence_lengths': {},
                'calibration_period_analysis': calibration_stats
            }
        
        logger.info(f"Created {stats['total_sequences']} sequences")
        logger.info(f"Found {stats['total_gaps']} gaps > {self.small_gap_max_seconds / 60} minutes")
        
        if calibration_stats['calibration_periods_detected'] > 0:
            logger.info(f"Detected {calibration_stats['calibration_periods_detected']} calibration periods")
            logger.info(f"Removed {calibration_stats['total_records_marked_for_removal']} records after calibration")
        
        columns_to_remove = ['time_diff_seconds', 'is_gap', 'is_calibration_gap', 'remove_due_to_calibration']
        df = df.drop([col for col in columns_to_remove if col in df.columns])
        
        return df, stats, current_last_sequence_id

    def _create_sequences_for_user(
        self, 
        user_df: pl.DataFrame, 
        last_sequence_id: int = 0, 
        user_id: Optional[str] = None, 
        field_categories_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        stats = {
            'calibration_periods_detected': 0,
            'total_records_marked_for_removal': 0
        }
        
        if len(user_df) == 0:
            return user_df, stats, last_sequence_id

        def _apply_gap_detection(df_to_process: pl.DataFrame) -> pl.DataFrame:
            """
            Internal helper to detect gaps based on major time gaps or glucose gaps.
            """
            # Calculate time differences between consecutive records
            res_df = df_to_process.with_columns([
                pl.col(ts_col).diff().dt.total_seconds().alias('time_diff_seconds')
            ])
            
            # 1. Identify major gaps (calibration gaps)
            res_df = res_df.with_columns([
                (pl.col('time_diff_seconds') > self.calibration_period_seconds).fill_null(False).alias('is_calibration_gap')
            ])
            
            # 2. Identify glucose gaps
            # We split only if GLUCOSE has a gap > small_gap_max_seconds
            is_gap_glucose = pl.lit(False)
            if glucose_col in res_df.columns:
                # Optimized vectorized glucose gap detection
                # 1. Get timestamps of non-null glucose
                ts_at_glucose = pl.when(pl.col(glucose_col).is_not_null()).then(pl.col(ts_col)).otherwise(None)
                
                # 2. Forward fill to get the timestamp of the last non-null glucose
                # We shift by 1 to compare with the PREVIOUS non-null glucose
                prev_glucose_ts = ts_at_glucose.shift(1).forward_fill()
                
                # 3. Mark the current row if it's non-null AND the gap to the last non-null is too large
                is_gap_glucose = (
                    pl.col(glucose_col).is_not_null() & 
                    ((pl.col(ts_col) - prev_glucose_ts).dt.total_seconds() > self.small_gap_max_seconds)
                ).fill_null(False)
            
            # Combine: split ONLY if glucose gap > small_gap_max_seconds
            # We ignore generic time gaps or gaps in other fields to avoid fragmentation.
            res_df = res_df.with_columns([
                is_gap_glucose.alias('is_gap')
            ])
            return res_df

        # Initial gap detection
        df = _apply_gap_detection(user_df)
        
        stats['calibration_periods_detected'] = int(df['is_calibration_gap'].sum())
        
        should_remove_calibration = field_categories_dict.get('remove_after_calibration', True) if field_categories_dict else True
        
        if stats['calibration_periods_detected'] > 0 and should_remove_calibration:
            calibration_indices = df.with_row_index().filter(pl.col('is_calibration_gap'))['index'].to_list()
            
            removal_windows: List[Tuple[datetime, datetime]] = []
            timestamps = df[ts_col].to_list()
            
            for idx in calibration_indices:
                start_removal = timestamps[idx]
                end_removal = start_removal + timedelta(hours=self.remove_after_calibration_hours)
                removal_windows.append((start_removal, end_removal))
            
            df = df.with_columns(pl.lit(True).alias('keep_record'))
            for start, end in removal_windows:
                df = df.with_columns(
                    pl.when((pl.col(ts_col) >= start) & (pl.col(ts_col) < end))
                    .then(False)
                    .otherwise(pl.col('keep_record'))
                    .alias('keep_record')
                )
                
            stats['total_records_marked_for_removal'] = int((~df['keep_record']).sum())
            df = df.filter(pl.col('keep_record')).drop('keep_record')
            
            # Re-detect gaps after calibration removal
            if len(df) > 0:
                df = _apply_gap_detection(df)
            
        if len(df) > 0:
            df = df.with_columns([
                pl.col('is_gap').cum_sum().alias('local_sequence_id')
            ])
            
            df = df.with_columns([
                (pl.col('local_sequence_id') + last_sequence_id + 1).alias(seq_id_col)
            ]).drop('local_sequence_id')
            
            max_sequence_id = df[seq_id_col].max()
            last_sequence_id = int(max_sequence_id) if max_sequence_id is not None else last_sequence_id
        
        return df, stats, last_sequence_id



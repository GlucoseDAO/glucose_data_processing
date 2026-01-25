"""Interpolation logic for filling small gaps."""

import polars as pl
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from processing.core.fields import StandardFieldNames, INTERPOLATED_EVENT_TYPE, INSERTED_EVENT_TYPE

class ValueInterpolator:
    """
    Interpolates missing values for small gaps within sequences.
    """
    
    def __init__(self, expected_interval_minutes: int, small_gap_max_minutes: int) -> None:
        self.expected_interval_minutes = expected_interval_minutes
        self.small_gap_max_minutes = small_gap_max_minutes
        self.small_gap_max_seconds = small_gap_max_minutes * 60

    def interpolate_missing_values(
        self, 
        df: pl.DataFrame, 
        field_categories_dict: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Main entry point for interpolation.
        """
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        if df.height == 0 or seq_id_col not in df.columns:
            logger.info("Empty dataframe or missing sequence_id - skipping interpolation")
            return df, {
                'total_interpolations': 0,
                'total_interpolated_data_points': 0,
                'sequences_processed': 0,
                'small_gaps_filled': 0,
                'large_gaps_skipped': 0
            }

        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        event_type_col = StandardFieldNames.EVENT_TYPE
        user_id_col = StandardFieldNames.USER_ID
        if field_categories_dict is None:
            field_categories_dict = {
                'continuous': [glucose_col],
                'occasional': [],
                'service': []
            }
        
        continuous_fields = field_categories_dict.get('continuous', [])
        if glucose_col in df.columns and glucose_col not in continuous_fields:
            continuous_fields.append(glucose_col)
        
        fields_to_interpolate = [f for f in continuous_fields if f in df.columns]
        fill_during_interp_fields = [f for f in field_categories_dict.get('fill_during_interpolation', []) if f in df.columns]
        
        if not fields_to_interpolate and not fill_during_interp_fields:
            logger.info("No fields to interpolate or fill found - skipping interpolation")
            return df, {
                'total_interpolations': 0,
                'total_interpolated_data_points': 0,
                'sequences_processed': 0,
                'small_gaps_filled': 0,
                'large_gaps_skipped': 0
            }
        
        if fields_to_interpolate:
            logger.info(f"Interpolating small gaps for continuous fields: {', '.join(fields_to_interpolate)}...")
        if fill_during_interp_fields:
            logger.info(f"Filling constant values for fields: {', '.join(fill_during_interp_fields)}...")
        
        field_safe_names: Dict[str, str] = {}
        field_stats_keys: Dict[str, str] = {}
        for field in fields_to_interpolate:
            safe_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            field_safe_names[field] = safe_name
            field_stats_keys[field] = safe_name.lower()
        
        interpolation_stats: Dict[str, Any] = {
            'total_interpolations': 0,
            'total_interpolated_data_points': 0,
            'sequences_processed': 0,
            'small_gaps_filled': 0,
            'large_gaps_skipped': 0
        }
        
        for field in fields_to_interpolate:
            interpolation_stats[f'{field_stats_keys[field]}_interpolations'] = 0
        
        interpolation_stats['sequences_processed'] = df[seq_id_col].n_unique()
        per_field_interpolations = {field: 0 for field in fields_to_interpolate}
        
        # Step 0: Fill fields that should be constant during interpolation if values on both sides match
        for field in fill_during_interp_fields:
            # Fill nulls if value before and after are the same within a small gap
            ts_at_non_null = pl.when(pl.col(field).is_not_null()).then(pl.col(ts_col)).otherwise(None)
            gap_size = (
                ts_at_non_null.backward_fill().over(seq_id_col) - 
                ts_at_non_null.forward_fill().over(seq_id_col)
            ).dt.total_seconds()
            is_small_gap = (gap_size > 0) & (gap_size <= self.small_gap_max_seconds)
            
            v_prev = pl.col(field).forward_fill().over(seq_id_col)
            v_next = pl.col(field).backward_fill().over(seq_id_col)
            
            # We fill only if v_prev == v_next and they are NOT null
            df = df.with_columns(
                pl.when(pl.col(field).is_null() & is_small_gap & (v_prev == v_next) & v_prev.is_not_null())
                .then(v_prev)
                .otherwise(pl.col(field))
                .alias(field)
            )

        # Step 1: For each continuous field, fill missing values at existing timestamps
        # This fills nulls in continuous fields where other modalities might have data
        for field in fields_to_interpolate:
            # Vectorized interpolation restricted by gap size
            ts_at_non_null = pl.when(pl.col(field).is_not_null()).then(pl.col(ts_col)).otherwise(None)
            
            # Calculate gap size for each null point by looking at nearest non-null neighbors
            gap_size = (
                ts_at_non_null.backward_fill().over(seq_id_col) - 
                ts_at_non_null.forward_fill().over(seq_id_col)
            ).dt.total_seconds()
            
            # Mask for small gaps where we want to interpolate
            is_small_gap = (gap_size > 0) & (gap_size <= self.small_gap_max_seconds)
            
            # Linear interpolation
            interpolated_values = pl.col(field).interpolate().over(seq_id_col)
            
            # We want to mark rows as interpolated only if they were null, within a small gap,
            # and the interpolation actually produced a non-null value.
            # CRITICAL: We evaluate this before updating the column to avoid null_mask becoming false.
            will_be_filled = pl.col(field).is_null() & is_small_gap & (interpolated_values.is_not_null())
            
            # Update statistics using the condition before we update the dataframe
            interpolated_count = df.select(will_be_filled.sum()).item()
            per_field_interpolations[field] = int(interpolated_count) if interpolated_count is not None else 0

            update_exprs = [
                pl.when(pl.col(field).is_null() & is_small_gap)
                .then(interpolated_values)
                .otherwise(pl.col(field))
                .alias(field)
            ]
            
            # Update event type for interpolated values if column exists
            if event_type_col in df.columns:
                update_exprs.append(
                    pl.when(will_be_filled)
                    .then(pl.lit(INTERPOLATED_EVENT_TYPE))
                    .otherwise(pl.col(event_type_col))
                    .alias(event_type_col)
                )
            
            df = df.with_columns(update_exprs)
        
        for field in fields_to_interpolate:
            stats_key = field_stats_keys[field]
            interpolation_stats[f'{stats_key}_interpolations'] = per_field_interpolations[field]
        
        # Step 2: Process timestamp-based gaps
        df_with_diffs = df.with_row_index('row_idx').with_columns([
            (pl.col(ts_col).diff().over(seq_id_col).dt.total_seconds() / 60.0).alias('time_diff_minutes')
        ])
        
        df_with_gaps = df_with_diffs.with_columns([
            (
                (pl.col('time_diff_minutes') > self.expected_interval_minutes) &
                (pl.col('time_diff_minutes') <= self.small_gap_max_minutes)
            ).alias('is_small_gap'),
            (pl.col('time_diff_minutes') > self.small_gap_max_minutes).alias('is_large_gap')
        ])
        
        small_gaps_df = df_with_gaps.filter(pl.col('is_small_gap'))
        large_gaps_df = df_with_gaps.filter(pl.col('is_large_gap'))
        
        timestamp_based_gaps = small_gaps_df.height
        interpolation_stats['small_gaps_filled'] = timestamp_based_gaps
        interpolation_stats['large_gaps_skipped'] = large_gaps_df.height
        
        total_interpolations = sum(per_field_interpolations.values())
        interpolation_stats['total_interpolations'] = total_interpolations
        
        if small_gaps_df.height > 0:
            prev_cols = [
                pl.col(ts_col).shift(1).over(seq_id_col).alias('prev_timestamp')
            ]
            
            for field in fields_to_interpolate:
                safe_name = field_safe_names[field]
                prev_cols.append(pl.col(field).shift(1).over(seq_id_col).alias(f'prev_{safe_name}'))
            
            # Use safe names for fill_during_interp fields too
            fill_field_safe_names: Dict[str, str] = {}
            for field in fill_during_interp_fields:
                safe_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") + "_fill"
                fill_field_safe_names[field] = safe_name
                prev_cols.append(pl.col(field).shift(1).over(seq_id_col).alias(f'prev_{safe_name}'))
                # We also need the current value to check if it matches
                prev_cols.append(pl.col(field).alias(f'curr_{safe_name}'))

            if user_id_col in df_with_gaps.columns and user_id_col not in fill_during_interp_fields:
                prev_cols.append(pl.col(user_id_col).shift(1).over(seq_id_col).alias('prev_user_id'))
            
            df_with_prev = df_with_gaps.with_columns(prev_cols)
            
            gaps_to_process = df_with_prev.filter(pl.col('is_small_gap')).with_columns([
                ((pl.col('time_diff_minutes') / self.expected_interval_minutes).cast(pl.Int64) - 1)
                .alias('missing_points')
            ]).filter(pl.col('missing_points') > 0)
            
            if gaps_to_process.height > 0:
                # Optimized: Use pl.int_ranges instead of map_elements
                gaps_with_j = gaps_to_process.with_columns([
                    pl.int_ranges(1, pl.col('missing_points') + 1).alias('j_values')
                ])
                
                gaps_exploded = gaps_with_j.explode('j_values').with_columns([
                    pl.col('j_values').alias('j')
                ])
                
                interpolated_cols = [
                    (pl.col('prev_timestamp') + 
                     pl.duration(minutes=pl.col('j') * self.expected_interval_minutes))
                    .alias(ts_col),
                    ((pl.col('j').cast(pl.Float64) * self.expected_interval_minutes) / 
                     pl.col('time_diff_minutes').cast(pl.Float64))
                    .alias('alpha'),
                    pl.col(seq_id_col),
                ]
                
                for field in fields_to_interpolate:
                    safe_name = field_safe_names[field]
                    interpolated_cols.extend([
                        pl.col(f'prev_{safe_name}'),
                        pl.col(field).alias(f'curr_{safe_name}'),
                    ])
                
                for field in fill_during_interp_fields:
                    safe_name = fill_field_safe_names[field]
                    interpolated_cols.extend([
                        pl.col(f'prev_{safe_name}'),
                        pl.col(f'curr_{safe_name}'),
                    ])
                
                if 'prev_user_id' in gaps_exploded.columns and user_id_col not in fill_during_interp_fields:
                    interpolated_cols.append(pl.col('prev_user_id'))
                
                gaps_calculated = gaps_exploded.select(interpolated_cols)
                
                cast_exprs = []
                for field in fields_to_interpolate:
                    safe_name = field_safe_names[field]
                    cast_exprs.extend([
                        pl.col(f'prev_{safe_name}').cast(pl.Float64, strict=False).alias(f'prev_{safe_name}_num'),
                        pl.col(f'curr_{safe_name}').cast(pl.Float64, strict=False).alias(f'curr_{safe_name}_num'),
                    ])
                
                gaps_calculated = gaps_calculated.with_columns(cast_exprs)
                
                interpolated_field_exprs = []
                for field in fields_to_interpolate:
                    safe_name = field_safe_names[field]
                    prev_col_num = f'prev_{safe_name}_num'
                    curr_col_num = f'curr_{safe_name}_num'
                    
                    interpolated_field_exprs.append(
                        pl.when(
                            (pl.col(prev_col_num).is_not_null()) & 
                            (pl.col(curr_col_num).is_not_null())
                        ).then(
                            pl.col(prev_col_num) + 
                            pl.col('alpha') * (pl.col(curr_col_num) - pl.col(prev_col_num))
                        ).otherwise(None).alias(field)
                    )
                
                interpolated_df = gaps_calculated.with_columns(interpolated_field_exprs)
                
                final_cols = [
                    pl.col(ts_col),
                    pl.col(seq_id_col),
                ]
                
                for field in fields_to_interpolate:
                    final_cols.append(pl.col(field))
                
                if event_type_col in df.columns:
                    final_cols.append(pl.lit(INTERPOLATED_EVENT_TYPE).alias(event_type_col))
                
                for field in fill_during_interp_fields:
                    safe_name = fill_field_safe_names[field]
                    prev_col = f'prev_{safe_name}'
                    curr_col = f'curr_{safe_name}'
                    final_cols.append(
                        pl.when((pl.col(prev_col) == pl.col(curr_col)) & pl.col(prev_col).is_not_null())
                        .then(pl.col(prev_col))
                        .otherwise(None)
                        .alias(field)
                    )

                if 'prev_user_id' in gaps_calculated.columns and user_id_col not in fill_during_interp_fields:
                    final_cols.append(
                        pl.when(pl.col('prev_user_id').is_not_null())
                        .then(pl.col('prev_user_id'))
                        .otherwise(pl.lit(''))
                        .alias(user_id_col)
                    )
                
                original_schema = df.schema
                existing_col_names = [ts_col, seq_id_col] + fields_to_interpolate + fill_during_interp_fields
                if event_type_col in df.columns:
                    existing_col_names.append(event_type_col)
                if 'prev_user_id' in gaps_calculated.columns:
                    existing_col_names.append(user_id_col)
                
                for col in df.columns:
                    if col not in existing_col_names:
                        col_type = original_schema[col]
                        is_occasional = col in field_categories_dict.get('occasional', [])
                        is_service = col in field_categories_dict.get('service', [])
                        
                        if is_occasional or is_service:
                            if col_type == pl.Utf8 or col_type == pl.String:
                                final_cols.append(pl.lit('').cast(col_type).alias(col))
                            else:
                                final_cols.append(pl.lit(None).cast(col_type).alias(col))
                        else:
                            if col_type == pl.Utf8 or col_type == pl.String:
                                final_cols.append(pl.lit('').cast(col_type).alias(col))
                            else:
                                final_cols.append(pl.lit(None).cast(col_type).alias(col))
                
                interpolated_df = interpolated_df.select(final_cols)
                interpolated_df = interpolated_df.select(df.columns)
                
                interpolation_stats['total_interpolated_data_points'] = len(interpolated_df)
                
                timestamp_based_interpolations = 0
                for field in fields_to_interpolate:
                    stats_key = field_stats_keys[field]
                    field_interpolations = interpolated_df.filter(pl.col(field).is_not_null()).height
                    interpolation_stats[f'{stats_key}_interpolations'] = interpolation_stats.get(f'{stats_key}_interpolations', 0) + field_interpolations
                    timestamp_based_interpolations += field_interpolations
                
                interpolation_stats['total_interpolations'] = interpolation_stats.get('total_interpolations', 0) + timestamp_based_interpolations
                
                df = pl.concat([df, interpolated_df], how='vertical_relaxed')
        
        if user_id_col in df.columns:
            df = df.sort([user_id_col, seq_id_col, ts_col])
        else:
            df = df.sort([seq_id_col, ts_col])
            
        total_rows = len(df)
        if total_rows > 0:
            for field in fields_to_interpolate:
                stats_key = field_stats_keys[field]
                count = interpolation_stats.get(f'{stats_key}_interpolations', 0)
                percentage = (count / total_rows) * 100
                interpolation_stats[f'{stats_key}_interpolations_pct'] = round(percentage, 2)
        
        logger.info(f"Identified and processed {interpolation_stats['small_gaps_filled']} small gaps")
        logger.info(f"Created {interpolation_stats['total_interpolated_data_points']} interpolated (inserted) data points")
        
        for field in fields_to_interpolate:
            count = interpolation_stats.get(f'{field_stats_keys[field]}_interpolations', 0)
            pct = interpolation_stats.get(f'{field_stats_keys[field]}_interpolations_pct', 0)
            logger.info(f"Interpolated {field}: {count:,} ({pct}%)")
            
        logger.info(f"Skipped {interpolation_stats['large_gaps_skipped']} large gaps")
        logger.info(f"Processed {interpolation_stats['sequences_processed']} sequences")
        
        return df, interpolation_stats


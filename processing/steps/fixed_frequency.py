"""Logic for creating fixed-frequency data with consistent intervals."""

import polars as pl
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from processing.core.fields import StandardFieldNames

class FixedFreqGenerator:
    """
    Creates fixed-frequency data by aligning sequences to round minutes 
    and ensuring consistent intervals.
    """
    
    def __init__(self, expected_interval_minutes: int) -> None:
        self.expected_interval_minutes = expected_interval_minutes

    def create_fixed_frequency_data(
        self, 
        df: pl.DataFrame, 
        field_categories_dict: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Main entry point for fixed-frequency creation.
        """
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        if df.height == 0 or seq_id_col not in df.columns:
            logger.info("Empty dataframe or missing sequence_id - skipping fixed-frequency creation")
            return df, {
                'sequences_processed': 0,
                'total_records_before': 0,
                'total_records_after': 0,
                'glucose_interpolations': 0,
                'carb_shifted_records': 0,
                'insulin_shifted_records': 0,
                'time_adjustments': 0,
                'data_density_before': {
                    'avg_points_per_interval': 0.0,
                    'total_intervals': 0,
                    'total_points': 0
                },
                'data_density_after': {},
                'density_change_explanation': {}
            }

        logger.info(f"Creating fixed-frequency data with {self.expected_interval_minutes}-minute intervals...")
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        before_density_stats = self._calculate_data_density(df, self.expected_interval_minutes)
        
        fixed_freq_stats: Dict[str, Any] = {
            'sequences_processed': 0,
            'total_records_before': len(df),
            'total_records_after': 0,
            'glucose_interpolations': 0,
            'carb_shifted_records': 0,
            'insulin_shifted_records': 0,
            'time_adjustments': 0,
            'data_density_before': before_density_stats,
            'data_density_after': {},
            'density_change_explanation': {}
        }
        
        unique_sequences = df[seq_id_col].unique().to_list()
        all_fixed_sequences: List[pl.DataFrame] = []
        
        for seq_id in unique_sequences:
            seq_data = df.filter(pl.col(seq_id_col) == seq_id).sort(ts_col)
            
            if len(seq_data) < 2:
                all_fixed_sequences.append(seq_data.select(df.columns).cast(df.schema))
                continue
                
            fixed_freq_stats['sequences_processed'] += 1
            
            fixed_seq_data = self._create_fixed_frequency_sequence(seq_data, seq_id, fixed_freq_stats, field_categories_dict)
            fixed_seq_data = fixed_seq_data.select(df.columns).cast(df.schema)
            all_fixed_sequences.append(fixed_seq_data)
        
        if all_fixed_sequences:
            df_fixed = pl.concat(all_fixed_sequences).sort([seq_id_col, ts_col])
        else:
            df_fixed = df
        
        fixed_freq_stats['total_records_after'] = len(df_fixed)
        after_density_stats = self._calculate_data_density(df_fixed, self.expected_interval_minutes)
        fixed_freq_stats['data_density_after'] = after_density_stats
        
        density_change_explanation = self._calculate_density_change_explanation(
            fixed_freq_stats['data_density_before'],
            fixed_freq_stats['data_density_after'],
            fixed_freq_stats['total_records_before'],
            fixed_freq_stats['total_records_after']
        )
        fixed_freq_stats['density_change_explanation'] = density_change_explanation
        
        logger.info(f"Processed {fixed_freq_stats['sequences_processed']} sequences")
        logger.info(f"Time adjustments made: {fixed_freq_stats['time_adjustments']}")
        logger.info(f"Glucose interpolations: {fixed_freq_stats['glucose_interpolations']}")
        logger.info(f"Insulin records shifted: {fixed_freq_stats['insulin_shifted_records']}")
        logger.info(f"Carb records shifted: {fixed_freq_stats['carb_shifted_records']}")
        logger.info(f"Records before: {fixed_freq_stats['total_records_before']:,}")
        logger.info(f"Records after: {fixed_freq_stats['total_records_after']:,}")
        
        before_density = fixed_freq_stats['data_density_before']
        after_density = fixed_freq_stats['data_density_after']
        explanation = fixed_freq_stats['density_change_explanation']
        
        logger.info(f"Data density: {before_density['avg_points_per_interval']:.2f} -> {after_density['avg_points_per_interval']:.2f} points/interval ({explanation.get('density_change_pct', 0):+.1f}%)")
        logger.info(f"Change explained by density: {explanation.get('explained_pct', 0):.1f}%")
        
        logger.info("Fixed-frequency data creation complete")
        
        return df_fixed, fixed_freq_stats

    def _calculate_data_density(self, df: pl.DataFrame, interval_minutes: int) -> Dict[str, Any]:
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        if len(df) == 0:
            return {
                'avg_points_per_interval': 0.0,
                'total_intervals': 0,
                'total_points': 0
            }
        
        interval_seconds = interval_minutes * 60
        total_points = 0
        total_intervals = 0
        
        for seq_id in df[seq_id_col].unique().to_list():
            seq_data = df.filter(pl.col(seq_id_col) == seq_id).sort(ts_col)
            
            if len(seq_data) < 2:
                total_points += 1
                total_intervals += 1
                continue
            
            first_ts = seq_data[ts_col].min()
            last_ts = seq_data[ts_col].max()
            if first_ts is None or last_ts is None:
                continue
            duration_seconds = (last_ts - first_ts).total_seconds()
            
            num_intervals = max(1, int(duration_seconds / interval_seconds) + 1)
            num_points = len(seq_data)
            
            total_points += num_points
            total_intervals += num_intervals
        
        return {
            'avg_points_per_interval': total_points / total_intervals if total_intervals > 0 else 0.0,
            'total_intervals': total_intervals,
            'total_points': total_points
        }

    def _calculate_density_change_explanation(
        self, 
        before_density: Dict[str, Any], 
        after_density: Dict[str, Any], 
        records_before: int, 
        records_after: int
    ) -> Dict[str, Any]:
        if before_density['avg_points_per_interval'] == 0:
            return {
                'records_change_pct': 0.0,
                'expected_change_pct': 0.0,
                'explained_pct': 0.0,
                'unexplained_pct': 0.0,
                'expected_records_after': records_before,
                'unexplained_records': 0
            }
        
        records_change_pct = ((records_after - records_before) / records_before * 100) if records_before > 0 else 0.0
        density_change_pct = ((after_density['avg_points_per_interval'] - before_density['avg_points_per_interval']) / 
                              before_density['avg_points_per_interval'] * 100) if before_density['avg_points_per_interval'] > 0 else 0.0
        
        density_ratio = after_density['avg_points_per_interval'] / before_density['avg_points_per_interval'] if before_density['avg_points_per_interval'] > 0 else 1.0
        expected_records_after = int(records_before * density_ratio)
        expected_change_pct = (density_ratio - 1) * 100
        
        actual_change = records_after - records_before
        expected_change = expected_records_after - records_before
        unexplained_change = actual_change - expected_change
        
        explained_pct = (abs(expected_change) / abs(actual_change) * 100) if actual_change != 0 else 100.0
        unexplained_pct = 100.0 - explained_pct if actual_change != 0 else 0.0
        
        return {
            'records_change_pct': records_change_pct,
            'expected_change_pct': expected_change_pct,
            'explained_pct': explained_pct,
            'unexplained_pct': unexplained_pct,
            'expected_records_after': expected_records_after,
            'unexplained_records': unexplained_change,
            'density_change_pct': density_change_pct
        }

    def _create_fixed_frequency_sequence(
        self, 
        seq_data: pl.DataFrame, 
        seq_id: int, 
        stats: Dict[str, Any], 
        field_categories_dict: Optional[Dict[str, List[str]]] = None
    ) -> pl.DataFrame:
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        event_type_col = StandardFieldNames.EVENT_TYPE
        user_id_col = StandardFieldNames.USER_ID
        first_timestamp = seq_data[ts_col].min()
        last_timestamp = seq_data[ts_col].max()
        if first_timestamp is None or last_timestamp is None:
            return seq_data

        # Align the fixed-frequency grid to the dominant CGM seconds offset (important for Medtronic).
        #
        # Many datasets have glucose timestamps with a stable seconds offset (e.g. :16) while
        # event rows (insulin/carb) can be logged at different seconds (e.g. :18).
        # If we anchor the grid to the first row in the sequence, we can get a constant phase shift
        # and the resampled/interpolated glucose values will look mismatched vs raw CGM.
        #
        # Strategy:
        # - Prefer anchoring to timestamps where we have CGM glucose values (event_type == "CGM")
        # - Fall back to any glucose values
        # - Fall back to the sequence start
        anchor_timestamp = first_timestamp
        anchor_seconds = 0
        try:
            if glucose_col in seq_data.columns:
                glucose_points = seq_data.select([ts_col, glucose_col] + ([event_type_col] if event_type_col in seq_data.columns else [])).with_columns(
                    pl.col(glucose_col).cast(pl.Float64, strict=False).alias("_glucose_numeric")
                ).filter(pl.col("_glucose_numeric").is_not_null())

                if event_type_col in glucose_points.columns:
                    cgm_points = glucose_points.filter(pl.col(event_type_col) == "CGM")
                    if len(cgm_points) > 0:
                        anchor_timestamp = cgm_points[ts_col].min()
                        points_for_seconds = cgm_points
                    elif len(glucose_points) > 0:
                        anchor_timestamp = glucose_points[ts_col].min()
                        points_for_seconds = glucose_points
                elif len(glucose_points) > 0:
                    anchor_timestamp = glucose_points[ts_col].min()
                    points_for_seconds = glucose_points

                # Preserve a non-zero seconds offset only when it's stable (dominant across points).
                # Otherwise keep the classic "round minute" grid (seconds == 0), which tests assume.
                if 'points_for_seconds' in locals() and len(points_for_seconds) > 0:
                    sec_counts = (
                        points_for_seconds
                        .select(pl.col(ts_col).dt.second().alias("sec"))
                        .group_by("sec")
                        .len()
                        .sort("len", descending=True)
                    )
                    if sec_counts.height > 0:
                        top_sec, top_len = sec_counts.row(0)
                        try:
                            top_sec_int = int(top_sec)
                            top_len_int = int(top_len)
                        except Exception:
                            top_sec_int = 0
                            top_len_int = 0
                        if top_sec_int != 0 and top_len_int / max(1, len(points_for_seconds)) >= 0.80:
                            anchor_seconds = top_sec_int
        except Exception:
            anchor_timestamp = first_timestamp

        if anchor_timestamp is None:
            anchor_timestamp = first_timestamp

        # Align start to a "round minute" boundary (seconds == 0) in the normalized space.
        # If we detected a stable CGM seconds offset, we round relative to that offset so the grid
        # lands on real CGM timestamps (e.g. hh:mm:53).
        target_second = anchor_seconds
        normalized = anchor_timestamp.replace(microsecond=0) - timedelta(seconds=target_second)
        if normalized.second >= 30:
            normalized = normalized + timedelta(seconds=(60 - normalized.second))
        else:
            normalized = normalized - timedelta(seconds=normalized.second)
        normalized = normalized.replace(microsecond=0)
        aligned_start = (normalized + timedelta(seconds=target_second)).replace(second=target_second, microsecond=0)
        
        total_duration = (last_timestamp - aligned_start).total_seconds()
        num_intervals = int(total_duration / (self.expected_interval_minutes * 60)) + 1
        
        fixed_timestamps_list = [
            aligned_start + timedelta(minutes=i * self.expected_interval_minutes)
            for i in range(num_intervals)
            if aligned_start + timedelta(minutes=i * self.expected_interval_minutes) <= last_timestamp
        ]
        
        fixed_timestamps = pl.DataFrame({
            ts_col: fixed_timestamps_list,
            seq_id_col: [seq_id] * len(fixed_timestamps_list)
        })
        
        service_fields: List[str] = field_categories_dict.get('service', []).copy() if field_categories_dict else []
        if field_categories_dict is None:
            continuous_fields = [glucose_col] if glucose_col in seq_data.columns else []
        else:
            continuous_fields = [f for f in field_categories_dict.get('continuous', []) if f in seq_data.columns]
            if glucose_col in seq_data.columns and glucose_col not in continuous_fields:
                continuous_fields.append(glucose_col)
        
        result_df = self._interpolate_continuous_fields_linear(fixed_timestamps, seq_data, stats, continuous_fields)
        
        occasional_fields: List[str] = []
        if field_categories_dict is not None:
            occasional_fields = [f for f in field_categories_dict.get('occasional', []) if f in seq_data.columns]
        
        event_cols: List[str] = []
        seen_cols = set()
        result_existing = set(result_df.columns)
        for col in occasional_fields + service_fields + [event_type_col, user_id_col]:
            if col in result_existing:
                continue
            if col in seq_data.columns and col not in continuous_fields and col not in seen_cols:
                event_cols.append(col)
                seen_cols.add(col)
        
        if event_cols:
            numeric_hint: Optional[set[str]] = set(occasional_fields) if occasional_fields else None
            events_df = self._shift_events_rounding(
                seq_data,
                event_cols,
                stats,
                fixed_timestamps_list,
                numeric_cols_hint=numeric_hint,
            )
            result_df = result_df.join(events_df, on=ts_col, how='left')
        
        original_cols = set(seq_data.columns)
        result_cols = set(result_df.columns)
        missing_cols = original_cols - result_cols
        
        if missing_cols:
            for col in missing_cols:
                col_type = seq_data.schema[col]
                if col_type == pl.Utf8 or col_type == pl.String:
                    result_df = result_df.with_columns([pl.lit('').cast(col_type).alias(col)])
                else:
                    result_df = result_df.with_columns([pl.lit(None).cast(col_type).alias(col)])
        
        result_df = result_df.select(seq_data.columns)
        
        return result_df

    def _interpolate_continuous_fields_linear(
        self, 
        fixed_timestamps: pl.DataFrame, 
        seq_data: pl.DataFrame, 
        stats: Dict[str, Any], 
        continuous_fields: List[str]
    ) -> pl.DataFrame:
        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        if not continuous_fields:
            return fixed_timestamps
        
        result_df = fixed_timestamps
        
        for field in continuous_fields:
            if field not in seq_data.columns:
                result_df = result_df.with_columns([pl.lit(None).cast(pl.Float64, strict=False).alias(field)])
                continue
            
            seq_data_ts = seq_data.select([ts_col, field]).with_columns([
                pl.col(ts_col).alias('ts_orig'),
                pl.col(field).cast(pl.Float64, strict=False)
            ]).filter(pl.col(field).is_not_null())
            
            if len(seq_data_ts) == 0:
                result_df = result_df.with_columns([pl.lit(None).cast(pl.Float64, strict=False).alias(field)])
                continue
            
            safe_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            
            forward = result_df.select([ts_col]).join_asof(
                seq_data_ts, 
                on=ts_col, 
                strategy='forward'
            ).rename({field: f'{safe_name}_next', 'ts_orig': f'ts_{safe_name}_next'})
            
            backward = result_df.select([ts_col]).join_asof(
                seq_data_ts, 
                on=ts_col, 
                strategy='backward'
            ).select([ts_col, field, 'ts_orig']).rename({field: f'{safe_name}_prev', 'ts_orig': f'ts_{safe_name}_prev'})
            
            combined = forward.join(backward.select([ts_col, f'{safe_name}_prev', f'ts_{safe_name}_prev']), on=ts_col, how='left')
            
            combined = combined.with_columns([
                pl.when(pl.col(f'ts_{safe_name}_prev') == pl.col(f'ts_{safe_name}_next')) # Exact match
                .then(pl.col(f'{safe_name}_prev'))
                .when(pl.col(f'ts_{safe_name}_prev').is_null()) # No prev
                .then(pl.col(f'{safe_name}_next'))
                .when(pl.col(f'ts_{safe_name}_next').is_null()) # No next
                .then(pl.col(f'{safe_name}_prev'))
                .otherwise(
                    pl.col(f'{safe_name}_prev') + (
                        (pl.col(f'{safe_name}_next') - pl.col(f'{safe_name}_prev')) * 
                        (pl.col(ts_col) - pl.col(f'ts_{safe_name}_prev')).dt.total_seconds() / 
                        (pl.col(f'ts_{safe_name}_next') - pl.col(f'ts_{safe_name}_prev')).dt.total_seconds()
                    )
                ).alias(field)
            ])
            
            # Merge field into result_df
            result_df = result_df.join(
                combined.select([ts_col, field]),
                on=ts_col,
                how='left'
            )
            
            if field == glucose_col:
                interpolated_count = combined.filter(
                    pl.col(field).is_not_null() & 
                    (pl.col(f'ts_{safe_name}_next') != pl.col(ts_col))
                ).height
                stats['glucose_interpolations'] += interpolated_count
        
        return result_df

    def _shift_events_rounding(
        self,
        seq_data: pl.DataFrame,
        cols: List[str],
        stats: Dict[str, Any],
        fixed_timestamps_list: List[datetime],
        numeric_cols_hint: Optional[set[str]] = None,
    ) -> pl.DataFrame:
        ts_col = StandardFieldNames.TIMESTAMP
        user_id_col = StandardFieldNames.USER_ID
        event_type_col = StandardFieldNames.EVENT_TYPE

        if len(fixed_timestamps_list) == 0:
            return pl.DataFrame({ts_col: []})
        
        events = seq_data.select([ts_col] + cols)
        events = events.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols]))
        
        if len(events) == 0:
            return pl.DataFrame({ts_col: fixed_timestamps_list[:0]})
        
        fixed_timestamps_df = pl.DataFrame({
            'fixed_timestamp': fixed_timestamps_list
        }).sort('fixed_timestamp')
        
        events_shifted = events.join_asof(
            fixed_timestamps_df,
            left_on=ts_col,
            right_on='fixed_timestamp',
            strategy='nearest'
        ).with_columns([
            pl.col('fixed_timestamp').alias(ts_col)
        ]).drop('fixed_timestamp')
        
        numeric_cols: List[str] = []
        if numeric_cols_hint is not None:
            numeric_cols = [
                c for c in cols
                if c in numeric_cols_hint and c not in {event_type_col, user_id_col} and not c.endswith('_id')
            ]
        else:
            for c in cols:
                if c in {event_type_col, user_id_col} or c.endswith('_id'):
                    continue
                try:
                    any_numeric = seq_data.select(
                        pl.col(c).cast(pl.Float64, strict=False).is_not_null().any()
                    ).item()
                except (ValueError, TypeError, AttributeError):
                    any_numeric = False
                if any_numeric:
                    numeric_cols.append(c)
        
        cast_exprs = []
        for col in cols:
            if col in numeric_cols:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
            else:
                cast_exprs.append(pl.col(col))
        
        if cast_exprs:
            events_shifted = events_shifted.with_columns(cast_exprs)
        
        if 'carb_shifted_records' not in stats:
            stats['carb_shifted_records'] = 0
        if 'insulin_shifted_records' not in stats:
            stats['insulin_shifted_records'] = 0
            
        for col in cols:
            if col in numeric_cols:
                stats['insulin_shifted_records'] += events_shifted.filter(pl.col(col).is_not_null()).height
        
        agg_exprs = []
        for col in cols:
            if col in numeric_cols:
                agg_exprs.append(
                    pl.when(pl.col(col).is_not_null().any())
                    .then(pl.col(col).sum())
                    .otherwise(None)
                    .alias(col)
                )
            else:
                agg_exprs.append(pl.col(col).first().alias(col))
        
        shifted = events_shifted.group_by(ts_col).agg(agg_exprs)

        stabilize_exprs = []
        for c in cols:
            if c not in shifted.columns:
                continue
            if c in numeric_cols:
                stabilize_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
            else:
                stabilize_exprs.append(pl.col(c).cast(pl.Utf8, strict=False).alias(c))
        
        if stabilize_exprs:
            shifted = shifted.with_columns(stabilize_exprs)
        
        return shifted


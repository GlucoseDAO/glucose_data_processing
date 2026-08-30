"""Statistics management for glucose data processing."""

import polars as pl
from typing import Dict, Any, List, Optional, Union, Sequence
from loguru import logger
from processing.core.fields import StandardFieldNames, INTERPOLATED_EVENT_TYPE, INSERTED_EVENT_TYPE
from formats.base_converter import CSVFormatConverter
from pathlib import Path

class StatsManager:
    """
    Generates and aggregates statistics about the processed data.
    """
    
    def __init__(self, original_record_count: int = 0) -> None:
        self.original_record_count = original_record_count

    def get_statistics(
        self, 
        df: pl.DataFrame, 
        gap_stats: Dict[str, Any], 
        interp_stats: Dict[str, Any], 
        filter_stats: Optional[Dict[str, Any]] = None, 
        glucose_filter_stats: Optional[Dict[str, Any]] = None, 
        fixed_freq_stats: Optional[Dict[str, Any]] = None,
        cleaning_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics.
        """
        # Mapping standard names to display names if available
        field_map = CSVFormatConverter.get_field_to_display_name_map()
        
        def get_col(std_name: str) -> str:
            # Check if standard name is in columns
            if std_name in df.columns:
                return std_name
            # Check if display name is in columns
            disp_name = field_map.get(std_name)
            if disp_name and disp_name in df.columns:
                return disp_name
            return std_name

        ts_col = get_col(StandardFieldNames.TIMESTAMP)
        seq_id_col = get_col(StandardFieldNames.SEQUENCE_ID)
        event_type_col = get_col(StandardFieldNames.EVENT_TYPE)
        
        # Service fields that we don't calculate completeness for (except glucose)
        service_fields = {
            ts_col, seq_id_col, event_type_col,
            get_col(StandardFieldNames.USER_ID),
            get_col(StandardFieldNames.DATASET_NAME)
        }

        date_range = {'start': 'N/A', 'end': 'N/A'}
        if ts_col in df.columns:
            ts_dtype = df.schema.get(ts_col)
            valid_timestamps = df.filter(pl.col(ts_col).is_not_null())
            if len(valid_timestamps) > 0:
                if ts_dtype == pl.Datetime:
                    timestamps = valid_timestamps[ts_col].dt.strftime('%Y-%m-%dT%H:%M:%S').sort()
                else:
                    timestamps = valid_timestamps[ts_col].cast(pl.Utf8, strict=False).sort()
                date_range = {'start': timestamps[0], 'end': timestamps[-1]}
        
        if seq_id_col in df.columns:
            sequence_counts = df.group_by(seq_id_col).len().sort(seq_id_col)
            seq_lens = sequence_counts['len']
            
            sequence_lengths_stats = {
                'count': len(seq_lens),
                'mean': seq_lens.mean() if not seq_lens.is_empty() else 0,
                'std': seq_lens.std() if not seq_lens.is_empty() else 0,
                'min': seq_lens.min() if not seq_lens.is_empty() else None,
                '25%': seq_lens.quantile(0.25) if not seq_lens.is_empty() else 0,
                '50%': seq_lens.median() if not seq_lens.is_empty() else 0,
                '75%': seq_lens.quantile(0.75) if not seq_lens.is_empty() else 0,
                'max': seq_lens.max() if not seq_lens.is_empty() else 0
            }
            
            if not seq_lens.is_empty():
                counts_df = seq_lens.value_counts().sort("len")
                sequences_by_length = dict(zip(counts_df["len"].to_list(), counts_df["count"].to_list()))
            else:
                sequences_by_length = {}
            
            all_lengths = seq_lens.to_list() if not seq_lens.is_empty() else []
            total_sequences = df[seq_id_col].n_unique()
        else:
            sequence_lengths_stats = {
                'count': 0, 'mean': 0, 'std': 0, 'min': 0, '25%': 0, '50%': 0, '75%': 0, 'max': 0
            }
            sequences_by_length = {}
            all_lengths = []
            total_sequences = 0

        stats = {
            'dataset_overview': {
                'total_records': len(df),
                'total_sequences': total_sequences,
                'date_range': date_range,
                'original_records': self.original_record_count if self.original_record_count > 0 else len(df)
            },
            'sequence_analysis': {
                'sequence_lengths': sequence_lengths_stats,
                'longest_sequence': sequence_lengths_stats['max'],
                'shortest_sequence': sequence_lengths_stats['min'],  # None when no sequences
                'sequences_by_length': sequences_by_length,
                'all_lengths': all_lengths
            },
            'gap_analysis': gap_stats,
            'cleaning_analysis': cleaning_stats if cleaning_stats else {},
            'interpolation_analysis': interp_stats,
            'calibration_removal_analysis': {},
            'filtering_analysis': filter_stats if filter_stats else {},
            'replacement_analysis': {},
            'fixed_frequency_analysis': fixed_freq_stats if fixed_freq_stats else {},
            'glucose_filtering_analysis': glucose_filter_stats if glucose_filter_stats else {},
            'data_quality': {}
        }
        
        # Calculate completeness for all non-service fields
        quality = {}
        if len(df) > 0:
            for col in df.columns:
                if col not in service_fields:
                    completeness = (1 - df[col].null_count() / len(df)) * 100
                    quality[f"{col}_data_completeness"] = completeness
        
        # Add special records counts if columns exist
        if event_type_col in df.columns:
            quality['interpolated_records'] = df.filter(pl.col(event_type_col) == INTERPOLATED_EVENT_TYPE).height
            quality['inserted_records'] = df.filter(pl.col(event_type_col) == INSERTED_EVENT_TYPE).height
        else:
            quality['interpolated_records'] = 0
            quality['inserted_records'] = 0

        stats['data_quality'] = quality
        
        return stats


    def aggregate_statistics(self, all_statistics: List[Dict[str, Any]], csv_folders: Sequence[Union[str, Path]]) -> Dict[str, Any]:
        """
        Aggregate statistics from multiple databases.
        """
        aggregated = {
            'multi_database_info': {
                'total_databases': len(all_statistics),
                'database_paths': [str(p) for p in csv_folders],
                'databases_processed': []
            },
            'dataset_overview': {
                'total_records': 0,
                'total_sequences': 0,
                'date_range': {'start': None, 'end': None},
                'original_records': 0
            },
            'sequence_analysis': {
                'sequence_lengths': {
                    'count': 0,
                    'mean': 0,
                    'std': 0,
                    'min': float('inf'),
                    '25%': 0,
                    '50%': 0,
                    '75%': 0,
                    'max': 0
                },
                'longest_sequence': 0,
                'shortest_sequence': float('inf'),
                'sequences_by_length': {}
            },
            'gap_analysis': {
                'total_sequences': 0,
                'gap_positions': 0,
                'total_gaps': 0,
                'sequence_lengths': {},
                'calibration_period_analysis': {
                    'calibration_periods_detected': 0,
                    'sequences_marked_for_removal': 0,
                    'total_records_marked_for_removal': 0
                }
            },
            'cleaning_analysis': {
                'removed_records': 0
            },
            'interpolation_analysis': {
                'total_interpolations': 0,
                'total_interpolated_data_points': 0,
                'sequences_processed': 0,
                'small_gaps_filled': 0,
                'large_gaps_skipped': 0
            },
            'calibration_removal_analysis': {},
            'filtering_analysis': {
                'original_sequences': 0,
                'filtered_sequences': 0,
                'removed_sequences': 0,
                'original_records': 0,
                'filtered_records': 0,
                'removed_records': 0
            },
            'replacement_analysis': {},
            'fixed_frequency_analysis': {
                'sequences_processed': 0,
                'total_records_before': 0,
                'total_records_after': 0,
                'time_adjustments': 0
            },
            'glucose_filtering_analysis': {},
            'data_quality': {}
        }
        
        all_sequence_lengths = []
        
        for idx, stats in enumerate(all_statistics):
            db_info = stats.get('database_info', {})
            # Ensure name and index are set
            db_info.setdefault('database_name', str(csv_folders[idx]) if idx < len(csv_folders) else f"Dataset {idx+1}")
            db_info.setdefault('database_index', idx + 1)
            
            # Extract sequence range if not present but available in stats
            if 'sequence_id_range' not in db_info:
                overview = stats.get('dataset_overview', {})
                # This might be tricky without the actual DF, but we can try to guess from sequence_analysis
                db_info['sequence_id_range'] = {'min': 'N/A', 'max': 'N/A'}

            aggregated['multi_database_info']['databases_processed'].append(db_info)
            
            overview = stats.get('dataset_overview', {})
            aggregated['dataset_overview']['total_records'] += overview.get('total_records', 0)
            aggregated['dataset_overview']['total_sequences'] += overview.get('total_sequences', 0)
            aggregated['dataset_overview']['original_records'] += overview.get('original_records', 0)
            
            date_range = overview.get('date_range', {})
            if date_range.get('start') and date_range['start'] != 'N/A':
                if aggregated['dataset_overview']['date_range']['start'] is None:
                    aggregated['dataset_overview']['date_range']['start'] = date_range['start']
                else:
                    aggregated['dataset_overview']['date_range']['start'] = min(
                        aggregated['dataset_overview']['date_range']['start'],
                        date_range['start']
                    )
            
            if date_range.get('end') and date_range['end'] != 'N/A':
                if aggregated['dataset_overview']['date_range']['end'] is None:
                    aggregated['dataset_overview']['date_range']['end'] = date_range['end']
                else:
                    aggregated['dataset_overview']['date_range']['end'] = max(
                        aggregated['dataset_overview']['date_range']['end'],
                        date_range['end']
                    )
            
            seq_analysis = stats.get('sequence_analysis', {})
            if 'all_lengths' in seq_analysis and seq_analysis['all_lengths']:
                all_sequence_lengths.extend(seq_analysis['all_lengths'])
            elif 'sequence_lengths' in seq_analysis and 'all_lengths' in seq_analysis['sequence_lengths']:
                all_sequence_lengths.extend(seq_analysis['sequence_lengths']['all_lengths'])
            elif 'sequence_lengths' in stats.get('gap_analysis', {}):
                sequence_lengths_dict = stats['gap_analysis']['sequence_lengths']
                if isinstance(sequence_lengths_dict, dict):
                    all_sequence_lengths.extend(list(sequence_lengths_dict.values()))
            
            aggregated['sequence_analysis']['longest_sequence'] = max(
                aggregated['sequence_analysis']['longest_sequence'],
                seq_analysis.get('longest_sequence', 0)
            )
            
            shortest = seq_analysis.get('shortest_sequence')
            if shortest is not None and shortest < aggregated['sequence_analysis']['shortest_sequence']:
                aggregated['sequence_analysis']['shortest_sequence'] = shortest
            
            gap_analysis = stats.get('gap_analysis', {})
            aggregated['gap_analysis']['total_sequences'] += gap_analysis.get('total_sequences', 0)
            aggregated['gap_analysis']['total_gaps'] += gap_analysis.get('total_gaps', 0)
            
            calib_analysis = gap_analysis.get('calibration_period_analysis', {})
            aggregated['gap_analysis']['calibration_period_analysis']['calibration_periods_detected'] += calib_analysis.get('calibration_periods_detected', 0)
            aggregated['gap_analysis']['calibration_period_analysis']['sequences_marked_for_removal'] += calib_analysis.get('sequences_marked_for_removal', 0)
            aggregated['gap_analysis']['calibration_period_analysis']['total_records_marked_for_removal'] += calib_analysis.get('total_records_marked_for_removal', 0)
            
            cleaning_analysis = stats.get('cleaning_analysis', {})
            aggregated['cleaning_analysis']['removed_records'] += cleaning_analysis.get('removed_records', 0)
            
            interp_analysis = stats.get('interpolation_analysis', {})
            aggregated['interpolation_analysis']['total_interpolations'] += interp_analysis.get('total_interpolations', 0)
            aggregated['interpolation_analysis']['total_interpolated_data_points'] += interp_analysis.get('total_interpolated_data_points', 0)
            
            # Aggregate all field-specific interpolations
            for key, val in interp_analysis.items():
                if key.endswith('_interpolations') and key != 'total_interpolations':
                    if key not in aggregated['interpolation_analysis']:
                        aggregated['interpolation_analysis'][key] = 0
                    aggregated['interpolation_analysis'][key] += val

            aggregated['interpolation_analysis']['sequences_processed'] += interp_analysis.get('sequences_processed', 0)
            aggregated['interpolation_analysis']['small_gaps_filled'] += interp_analysis.get('small_gaps_filled', 0)
            aggregated['interpolation_analysis']['large_gaps_skipped'] += interp_analysis.get('large_gaps_skipped', 0)
            
            filter_analysis = stats.get('filtering_analysis', {})
            if filter_analysis:
                aggregated['filtering_analysis']['original_sequences'] += filter_analysis.get('original_sequences', 0)
                aggregated['filtering_analysis']['filtered_sequences'] += filter_analysis.get('filtered_sequences', 0)
                aggregated['filtering_analysis']['removed_sequences'] += filter_analysis.get('removed_sequences', 0)
                aggregated['filtering_analysis']['original_records'] += filter_analysis.get('original_records', 0)
                aggregated['filtering_analysis']['filtered_records'] += filter_analysis.get('filtered_records', 0)
                aggregated['filtering_analysis']['removed_records'] += filter_analysis.get('removed_records', 0)
            
            fixed_freq_analysis = stats.get('fixed_frequency_analysis', {})
            if fixed_freq_analysis:
                for key, val in fixed_freq_analysis.items():
                    if isinstance(val, (int, float)) and key not in ['data_density_before', 'data_density_after']:
                        if key not in aggregated['fixed_frequency_analysis']:
                            aggregated['fixed_frequency_analysis'][key] = 0
                        aggregated['fixed_frequency_analysis'][key] += val
                
                if 'data_density_before' in fixed_freq_analysis and 'data_density_after' in fixed_freq_analysis:
                    before_density = fixed_freq_analysis['data_density_before']
                    after_density = fixed_freq_analysis['data_density_after']
                    
                    if 'data_density_before' not in aggregated['fixed_frequency_analysis'] or isinstance(aggregated['fixed_frequency_analysis']['data_density_before'], (int, float)):
                        aggregated['fixed_frequency_analysis']['data_density_before'] = {'total_points': 0, 'total_intervals': 0}
                    if 'data_density_after' not in aggregated['fixed_frequency_analysis'] or isinstance(aggregated['fixed_frequency_analysis']['data_density_after'], (int, float)):
                        aggregated['fixed_frequency_analysis']['data_density_after'] = {'total_points': 0, 'total_intervals': 0}
                    
                    agg_before = aggregated['fixed_frequency_analysis']['data_density_before']
                    agg_after = aggregated['fixed_frequency_analysis']['data_density_after']
                    
                    agg_before['total_points'] += before_density.get('total_points', 0)
                    agg_before['total_intervals'] += before_density.get('total_intervals', 0)
                    agg_after['total_points'] += after_density.get('total_points', 0)
                    agg_after['total_intervals'] += after_density.get('total_intervals', 0)
            
            # Aggregate data quality
            quality = stats.get('data_quality', {})
            if quality:
                recs = overview.get('total_records', 0)
                for key, val in quality.items():
                    if key.endswith('_data_completeness'):
                        if key not in aggregated['data_quality']:
                            aggregated['data_quality'][key] = 0
                        aggregated['data_quality'][key] += val * recs
                    elif key in ['interpolated_records', 'inserted_records']:
                        if key not in aggregated['data_quality']:
                            aggregated['data_quality'][key] = 0
                        aggregated['data_quality'][key] += val
        
        # Calculate final averages for completeness
        total_agg_recs = aggregated['dataset_overview']['total_records']
        if total_agg_recs > 0:
            for key in list(aggregated['data_quality'].keys()):
                if key.endswith('_data_completeness'):
                    aggregated['data_quality'][key] /= total_agg_recs
            
            # Recalculate interpolation percentages for aggregated stats
            interp_agg = aggregated['interpolation_analysis']
            for key in list(interp_agg.keys()):
                if key.endswith('_interpolations') and key != 'total_interpolations':
                    val = interp_agg[key]
                    pct_key = f"{key}_pct"
                    interp_agg[pct_key] = round((val / total_agg_recs) * 100, 2)
        
        if 'fixed_frequency_analysis' in aggregated and isinstance(aggregated['fixed_frequency_analysis'].get('data_density_before'), dict):
            before_density = aggregated['fixed_frequency_analysis']['data_density_before']
            after_density = aggregated['fixed_frequency_analysis']['data_density_after']
            
            if before_density.get('total_intervals', 0) > 0:
                before_density['avg_points_per_interval'] = before_density['total_points'] / before_density['total_intervals']
            if after_density.get('total_intervals', 0) > 0:
                after_density['avg_points_per_interval'] = after_density['total_points'] / after_density['total_intervals']
        
        if all_sequence_lengths:
            s_series = pl.Series("lens", all_sequence_lengths)
            aggregated['sequence_analysis']['sequence_lengths']['count'] = len(all_sequence_lengths)
            aggregated['sequence_analysis']['sequence_lengths']['mean'] = float(s_series.mean())
            aggregated['sequence_analysis']['sequence_lengths']['std'] = float(s_series.std()) if len(all_sequence_lengths) > 1 else 0.0
            aggregated['sequence_analysis']['sequence_lengths']['min'] = int(s_series.min())
            aggregated['sequence_analysis']['sequence_lengths']['25%'] = float(s_series.quantile(0.25))
            aggregated['sequence_analysis']['sequence_lengths']['50%'] = float(s_series.median())
            aggregated['sequence_analysis']['sequence_lengths']['75%'] = float(s_series.quantile(0.75))
            aggregated['sequence_analysis']['sequence_lengths']['max'] = int(s_series.max())
            # CRITICAL: preserve all_lengths for further aggregation
            aggregated['sequence_analysis']['all_lengths'] = all_sequence_lengths
        
        if aggregated['sequence_analysis']['shortest_sequence'] == float('inf'):
            aggregated['sequence_analysis']['shortest_sequence'] = 0
            
        return aggregated

def print_statistics(stats: Dict[str, Any], preprocessor_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Print formatted statistics and return as string.
    """
    lines = []
    lines.append("\n" + "="*60)
    lines.append("GLUCOSE DATA PREPROCESSING STATISTICS")
    lines.append("="*60)
    
    if 'multi_database_info' in stats and stats['multi_database_info'].get('total_databases', 0) > 1:
        multi_db_info = stats['multi_database_info']
        lines.append(f"\nMULTI-DATABASE PROCESSING SUMMARY:")
        lines.append(f"   Total Databases Combined: {multi_db_info['total_databases']}")
        
        lines.append(f"\n   Processed Databases Details:")
        for db in multi_db_info['databases_processed']:
            db_idx = db.get('database_index', 'N/A')
            db_name = db.get('database_name', 'Unknown')
            seq_range = db.get('sequence_id_range', {})
            
            # Use folder name for display
            display_name = Path(db_name).name if '/' in db_name or '\\' in db_name else db_name
            
            # Show sequence ID range if available and valid
            seq_info = ""
            if seq_range.get('min') is not None and seq_range.get('min') != 'N/A':
                seq_info = f" (Sequence IDs: {seq_range['min']} - {seq_range['max']})"
            
            lines.append(f"      {db_idx}. {display_name}{seq_info}")
    
    if preprocessor_params:
        lines.append(f"\nPARAMETERS USED:")
        for k, v in preprocessor_params.items():
            lines.append(f"   {k.replace('_', ' ').title()}: {v}")
    
    overview = stats.get('dataset_overview', {})
    lines.append(f"\nDATASET OVERVIEW:")
    lines.append(f"   Total Records: {overview.get('total_records', 0):,}")
    lines.append(f"   Total Sequences: {overview.get('total_sequences', 0):,}")
    
    date_range = overview.get('date_range', {})
    lines.append(f"   Date Range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
    
    original_records = overview.get('original_records', overview.get('total_records', 0))
    final_records = overview.get('total_records', 0)
    preservation_percentage = (final_records / original_records * 100) if original_records > 0 else 100
    lines.append(f"   Data Preservation: {preservation_percentage:.1f}% ({final_records:,}/{original_records:,} records)")
    
    seq_analysis = stats.get('sequence_analysis', {})
    lines.append(f"\nSEQUENCE ANALYSIS:")
    lines.append(f"   Longest Sequence: {seq_analysis.get('longest_sequence', 0):,} records")
    lines.append(f"   Shortest Sequence: {seq_analysis.get('shortest_sequence', 0):,} records")
    
    seq_lengths = seq_analysis.get('sequence_lengths', {})
    lines.append(f"   Average Sequence Length: {seq_lengths.get('mean', 0):.1f} records")
    lines.append(f"   Median Sequence Length: {seq_lengths.get('50%', 0):.1f} records")
    
    gap_analysis = stats.get('gap_analysis', {})
    if gap_analysis:
        lines.append(f"\nGAP ANALYSIS:")
        lines.append(f"   Total Gaps: {gap_analysis.get('total_gaps', 0):,}")
        lines.append(f"   Sequences Created: {gap_analysis.get('total_sequences', 0):,}")
    
    if 'calibration_period_analysis' in gap_analysis and gap_analysis['calibration_period_analysis']:
        calib_analysis = gap_analysis['calibration_period_analysis']
        lines.append(f"\nCALIBRATION PERIOD ANALYSIS:")
        lines.append(f"   Calibration Periods Detected: {calib_analysis.get('calibration_periods_detected', 0):,}")
        lines.append(f"   Records Removed After Calibration: {calib_analysis.get('total_records_marked_for_removal', 0):,}")
    
    cleaning_analysis = stats.get('cleaning_analysis', {})
    if cleaning_analysis and cleaning_analysis.get('removed_records', 0) > 0:
        lines.append(f"\nDATA CLEANING ANALYSIS:")
        lines.append(f"   Records Removed In Large Gaps: {cleaning_analysis.get('removed_records', 0):,}")
    
    interp_analysis = stats.get('interpolation_analysis', {})
    if interp_analysis:
        lines.append(f"\nINTERPOLATION ANALYSIS:")
        lines.append(f"   Small Gaps Identified and Processed: {interp_analysis.get('small_gaps_filled', 0):,}")
        lines.append(f"   Inserted Data Points Created: {interp_analysis.get('total_interpolated_data_points', 0):,}")
        
        # Print field-specific interpolations
        for key, val in interp_analysis.items():
            if key.endswith('_interpolations') and key != 'total_interpolations':
                field_name = key.replace('_interpolations', '').replace('_', ' ').title()
                pct_key = f"{key}_pct"
                pct = interp_analysis.get(pct_key, 0.0)
                if val > 0:
                    lines.append(f"   {field_name} Interpolations: {val:,} ({pct}%)")
        
        lines.append(f"   Total Field Interpolations: {interp_analysis.get('total_interpolations', 0):,}")
    
    filter_analysis = stats.get('filtering_analysis', {})
    if filter_analysis:
        lines.append(f"\nSEQUENCE FILTERING ANALYSIS:")
        lines.append(f"   Original Sequences: {filter_analysis.get('original_sequences', 0):,}")
        lines.append(f"   Sequences After Filtering: {filter_analysis.get('filtered_sequences', 0):,}")
        lines.append(f"   Sequences Removed: {filter_analysis.get('removed_sequences', 0):,}")
        lines.append(f"   Records Before Filtering: {filter_analysis.get('original_records', 0):,}")
        lines.append(f"   Records After Filtering: {filter_analysis.get('filtered_records', 0):,}")
        lines.append(f"   Records Removed: {filter_analysis.get('removed_records', 0):,}")
    
    fixed_freq_analysis = stats.get('fixed_frequency_analysis', {})
    if fixed_freq_analysis:
        lines.append(f"\nFIXED-FREQUENCY ANALYSIS:")
        lines.append(f"   Sequences Processed: {fixed_freq_analysis.get('sequences_processed', 0):,}")
        lines.append(f"   Records Before: {fixed_freq_analysis.get('total_records_before', 0):,}")
        lines.append(f"   Records After: {fixed_freq_analysis.get('total_records_after', 0):,}")
        
        # Print other fixed-frequency metrics
        skip_keys = {
            'sequences_processed', 'total_records_before', 'total_records_after',
            'data_density_before', 'data_density_after', 'density_change_explanation'
        }
        for key, val in fixed_freq_analysis.items():
            if key not in skip_keys and isinstance(val, (int, float)) and val > 0:
                name = key.replace('_', ' ').title()
                lines.append(f"   {name}: {val:,}")
        
        if 'data_density_before' in fixed_freq_analysis and 'data_density_after' in fixed_freq_analysis:
            before_density = fixed_freq_analysis['data_density_before']
            after_density = fixed_freq_analysis['data_density_after']
            if isinstance(before_density, dict) and isinstance(after_density, dict):
                lines.append(f"\n   DATA DENSITY:")
                lines.append(f"      Before: {before_density.get('avg_points_per_interval', 0.0):.2f} points/interval")
                lines.append(f"      After: {after_density.get('avg_points_per_interval', 0.0):.2f} points/interval")
    
    quality = stats.get('data_quality', {})
    lines.append(f"\nDATA QUALITY:")
    
    # Sort quality metrics for consistent output
    sorted_quality = sorted(quality.items())
    for key, val in sorted_quality:
        if key.endswith('_data_completeness') and val > 0:
            field_name = key.replace('_data_completeness', '').replace('_', ' ').title()
            lines.append(f"   {field_name} Data Completeness: {val:.1f}%")
            
    # Print special record counts
    if quality.get('interpolated_records', 0) > 0:
        lines.append(f"   Interpolated Records (Existing rows): {quality['interpolated_records']:,}")
    if quality.get('inserted_records', 0) > 0:
        lines.append(f"   Inserted Records (New rows): {quality['inserted_records']:,}")
    
    lines.append("\n" + "="*60)
    
    output = "\n".join(lines)
    return output


#!/usr/bin/env python3
"""
Glucose Data Preprocessor for Machine Learning

This script processes glucose monitoring data for ML training by:
1. Detecting time gaps in the data
2. Interpolating missing values for gaps <= 10 minutes
3. Creating sequence IDs for continuous data segments
4. Providing statistics about the processed data
"""

import polars as pl
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import csv
import warnings
import yaml

warnings.filterwarnings('ignore')


class GlucoseMLPreprocessor:
    """
    Preprocessor for glucose monitoring data to prepare it for machine learning.
    """
    
    @classmethod
    def from_config_file(cls, config_path: str, **cli_overrides):
        """
        Create a GlucoseMLPreprocessor instance from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            **cli_overrides: Command line arguments that override config values
            
        Returns:
            GlucoseMLPreprocessor instance with loaded configuration
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Extract glucose replacement values
        glucose_config = config.get('glucose_value_replacement', {})
        high_value = glucose_config.get('high_value', 401)
        low_value = glucose_config.get('low_value', 39)
        
        # Create instance with config values, overridden by CLI arguments
        return cls(
            expected_interval_minutes=cli_overrides.get('expected_interval_minutes', config.get('expected_interval_minutes', 5)),
            small_gap_max_minutes=cli_overrides.get('small_gap_max_minutes', config.get('small_gap_max_minutes', 15)),
            remove_calibration=cli_overrides.get('remove_calibration', config.get('remove_calibration', True)),
            min_sequence_len=cli_overrides.get('min_sequence_len', config.get('min_sequence_len', 200)),
            save_intermediate_files=cli_overrides.get('save_intermediate_files', config.get('save_intermediate_files', False)),
            calibration_period_minutes=cli_overrides.get('calibration_period_minutes', config.get('calibration_period_minutes', 165)),
            remove_after_calibration_hours=cli_overrides.get('remove_after_calibration_hours', config.get('remove_after_calibration_hours', 24)),
            high_glucose_value=cli_overrides.get('high_glucose_value', high_value),
            low_glucose_value=cli_overrides.get('low_glucose_value', low_value),
            glucose_only=cli_overrides.get('glucose_only', config.get('glucose_only', False)),
            config=config
        )
    
    def __init__(self, expected_interval_minutes: int = 5, small_gap_max_minutes: int = 15, remove_calibration: bool = True, min_sequence_len: int = 200, save_intermediate_files: bool = False, calibration_period_minutes: int = 60*2 + 45, remove_after_calibration_hours: int = 24, high_glucose_value: int = 401, low_glucose_value: int = 39, glucose_only: bool = False, config: Optional[Dict] = None):
        """
        Initialize the preprocessor.
        
        Args:
            expected_interval_minutes: Expected data collection interval for time discretization (default: 5 minutes)
            small_gap_max_minutes: Maximum gap size to interpolate (default: 15 minutes)
            remove_calibration: If True, remove all Calibration Event Type rows (default: True)
            min_sequence_len: Minimum sequence length to keep for ML training (default: 200)
            save_intermediate_files: If True, save intermediate files after each processing step (default: False)
            calibration_period_minutes: Gap duration considered as calibration period (default: 165 minutes)
            remove_after_calibration_hours: Hours of data to remove after calibration period (default: 24 hours)
            high_glucose_value: Numeric value to replace 'High' glucose readings (default: 401 mg/dL)
            low_glucose_value: Numeric value to replace 'Low' glucose readings (default: 39 mg/dL)
            glucose_only: If True, output only glucose data with simplified fields (default: False)
            config: Optional configuration dictionary from YAML file
        """
        self.expected_interval_minutes = expected_interval_minutes
        self.small_gap_max_minutes = small_gap_max_minutes
        self.remove_calibration = remove_calibration
        self.min_sequence_len = min_sequence_len
        self.save_intermediate_files = save_intermediate_files
        self.calibration_period_minutes = calibration_period_minutes
        self.remove_after_calibration_hours = remove_after_calibration_hours
        self.high_glucose_value = high_glucose_value
        self.low_glucose_value = low_glucose_value
        self.glucose_only = glucose_only
        self.config = config
        self.expected_interval_seconds = expected_interval_minutes * 60
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        self.calibration_period_seconds = calibration_period_minutes * 60
    
    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object for sorting."""
        if not timestamp_str or timestamp_str.strip() == "":
            return None
        
        # Handle the format "2019-10-28 0:52:15" or "2019-10-14T16:42:37"
        timestamp_str = timestamp_str.strip()
        
        # Try different timestamp formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",  # ISO format with T
            "%Y-%m-%d %H:%M:%S",  # Space format
            "%Y-%m-%d %H:%M:%S.%f",  # With microseconds
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def process_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single CSV file and extract required fields."""
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Skip rows without timestamp
                    timestamp = row.get('Timestamp (YYYY-MM-DDThh:mm:ss)', '').strip()
                    if not timestamp:
                        continue
                    
                    # Extract required fields
                    record = {
                        'Timestamp (YYYY-MM-DDThh:mm:ss)': timestamp,
                        'Event Type': row.get('Event Type', '').strip(),
                        'Glucose Value (mg/dL)': row.get('Glucose Value (mg/dL)', '').strip(),
                        'Insulin Value (u)': row.get('Insulin Value (u)', '').strip(),
                        'Carb Value (grams)': row.get('Carb Value (grams)', '').strip()
                    }
                    
                    data.append(record)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return data
    
    def consolidate_glucose_data(self, csv_folder: str, output_file: str = None) -> pl.DataFrame:
        """Consolidate all CSV files in the folder into a single DataFrame.
        
        Args:
            csv_folder: Path to folder containing CSV files
            output_file: Optional path to save consolidated data
            
        Returns:
            DataFrame with consolidated and sorted data
        """
        csv_path = Path(csv_folder)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV folder not found: {csv_folder}")
        
        if not csv_path.is_dir():
            raise ValueError(f"Input must be a directory containing CSV files, got: {csv_folder}")
        
        all_data = []
        
        # Get all CSV files
        csv_files = list(csv_path.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {csv_folder}")
        
        print(f"Found {len(csv_files)} CSV files to consolidate")
        
        for csv_file in sorted(csv_files):
            print(f"Processing: {csv_file.name}")
            file_data = self.process_csv_file(csv_file)
            all_data.extend(file_data)
            print(f"  ✓ Extracted {len(file_data)} records")
        
        print(f"\nTotal records collected: {len(all_data):,}")
        
        if not all_data:
            raise ValueError("No valid data found in CSV files!")
        
        # Convert to DataFrame for easier sorting
        df = pl.DataFrame(all_data)
        
        # Parse timestamps and sort
        print("Parsing timestamps and sorting...")
        df = df.with_columns(
            pl.col('Timestamp (YYYY-MM-DDThh:mm:ss)').map_elements(self.parse_timestamp, return_dtype=pl.Datetime).alias('parsed_timestamp')
        )
        
        # Remove rows where timestamp parsing failed
        df = df.filter(pl.col('parsed_timestamp').is_not_null())
        
        print(f"Records with valid timestamps: {len(df):,}")
        
        # Sort by timestamp (oldest first)
        df = df.sort('parsed_timestamp')
        
        # Rename parsed_timestamp to timestamp for consistency with other methods
        df = df.rename({'parsed_timestamp': 'timestamp'})
        
        # Write to output file
        if output_file:
            print(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
        
        print(f"✓ Consolidation complete!")
        print(f"Total records in output: {len(df):,}")
        
        # Show date range
        if len(df) > 0:
            first_date = df['Timestamp (YYYY-MM-DDThh:mm:ss)'][0]
            last_date = df['Timestamp (YYYY-MM-DDThh:mm:ss)'][-1]
            print(f"Date range: {first_date} to {last_date}")

        return df
        
    
    def detect_gaps_and_sequences(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Detect time gaps and create sequence IDs, marking calibration periods and sequences for removal.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Tuple of (DataFrame with sequence IDs and removal flags, statistics dictionary)
        """
        print("Detecting gaps and creating sequences...")
        
        # Sort by timestamp to ensure proper order
        df = df.sort('timestamp')
        
        # Calculate time differences and create sequence IDs
        df = df.with_columns([
            pl.col('timestamp').diff().dt.total_seconds().alias('time_diff_seconds'),
        ]).with_columns([
            (pl.col('time_diff_seconds') > self.small_gap_max_seconds).alias('is_gap'),
        ]).with_columns([
            pl.col('is_gap').cum_sum().alias('sequence_id')
        ])
        
        # Detect calibration periods and mark sequences for removal
        calibration_stats = {
            'calibration_periods_detected': 0,
            'sequences_marked_for_removal': 0,
            'total_records_marked_for_removal': 0
        }
        
        # Check if calibration period detection is enabled
        if (self.calibration_period_minutes > self.small_gap_max_minutes and 
            self.remove_after_calibration_hours > 0):
            
            print(f"Detecting calibration periods (gaps >= {self.calibration_period_minutes} minutes)...")
            
            # Find gaps that are calibration periods (>= calibration_period_minutes)
            calibration_gaps = df.filter(
                (pl.col('is_gap') == True) & 
                (pl.col('time_diff_seconds') >= self.calibration_period_seconds)
            )
            
            calibration_stats['calibration_periods_detected'] = len(calibration_gaps)
            
            if len(calibration_gaps) > 0:
                print(f"Found {len(calibration_gaps)} calibration periods")
                
                # Mark sequences after calibration periods for removal
                removal_start_times = []
                for row in calibration_gaps.iter_rows(named=True):
                    # Get the sequence that starts after this calibration gap
                    calibration_end_time = row['timestamp']
                    removal_start_time = calibration_end_time
                    removal_end_time = calibration_end_time + timedelta(hours=self.remove_after_calibration_hours)
                    removal_start_times.append((removal_start_time, removal_end_time))
                
                # Create removal flag for records in the specified time period after calibration
                def should_remove_record(timestamp, removal_periods):
                    for start_time, end_time in removal_periods:
                        if start_time <= timestamp <= end_time:
                            return True
                    return False
                
                # Add removal flag
                df = df.with_columns([
                    pl.col('timestamp').map_elements(
                        lambda ts: should_remove_record(ts, removal_start_times),
                        return_dtype=pl.Boolean
                    ).alias('remove_after_calibration')
                ])
                
                # Count records marked for removal
                records_to_remove = df.filter(pl.col('remove_after_calibration') == True)
                calibration_stats['total_records_marked_for_removal'] = len(records_to_remove)
                
                # Count sequences that will be affected
                affected_sequences = records_to_remove['sequence_id'].unique()
                calibration_stats['sequences_marked_for_removal'] = len(affected_sequences)
                
                print(f"Marked {calibration_stats['total_records_marked_for_removal']:,} records for removal")
                print(f"Affected {calibration_stats['sequences_marked_for_removal']} sequences")
                
                # Actually remove the marked records
                df = df.filter(pl.col('remove_after_calibration') != True)
                print(f"Removed {calibration_stats['total_records_marked_for_removal']:,} records after calibration periods")
                
                # Recalculate sequence IDs after removal (remove the temporary removal flag column)
                df = df.drop('remove_after_calibration')
                df = df.with_columns([
                    pl.col('timestamp').diff().dt.total_seconds().alias('time_diff_seconds'),
                ]).with_columns([
                    (pl.col('time_diff_seconds') > self.small_gap_max_seconds).alias('is_gap'),
                ]).with_columns([
                    pl.col('is_gap').cum_sum().alias('sequence_id')
                ])
            else:
                # No calibration periods found, remove the temporary removal flag column
                df = df.drop('remove_after_calibration')
        else:
            # Calibration period detection disabled, no removal flag needed
            pass
        
        # Calculate statistics
        sequence_counts = df.group_by('sequence_id').count().sort('sequence_id')
        stats = {
            'total_sequences': df['sequence_id'].max() + 1,
            'gap_positions': df['is_gap'].sum(),
            'total_gaps': df['is_gap'].sum(),
            'sequence_lengths': dict(zip(sequence_counts['sequence_id'].to_list(), sequence_counts['count'].to_list())),
            'calibration_period_analysis': calibration_stats
        }
        
        print(f"Created {stats['total_sequences']} sequences")
        print(f"Found {stats['total_gaps']} gaps > {self.small_gap_max_minutes} minutes")
        
        # Remove temporary columns
        df = df.drop(['time_diff_seconds', 'is_gap'])
        
        return df, stats
    
    def interpolate_missing_values(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Interpolate only small gaps (1-2 missing data points) within sequences.
        Large gaps are treated as sequence boundaries and not interpolated.
        
        Args:
            df: DataFrame with sequence IDs and timestamp data
            
        Returns:
            Tuple of (DataFrame with interpolated values, interpolation statistics)
        """
        print("Interpolating small gaps only...")
        
        interpolation_stats = {
            'total_interpolations': 0,
            'total_interpolated_data_points': 0,
            'glucose_value_mg/dl_interpolations': 0,
            'insulin_value_u_interpolations': 0,
            'carb_value_grams_interpolations': 0,
            'sequences_processed': 0,
            'small_gaps_filled': 0,
            'large_gaps_skipped': 0
        }
        
        # Process each sequence separately
        unique_sequences = df['sequence_id'].unique().to_list()
        
        for seq_id in unique_sequences:
            seq_data = df.filter(pl.col('sequence_id') == seq_id).sort('timestamp')
            
            if len(seq_data) < 2:
                continue
                
            interpolation_stats['sequences_processed'] += 1
            
            # Get time differences as list for processing
            time_diffs = seq_data['timestamp'].diff().dt.total_seconds() / 60.0
            time_diffs_list = time_diffs.to_list()
            
            # Find small gaps (1-2 missing data points = expected_interval to small_gap_max_minutes)
            small_gaps = [(i, diff) for i, diff in enumerate(time_diffs_list) 
                         if i > 0 and self.expected_interval_minutes < diff <= self.small_gap_max_minutes]
            large_gaps = [(i, diff) for i, diff in enumerate(time_diffs_list) 
                         if i > 0 and diff > self.small_gap_max_minutes]
            
            interpolation_stats['small_gaps_filled'] += len(small_gaps)
            interpolation_stats['large_gaps_skipped'] += len(large_gaps)
            
            # Only interpolate small gaps
            if small_gaps:
                # Convert to pandas for easier interpolation logic, then back to polars
                seq_pandas = seq_data.to_pandas()
                new_rows = []
                
                for gap_idx, time_diff_minutes in small_gaps:
                    if gap_idx > 0:
                        prev_row = seq_pandas.iloc[gap_idx-1]
                        current_row = seq_pandas.iloc[gap_idx]
                        
                        # Calculate number of missing points
                        missing_points = int(time_diff_minutes / self.expected_interval_minutes) - 1
                        
                        if missing_points > 0:
                            # Create interpolated points
                            for j in range(1, missing_points + 1):
                                interpolated_time = prev_row['timestamp'] + timedelta(minutes=self.expected_interval_minutes*j)
                                
                                # Interpolate numeric values - include all columns from original data
                                new_row = {
                                    'Timestamp (YYYY-MM-DDThh:mm:ss)': interpolated_time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    'Event Type': 'Interpolated',
                                    'Glucose Value (mg/dL)': None,  # Default to None (will be converted to proper type)
                                    'Insulin Value (u)': None,      # Default to None (will be converted to proper type)
                                    'Carb Value (grams)': None,     # Default to None (will be converted to proper type)
                                    'timestamp': interpolated_time,
                                    'sequence_id': seq_id
                                }
                                
                                # Linear interpolation for numeric columns
                                numeric_cols = ['Glucose Value (mg/dL)', 'Insulin Value (u)', 'Carb Value (grams)']
                                interpolations_made = 0
                                for col in numeric_cols:
                                    prev_val = prev_row[col]
                                    curr_val = current_row[col]
                                    
                                    # Check if both values are valid and numeric
                                    try:
                                        prev_numeric = float(prev_val) if prev_val is not None and str(prev_val).strip() != '' else None
                                        curr_numeric = float(curr_val) if curr_val is not None and str(curr_val).strip() != '' else None
                                        
                                        if prev_numeric is not None and curr_numeric is not None:
                                            # Linear interpolation
                                            alpha = j / (missing_points + 1)
                                            interpolated_value = prev_numeric + alpha * (curr_numeric - prev_numeric)
                                            new_row[col] = interpolated_value  # Keep as numeric value
                                            interpolation_stats[f'{col.lower().replace(" ", "_").replace("(", "").replace(")", "")}_interpolations'] += 1
                                            interpolations_made += 1
                                    except (ValueError, TypeError):
                                        # Keep empty string for non-numeric values
                                        pass
                                
                                # Count this as one interpolated data point if any field was interpolated
                                if interpolations_made > 0:
                                    interpolation_stats['total_interpolations'] += 1
                                
                                new_rows.append(new_row)
                                
                                # Count this as one interpolated data point (row created)
                                interpolation_stats['total_interpolated_data_points'] += 1
                
                # Add interpolated rows to the sequence
                if new_rows:
                    interpolated_df = pl.DataFrame(new_rows)
                    # Ensure all columns have the same types as the original sequence
                    interpolated_df = interpolated_df.with_columns([
                        pl.col('sequence_id').cast(seq_data['sequence_id'].dtype),
                        # Ensure numeric columns have the same type as original data
                        pl.col('Glucose Value (mg/dL)').cast(seq_data['Glucose Value (mg/dL)'].dtype, strict=False),
                        pl.col('Insulin Value (u)').cast(seq_data['Insulin Value (u)'].dtype, strict=False),
                        pl.col('Carb Value (grams)').cast(seq_data['Carb Value (grams)'].dtype, strict=False)
                    ])
                    seq_data = pl.concat([seq_data, interpolated_df]).sort(['sequence_id', 'timestamp'])
            
            # Update the main DataFrame
            df = df.filter(pl.col('sequence_id') != seq_id)  # Remove original sequence data
            df = pl.concat([df, seq_data])
        
        # Sort by sequence_id and timestamp
        df = df.sort(['sequence_id', 'timestamp'])
        
        print(f"Identified and processed {interpolation_stats['small_gaps_filled']} small gaps")
        print(f"Created {interpolation_stats['total_interpolated_data_points']} interpolated data points")
        print(f"Interpolated {interpolation_stats['total_interpolations']} missing field values")
        print(f"Skipped {interpolation_stats['large_gaps_skipped']} large gaps")
        print(f"Processed {interpolation_stats['sequences_processed']} sequences")
        
        return df, interpolation_stats
    
    def remove_calibration_values(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Remove all rows with Calibration Event Type to create gaps that can be interpolated.
        
        Args:
            df: DataFrame with glucose data
            
        Returns:
            Tuple of (DataFrame with calibration rows removed, statistics dictionary)
        """
        print("Removing calibration events to create interpolatable gaps...")
        
        removal_stats = {
            'calibration_events_removed': 0,
            'records_before_removal': len(df),
            'records_after_removal': 0,
            'calibration_removal_enabled': self.remove_calibration
        }
        
        if not self.remove_calibration:
            print("Calibration removal is disabled - keeping all calibration events")
            removal_stats['records_after_removal'] = len(df)
            return df, removal_stats
        
        # Count calibration events before removal
        calibration_events = df.filter(pl.col('Event Type') == 'Calibration')
        removal_stats['calibration_events_removed'] = len(calibration_events)
        
        if len(calibration_events) == 0:
            print("No calibration events found")
            removal_stats['records_after_removal'] = len(df)
            return df, removal_stats
        
        print(f"Found {len(calibration_events)} calibration events to remove")
        
        # Remove all calibration events
        df_filtered = df.filter(pl.col('Event Type') != 'Calibration')
        removal_stats['records_after_removal'] = len(df_filtered)
        
        print(f"Removed {removal_stats['calibration_events_removed']} calibration events")
        print(f"Records before removal: {removal_stats['records_before_removal']:,}")
        print(f"Records after removal: {removal_stats['records_after_removal']:,}")
        print("✓ Calibration events removed - gaps can now be interpolated")
        
        return df_filtered, removal_stats
    
    def replace_high_low_values(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Replace 'High' with 401 and 'Low' with 39 in the glucose field.
        
        Args:
            df: DataFrame with glucose data
            
        Returns:
            Tuple of (DataFrame with replaced values, statistics dictionary)
        """
        print("Replacing High/Low glucose values with numeric equivalents...")
        
        replacement_stats = {
            'high_replacements': 0,
            'low_replacements': 0,
            'total_replacements': 0,
            'glucose_field_converted_to_float': False
        }
        
        # Count High and Low values before replacement
        high_count = df.filter(pl.col('Glucose Value (mg/dL)') == 'High').height
        low_count = df.filter(pl.col('Glucose Value (mg/dL)') == 'Low').height
        
        replacement_stats['high_replacements'] = high_count
        replacement_stats['low_replacements'] = low_count
        replacement_stats['total_replacements'] = high_count + low_count
        
        # Replace High and Low with configurable values, then convert to float
        df = df.with_columns([
            pl.col('Glucose Value (mg/dL)')
            .str.replace('High', str(self.high_glucose_value))
            .str.replace('Low', str(self.low_glucose_value))
            .cast(pl.Float64, strict=False)
            .alias('Glucose Value (mg/dL)')
        ])
        
        replacement_stats['glucose_field_converted_to_float'] = True
        
        print(f"Replaced {replacement_stats['high_replacements']} 'High' values with {self.high_glucose_value}")
        print(f"Replaced {replacement_stats['low_replacements']} 'Low' values with {self.low_glucose_value}")
        print(f"Total replacements: {replacement_stats['total_replacements']}")
        print("✓ Glucose field converted to Float64 type")
        
        return df, replacement_stats
    
    def filter_sequences_by_length(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Filter out sequences that are shorter than the minimum required length.
        
        Args:
            df: DataFrame with sequence IDs and processed data
            
        Returns:
            Tuple of (filtered DataFrame, filtering statistics dictionary)
        """
        print(f"Filtering sequences with length < {self.min_sequence_len}...")
        
        # Calculate sequence lengths
        sequence_counts = df.group_by('sequence_id').count().sort('sequence_id')
        
        # Find sequences to keep (longer than or equal to min_sequence_len)
        sequences_to_keep = sequence_counts.filter(pl.col('count') >= self.min_sequence_len)
        
        filtering_stats = {
            'original_sequences': sequence_counts.height,
            'filtered_sequences': sequences_to_keep.height,
            'removed_sequences': sequence_counts.height - sequences_to_keep.height,
            'original_records': len(df),
            'filtered_records': 0,  # Will be calculated after filtering
            'removed_records': 0    # Will be calculated after filtering
        }
        
        if len(sequences_to_keep) == 0:
            print("⚠️  Warning: No sequences meet the minimum length requirement!")
            return df, filtering_stats
        
        # Filter the DataFrame to keep only sequences that meet the length requirement
        valid_sequence_ids = sequences_to_keep['sequence_id'].to_list()
        filtered_df = df.filter(pl.col('sequence_id').is_in(valid_sequence_ids))
        
        # Update filtering statistics
        filtering_stats['filtered_records'] = len(filtered_df)
        filtering_stats['removed_records'] = len(df) - len(filtered_df)
        
        print(f"Original sequences: {filtering_stats['original_sequences']}")
        print(f"Sequences after filtering: {filtering_stats['filtered_sequences']}")
        print(f"Sequences removed: {filtering_stats['removed_sequences']}")
        print(f"Original records: {filtering_stats['original_records']:,}")
        print(f"Records after filtering: {filtering_stats['filtered_records']:,}")
        print(f"Records removed: {filtering_stats['removed_records']:,}")
        
        # Show statistics about removed sequences
        if filtering_stats['removed_sequences'] > 0:
            removed_sequences = sequence_counts.filter(pl.col('count') < self.min_sequence_len)
            if len(removed_sequences) > 0:
                min_len_removed = removed_sequences['count'].min()
                max_len_removed = removed_sequences['count'].max()
                avg_len_removed = removed_sequences['count'].mean()
                print(f"Removed sequence lengths - Min: {min_len_removed}, Max: {max_len_removed}, Avg: {avg_len_removed:.1f}")
        
        return filtered_df, filtering_stats
    
    def filter_glucose_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Filter to keep only glucose data with simplified fields.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Tuple of (filtered DataFrame with only glucose data, filtering statistics)
        """
        print("Filtering to glucose-only data...")
        
        filtering_stats = {
            'original_records': len(df),
            'glucose_only_enabled': self.glucose_only,
            'records_after_filtering': 0,
            'records_removed': 0,
            'fields_removed': []
        }
        
        if not self.glucose_only:
            print("Glucose-only filtering is disabled - keeping all data")
            filtering_stats['records_after_filtering'] = len(df)
            return df, filtering_stats
        
        # Filter to keep only rows with non-null glucose values
        df_filtered = df.filter(pl.col('Glucose Value (mg/dL)').is_not_null())
        
        # Remove specified fields
        fields_to_remove = ['Event Type', 'Insulin Value (u)', 'Carb Value (grams)']
        existing_fields_to_remove = [field for field in fields_to_remove if field in df_filtered.columns]
        
        if existing_fields_to_remove:
            df_filtered = df_filtered.drop(existing_fields_to_remove)
            filtering_stats['fields_removed'] = existing_fields_to_remove
        
        # Update statistics
        filtering_stats['records_after_filtering'] = len(df_filtered)
        filtering_stats['records_removed'] = len(df) - len(df_filtered)
        
        print(f"Original records: {filtering_stats['original_records']:,}")
        print(f"Records with glucose values: {filtering_stats['records_after_filtering']:,}")
        print(f"Records removed (no glucose): {filtering_stats['records_removed']:,}")
        if filtering_stats['fields_removed']:
            print(f"Fields removed: {', '.join(filtering_stats['fields_removed'])}")
        print("✓ Glucose-only filtering complete")
        
        return df_filtered, filtering_stats
    
    def prepare_ml_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare final DataFrame for machine learning with sequence_id as first column.
        
        Args:
            df: Processed DataFrame with sequence IDs
            
        Returns:
            Final DataFrame ready for ML training
        """
        print("Preparing final ML dataset...")
        
        # Convert timestamp back to string format for output and convert all numeric fields to Float64
        columns_to_cast = []
        
        # Always convert timestamp
        columns_to_cast.append(
            pl.col('timestamp').dt.strftime('%Y-%m-%dT%H:%M:%S').alias('Timestamp (YYYY-MM-DDThh:mm:ss)')
        )
        
        # Convert numeric fields to Float64 if they exist
        if 'Glucose Value (mg/dL)' in df.columns:
            columns_to_cast.append(pl.col('Glucose Value (mg/dL)').cast(pl.Float64, strict=False))
        if 'Insulin Value (u)' in df.columns:
            columns_to_cast.append(pl.col('Insulin Value (u)').cast(pl.Float64, strict=False))
        if 'Carb Value (grams)' in df.columns:
            columns_to_cast.append(pl.col('Carb Value (grams)').cast(pl.Float64, strict=False))
        
        df = df.with_columns(columns_to_cast)
        
        # Reorder columns with sequence_id first
        ml_columns = ['sequence_id'] + [col for col in df.columns if col not in ['sequence_id', 'timestamp']]
        ml_df = df.select(ml_columns)
        
        return ml_df
    
    def get_statistics(self, df: pl.DataFrame, gap_stats: Dict, interp_stats: Dict, removal_stats: Dict = None, filter_stats: Dict = None, replacement_stats: Dict = None, glucose_filter_stats: Dict = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the processed data.
        
        Args:
            df: Final processed DataFrame
            gap_stats: Gap detection statistics
            interp_stats: Interpolation statistics
            removal_stats: Calibration removal statistics
            filter_stats: Sequence filtering statistics
            replacement_stats: High/Low value replacement statistics
            glucose_filter_stats: Glucose-only filtering statistics
            
        Returns:
            Dictionary with comprehensive statistics
        """
        # Get date range from timestamp column if available
        date_range = {'start': 'N/A', 'end': 'N/A'}
        if 'Timestamp (YYYY-MM-DDThh:mm:ss)' in df.columns:
            valid_timestamps = df.filter(pl.col('Timestamp (YYYY-MM-DDThh:mm:ss)').is_not_null())
            if len(valid_timestamps) > 0:
                timestamps = valid_timestamps['Timestamp (YYYY-MM-DDThh:mm:ss)'].sort()
                date_range = {
                    'start': timestamps[0],
                    'end': timestamps[-1]
                }
        
        # Calculate sequence statistics
        sequence_counts = df.group_by('sequence_id').count().sort('sequence_id')
        sequence_lengths = sequence_counts['count'].to_list()
        
        stats = {
            'dataset_overview': {
                'total_records': len(df),
                'total_sequences': df['sequence_id'].n_unique(),
                'date_range': date_range
            },
            'sequence_analysis': {
                'sequence_lengths': {
                    'count': len(sequence_lengths),
                    'mean': sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
                    'std': np.std(sequence_lengths) if sequence_lengths else 0,
                    'min': min(sequence_lengths) if sequence_lengths else 0,
                    '25%': np.percentile(sequence_lengths, 25) if sequence_lengths else 0,
                    '50%': np.percentile(sequence_lengths, 50) if sequence_lengths else 0,
                    '75%': np.percentile(sequence_lengths, 75) if sequence_lengths else 0,
                    'max': max(sequence_lengths) if sequence_lengths else 0
                },
                'longest_sequence': max(sequence_lengths) if sequence_lengths else 0,
                'shortest_sequence': min(sequence_lengths) if sequence_lengths else 0,
                'sequences_by_length': dict(zip(*np.unique(sequence_lengths, return_counts=True)))
            },
            'gap_analysis': gap_stats,
            'interpolation_analysis': interp_stats,
            'calibration_removal_analysis': removal_stats if removal_stats else {},
            'filtering_analysis': filter_stats if filter_stats else {},
            'replacement_analysis': replacement_stats if replacement_stats else {},
            'glucose_filtering_analysis': glucose_filter_stats if glucose_filter_stats else {},
            'data_quality': {
                'glucose_data_completeness': (1 - df['Glucose Value (mg/dL)'].null_count() / len(df)) * 100 if 'Glucose Value (mg/dL)' in df.columns else 0,
                'insulin_data_completeness': (1 - df['Insulin Value (u)'].null_count() / len(df)) * 100 if 'Insulin Value (u)' in df.columns else 0,
                'carb_data_completeness': (1 - df['Carb Value (grams)'].null_count() / len(df)) * 100 if 'Carb Value (grams)' in df.columns else 0,
                'interpolated_records': df.filter(pl.col('Event Type') == 'Interpolated').height if 'Event Type' in df.columns else 0
            }
        }
        
        return stats
    
    def process(self, csv_folder: str, output_file: str = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline with mandatory consolidation.
        
        Args:
            csv_folder: Path to folder containing CSV files (consolidation is mandatory)
            output_file: Optional path to save processed data
            
        Returns:
            Tuple of (processed DataFrame, statistics dictionary)
        """
        print("🍯 Starting glucose data preprocessing for ML...")
        print(f"⚙️  Time discretization interval: {self.expected_interval_minutes} minutes")
        print(f"⚙️  Small gap max (interpolation limit): {self.small_gap_max_minutes} minutes")
        print(f"⚙️  Save intermediate files: {self.save_intermediate_files}")
        print("-" * 50)
        
        # Step 1: Consolidate CSV files (mandatory step)
        if self.save_intermediate_files:
            consolidated_file = "consolidated_data.csv"
        else:
            consolidated_file = None
        
        print("STEP 1: Consolidating CSV files (mandatory step)...")
        df = self.consolidate_glucose_data(csv_folder, consolidated_file)
        
        if self.save_intermediate_files:
            print(f"💾 Consolidated data saved to: {consolidated_file}")
        
        print("-" * 40)
        
        # Step 2: Replace High/Low glucose values
        print("STEP 2: Replacing High/Low glucose values...")
        df, replacement_stats = self.replace_high_low_values(df)
        print("✓ High/Low value replacement complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step2_high_low_replaced.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"💾 Data with High/Low replacements saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 3: Remove calibration events
        print("STEP 3: Removing calibration events...")
        df, removal_stats = self.remove_calibration_values(df)
        print("✓ Calibration event removal complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step3_calibrations_removed.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"💾 Data with calibrations removed saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 4: Detect gaps and create sequences
        print("STEP 4: Detecting gaps and creating sequences...")
        df, gap_stats = self.detect_gaps_and_sequences(df)
        print("✓ Gap detection and sequence creation complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step4_sequences_created.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"💾 Data with sequences saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 5: Interpolate missing values
        print("STEP 5: Interpolating missing values...")
        df, interp_stats = self.interpolate_missing_values(df)
        print("✓ Missing value interpolation complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step5_interpolated_values.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"💾 Data with interpolated values saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 6: Filter sequences by minimum length
        print("STEP 6: Filtering sequences by minimum length...")
        df, filter_stats = self.filter_sequences_by_length(df)
        print("✓ Sequence filtering complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step6_filtered_sequences.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"💾 Filtered data saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 7: Filter to glucose-only data (if requested)
        print("STEP 7: Filtering to glucose-only data...")
        df, glucose_filter_stats = self.filter_glucose_only(df)
        print("✓ Glucose-only filtering complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step7_glucose_only.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"💾 Glucose-only data saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 8: Prepare final ML dataset
        print("STEP 8: Preparing final ML dataset...")
        ml_df = self.prepare_ml_data(df)
        print("✓ ML dataset preparation complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step8_ml_ready.csv"
            ml_df.write_csv(intermediate_file, null_value="")
            print(f"💾 ML-ready data saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Generate statistics
        stats = self.get_statistics(ml_df, gap_stats, interp_stats, removal_stats, filter_stats, replacement_stats, glucose_filter_stats)
        
        # Save final output if specified
        if output_file:
            ml_df.write_csv(output_file, null_value="")
            print(f"💾 Final processed data saved to: {output_file}")
        
        print("-" * 50)
        print("✅ Preprocessing completed successfully!")
        
        return ml_df, stats
    


def print_statistics(stats: Dict[str, Any], preprocessor: 'GlucoseMLPreprocessor' = None) -> None:
    """
    Print formatted statistics about the processed data.
    
    Args:
        stats: Statistics dictionary from preprocessor
        preprocessor: Optional preprocessor instance to show parameters
    """
    print("\n" + "="*60)
    print("GLUCOSE DATA PREPROCESSING STATISTICS")
    print("="*60)
    
    # Show parameters if preprocessor is provided
    if preprocessor:
        print(f"\n⚙️  PARAMETERS USED:")
        print(f"   Time Discretization Interval: {preprocessor.expected_interval_minutes} minutes")
        print(f"   Small Gap Max (Interpolation Limit): {preprocessor.small_gap_max_minutes} minutes")
        print(f"   Remove Calibration Events: {preprocessor.remove_calibration}")
        print(f"   Minimum Sequence Length: {preprocessor.min_sequence_len}")
        print(f"   Calibration Period Threshold: {preprocessor.calibration_period_minutes} minutes")
        print(f"   Remove After Calibration: {preprocessor.remove_after_calibration_hours} hours")
    
    # Dataset Overview
    overview = stats['dataset_overview']
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total Records: {overview['total_records']:,}")
    print(f"   Total Sequences: {overview['total_sequences']:,}")
    print(f"   Date Range: {overview['date_range']['start']} to {overview['date_range']['end']}")
    
    # Sequence Analysis
    seq_analysis = stats['sequence_analysis']
    print(f"\n🔗 SEQUENCE ANALYSIS:")
    print(f"   Longest Sequence: {seq_analysis['longest_sequence']:,} records")
    print(f"   Shortest Sequence: {seq_analysis['shortest_sequence']:,} records")
    print(f"   Average Sequence Length: {seq_analysis['sequence_lengths']['mean']:.1f} records")
    print(f"   Median Sequence Length: {seq_analysis['sequence_lengths']['50%']:.1f} records")
    
    # Gap Analysis
    gap_analysis = stats['gap_analysis']
    print(f"\n⏰ GAP ANALYSIS:")
    print(f"   Total Gaps > {preprocessor.small_gap_max_minutes if preprocessor else 'N/A'} minutes: {gap_analysis['total_gaps']:,}")
    print(f"   Sequences Created: {gap_analysis['total_sequences']:,}")
    
    # Calibration Period Analysis
    if 'calibration_period_analysis' in gap_analysis:
        calib_analysis = gap_analysis['calibration_period_analysis']
        print(f"\n🔬 CALIBRATION PERIOD ANALYSIS:")
        print(f"   Calibration Periods Detected: {calib_analysis['calibration_periods_detected']:,}")
        print(f"   Records Removed After Calibration: {calib_analysis['total_records_marked_for_removal']:,}")
        print(f"   Sequences Affected: {calib_analysis['sequences_marked_for_removal']:,}")
    
    # High/Low Value Replacement Analysis
    if 'replacement_analysis' in stats and stats['replacement_analysis']:
        replacement_analysis = stats['replacement_analysis']
        print(f"\n🔄 HIGH/LOW VALUE REPLACEMENT ANALYSIS:")
        print(f"   High Values Replaced (→ 401): {replacement_analysis['high_replacements']:,}")
        print(f"   Low Values Replaced (→ 39): {replacement_analysis['low_replacements']:,}")
        print(f"   Total Replacements: {replacement_analysis['total_replacements']:,}")
        print(f"   Glucose Field Type: {'Float64' if replacement_analysis['glucose_field_converted_to_float'] else 'String'}")
    
    # Interpolation Analysis
    interp_analysis = stats['interpolation_analysis']
    print(f"\n🔧 INTERPOLATION ANALYSIS:")
    print(f"   Small Gaps Identified and Processed: {interp_analysis['small_gaps_filled']:,}")
    print(f"   Interpolated Data Points Created: {interp_analysis['total_interpolated_data_points']:,}")
    print(f"   Total Field Interpolations: {interp_analysis['total_interpolations']:,}")
    print(f"   Glucose Interpolations: {interp_analysis['glucose_value_mg/dl_interpolations']:,}")
    print(f"   Insulin Interpolations: {interp_analysis['insulin_value_u_interpolations']:,}")
    print(f"   Carb Interpolations: {interp_analysis['carb_value_grams_interpolations']:,}")
    print(f"   Large Gaps Skipped: {interp_analysis['large_gaps_skipped']:,}")
    print(f"   Sequences Processed: {interp_analysis['sequences_processed']:,}")
    
    # Calibration Removal Analysis
    if 'calibration_removal_analysis' in stats and stats['calibration_removal_analysis']:
        removal_analysis = stats['calibration_removal_analysis']
        print(f"\n🗑️  CALIBRATION REMOVAL ANALYSIS:")
        print(f"   Calibration Events Removed: {removal_analysis['calibration_events_removed']:,}")
        print(f"   Records Before Removal: {removal_analysis['records_before_removal']:,}")
        print(f"   Records After Removal: {removal_analysis['records_after_removal']:,}")
        print(f"   Removal Enabled: {removal_analysis['calibration_removal_enabled']}")
    
    # Filtering Analysis
    if 'filtering_analysis' in stats and stats['filtering_analysis']:
        filter_analysis = stats['filtering_analysis']
        print(f"\n🔍 SEQUENCE FILTERING ANALYSIS:")
        print(f"   Original Sequences: {filter_analysis['original_sequences']:,}")
        print(f"   Sequences After Filtering: {filter_analysis['filtered_sequences']:,}")
        print(f"   Sequences Removed: {filter_analysis['removed_sequences']:,}")
        print(f"   Original Records: {filter_analysis['original_records']:,}")
        print(f"   Records After Filtering: {filter_analysis['filtered_records']:,}")
        print(f"   Records Removed: {filter_analysis['removed_records']:,}")
    
    # Glucose Filtering Analysis
    if 'glucose_filtering_analysis' in stats and stats['glucose_filtering_analysis']:
        glucose_filter_analysis = stats['glucose_filtering_analysis']
        print(f"\n🍯 GLUCOSE-ONLY FILTERING ANALYSIS:")
        print(f"   Glucose-Only Mode Enabled: {glucose_filter_analysis['glucose_only_enabled']}")
        print(f"   Original Records: {glucose_filter_analysis['original_records']:,}")
        print(f"   Records After Filtering: {glucose_filter_analysis['records_after_filtering']:,}")
        print(f"   Records Removed (No Glucose): {glucose_filter_analysis['records_removed']:,}")
        if glucose_filter_analysis['fields_removed']:
            print(f"   Fields Removed: {', '.join(glucose_filter_analysis['fields_removed'])}")
    
    # Data Quality
    quality = stats['data_quality']
    print(f"\n✅ DATA QUALITY:")
    print(f"   Glucose Data Completeness: {quality['glucose_data_completeness']:.1f}%")
    print(f"   Insulin Data Completeness: {quality['insulin_data_completeness']:.1f}%")
    print(f"   Carb Data Completeness: {quality['carb_data_completeness']:.1f}%")
    print(f"   Interpolated Records: {quality['interpolated_records']:,}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Configuration
    CSV_FOLDER = "000-csv"  # Folder containing multiple CSV files
    CONSOLIDATED_FILE = "consolidated_glucose_data.csv"  # Consolidated data file
    OUTPUT_FILE = "glucose_ml_ready.csv"  # Final ML-ready output
    
    # Processing mode - consolidation is now mandatory
    
    # Initialize preprocessor with configurable parameters
    preprocessor = GlucoseMLPreprocessor(
        expected_interval_minutes=5,   # Time discretization interval
        small_gap_max_minutes=15,      # Maximum gap size to interpolate
        remove_calibration=True,       # Remove calibration events to create interpolatable gaps
        min_sequence_len=200,          # Minimum sequence length to keep for ML training
        calibration_period_minutes=60*2 + 45,  # Gap duration considered as calibration period (2h 45m)
        remove_after_calibration_hours=24      # Hours of data to remove after calibration period
    )
    
    # Process data
    try:
       
        # Start from CSV folder and consolidate (mandatory step)
        print("Starting glucose data processing from CSV folder...")
        ml_data, statistics = preprocessor.process(CSV_FOLDER, OUTPUT_FILE)
        
        # Print statistics
        print_statistics(statistics, preprocessor)
        
        # Show sample of processed data
        print(f"\n📋 SAMPLE OF PROCESSED DATA:")
        print(ml_data.head(10))
        
        print(f"\n💾 Output file: {OUTPUT_FILE}")
        print(f"📈 Ready for machine learning training!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

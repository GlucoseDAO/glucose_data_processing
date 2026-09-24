#!/usr/bin/env python3
"""
Database converters for different glucose monitoring database types.

This module provides converters that handle the consolidation and processing
of different database structures (mono-user vs multi-user).
"""

import csv
from abc import ABC, abstractmethod
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import polars as pl
from loguru import logger
from formats.base_converter import CSVFormatConverter
from formats.format_detector import CSVFormatDetector


DATA_FILE_SUFFIXES: frozenset[str] = frozenset({".csv", ".txt"})

# Reasons a data file contributed no rows, used for the aggregated end-of-run report.
FILE_MATCHED = "matched a converter"
FILE_NO_CONVERTER = "no converter recognised the header"
FILE_NO_HEADER = "header line not found"
FILE_READ_ERROR = "read error"


# Event quantities that add up when several rows share a timestamp (two doses entered at
# the same minute are two doses). Everything else keeps its first value.
SUMMED_EVENT_FIELDS: frozenset[str] = frozenset({"fast_acting_insulin_u", "long_acting_insulin_u", "carb_grams"})


def merge_same_timestamp_rows(df: pl.DataFrame) -> pl.DataFrame:
    """
    Collapse rows sharing (timestamp, user_id) into one, sorted by (timestamp, user_id).

    - Exact duplicate rows are dropped first: consecutive exports overlap in time, so the
      same record can arrive twice.
    - ``SUMMED_EVENT_FIELDS`` are summed over the remaining rows (null if none has a value).
    - Every other column takes its first value in the frame's current order.

    Row converters write "" for fields a row does not carry (an EGV row's insulin), so in
    string columns "" is treated as missing; otherwise it would win over a real value.
    """
    group_cols = ['timestamp', 'user_id']
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in df.columns if c in SUMMED_EVENT_FIELDS]
    ).with_columns(
        [pl.col(c).replace("", None) for c, dtype in df.schema.items() if dtype == pl.String and c not in SUMMED_EVENT_FIELDS]
    ).unique(maintain_order=True)

    def present(col: str) -> pl.Expr:
        return pl.col(col).filter(pl.col(col).is_not_null())

    agg_exprs = [
        pl.when(pl.col(col).is_not_null().any()).then(pl.col(col).sum()).alias(col)
        if col in SUMMED_EVENT_FIELDS
        else present(col).first().alias(col)
        for col in df.columns
        if col not in group_cols
    ]
    return df.group_by(group_cols).agg(agg_exprs).sort(group_cols)


class DatabaseConverter(ABC):
    """Base class for database converters."""
    
    def __init__(self, config: Dict[str, Any], output_fields: Optional[List[str]] = None, database_type: Optional[str] = None):
        """
        Initialize the database converter.
        
        Args:
            config: Configuration dictionary with database-specific settings
            output_fields: List of field names to include in converter output.
                          If None, uses default fields matching current usage.
            database_type: String identifier for the database type (e.g., 'uom', 'ai_readi')
        """
        self.config = config
        self.output_fields = output_fields
        self.database_type = database_type
        self.format_detector = CSVFormatDetector(output_fields)
        # How each examined data file fared, keyed by one of the FILE_* reasons.
        self.file_report: Counter[str] = Counter()
        # Subjects whose rows were all removed by database-specific filtering.
        self.subjects_without_trace_rows: int = 0

    def describe_file_report(self) -> str:
        """Summarise how many examined files matched a converter, grouped by reason."""
        total = sum(self.file_report.values())
        matched = self.file_report.get(FILE_MATCHED, 0)
        others = ", ".join(
            f"{reason}: {count}" for reason, count in sorted(self.file_report.items()) if reason != FILE_MATCHED
        )
        report = f"{matched} of {total} data files matched a converter" + (f" ({others})" if others else "")
        if self.subjects_without_trace_rows:
            report += (
                f"; {self.subjects_without_trace_rows} subject folder(s) had no rows left after "
                f"{self.database_type}-specific filtering"
            )
        return report

    def _read_converted_rows(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Detect the format of one CSV/TXT file and convert its rows.

        Files that no converter recognises are skipped and counted in ``file_report``;
        the caller decides whether an empty result is fatal.
        """
        data: List[Dict[str, Any]] = []
        try:
            converter = self.format_detector.detect_format(file_path)
            if converter is None:
                logger.debug(f"No converter recognised {file_path}, skipping file")
                self.file_report[FILE_NO_CONVERTER] += 1
                return data

            with open(file_path, 'r', encoding='utf-8-sig') as file:  # utf-8-sig handles BOM
                lines = file.readlines()

            # Find the line with headers
            header_line_num = None
            delimiter = converter.get_csv_delimiter()
            for line_num in range(min(15, len(lines))):
                line = lines[line_num].strip()
                if not line:
                    continue
                headers = next(csv.reader(StringIO(line), delimiter=delimiter))
                # Clean headers: remove quotes and strip whitespace
                headers = [col.strip().strip('"') for col in headers]
                if converter.can_handle(headers):
                    header_line_num = line_num
                    break

            if header_line_num is None:
                logger.info(f"Could not find headers for {file_path}")
                self.file_report[FILE_NO_HEADER] += 1
                return data

            header_line = lines[header_line_num].strip()
            # Fallback heuristic if converter didn't specify
            if delimiter == "," and header_line.count(";") > header_line.count(","):
                delimiter = ";"
            reader = csv.DictReader(StringIO(''.join(lines[header_line_num:])), delimiter=delimiter)

            for row in reader:
                converted_record = converter.convert_row(row)
                if converted_record is not None:
                    data.append(converted_record)
            self.file_report[FILE_MATCHED] += 1
        except Exception as e:
            logger.info(f"Error processing {file_path}: {e}")
            self.file_report[FILE_READ_ERROR] += 1

        return data

    @staticmethod
    def _records_to_frame(records: List[Dict[str, Any]]) -> pl.DataFrame:
        """Build an all-string DataFrame from converted records, filling missing output fields."""
        output_fields = CSVFormatConverter.get_output_fields()
        for record in records:
            for field in output_fields:
                if field not in record:
                    record[field] = None
            # Coerce any non-string values to strings to avoid Polars schema inference conflicts
            for k, v in list(record.items()):
                if v is None or isinstance(v, str):
                    continue
                record[k] = str(v)
        all_columns: set[str] = set()
        for record in records:
            all_columns.update(record.keys())
        schema_overrides = {col: pl.Utf8 for col in sorted(all_columns)}
        return pl.DataFrame(records, schema_overrides=schema_overrides)

    @staticmethod
    def _parse_timestamps(df: pl.DataFrame) -> pl.DataFrame:
        """Parse string timestamps in the supported layouts and drop rows that fail to parse."""
        if df['timestamp'].dtype in [pl.Utf8, pl.String]:
            df = df.with_columns(
                pl.coalesce(
                    pl.col('timestamp').str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False),
                    pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
                ).alias('timestamp')
            )
        return df.filter(pl.col('timestamp').is_not_null())

    def _get_start_with_user_id(self) -> Optional[str]:
        """Get the start_with_user_id parameter for this database from config."""
        if not self.database_type:
            return None
        db_configs = self.config.get("database_configs", {})
        db_config = db_configs.get(self.database_type, {})
        return str(db_config.get("start_with_user_id")) if "start_with_user_id" in db_config else None

    @abstractmethod
    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        """
        Consolidate data from the database folder.
        
        Args:
            data_folder: Path to folder containing data files
            output_file: Optional path to save consolidated data
            
        Returns:
            Consolidated DataFrame
        """
        pass

    @abstractmethod
    def iter_user_event_frames(self, data_folder: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        """
        Iterate over users and return a DataFrame for each user.
        
        Args:
            data_folder: Path to the data folder
            interval_minutes: Expected interval between readings
            
        Returns:
            An iterable of DataFrames, one per user
        """
        pass
    
    def _enforce_output_schema(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """
        Enforce that all default output fields are present in the DataFrame.
        Adds missing columns with null/empty string values to ensure schema consistency.
        
        Args:
            df: DataFrame or LazyFrame to enforce schema on
            
        Returns:
            DataFrame or LazyFrame with all default output fields present
        """
        # Get output fields from CSVFormatConverter (standard names)
        output_fields = CSVFormatConverter.get_output_fields()
        
        # For LazyFrame, use collect_schema() to avoid performance warning
        is_lazy = isinstance(df, pl.LazyFrame)
        existing_columns = df.collect_schema().names() if is_lazy else df.columns
        
        # Add user_id for multi-user databases (it's added during processing)
        required_fields = output_fields.copy()
        
        # Add missing columns with empty-string placeholders.
        for field in required_fields:
            if field not in existing_columns:
                df = df.with_columns(pl.lit(None).alias(field))
        
        # Update existing columns list after additions
        existing_columns = df.collect_schema().names() if is_lazy else df.columns
        
        # Ensure columns are in the correct order: timestamp first, then other fields
        ordered_columns = []
        
        # Add required fields in order
        for field in required_fields:
            if field in existing_columns:
                ordered_columns.append(field)
        
        # Add any remaining columns
        for col in existing_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        # Reorder columns
        df = df.select(ordered_columns)
        
        return df
    
    @abstractmethod
    def get_database_name(self) -> str:
        """
        Get the name of the database type this converter handles.
        
        Returns:
            String name of the database type
        """
        pass


class MonoUserDatabaseConverter(DatabaseConverter):
    """Converter for mono-user databases (Dexcom, Libre3)."""
    
    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        """
        Consolidate mono-user data from multiple CSV files.
        
        Args:
            data_folder: Path to folder containing CSV files
            output_file: Optional path to save consolidated data
            
        Returns:
            Consolidated DataFrame with processed data
        """
        all_dfs = list(self.iter_user_event_frames(data_folder, interval_minutes=5))
        if not all_dfs:
            raise ValueError(f"No valid data found in {data_folder}: {self.describe_file_report()}")

        df = pl.concat(all_dfs, how="diagonal_relaxed")
        
        # Write to output file
        if output_file:
            logger.info(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
        
        logger.info(f"OK: Consolidation complete!")
        logger.info(f"Total records in output: {len(df):,}")
        
        # Show date range
        if len(df) > 0 and 'timestamp' in df.columns:
            # Format timestamp for display
            first_date = df['timestamp'][0].strftime('%Y-%m-%dT%H:%M:%S') if hasattr(df['timestamp'][0], 'strftime') else str(df['timestamp'][0])
            last_date = df['timestamp'][-1].strftime('%Y-%m-%dT%H:%M:%S') if hasattr(df['timestamp'][-1], 'strftime') else str(df['timestamp'][-1])
            logger.info(f"Date range: {first_date} to {last_date}")

        return df

    def iter_user_event_frames(self, data_folder: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        """
        Iterate over users and yield one DataFrame per user.

        Users are identified in this order:
        - a ``user_id`` the row converter set (e.g. ``PtID`` in shared clinical-trial files);
        - otherwise the first-level subfolder the file sits in, so a root holding one folder
          per subject (BIG IDEAs, D1NAMO) yields one user per folder;
        - otherwise, for files directly in ``data_folder``, the folder's own name.
        """
        csv_path = Path(data_folder)

        if not csv_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        if not csv_path.is_dir():
            raise ValueError(f"Input must be a directory containing CSV files, got: {data_folder}")

        subject_groups = self._group_files_by_subject(csv_path)
        if not subject_groups:
            logger.warning(f"No CSV or TXT files found in directory: {data_folder}")
            return

        n_files = sum(len(files) for _, files in subject_groups)
        logger.info(f"Found {n_files} files in {len(subject_groups)} subject folder(s) to consolidate")

        for folder_user_id, files in subject_groups:
            records: List[Dict[str, Any]] = []
            for data_file in files:
                for record in self._read_converted_rows(data_file):
                    if not record.get('user_id'):
                        record['user_id'] = folder_user_id
                    records.append(record)

            if not records:
                continue

            df = self._enforce_output_schema(self._records_to_frame(records))
            df = self._parse_timestamps(df)
            # Before de-duplication, so a dropped row cannot lend its event_type to a kept
            # row that shares its timestamp.
            df = self._drop_non_trace_rows(df)
            if df.height == 0:
                self.subjects_without_trace_rows += 1
                continue

            df = merge_same_timestamp_rows(df)

            user_frames = df.partition_by('user_id', as_dict=True, maintain_order=True)
            for (user_id,), user_df in sorted(user_frames.items(), key=lambda item: item[0]):
                logger.info(f"Consolidated {len(user_df):,} records for user {user_id}")
                yield self._apply_database_specific_processing(user_df)

    def _drop_non_trace_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove converted rows that must not reach the glucose trace (overridden by
        subclasses). Runs on string columns, before per-timestamp de-duplication.
        """
        return df

    @staticmethod
    def _group_files_by_subject(root: Path) -> List[Tuple[str, List[Path]]]:
        """
        Group data files by subject: files directly under ``root`` belong to ``root.name``,
        files anywhere below a first-level subfolder belong to that subfolder's name.
        Groups and files are sorted for a deterministic processing order.
        """
        groups: Dict[str, List[Path]] = {}
        for data_file in sorted(root.glob("**/*")):
            if not data_file.is_file() or data_file.suffix.lower() not in DATA_FILE_SUFFIXES:
                continue
            relative = data_file.relative_to(root)
            subject = root.name if len(relative.parts) == 1 else relative.parts[0]
            groups.setdefault(subject, []).append(data_file)
        return sorted(groups.items(), key=lambda item: item[0])

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply database-specific processing (to be overridden by subclasses).
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        return df
    
    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Mono-User Database"


class MultiUserDatabaseConverter(DatabaseConverter):
    """Converter for multi-user databases (UoM, Zendo)."""
    
    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        """
        Consolidate multi-user data from multiple CSV files.
        """
        # For multi-user, we use the iterator to collect all data
        all_dfs = []
        for user_df in self.iter_user_event_frames(data_folder, interval_minutes=5):
            all_dfs.append(user_df)
            
        if not all_dfs:
            raise ValueError("No valid data found in data files!")
            
        # Combine all user dataframes
        df = pl.concat(all_dfs)
        
        # Sort by user_id and timestamp (each user's data sorted individually)
        df = df.sort(['user_id', 'timestamp'])
        
        # Write to output file
        if output_file:
            logger.info(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
        
        logger.info(f"OK: Multi-user consolidation complete!")
        logger.info(f"Total records in output: {len(df):,}")
        
        # Show user statistics
        user_counts = df.group_by('user_id').len().sort('user_id')
        logger.info(f"Users processed: {len(user_counts)}")
        for row in user_counts.iter_rows(named=True):
            logger.info(f"  User {row['user_id']}: {row['len']:,} records")

        return df

    def iter_user_event_frames(self, data_folder: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        """
        Iterate over users and return a DataFrame for each user.
        """
        data_path = Path(data_folder)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        if not data_path.is_dir():
            raise ValueError(f"Input must be a directory, got: {data_folder}")
        
        # Process each user separately (sorted for deterministic processing order)
        users_processed = self._identify_users(data_path)
        
        # Apply start_with_user_id skipping if specified
        start_user_id = self._get_start_with_user_id()
        sorted_users_list = sorted(users_processed.items(), key=lambda x: x[0])
        
        if start_user_id:
            start_index = 0
            found = False
            for i, (user_id, _) in enumerate(sorted_users_list):
                if user_id == start_user_id:
                    start_index = i
                    found = True
                    break
            if found:
                logger.info(f"Skipping users before {start_user_id} (found at index {start_index})")
                sorted_users_list = sorted_users_list[start_index:]
            else:
                logger.info(f"Warning: start_with_user_id '{start_user_id}' not found in database. Processing all users.")

        # Apply first_n_users filtering if specified
        first_n_users = self.config.get('first_n_users')
        if first_n_users and first_n_users > 0:
            final_users_list = sorted_users_list[:first_n_users]
            logger.info(f"Found {len(users_processed)} users to process (limited to first {first_n_users} users)")
        else:
            final_users_list = sorted_users_list
            logger.info(f"Found {len(users_processed)} users to process")
        
        for user_id, user_files in final_users_list:
            user_data = self._process_user_data(user_id, user_files)
            if not user_data:
                continue
                
            df = self._enforce_output_schema(self._records_to_frame(user_data))
            df = self._parse_timestamps(df)

            # Sort by user_id and timestamp
            df = df.sort(['user_id', 'timestamp'])
            
            # Apply database-specific processing
            df = self._apply_database_specific_processing(df)
            
            yield df

    def _identify_users(self, data_path: Path) -> Dict[str, List[Path]]:
        """
        Identify users and their associated files.
        
        Args:
            data_path: Path to the data folder
            
        Returns:
            Dictionary mapping user_id to list of file paths
        """
        users = {}
        
        # Get all CSV files (sorted for deterministic processing order)
        csv_files = sorted(data_path.glob("**/*.csv"))
        
        # Deduplicate files by stem to avoid processing same data in different folders
        seen_stems = set()
        unique_files = []
        for f in csv_files:
            if f.stem not in seen_stems:
                unique_files.append(f)
                seen_stems.add(f.stem)
        
        for csv_file in unique_files:
            user_id = self._extract_user_id_from_filename(csv_file)
            if user_id:
                if user_id not in users:
                    users[user_id] = []
                users[user_id].append(csv_file)
        
        # Sort files within each user for deterministic processing
        for user_id in users:
            users[user_id] = sorted(users[user_id])
        
        return users
    
    def _extract_user_id_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Extract user ID from filename (to be overridden by subclasses).
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            User ID string or None if not found
        """
        return None
    
    def _process_user_data(self, user_id: str, user_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Process all files for a single user.
        
        Args:
            user_id: User identifier
            user_files: List of file paths for this user
            
        Returns:
            List of processed records for this user
        """
        user_data = []
        
        # Sort files for deterministic processing order
        for file_path in sorted(user_files):
            file_data = self._process_csv_file(file_path, user_id)
            user_data.extend(file_data)
        
        return user_data
    
    def _process_csv_file(self, file_path: Path, user_id: str) -> List[Dict[str, Any]]:
        """Process a single CSV file for one user and tag every record with that user_id."""
        data = self._read_converted_rows(file_path)
        for record in data:
            record['user_id'] = user_id
        return data

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply database-specific processing (to be overridden by subclasses).
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        return df
    
    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Multi-User Database"

#!/usr/bin/env python3
"""
Database converters for different glucose monitoring database types.

This module provides converters that handle the consolidation and processing
of different database structures (mono-user vs multi-user).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import polars as pl
from loguru import logger
from formats.base_converter import CSVFormatConverter
from formats.format_detector import CSVFormatDetector


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
        # For backward compatibility and simple cases, we just use the first user
        # (which is the only user in MonoUser)
        all_dfs = list(self.iter_user_event_frames(data_folder, interval_minutes=5))
        if not all_dfs:
            raise ValueError("No valid data found in CSV files!")
        
        df = all_dfs[0]
        
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
        Iterate over users and return a DataFrame for each user.
        For MonoUserDatabaseConverter, this yields a single DataFrame for the entire folder.
        """
        csv_path = Path(data_folder)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        if not csv_path.is_dir():
            raise ValueError(f"Input must be a directory containing CSV files, got: {data_folder}")
        
        # Use folder name as user_id for mono-user databases
        user_id = csv_path.name
        
        all_data = []
        
        # Get all CSV and TXT files (including in subdirectories, sorted for deterministic processing order)
        csv_files = list(csv_path.glob("**/*.csv"))
        txt_files = list(csv_path.glob("**/*.txt"))
        all_files = sorted(csv_files + txt_files)
        
        if not all_files:
            logger.warning(f"No CSV or TXT files found in directory: {data_folder}")
            return
        
        logger.info(f"Found {len(all_files)} files to consolidate for user {user_id}")
        
        for data_file in all_files:
            file_data = self._process_csv_file(data_file)
            # Add user_id to each record
            for record in file_data:
                record['user_id'] = user_id
            all_data.extend(file_data)
        
        if not all_data:
            return
            
        # Ensure all records have all required fields before DataFrame creation.
        output_fields = CSVFormatConverter.get_output_fields()
        for record in all_data:
            for field in output_fields:
                if field not in record:
                    record[field] = None
            # Coerce any non-string values to strings to avoid Polars schema inference conflicts
            for k, v in list(record.items()):
                if v is None or isinstance(v, str):
                    continue
                record[k] = str(v)
        
        # Convert to DataFrame with an explicit string schema
        all_columns: set[str] = set()
        for record in all_data:
            all_columns.update(record.keys())
        schema_overrides = {col: pl.Utf8 for col in all_columns}
        df = pl.DataFrame(all_data, schema_overrides=schema_overrides)
        
        # Enforce output schema
        df = self._enforce_output_schema(df)
        
        # Parse timestamps and sort
        logger.info(f"Parsing timestamps and sorting for user {user_id}...")
        timestamp_col_type = df['timestamp'].dtype
        if timestamp_col_type in [pl.Utf8, pl.String]:
            df = df.with_columns(
                pl.coalesce(
                    pl.col('timestamp').str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False),
                    pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
                ).alias('timestamp')
            )
        
        # Remove rows where timestamp parsing failed
        df = df.filter(pl.col('timestamp').is_not_null())
        
        # Sort by timestamp (oldest first)
        df = df.sort('timestamp')
        
        # De-duplicate records with identical timestamps
        logger.info(f"De-duplicating records for user {user_id}...")
        group_cols = ['timestamp', 'user_id']
            
        agg_exprs = []
        for col in df.columns:
            if col in group_cols:
                continue
            agg_exprs.append(
                pl.col(col).filter(pl.col(col).is_not_null()).first().alias(col)
            )
        
        df = df.group_by(group_cols).agg(agg_exprs).sort(group_cols)
        
        # Apply database-specific processing
        df = self._apply_database_specific_processing(df)
        
        yield df
    
    def _process_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single CSV file and extract required fields using format detection."""
        data = []
        
        try:
            # Detect the format of the CSV file
            converter = self.format_detector.detect_format(file_path)
            
            if converter is None:
                logger.info(f"Warning: Could not detect format for {file_path}, skipping file")
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
                    
                    # Parse CSV line properly using the converter's delimiter
                    import csv
                    from io import StringIO
                    csv_reader = csv.reader(StringIO(line), delimiter=delimiter)
                    headers = next(csv_reader)
                    
                    # Clean headers: remove quotes and strip whitespace
                    headers = [col.strip().strip('"') for col in headers]
                    
                    if converter.can_handle(headers):
                        header_line_num = line_num
                        break
                
                if header_line_num is None:
                    logger.info(f"Could not find headers for {file_path}")
                    return data
                
                # Create a new file-like object starting from the header line
                from io import StringIO
                csv_content = ''.join(lines[header_line_num:])
                csv_file = StringIO(csv_content)
                
                import csv
                header_line = lines[header_line_num].strip()
                delimiter = converter.get_csv_delimiter()
                # Fallback heuristic if converter didn't specify
                if delimiter == "," and header_line.count(";") > header_line.count(","):
                    delimiter = ";"
                reader = csv.DictReader(csv_file, delimiter=delimiter)
                
                for row in reader:
                    # Use the appropriate converter to process the row
                    converted_record = converter.convert_row(row)
                    if converted_record is not None:
                        data.append(converted_record)
                    
        except Exception as e:
            logger.info(f"Error processing {file_path}: {e}")
        
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
                
            # Ensure all records have all required fields
            output_fields = CSVFormatConverter.get_output_fields()
            for record in user_data:
                for field in output_fields:
                    if field not in record:
                        record[field] = None
                # Coerce any non-string values to strings to avoid Polars schema inference conflicts
                for k, v in list(record.items()):
                    if v is None or isinstance(v, str):
                        continue
                    record[k] = str(v)
            
            # Convert to DataFrame with an explicit string schema
            all_columns: set[str] = set()
            for record in user_data:
                all_columns.update(record.keys())
            schema_overrides = {col: pl.Utf8 for col in all_columns}
            df = pl.DataFrame(user_data, schema_overrides=schema_overrides)
            
            # Enforce output schema
            df = self._enforce_output_schema(df)
            
            # Parse timestamps and sort
            timestamp_col_type = df['timestamp'].dtype
            if timestamp_col_type in [pl.Utf8, pl.String]:
                df = df.with_columns(
                    pl.coalesce(
                        pl.col('timestamp').str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False),
                        pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
                        pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
                    ).alias('timestamp')
                )
            
            # Remove rows where timestamp parsing failed
            df = df.filter(pl.col('timestamp').is_not_null())
            
            # Sort by user_id and timestamp
            df = df.sort(['user_id', 'timestamp'])
            
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
        """Process a single CSV file and extract required fields using format detection."""
        data = []
        
        try:
            # Detect the format of the CSV file
            converter = self.format_detector.detect_format(file_path)
            
            if converter is None:
                logger.info(f"Warning: Could not detect format for {file_path}, skipping file")
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
                    
                    # Parse CSV line properly using the converter's delimiter
                    import csv
                    from io import StringIO
                    csv_reader = csv.reader(StringIO(line), delimiter=delimiter)
                    headers = next(csv_reader)
                    
                    # Clean headers: remove quotes and strip whitespace
                    headers = [col.strip().strip('"') for col in headers]
                    
                    if converter.can_handle(headers):
                        header_line_num = line_num
                        break
                
                if header_line_num is None:
                    logger.info(f"Could not find headers for {file_path}")
                    return data
                
                # Create a new file-like object starting from the header line
                from io import StringIO
                csv_content = ''.join(lines[header_line_num:])
                csv_file = StringIO(csv_content)
                
                import csv
                header_line = lines[header_line_num].strip()
                delimiter = converter.get_csv_delimiter()
                # Fallback heuristic if converter didn't specify
                if delimiter == "," and header_line.count(";") > header_line.count(","):
                    delimiter = ";"
                reader = csv.DictReader(csv_file, delimiter=delimiter)
                
                for row in reader:
                    # Use the appropriate converter to process the row
                    converted_record = converter.convert_row(row)
                    if converted_record is not None:
                        # Add user_id to the record
                        converted_record['user_id'] = user_id
                        data.append(converted_record)
                    
        except Exception as e:
            logger.info(f"Error processing {file_path}: {e}")
        
        return data
    
    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Multi-User Database"

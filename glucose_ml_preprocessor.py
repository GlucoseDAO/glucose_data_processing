#!/usr/bin/env python3
"""
Glucose Data Preprocessor for Machine Learning

This script processes glucose monitoring data for ML training by:
1. Detecting time gaps in the data
2. Interpolating missing values for gaps <= 10 minutes
3. Creating sequence IDs for continuous data segments
4. Creating fixed-frequency data with consistent intervals
5. Providing statistics about the processed data
"""

import polars as pl
from typing import Tuple, Dict, Any, List, Optional, Iterable, Union, cast
from datetime import datetime
from pathlib import Path
from loguru import logger
import warnings
import yaml
import sys
import json
import concurrent.futures
import os
import contextlib
import multiprocessing

# Import database detection and conversion classes
from formats import DatabaseDetector
from formats.base_converter import CSVFormatConverter

# Import modular processing components
from processing.core.fields import StandardFieldNames, INTERPOLATED_EVENT_TYPE
from processing.core.config import extract_field_categories, get_schema_file
from processing.steps.gap_detection import GapDetector
from processing.steps.data_cleaning import DataCleaner
from processing.steps.interpolation import ValueInterpolator
from processing.steps.filtering import SequenceFilter
from processing.steps.fixed_frequency import FixedFreqGenerator
from processing.steps.ml_prep import MLDataPreparer
from processing.stats_manager import StatsManager

warnings.filterwarnings('ignore')

# Constants for common field names and literal values
DEFAULT_STREAMING_MAX_BUFFER_MB = 256
DEFAULT_STREAMING_FLUSH_MAX_USERS = 10
MIN_BUFFER_MB = 32

_worker_lock = None

def _init_worker_process(params: Dict[str, Any]) -> None:
    """Initialize worker process: configure logging and shared state."""
    verbose = params.get('verbose', False)
    config = params.get('config')
    
    global _worker_lock
    _worker_lock = params.get('lock')
    
    # Configure loguru to show only the message, matching the CLI behavior
    logger.remove()
    if verbose:
        logger.add(sys.stdout, format="{message}")
    
    if config:
        CSVFormatConverter.initialize_from_config(config)

def _run_processing_pipeline(
    df: pl.DataFrame,
    last_sequence_id: int,
    field_categories_dict: Optional[Dict[str, Any]],
    gap_detector: GapDetector,
    data_cleaner: DataCleaner,
    interpolator: ValueInterpolator,
    filter_step: SequenceFilter,
    fixed_freq_generator: FixedFreqGenerator,
    ml_preparer: MLDataPreparer,
    stats_manager: StatsManager,
    create_fixed_frequency: bool,
    log_steps: bool = False,
    last_step: int = 0,
    save_intermediate: bool = False,
    output_dir: Optional[Path] = None,
    output_prefix: Optional[str] = None,
    expected_standard_cols: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
    """
    Common processing pipeline used by both sequential and parallel processing modes.
    Steps:
    2. Data cleaning (removing covariates in large glucose gaps)
    3. Gap detection and sequence creation
    4. Interpolation
    5. Filtering by length
    6. Fixed-frequency data creation
    7. Filtering to glucose-only data
    8. ML data preparation
    """
    cleaning_stats = {}
    gap_stats = {}
    interp_stats = {}
    filter_stats = {}
    fixed_freq_stats = {}
    glucose_filter_stats = {}

    def _save_intermediate(current_df: pl.DataFrame, step_num: int):
        if not save_intermediate or output_prefix is None:
            return
        
        target_dir = output_dir if output_dir is not None else Path("OUTPUT")
        target_dir.mkdir(exist_ok=True, parents=True)
        
        # Sanitize output_prefix to avoid path issues with .zip in name
        clean_prefix = str(output_prefix).replace(".zip", "_zip").replace(".", "_").replace("/", "_").replace("\\", "_")
        filename = f"{clean_prefix}_step_{step_num}.csv"
        target_path = target_dir / filename

        # Align columns to expected_standard_cols if available
        if expected_standard_cols:
            missing = [c for c in expected_standard_cols if c not in current_df.columns]
            if missing:
                current_df = current_df.with_columns([pl.lit(None).alias(c) for c in missing])
            
            # Use only expected columns and in correct order
            available_expected = [c for c in expected_standard_cols if c in current_df.columns]
            current_df = current_df.select(available_expected)
        
        # Use a global lock if available to prevent interleaved writes from multiple processes
        global _worker_lock
        lock_ctx = _worker_lock if _worker_lock is not None else contextlib.nullcontext()
        
        with lock_ctx:
            # When accumulating, we only write header if the file is new
            include_header = not target_path.exists()
            with open(target_path, "ab") as f:
                current_df.write_csv(f, include_header=include_header)

    # Internal helper to handle early exit with data preparation
    def early_exit(current_df: pl.DataFrame, c_stats, g_stats, i_stats, f_stats, ff_stats, gf_stats, last_seq_id):
        # Always prepare data (cast and rename) for consistent output even on early exit
        new_last_seq_id = last_seq_id
        if StandardFieldNames.SEQUENCE_ID not in current_df.columns:
            # If we are exiting before Step 3 (Gap Detection), we haven't assigned sequence IDs yet.
            # To allow multi-user merging to work correctly in Step 1 and 2, 
            # we assign a temporary sequence ID of 1 to this user's chunk.
            # This ensures that when handle_result() remaps IDs, each user gets a unique ID.
            current_df = current_df.with_columns(pl.lit(1).alias(StandardFieldNames.SEQUENCE_ID))
            new_last_seq_id = 1
        
        ml_df = ml_preparer.prepare_ml_data(current_df, field_categories_dict)
        stats = stats_manager.get_statistics(ml_df, g_stats, i_stats, f_stats, gf_stats, ff_stats, cleaning_stats=c_stats)
        return ml_df, stats, new_last_seq_id

    if last_step == 1:
        _save_intermediate(df, 1)
        return early_exit(df, cleaning_stats, gap_stats, interp_stats, filter_stats, fixed_freq_stats, glucose_filter_stats, last_sequence_id)

    # STEP 2: Data cleaning (removing covariates in large glucose gaps)
    if log_steps:
        logger.info("STEP 2: Data cleaning (removing covariates in large glucose gaps)...")
    df, cleaning_stats = data_cleaner.clean_remote_data(df, field_categories_dict)
    _save_intermediate(df, 2)
    if last_step == 2:
        return early_exit(df, cleaning_stats, gap_stats, interp_stats, filter_stats, fixed_freq_stats, glucose_filter_stats, last_sequence_id)

    # STEP 3: Detecting gaps and creating sequences
    if log_steps:
        logger.info("STEP 3: Detecting gaps and creating sequences...")
    df, gap_stats, last_sequence_id = gap_detector.detect_gaps_and_sequences(df, last_sequence_id, field_categories_dict)
    _save_intermediate(df, 3)
    if last_step == 3:
        return early_exit(df, cleaning_stats, gap_stats, interp_stats, filter_stats, fixed_freq_stats, glucose_filter_stats, last_sequence_id)
    
    # STEP 4: Interpolating missing values
    if log_steps:
        logger.info("STEP 4: Interpolating missing values...")
    df, interp_stats = interpolator.interpolate_missing_values(df, field_categories_dict)
    _save_intermediate(df, 4)
    if last_step == 4:
        return early_exit(df, cleaning_stats, gap_stats, interp_stats, filter_stats, fixed_freq_stats, glucose_filter_stats, last_sequence_id)
    
    # STEP 5: Filtering sequences by minimum length
    if log_steps:
        logger.info("STEP 5: Filtering sequences by minimum length...")
    df, filter_stats = filter_step.filter_sequences_by_length(df)
    _save_intermediate(df, 5)
    if last_step == 5:
        return early_exit(df, cleaning_stats, gap_stats, interp_stats, filter_stats, fixed_freq_stats, glucose_filter_stats, last_sequence_id)
    
    # STEP 6: Creating fixed-frequency data
    if create_fixed_frequency:
        if log_steps:
            logger.info("STEP 6: Creating fixed-frequency data...")
        df, fixed_freq_stats = fixed_freq_generator.create_fixed_frequency_data(df, field_categories_dict)
        _save_intermediate(df, 6)
    if last_step == 6:
        return early_exit(df, cleaning_stats, gap_stats, interp_stats, filter_stats, fixed_freq_stats, glucose_filter_stats, last_sequence_id)
    
    # STEP 7: Filtering to glucose-only data
    if log_steps:
        logger.info("STEP 7: Filtering to glucose-only data...")
    df, glucose_filter_stats = filter_step.filter_glucose_only(df)
    if filter_step.glucose_only:
        _save_intermediate(df, 7)
    if last_step == 7:
        return early_exit(df, cleaning_stats, gap_stats, interp_stats, filter_stats, fixed_freq_stats, glucose_filter_stats, last_sequence_id)
    
    # STEP 8: Preparing final ML dataset
    if log_steps:
        logger.info("STEP 8: Preparing final ML dataset...")
    ml_df = ml_preparer.prepare_ml_data(df, field_categories_dict)
    _save_intermediate(df, 8)
    
    stats = stats_manager.get_statistics(
        ml_df, gap_stats, interp_stats, filter_stats, glucose_filter_stats, fixed_freq_stats, cleaning_stats=cleaning_stats
    )
    
    return ml_df, stats, last_sequence_id

def _process_user_frame_task(
    user_df: pl.DataFrame,
    params: Dict[str, Any],
    field_categories_dict: Optional[Dict[str, Any]],
    expected_cols: List[str],
    expected_standard_cols: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
    """
    Standalone task for processing a single user's data in a worker process.
    """
    save_intermediate_files = params.get('save_intermediate_files', False)
    # Initialize components
    gap_detector = GapDetector(
        small_gap_max_minutes=params['small_gap_max_minutes'],
        calibration_period_minutes=params['calibration_period_minutes'],
        remove_after_calibration_hours=params['remove_after_calibration_hours']
    )
    data_cleaner = DataCleaner(
        small_gap_max_minutes=params['small_gap_max_minutes']
    )
    interpolator = ValueInterpolator(
        expected_interval_minutes=params['expected_interval_minutes'],
        small_gap_max_minutes=params['small_gap_max_minutes']
    )
    filter_step = SequenceFilter(
        min_sequence_len=params['min_sequence_len'],
        glucose_only=params['glucose_only']
    )
    fixed_freq_generator = FixedFreqGenerator(
        expected_interval_minutes=params['expected_interval_minutes']
    )
    ml_preparer = MLDataPreparer(config=params.get('config', {}))
    stats_manager = StatsManager(original_record_count=len(user_df))

    # Process using the common pipeline
    # Start with sequence ID 0 for local count
    ml_df, user_stats, user_max_id = _run_processing_pipeline(
        user_df, 0, field_categories_dict,
        gap_detector, data_cleaner, interpolator, filter_step, fixed_freq_generator, ml_preparer, stats_manager,
        params['create_fixed_frequency'],
        last_step=params.get('last_step', 0),
        save_intermediate=save_intermediate_files,
        output_dir=params.get('output_dir'),
        output_prefix=params.get('output_prefix'),
        expected_standard_cols=expected_standard_cols
    )
    
    # We return the max sequence ID from gap detection to maintain parity with sequential processing
    # where sequence IDs increment even for filtered out sequences.
    return ml_df, user_stats, user_max_id

class GlucoseMLPreprocessor:
    """
    Preprocessor for glucose monitoring data to prepare it for machine learning.
    Acts as a facade delegating work to specialized processing steps.
    """
    
    @classmethod
    def from_config_file(cls, config_path: Path, **cli_overrides: Any) -> "GlucoseMLPreprocessor":
        """
        Create a GlucoseMLPreprocessor instance from a YAML configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        CSVFormatConverter.initialize_from_config(config)
        
        dexcom_config = config.get('dexcom', {})
        high_value = dexcom_config.get('high_glucose_value', 401)
        low_value = dexcom_config.get('low_glucose_value', 39)
        
        return cls(
            expected_interval_minutes=cli_overrides.get('expected_interval_minutes', config.get('expected_interval_minutes', 5)),
            small_gap_max_minutes=cli_overrides.get('small_gap_max_minutes', config.get('small_gap_max_minutes', 15)),
            remove_calibration=cli_overrides.get('remove_calibration', dexcom_config.get('remove_calibration', True)),
            min_sequence_len=cli_overrides.get('min_sequence_len', config.get('min_sequence_len', 200)),
            save_intermediate_files=cli_overrides.get('save_intermediate_files', config.get('save_intermediate_files', False)),
            calibration_period_minutes=cli_overrides.get('calibration_period_minutes', dexcom_config.get('calibration_period_minutes', 165)),
            remove_after_calibration_hours=cli_overrides.get('remove_after_calibration_hours', dexcom_config.get('remove_after_calibration_hours', 24)),
            high_glucose_value=cli_overrides.get('high_glucose_value', high_value),
            low_glucose_value=cli_overrides.get('low_glucose_value', low_value),
            glucose_only=cli_overrides.get('glucose_only', config.get('glucose_only', False)),
            create_fixed_frequency=cli_overrides.get('create_fixed_frequency', config.get('create_fixed_frequency', True)),
            last_step=cli_overrides.get('last_step', config.get('last_step', 0)),
            verbose=cli_overrides.get('verbose', False),
            config=config,
            first_n_users=cli_overrides.get('first_n_users', config.get('first_n_users', None)),
            output_file=cli_overrides.get('output_file', config.get('output_file', None)),
            print_statistics=cli_overrides.get('print_statistics', config.get('print_statistics', True))
        )
    
    @staticmethod
    def _clear_intermediate_files(output_file: Path) -> None:
        """Clear existing intermediate files to start fresh for accumulation."""
        prefix = output_file.stem
        odir = output_file.parent
        for step in range(1, 9):
            target = odir / f"{prefix}_step_{step}.csv"
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass

    def __init__(
        self,
        expected_interval_minutes: int = 5,
        small_gap_max_minutes: int = 15,
        remove_calibration: bool = True,
        min_sequence_len: int = 200,
        save_intermediate_files: bool = False,
        calibration_period_minutes: int = 60*2 + 45,
        remove_after_calibration_hours: int = 24,
        high_glucose_value: int = 401,
        low_glucose_value: int = 39,
        glucose_only: bool = False,
        create_fixed_frequency: bool = True,
        last_step: int = 0,
        verbose: bool = False,
        config: Optional[Dict[str, Any]] = None,
        first_n_users: Optional[int] = None,
        output_file: Optional[str] = None,
        print_statistics: bool = True,
    ) -> None:
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
        self.create_fixed_frequency = create_fixed_frequency
        self.last_step = last_step
        self.verbose = verbose
        self.config = config if config is not None else {}
        self.output_file = Path(output_file) if output_file else None
        self.print_statistics = print_statistics
        if first_n_users is not None:
            self.config['first_n_users'] = first_n_users
        
        self.fields = StandardFieldNames()
        self._original_record_count: int = 0
        self._field_categories_dict: Optional[Dict[str, Any]] = None
        
        # Initialize specialized processing steps
        self.gap_detector = GapDetector(
            small_gap_max_minutes=small_gap_max_minutes,
            calibration_period_minutes=calibration_period_minutes,
            remove_after_calibration_hours=remove_after_calibration_hours
        )
        self.data_cleaner = DataCleaner(
            small_gap_max_minutes=small_gap_max_minutes
        )
        self.interpolator = ValueInterpolator(
            expected_interval_minutes=expected_interval_minutes,
            small_gap_max_minutes=small_gap_max_minutes
        )
        self.filter_step = SequenceFilter(
            min_sequence_len=min_sequence_len,
            glucose_only=glucose_only
        )
        self.fixed_freq_generator = FixedFreqGenerator(
            expected_interval_minutes=expected_interval_minutes
        )
        self.ml_preparer = MLDataPreparer(config=self.config)
        self.stats_manager = StatsManager()

    @staticmethod
    def extract_field_categories(database_type: str) -> Dict[str, Any]:
        """Legacy static method for field categories extraction."""
        return extract_field_categories(database_type)

    def _df_estimated_size_bytes(self, df: pl.DataFrame) -> int:
        try:
            return int(df.estimated_size())
        except (AttributeError, ValueError):
            return int(len(df) * max(1, len(df.columns)) * 16)

    def _streaming_buffer_max_bytes(self) -> int:
        mb = self.config.get("streaming_max_buffer_mb", DEFAULT_STREAMING_MAX_BUFFER_MB)
        try:
            mb_i = int(mb)
        except (ValueError, TypeError):
            mb_i = DEFAULT_STREAMING_MAX_BUFFER_MB
        return max(MIN_BUFFER_MB, mb_i) * 1024 * 1024

    def _streaming_flush_max_users(self) -> int:
        v = self.config.get("streaming_flush_max_users", DEFAULT_STREAMING_FLUSH_MAX_USERS)
        try:
            n = int(v)
        except (ValueError, TypeError):
            n = DEFAULT_STREAMING_FLUSH_MAX_USERS
        return max(1, n)

    def _write_csv_append(self, df: pl.DataFrame, *, output_file: Path, include_header: bool) -> None:
        if len(df) == 0:
            return
        with open(output_file, "ab") as f:
            df.write_csv(f, include_header=include_header)

    def _compute_expected_output_columns(self, database_types: List[str], use_display_names: bool = True) -> List[str]:
        field_to_display = CSVFormatConverter.get_field_to_display_name_map() if use_display_names else {}
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        user_id_col = StandardFieldNames.USER_ID
        dataset_name_col = StandardFieldNames.DATASET_NAME

        if bool(self.config.get("restrict_output_to_config_fields", False)):
            output_fields = CSVFormatConverter.get_output_fields()
            service_allow = self.config.get("service_fields_allowlist")
            service_keep = {str(x) for x in service_allow} if isinstance(service_allow, list) else set()

            cols = [seq_id_col]
            if user_id_col not in output_fields:
                cols.append(field_to_display.get(user_id_col, user_id_col))
            if len(database_types) > 1:
                cols.append(field_to_display.get(dataset_name_col, dataset_name_col))
            for c in output_fields:
                if c == seq_id_col:
                    continue
                cols.append(field_to_display.get(c, c))
            for c in sorted(service_keep):
                disp = field_to_display.get(c, c)
                if disp not in set(cols):
                    cols.append(disp)
            seen: set[str] = set()
            out: List[str] = []
            for c in cols:
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            return out

        cols: List[str] = [seq_id_col]
        # Always include user_id in standardized output
        cols.append(field_to_display.get(user_id_col, user_id_col))
        
        if len(database_types) > 1:
            cols.append(field_to_display.get(dataset_name_col, dataset_name_col))
        for f in CSVFormatConverter.get_output_fields():
            if f == seq_id_col or f == user_id_col:
                continue
            cols.append(field_to_display.get(f, f))

        extra: set[str] = set()
        for db in database_types:
            schema_file = get_schema_file(db)
            schema_path = Path(__file__).parent / "formats" / schema_file
            if not schema_path.exists():
                continue
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = yaml.safe_load(f)
            for standard_name in schema.get("field_categories", {}).keys():
                if standard_name == seq_id_col:
                    continue
                extra.add(str(field_to_display.get(standard_name, standard_name)))

        for c in sorted(extra):
            if c not in set(cols):
                cols.append(c)
        return cols

    def _process_streaming_from_converter(
        self,
        *,
        data_folder: Path,
        database_type: str,
        output_file: Path,
        last_sequence_id: int,
        field_categories_dict: Optional[Dict[str, List[str]]],
    ) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        db_detector = DatabaseDetector()
        output_fields = self.config.get("output_fields")
        database_converter = db_detector.get_database_converter(database_type, self.config or {}, output_fields=output_fields)
        if database_converter is None:
            raise ValueError(f"No converter available for database type: {database_type}")

        iter_fn = getattr(database_converter, "iter_user_event_frames", None)
        if not callable(iter_fn):
            raise ValueError(f"Converter for {database_type} does not support streaming frames")

        output_file.write_text("", encoding="utf-8")

        expected_cols = self._compute_expected_output_columns([database_type])
        expected_standard_cols = self._compute_expected_output_columns([database_type], use_display_names=False)
        max_bytes = self._streaming_buffer_max_bytes()
        max_users = self._streaming_flush_max_users()

        wrote_header = False
        current_last_sequence_id = last_sequence_id
        buffered: List[pl.DataFrame] = []
        buffered_bytes = 0
        buffered_users = 0

        total_records = 0
        total_sequences = 0
        original_records = 0
        min_ts: Optional[str] = None
        max_ts: Optional[str] = None
        
        # Stats accumulation
        all_user_stats: List[Dict[str, Any]] = []
        total_users_processed = 0
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        # Prepare for intermediate file saving if enabled
        lock = None
        if self.save_intermediate_files and output_file:
            # Use Manager().Lock() for ProcessPoolExecutor compatibility on all platforms
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            # Clear existing intermediate files to start fresh for accumulation
            self._clear_intermediate_files(output_file)

        params = {
            'expected_interval_minutes': self.expected_interval_minutes,
            'small_gap_max_minutes': self.small_gap_max_minutes,
            'remove_calibration': self.remove_calibration,
            'min_sequence_len': self.min_sequence_len,
            'calibration_period_minutes': self.calibration_period_minutes,
            'remove_after_calibration_hours': self.remove_after_calibration_hours,
            'glucose_only': self.glucose_only,
            'create_fixed_frequency': self.create_fixed_frequency,
            'last_step': self.last_step,
            'save_intermediate_files': self.save_intermediate_files,
            'output_dir': output_file.parent if output_file else Path("OUTPUT"),
            'output_prefix': output_file.stem if output_file else "processed",
            'lock': lock,
            'output_fields': output_fields,
            'config': self.config,
            'verbose': self.verbose
        }

        def flush() -> None:
            nonlocal wrote_header, buffered, buffered_bytes, buffered_users, total_records
            if not buffered:
                return
            frames = buffered
            buffered = []
            buffered_bytes = 0
            buffered_users = 0
            for frame in frames:
                total_records += len(frame)
                self._write_csv_append(frame, output_file=output_file, include_header=not wrote_header)
                wrote_header = True

        def handle_result(result: Tuple[pl.DataFrame, Dict[str, Any], int]) -> None:
            nonlocal current_last_sequence_id, total_sequences, original_records, min_ts, max_ts, buffered_bytes, buffered_users, total_users_processed
            ml_df, user_stats, user_max_id = result
            
            total_users_processed += 1
            # Log progress every 5 users
            if total_users_processed % 5 == 0:
                logger.info(f"   Processed {total_users_processed} users...")
            
            # Align columns to expected_cols
            missing = [c for c in expected_cols if c not in ml_df.columns]
            if missing:
                ml_df = ml_df.with_columns([pl.lit(None).alias(c) for c in missing])
            
            # Ensure correct order and only expected columns
            ml_df = ml_df.select(expected_cols)
            
            # Remap sequence IDs
            # Use user_max_id for remapping logic even if ml_df is empty or sequences were filtered
            if seq_id_col in ml_df.columns:
                # Always remap if we have sequence IDs, even if frame is empty (for consistency)
                ml_df = ml_df.with_columns([
                    pl.when(pl.col(seq_id_col) > 0)
                    .then(pl.col(seq_id_col) + current_last_sequence_id)
                    .otherwise(pl.col(seq_id_col))
                    .alias(seq_id_col)
                ])
            
            current_last_sequence_id += user_max_id
            
            # total_sequences should be the count of FINAL sequences kept
            if 'filtering_analysis' in user_stats and user_stats['filtering_analysis']:
                total_sequences += user_stats['filtering_analysis'].get('filtered_sequences', 0)
            else:
                total_sequences += user_stats['dataset_overview'].get('total_sequences', 0)
            
            all_user_stats.append(user_stats)
            original_records += user_stats['dataset_overview'].get('original_records', 0)
            
            # Update date range
            try:
                umin = user_stats['dataset_overview']['date_range'].get('start')
                umax = user_stats['dataset_overview']['date_range'].get('end')
                if umin and umin != "N/A":
                    min_ts = umin if (min_ts is None or umin < min_ts) else min_ts
                if umax and umax != "N/A":
                    max_ts = umax if (max_ts is None or umax > max_ts) else max_ts
            except (KeyError, AttributeError):
                pass
            
            buffered.append(ml_df)
            buffered_bytes += self._df_estimated_size_bytes(ml_df)
            buffered_users += 1
            
            if buffered_bytes >= max_bytes or buffered_users >= max_users:
                flush()

        # Use ProcessPoolExecutor for parallel processing of users
        max_workers = os.cpu_count() or 1
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker_process,
            initargs=(params,)
        ) as executor:
            # We use a sliding window of futures to avoid memory issues with thousands of users
            max_active_tasks = max_workers * 2
            futures = []
            
            # Cast iter_fn to the expected callable type for the linter
            typed_iter_fn = cast(Iterable[pl.DataFrame], iter_fn(data_folder, interval_minutes=self.expected_interval_minutes))
            for user_df in typed_iter_fn:
                if len(user_df) == 0:
                    continue
                
                # Submit new task
                futures.append(executor.submit(
                    _process_user_frame_task, 
                    user_df, 
                    params, 
                    field_categories_dict, 
                    expected_cols,
                    expected_standard_cols
                ))
                
                # If we reached the window size, wait for tasks in order to maintain sequence ID stability
                if len(futures) >= max_active_tasks:
                    handle_result(futures.pop(0).result())
            
            # Finalize remaining tasks
            while futures:
                handle_result(futures.pop(0).result())

        flush()

        # Use StatsManager to aggregate all collected user statistics
        if all_user_stats:
            stats = self.stats_manager.aggregate_statistics(all_user_stats, ["Streaming Chunk"] * len(all_user_stats))
            
            # If we only have one dataset, don't include multi-database info
            if "multi_database_info" in stats:
                del stats["multi_database_info"]
                
            # Update with overall counts and dates
            stats['dataset_overview']['total_records'] = total_records
            stats['dataset_overview']['total_sequences'] = total_sequences
            stats['dataset_overview']['original_records'] = original_records
            stats['dataset_overview']['date_range'] = {"start": min_ts or "N/A", "end": max_ts or "N/A"}
        else:
            stats = self.stats_manager.get_statistics(
                pl.DataFrame({seq_id_col: pl.Series([], dtype=pl.Int64)}),
                {}, {}, {}, {}, {}
            )

        placeholder = pl.DataFrame({seq_id_col: pl.Series([], dtype=pl.Int64)})
        return placeholder, stats, current_last_sequence_id

    def consolidate_glucose_data(self, data_folder: Path, output_file: Optional[Path] = None) -> pl.DataFrame:
        db_detector = DatabaseDetector()
        database_type = db_detector.detect_database_type(data_folder)
        logger.info(f"Detected database type: {database_type}")
        
        if database_type == 'unknown':
            raise ValueError(f"Could not detect database type for folder: {data_folder}")
        
        database_converter = db_detector.get_database_converter(database_type, self.config or {})
        if database_converter is None:
            raise ValueError(f"No converter available for database type: {database_type}")
        
        logger.info(f"Using {database_converter.get_database_name()}")
        df = database_converter.consolidate_data(data_folder, output_file)
        self._original_record_count = len(df)
        self.stats_manager.original_record_count = len(df)
        return df

    def clean_remote_data(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Wrapper for DataCleaner.clean_remote_data."""
        return self.data_cleaner.clean_remote_data(df, field_categories_dict)

    def detect_gaps_and_sequences(self, df: pl.DataFrame, last_sequence_id: int = 0, field_categories_dict: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """Wrapper for GapDetector.detect_gaps_and_sequences."""
        return self.gap_detector.detect_gaps_and_sequences(df, last_sequence_id, field_categories_dict)

    def interpolate_missing_values(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Wrapper for ValueInterpolator.interpolate_missing_values."""
        return self.interpolator.interpolate_missing_values(df, field_categories_dict)

    def filter_sequences_by_length(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Wrapper for SequenceFilter.filter_sequences_by_length."""
        return self.filter_step.filter_sequences_by_length(df)

    def create_fixed_frequency_data(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Wrapper for FixedFreqGenerator.create_fixed_frequency_data."""
        return self.fixed_freq_generator.create_fixed_frequency_data(df, field_categories_dict)

    def filter_glucose_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Wrapper for SequenceFilter.filter_glucose_only."""
        return self.filter_step.filter_glucose_only(df)

    def prepare_ml_data(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
        """Wrapper for MLDataPreparer.prepare_ml_data."""
        if field_categories_dict is None:
            field_categories_dict = self._field_categories_dict
        return self.ml_preparer.prepare_ml_data(df, field_categories_dict)

    def process(self, csv_folder: Path, output_file: Optional[Path] = None, last_sequence_id: int = 0, database_type: Optional[str] = None) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        logger.info("Starting glucose data preprocessing for ML...")
        
        if database_type is None:
            db_detector = DatabaseDetector()
            database_type = db_detector.detect_database_type(csv_folder)
            
        field_categories_dict = extract_field_categories(database_type) if database_type != 'unknown' else None
        self._field_categories_dict = field_categories_dict
        
        # Clear existing intermediate files if enabled
        if self.save_intermediate_files and output_file:
            self._clear_intermediate_files(output_file)

        db_detector = DatabaseDetector()
        database_converter = db_detector.get_database_converter(database_type, self.config or {})
        if (
            output_file
            and database_converter is not None
            and callable(getattr(database_converter, "iter_user_event_frames", None))
        ):
            return self._process_streaming_from_converter(
                data_folder=csv_folder,
                database_type=database_type,
                output_file=output_file,
                last_sequence_id=last_sequence_id,
                field_categories_dict=field_categories_dict,
            )
        
        logger.info("STEP 1: Consolidating CSV files (mandatory step)...")
        df = self.consolidate_glucose_data(csv_folder)
        
        # Use common processing pipeline for steps 2-8 (handles last_step internally)
        expected_standard_cols = self._compute_expected_output_columns([db_type], use_display_names=False)
        
        ml_df, stats, last_sequence_id = _run_processing_pipeline(
            df, last_sequence_id, field_categories_dict,
            self.gap_detector, self.data_cleaner, self.interpolator, self.filter_step, self.fixed_freq_generator, self.ml_preparer, self.stats_manager,
            self.create_fixed_frequency,
            log_steps=True,
            last_step=self.last_step,
            save_intermediate=self.save_intermediate_files,
            output_dir=output_file.parent if output_file else Path("OUTPUT"),
            output_prefix=output_file.stem if output_file else "processed",
            expected_standard_cols=expected_standard_cols
        )
        
        if output_file:
            ml_df.write_csv(output_file)
            logger.info(f"Final processed data saved to: {output_file}")
        
        return ml_df, stats, last_sequence_id

    def process_multiple_databases(self, csv_folders: List[Path], output_file: Optional[Path] = None, last_sequence_id: int = 0) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        logger.info(f"Starting multi-database processing for {len(csv_folders)} databases...")
        
        db_detector = DatabaseDetector()
        db_types: List[str] = []
        for p in csv_folders:
            try:
                db_types.append(db_detector.detect_database_type(p))
            except (ValueError, OSError):
                db_types.append("unknown")
        
        converters = [db_detector.get_database_converter(t, self.config or {}) for t in db_types]
        streaming_capable = any(c is not None and callable(getattr(c, "iter_user_event_frames", None)) for c in converters)
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        user_id_col = StandardFieldNames.USER_ID
        dataset_name_col = StandardFieldNames.DATASET_NAME
        dataset_name_display = CSVFormatConverter.get_display_name(dataset_name_col)

        if streaming_capable and output_file:
            expected_cols = self._compute_expected_output_columns(db_types)
            expected_standard_cols = self._compute_expected_output_columns(db_types, use_display_names=False)
            max_bytes = self._streaming_buffer_max_bytes()
            max_users = self._streaming_flush_max_users()
            output_file.write_text("", encoding="utf-8")

            current_last_sequence_id = last_sequence_id
            wrote_header = False
            buffered: List[pl.DataFrame] = []
            buffered_bytes = 0
            buffered_users = 0
            total_records = 0
            total_sequences = 0
            original_records = 0
            min_ts: Optional[str] = None
            max_ts: Optional[str] = None
            
            # Stats accumulation
            all_processing_stats: List[Dict[str, Any]] = []

            # Prepare for intermediate file saving if enabled
            lock = None
            if self.save_intermediate_files:
                # Use Manager().Lock() for potential ProcessPoolExecutor compatibility
                manager = multiprocessing.Manager()
                lock = manager.Lock()
                self._clear_intermediate_files(output_file)

            def flush() -> None:
                nonlocal wrote_header, buffered, buffered_bytes, buffered_users, total_records, total_sequences
                if not buffered:
                    return
                frames = buffered
                buffered = []
                buffered_bytes = 0
                buffered_users = 0
                for frame in frames:
                    missing = [c for c in expected_cols if c not in frame.columns]
                    if missing:
                        frame = frame.with_columns([pl.lit(None).alias(c) for c in missing])
                    frame = frame.select([c for c in expected_cols if c in frame.columns])
                    total_records += len(frame)
                    self._write_csv_append(frame, output_file=output_file, include_header=not wrote_header)
                    wrote_header = True

            for idx, (csv_folder, db_type, converter) in enumerate(zip(csv_folders, db_types, converters), 1):
                if converter is not None and callable(getattr(converter, "iter_user_event_frames", None)):
                    field_categories_dict = extract_field_categories(db_type) if db_type != "unknown" else None
                    # Use getattr to avoid linter error on base class attribute access
                    iter_fn = getattr(converter, "iter_user_event_frames")
                    for user_df in iter_fn(csv_folder, interval_minutes=self.expected_interval_minutes):
                        if len(user_df) == 0:
                            continue
                        
                        self.stats_manager.original_record_count = len(user_df)
                        original_records += len(user_df)
                        
                        try:
                            umin = user_df[ts_col].min()
                            umax = user_df[ts_col].max()
                            if umin is not None:
                                s = umin.strftime("%Y-%m-%dT%H:%M:%S")
                                min_ts = s if (min_ts is None or s < min_ts) else min_ts
                            if umax is not None:
                                s = umax.strftime("%Y-%m-%dT%H:%M:%S")
                                max_ts = s if (max_ts is None or s > max_ts) else max_ts
                        except (KeyError, AttributeError, ValueError):
                            pass

                        ml_df, user_stats, current_last_sequence_id = _run_processing_pipeline(
                            user_df, current_last_sequence_id, field_categories_dict,
                            self.gap_detector, self.data_cleaner, self.interpolator, self.filter_step, self.fixed_freq_generator, self.ml_preparer, self.stats_manager,
                            self.create_fixed_frequency,
                            last_step=self.last_step,
                            save_intermediate=self.save_intermediate_files,
                            output_dir=output_file.parent if output_file else Path("OUTPUT"),
                            output_prefix=output_file.stem if output_file else "combined",
                            expected_standard_cols=expected_standard_cols
                        )
                        
                        # Add dataset name in multi-database mode
                        ml_df = ml_df.with_columns(pl.lit(csv_folder.name).alias(dataset_name_display))

                        # Accumulate total sequences and stats
                        all_processing_stats.append(user_stats)
                        total_sequences += user_stats['dataset_overview']['total_sequences']

                        missing = [c for c in expected_cols if c not in ml_df.columns]
                        if missing:
                            ml_df = ml_df.with_columns([pl.lit(None).alias(c) for c in missing])
                        ml_df = ml_df.select([c for c in expected_cols if c in ml_df.columns])
                        buffered.append(ml_df)
                        buffered_bytes += self._df_estimated_size_bytes(ml_df)
                        buffered_users += 1
                        if buffered_bytes >= max_bytes or buffered_users >= max_users:
                            flush()
                else:
                    ml_df, stats, current_last_sequence_id = self.process(csv_folder, output_file=None, last_sequence_id=current_last_sequence_id)
                    
                    # Add dataset name in multi-database mode
                    ml_df = ml_df.with_columns(pl.lit(csv_folder.name).alias(dataset_name_display))

                    all_processing_stats.append(stats)
                    total_sequences += stats['dataset_overview']['total_sequences']
                    original_records += stats['dataset_overview'].get('original_records', 0)
                    
                    missing = [c for c in expected_cols if c not in ml_df.columns]
                    if missing:
                        ml_df = ml_df.with_columns([pl.lit(None).alias(c) for c in missing])
                    ml_df = ml_df.select([c for c in expected_cols if c in ml_df.columns])
                    buffered.append(ml_df)
                    buffered_bytes += self._df_estimated_size_bytes(ml_df)
                    buffered_users += 1
                    if buffered_bytes >= max_bytes or buffered_users >= max_users:
                        flush()
            flush()
            
            if all_processing_stats:
                aggregated_stats = self.stats_manager.aggregate_statistics(
                    all_processing_stats, ["Streaming Folder"] * len(all_processing_stats)
                )
                aggregated_stats['dataset_overview']['total_records'] = total_records
                aggregated_stats['dataset_overview']['original_records'] = original_records
                aggregated_stats['dataset_overview']['date_range'] = {"start": min_ts or "N/A", "end": max_ts or "N/A"}
            else:
                aggregated_stats = self.stats_manager.get_statistics(
                    pl.DataFrame({seq_id_col: pl.Series([], dtype=pl.Int64)}),
                    {}, {}, {}, {}, {}
                )
            
            placeholder = pl.DataFrame({seq_id_col: pl.Series([], dtype=pl.Int64)})
            return placeholder, aggregated_stats, current_last_sequence_id

        all_dataframes: List[pl.DataFrame] = []
        all_statistics: List[Dict[str, Any]] = []
        current_last_sequence_id = last_sequence_id
        
        for idx, csv_folder in enumerate(csv_folders, 1):
            ml_df, stats, current_last_sequence_id = self.process(csv_folder, output_file=None, last_sequence_id=current_last_sequence_id)
            if user_id_col in ml_df.columns:
                ml_df = ml_df.drop(user_id_col)
            
            # Add dataset name in multi-database mode
            ml_df = ml_df.with_columns(pl.lit(csv_folder.name).alias(dataset_name_display))

            max_seq_id = ml_df[seq_id_col].max() if len(ml_df) > 0 else current_last_sequence_id
            min_seq_id = ml_df[seq_id_col].min() if len(ml_df) > 0 else current_last_sequence_id
            stats['database_info'] = {
                'database_index': idx,
                'database_path': str(csv_folder),
                'sequence_id_range': {
                    'min': int(cast(Any, min_seq_id)) if min_seq_id is not None else current_last_sequence_id, 
                    'max': int(cast(Any, max_seq_id)) if max_seq_id is not None else current_last_sequence_id
                }
            }
            all_dataframes.append(ml_df)
            all_statistics.append(stats)
        
        combined_df = pl.concat(all_dataframes).sort([seq_id_col, ts_col])
        combined_stats = self.stats_manager.aggregate_statistics(all_statistics, csv_folders)
        
        if output_file:
            combined_df.write_csv(output_file)
        
        return combined_df, combined_stats, current_last_sequence_id


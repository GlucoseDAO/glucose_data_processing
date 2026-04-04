#!/usr/bin/env python3
"""
Command Line Interface for Glucose Data Preprocessing

This script provides a CLI wrapper around the glucose preprocessing functionality,
allowing users to process glucose data from CSV folders through command line arguments.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import typer
import polars as pl
from loguru import logger
from glucose_ml_preprocessor import GlucoseMLPreprocessor
from processing.stats_manager import print_statistics as sm_print_statistics
from formats import DatabaseDetector

app = typer.Typer(help="Glucose Data Preprocessing CLI")

def _generate_output_filename(input_folders: List[Path], user_output: Optional[Path]) -> Path:
    """
    Generate output filename based on source folder names.
    
    If user provided a custom output path, use that name but place it in OUTPUT folder.
    Otherwise, generate filename from source folder names.
    """
    output_dir = Path("OUTPUT")
    output_dir.mkdir(exist_ok=True)
    
    # If user provided a custom output path, use that filename
    if user_output is not None:
        return output_dir / user_output.name
    
    # Generate filename from source folder names
    source_names: List[str] = []
    for folder in input_folders:
        if folder.is_file() and folder.suffix.lower() == ".zip":
            # For zip files, use the filename without extension
            source_names.append(folder.stem)
        else:
            # For folders, use the folder name
            source_names.append(folder.name)
    
    # Sanitize names (remove invalid characters for filenames)
    sanitized_names = []
    invalid_chars = '<>:"/\\|?*'
    for name in source_names:
        sanitized = ''.join(c if c not in invalid_chars else '_' for c in name)
        sanitized_names.append(sanitized)
    
    # Create filename: combine source names with underscores
    if len(sanitized_names) == 1:
        filename = f"{sanitized_names[0]}_ml_ready.csv"
    else:
        # For multiple sources, combine them
        combined = "_".join(sanitized_names)
        filename = f"{combined}_ml_ready.csv"
    
    return output_dir / filename

def _resolve_config_file(config_file: Optional[Path]) -> Optional[Path]:
    """
    Resolve the config file path to use.

    Behavior:
    - If user provides --config, use it as-is.
    - Otherwise, if ./glucose_config.yaml exists, use it by default.
    - Otherwise, return None and rely on CLI args only.
    """
    if config_file:
        return config_file
    default_cfg = Path("glucose_config.yaml")
    if default_cfg.exists() and default_cfg.is_file():
        return default_cfg
    return None

@app.command()
def main(
    input_folders: List[Path] = typer.Argument(
        ..., 
        help="Path(s) to input datasets to process. Each input can be a folder (CSV datasets) or a .zip (AI-READI). Multiple inputs can be combined."
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file (command line args override config values)"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file name (will be saved in OUTPUT folder). If not provided, filename is generated from source folder names."
    ),
    interval_minutes: Optional[int] = typer.Option(
        None,
        "--interval", "-i",
        help="Time discretization interval in minutes (default: 5)"
    ),
    gap_max_minutes: Optional[int] = typer.Option(
        None,
        "--gap-max", "-g",
        help="Maximum gap size to interpolate in minutes (default: 15)"
    ),
    min_sequence_len: Optional[int] = typer.Option(
        None,
        "--min-length", "-l",
        help="Minimum sequence length to keep for ML training (default: 200)"
    ),
    remove_calibration: Optional[bool] = typer.Option(
        None,
        "--remove-calibration/--keep-calibration",
        help="Remove calibration events to create interpolatable gaps (default: True)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    show_stats: bool = typer.Option(
        True,
        "--stats/--no-stats",
        help="Show processing statistics"
    ),
    save_intermediate_files: Optional[bool] = typer.Option(
        None,
        "--save-intermediate", "-s",
        help="Save intermediate files after each processing step (default: False)"
    ),
    calibration_period_minutes: Optional[int] = typer.Option(
        None,
        "--calibration-period", "-p",
        help="Gap duration considered as calibration period in minutes (default: 165 minutes)"
    ),
    remove_after_calibration_hours: Optional[int] = typer.Option(
        None,
        "--remove-after-calibration", "-r",
        help="Hours of data to remove after calibration period (default: 24 hours)"
    ),
    glucose_only: Optional[bool] = typer.Option(
        None,
        "--glucose-only",
        help="Output only glucose data: remove Event Type, Insulin Value, and Carb Value fields, keep only rows with glucose values"
    ),
    create_fixed_frequency: Optional[bool] = typer.Option(
        None,
        "--fixed-frequency/--no-fixed-frequency",
        help="Create fixed-frequency data with consistent intervals (default: enabled)"
    ),
    last_step: Optional[int] = typer.Option(
        None,
        "--last-step",
        help="Last processing step to execute (1-7). 0 means all steps. (default: 0)"
    ),
    first_n_users: Optional[int] = typer.Option(
        None,
        "--first-n-users",
        help="Process only the first n users (for multi-user databases). If 0 or not specified, all users are processed."
    ),
    round_precision: Optional[int] = typer.Option(
        None,
        "--round-precision",
        help="Number of digits after the decimal point to round numeric fields. Can be negative for rounding to the left of the decimal point (default: 3)."
    )
) -> None:
    """
    Process glucose data from CSV folder(s) for machine learning.
    
    This tool consolidates CSV files from one or more input folders and processes them through
    the complete ML preprocessing pipeline including High/Low value replacement, calibration removal, 
    gap detection, interpolation, calibration period detection, and sequence filtering.
    
    When multiple folders are provided, sequence IDs are tracked and offset to ensure consistency
    across different databases, producing a single unified output file.
    """
    # Configure loguru to show only the message, matching the old print() behavior
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")
    
    # Validate input paths
    validated_folders: List[Path] = []
    for input_folder in input_folders:
        if not input_folder.exists():
            logger.error(f"Error: Input path '{input_folder}' does not exist")
            raise typer.Exit(1)
        
        if not input_folder.is_dir():
            # Allow zip-backed datasets (AI-READI)
            if not (input_folder.is_file() and input_folder.suffix.lower() == ".zip"):
                logger.error(
                    f"Error: Input must be a folder containing CSV files or a .zip dataset, got: '{input_folder}'"
                )
                raise typer.Exit(1)
        
        validated_folders.append(input_folder)
    
    # Initialize preprocessor info        
    try:
        resolved_config_file = _resolve_config_file(config_file)
        if verbose and resolved_config_file and not config_file:
            logger.info(f"Auto-loading default configuration from: {resolved_config_file}")

        # CLI arguments override config file values
        cli_overrides: Dict[str, Any] = {
            'verbose': verbose,
            'print_statistics': show_stats
        }
        
        # Only add optional overrides if they were explicitly provided via CLI
        if interval_minutes is not None:
            cli_overrides['expected_interval_minutes'] = interval_minutes
        if gap_max_minutes is not None:
            cli_overrides['small_gap_max_minutes'] = gap_max_minutes
        if remove_calibration is not None:
            cli_overrides['remove_calibration'] = remove_calibration
        if min_sequence_len is not None:
            cli_overrides['min_sequence_len'] = min_sequence_len
        if save_intermediate_files is not None:
            cli_overrides['save_intermediate_files'] = save_intermediate_files
        if calibration_period_minutes is not None:
            cli_overrides['calibration_period_minutes'] = calibration_period_minutes
        if remove_after_calibration_hours is not None:
            cli_overrides['remove_after_calibration_hours'] = remove_after_calibration_hours
        if glucose_only is not None:
            cli_overrides['glucose_only'] = glucose_only
        if create_fixed_frequency is not None:
            cli_overrides['create_fixed_frequency'] = create_fixed_frequency
        if last_step is not None:
            cli_overrides['last_step'] = last_step
        if first_n_users is not None and first_n_users > 0:
            cli_overrides['first_n_users'] = first_n_users
        if round_precision is not None:
            cli_overrides['round_precision'] = round_precision

        # Create preprocessor
        if resolved_config_file:
            if verbose:
                logger.info(f"Loading configuration from: {resolved_config_file}")
            preprocessor = GlucoseMLPreprocessor.from_config_file(resolved_config_file, **cli_overrides)
        else:
            # Use CLI arguments directly
            preprocessor = GlucoseMLPreprocessor(**cli_overrides)
        
        # Resolve output file priority:
        # 1. CLI --output option (filename only, placed in OUTPUT)
        # 2. Config file 'output_file' setting (filename only, placed in OUTPUT)
        # 3. Otherwise generate from dataset folder names (placed in OUTPUT)
        user_output = output_file or preprocessor.output_file
        final_output_file = _generate_output_filename(validated_folders, user_output)
            
        # Ensure output directory exists
        if final_output_file.parent and not final_output_file.parent.exists():
            if verbose:
                logger.info(f"Creating output directory: {final_output_file.parent}")
            final_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Process data
        if verbose:
            logger.info("Initializing glucose data preprocessor...")
            if len(validated_folders) == 1:
                logger.info(f"   Input folder: {validated_folders[0]}")
            else:
                logger.info(f"   Input folders ({len(validated_folders)}):")
                for i, folder in enumerate(validated_folders, 1):
                    logger.info(f"      {i}. {folder}")
            logger.info(f"   Output file: {final_output_file}")
            logger.info(f"   Time interval: {interval_minutes} minutes")
            logger.info(f"   Gap max: {gap_max_minutes} minutes")
            logger.info(f"   Min sequence length: {min_sequence_len}")
            logger.info(f"   Remove calibration events: {remove_calibration}")
            logger.info(f"   Calibration period: {calibration_period_minutes} minutes")
            logger.info(f"   Remove after calibration: {remove_after_calibration_hours} hours")
            logger.info(f"   Save intermediate files: {save_intermediate_files}")
            logger.info(f"   Glucose only mode: {glucose_only}")
            logger.info(f"   Fixed-frequency data: {create_fixed_frequency}")
            logger.info(f"   Last step: {last_step if last_step is not None and last_step > 0 else 'All'}")
            if len(validated_folders) == 1:
                logger.info("Starting glucose data processing pipeline...")
            else:
                logger.info(f"Starting multi-database processing pipeline for {len(validated_folders)} databases...")
        
        # Detect database type for single input
        database_type = None
        if len(validated_folders) == 1:
            try:
                db_detector = DatabaseDetector()
                database_type = db_detector.detect_database_type(validated_folders[0])
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not detect database type: {e}")

        # Process single or multiple databases
        if len(validated_folders) == 1:
            ml_data, statistics, _ = preprocessor.process(
                validated_folders[0], final_output_file, database_type=database_type
            )
        else:
            ml_data, statistics, _ = preprocessor.process_multiple_databases(
                validated_folders, final_output_file
            )
        
        # Show results
        logger.info(f"Processing completed successfully!")
        
        # Extract record and sequence counts safely
        try:
            overview = statistics.get("dataset_overview", {}) if isinstance(statistics, dict) else {}
            stats_records = int(overview.get("total_records", 0))
            stats_sequences = int(overview.get("total_sequences", 0))
        except (ValueError, TypeError, AttributeError):
            stats_records = 0
            stats_sequences = 0
        
        df_records = len(ml_data) if hasattr(ml_data, "__len__") else 0
        df_sequences = 0
        if isinstance(ml_data, pl.DataFrame) and "sequence_id" in ml_data.columns:
            df_sequences = int(ml_data["sequence_id"].n_unique())

        out_records = stats_records if (df_records == 0 and stats_records > 0) else df_records
        out_sequences = stats_sequences if (df_sequences == 0 and stats_sequences > 0) else df_sequences

        logger.info(f"Output: {out_records:,} records in {out_sequences:,} sequences")
        logger.info(f"Saved to: {final_output_file}")
        
        # Show statistics if requested
        if show_stats or preprocessor.print_statistics:
            params = {
                'expected_interval_minutes': preprocessor.expected_interval_minutes,
                'small_gap_max_minutes': preprocessor.small_gap_max_minutes,
                'remove_calibration': preprocessor.remove_calibration,
                'min_sequence_len': preprocessor.min_sequence_len,
                'calibration_period_minutes': preprocessor.calibration_period_minutes,
                'remove_after_calibration_hours': preprocessor.remove_after_calibration_hours,
                'create_fixed_frequency': preprocessor.create_fixed_frequency,
                'round_precision': preprocessor.round_precision
            }
            stats_output = sm_print_statistics(statistics, params)
            logger.info(stats_output)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if verbose:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
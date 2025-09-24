#!/usr/bin/env python3
"""
Command Line Interface for Glucose Data Preprocessing

This script provides a CLI wrapper around the glucose preprocessing functionality,
allowing users to process glucose data from CSV folders through command line arguments.
"""

import typer
from pathlib import Path
from typing import Optional
import sys
from glucose_ml_preprocessor import GlucoseMLPreprocessor, print_statistics

def main(
    input_folder: str = typer.Argument(
        ..., 
        help="Path to folder containing CSV files to process"
    ),
    output_file: str = typer.Option(
        "glucose_ml_ready.csv",
        "--output", "-o",
        help="Output file path for ML-ready data"
    ),
    interval_minutes: int = typer.Option(
        5,
        "--interval", "-i",
        help="Time discretization interval in minutes"
    ),
    gap_max_minutes: int = typer.Option(
        15,
        "--gap-max", "-g",
        help="Maximum gap size to interpolate in minutes"
    ),
    min_sequence_len: int = typer.Option(
        200,
        "--min-length", "-l",
        help="Minimum sequence length to keep for ML training"
    ),
    interpolate_calibration: bool = typer.Option(
        True,
        "--calibration/--no-calibration",
        help="Interpolate calibration glucose values to remove spikes"
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
    save_intermediate_files: bool = typer.Option(
        False,
        "--save-intermediate", "-s",
        help="Save intermediate files after each processing step"
    ),
    calibration_period_minutes: int = typer.Option(
        60*2 + 45,  # 2 hours 45 minutes
        "--calibration-period", "-c",
        help="Gap duration considered as calibration period in minutes (default: 165 minutes)"
    ),
    remove_after_calibration_hours: int = typer.Option(
        24,
        "--remove-after-calibration", "-r",
        help="Hours of data to remove after calibration period (default: 24 hours)"
    )
):
    """
    Process glucose data from CSV folder for machine learning.
    
    This tool consolidates CSV files from the input folder and processes them through
    the complete ML preprocessing pipeline including gap detection, interpolation,
    calibration smoothing, calibration period detection, and sequence filtering.
    
    Example:
        glucose-cli ./csv-folder --output ml_data.csv --verbose
        glucose-cli ./csv-folder --calibration-period 165 --remove-after-calibration 24
    """
    
    # Validate input path
    input_path_obj = Path(input_folder)
    if not input_path_obj.exists():
        typer.echo(f"❌ Error: Input folder '{input_folder}' does not exist", err=True)
        raise typer.Exit(1)
    
    if not input_path_obj.is_dir():
        typer.echo(f"❌ Error: Input must be a folder containing CSV files, got: '{input_folder}'", err=True)
        raise typer.Exit(1)
    
    # Initialize preprocessor
    if verbose:
        typer.echo("⚙️  Initializing glucose data preprocessor...")
        typer.echo(f"   📁 Input folder: {input_folder}")
        typer.echo(f"   📄 Output file: {output_file}")
        typer.echo(f"   ⏱️  Time interval: {interval_minutes} minutes")
        typer.echo(f"   📏 Gap max: {gap_max_minutes} minutes")
        typer.echo(f"   📊 Min sequence length: {min_sequence_len}")
        typer.echo(f"   🔧 Interpolate calibration: {interpolate_calibration}")
        typer.echo(f"   🕐 Calibration period: {calibration_period_minutes} minutes")
        typer.echo(f"   🗑️  Remove after calibration: {remove_after_calibration_hours} hours")
        typer.echo(f"   💾 Save intermediate files: {save_intermediate_files}")
    
    try:
        preprocessor = GlucoseMLPreprocessor(
            expected_interval_minutes=interval_minutes,
            small_gap_max_minutes=gap_max_minutes,
            interpolate_calibration=interpolate_calibration,
            min_sequence_len=min_sequence_len,
            save_intermediate_files=save_intermediate_files,
            calibration_period_minutes=calibration_period_minutes,
            remove_after_calibration_hours=remove_after_calibration_hours
        )
        
        # Process data
        if verbose:
            typer.echo("🔄 Starting glucose data processing pipeline...")
        
        ml_data, statistics = preprocessor.process(
            input_folder, output_file
        )
        
        # Show results
        typer.echo(f"✅ Processing completed successfully!")
        typer.echo(f"📊 Output: {len(ml_data):,} records in {ml_data['sequence_id'].n_unique():,} sequences")
        typer.echo(f"💾 Saved to: {output_file}")
        
        # Show statistics if requested
        if show_stats:
            if verbose:
                typer.echo("\n" + "="*60)
                typer.echo("DETAILED STATISTICS")
                typer.echo("="*60)
                print_statistics(statistics, preprocessor)
            else:
                # Show summary statistics only
                overview = statistics['dataset_overview']
                seq_analysis = statistics['sequence_analysis']
                typer.echo(f"\n📈 Summary:")
                typer.echo(f"   📅 Date range: {overview['date_range']['start']} to {overview['date_range']['end']}")
                typer.echo(f"   📏 Longest sequence: {seq_analysis['longest_sequence']:,} records")
                typer.echo(f"   📊 Average sequence: {seq_analysis['sequence_lengths']['mean']:.1f} records")
                
                # Show interpolation summary
                interp_analysis = statistics['interpolation_analysis']
                typer.echo(f"   🔧 Gaps processed: {interp_analysis['small_gaps_filled']:,} gaps")
                typer.echo(f"   🔧 Data points created: {interp_analysis['total_interpolated_data_points']:,} points")
                typer.echo(f"   🔧 Field interpolations: {interp_analysis['total_interpolations']:,} values")
                
                # Show calibration summary if enabled
                if interpolate_calibration and 'calibration_analysis' in statistics:
                    calib_analysis = statistics['calibration_analysis']
                    typer.echo(f"   🔬 Calibration interpolations: {calib_analysis['calibration_interpolations']:,}")
                
                # Show filtering summary
                if 'filtering_analysis' in statistics:
                    filter_analysis = statistics['filtering_analysis']
                    typer.echo(f"   🔍 Sequences filtered: {filter_analysis['removed_sequences']:,} removed")
                
                # Show calibration period summary
                gap_analysis = statistics['gap_analysis']
                if 'calibration_period_analysis' in gap_analysis:
                    calib_analysis = gap_analysis['calibration_period_analysis']
                    if calib_analysis['calibration_periods_detected'] > 0:
                        typer.echo(f"   🔬 Calibration periods: {calib_analysis['calibration_periods_detected']:,} detected, {calib_analysis['total_records_marked_for_removal']:,} records removed")
        
    except Exception as e:
        typer.echo(f"❌ Error during processing: {e}", err=True)
        if verbose:
            import traceback
            typer.echo(f"Traceback:\n{traceback.format_exc()}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
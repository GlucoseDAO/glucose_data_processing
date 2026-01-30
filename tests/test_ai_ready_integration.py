import pytest
import polars as pl
from pathlib import Path
from glucose_ml_preprocessor import GlucoseMLPreprocessor
import yaml

def test_ai_ready_integration():
    """
    Integration test for AI-READY dataset using a known small sample.
    Processes the sample and compares with a reference result.
    """
    # 1. Setup paths
    test_data_dir = Path("test_data/ai_ready")
    input_zip = test_data_dir / "ai_ready_small.zip"
    config_path = test_data_dir / "config.yaml"
    reference_csv = test_data_dir / "reference_result.csv"
    
    # Check if files exist
    assert input_zip.exists(), f"Test data zip not found at {input_zip}"
    assert config_path.exists(), f"Test config not found at {config_path}"
    assert reference_csv.exists(), f"Reference result not found at {reference_csv}"
    
    # 2. Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 3. Run preprocessor
    # We use the same parameters as used to generate the reference result
    preprocessor = GlucoseMLPreprocessor.from_config_file(config_path)
    
    # Process the data - we provide an output_file to force the same path as CLI (streaming)
    # and then read it back for comparison.
    temp_output = Path("OUTPUT/temp_integration_test.csv")
    output_df, stats, _ = preprocessor.process(input_zip, output_file=temp_output)
    
    # Since it was streaming, we read the result from the file
    output_df = pl.read_csv(temp_output)
    
    # 4. Load reference result
    ref_df = pl.read_csv(reference_csv)
    
    # 5. Compare results
    
    # A. Check row counts
    assert len(output_df) == len(ref_df), f"Row count mismatch: {len(output_df)} vs {len(ref_df)}"
    
    # B. Check column names
    assert sorted(output_df.columns) == sorted(ref_df.columns), f"Column names mismatch"
    
    # C. Check column order (not critical but good for exact reproduction)
    assert output_df.columns == ref_df.columns, f"Column order mismatch"
    
    # D. Check values
    # We'll use a join to compare row by row. 
    # Key columns for AI-READY are usually 'User ID' and 'Timestamp (YYYY-MM-DDThh:mm:ss)'
    ts_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    user_col = "User ID"
    
    # Ensure timestamps are comparable (strings vs strings)
    output_df = output_df.with_columns(pl.col(ts_col).cast(pl.Utf8))
    ref_df = ref_df.with_columns(pl.col(ts_col).cast(pl.Utf8))
    
    # Join on keys
    joined = output_df.join(ref_df, on=[user_col, ts_col], how="inner", suffix="_ref")
    
    assert len(joined) == len(ref_df), "Some rows from reference are missing in output"
    
    # Compare glucose values
    glucose_col = "Glucose Value (mg/dL)"
    if glucose_col in output_df.columns:
        diffs = joined.filter(
            (pl.col(glucose_col) != pl.col(f"{glucose_col}_ref")) & 
            (pl.col(glucose_col).is_not_null() | pl.col(f"{glucose_col}_ref").is_not_null())
        )
        assert len(diffs) == 0, f"Found {len(diffs)} differences in glucose values"

    # E. Check sequence IDs
    seq_col = "sequence_id"
    if seq_col in output_df.columns:
        diffs_seq = joined.filter(pl.col(seq_col) != pl.col(f"{seq_col}_ref"))
        assert len(diffs_seq) == 0, f"Found {len(diffs_seq)} differences in sequence IDs"

if __name__ == "__main__":
    # Allow running the test script directly
    test_ai_ready_integration()
    print("Integration test passed!")

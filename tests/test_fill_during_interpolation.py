import polars as pl
import pytest
from datetime import datetime, timedelta
from glucose_ml_preprocessor import GlucoseMLPreprocessor
from processing.core.fields import StandardFieldNames

def test_fill_during_interpolation():
    """Test that fields in fill_during_interpolation are correctly filled."""
    # Create preprocessor
    preprocessor = GlucoseMLPreprocessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15
    )
    
    # Create data with a gap
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=10) # 10 min gap
    ]
    
    # user_id and age should be filled if they match
    # recommended_split should also be filled
    df = pl.DataFrame({
        StandardFieldNames.TIMESTAMP: timestamps,
        StandardFieldNames.SEQUENCE_ID: [1, 1],
        StandardFieldNames.GLUCOSE_VALUE: [100.0, 120.0],
        StandardFieldNames.USER_ID: ["user1", "user1"],
        "age": [30, 30],
        "recommended_split": ["train", "train"],
        "other_field": ["X", "Y"], # Should NOT be filled if they differ
        StandardFieldNames.EVENT_TYPE: ["EGV", "EGV"]
    })
    
    field_categories = {
        'continuous': [StandardFieldNames.GLUCOSE_VALUE],
        'service': [StandardFieldNames.USER_ID, 'age', 'recommended_split'],
        'fill_during_interpolation': [StandardFieldNames.USER_ID, 'age', 'recommended_split', 'other_field']
    }
    
    result, stats = preprocessor.interpolator.interpolate_missing_values(df, field_categories)
    
    # Should have 3 rows (original 2 + 1 interpolated at 10:05)
    assert len(result) == 3
    
    # Check interpolated row
    interp_row = result.filter(pl.col(StandardFieldNames.TIMESTAMP) == base_time + timedelta(minutes=5))
    assert len(interp_row) == 1
    
    # Check filled fields
    assert interp_row[StandardFieldNames.USER_ID][0] == "user1"
    assert interp_row["age"][0] == 30
    assert interp_row["recommended_split"][0] == "train"
    
    # Check other_field (should be None or empty because values differed)
    assert interp_row["other_field"][0] is None or interp_row["other_field"][0] == ""

def test_fill_during_interpolation_existing_rows():
    """Test that fields in fill_during_interpolation are correctly filled in existing rows with nulls."""
    preprocessor = GlucoseMLPreprocessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15
    )
    
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=5),
        base_time + timedelta(minutes=10)
    ]
    
    # Row at 10:05 has nulls for service fields
    df = pl.DataFrame({
        StandardFieldNames.TIMESTAMP: timestamps,
        StandardFieldNames.SEQUENCE_ID: [1, 1, 1],
        StandardFieldNames.GLUCOSE_VALUE: [100.0, 110.0, 120.0],
        StandardFieldNames.USER_ID: ["user1", None, "user1"],
        "age": [30, None, 30],
        "recommended_split": ["train", "train", "train"], # Wait, if middle is train, it won't be filled anyway as it's not null
        StandardFieldNames.EVENT_TYPE: ["EGV", "EGV", "EGV"]
    })
    
    field_categories = {
        'continuous': [StandardFieldNames.GLUCOSE_VALUE],
        'service': [StandardFieldNames.USER_ID, 'age', 'recommended_split'],
        'fill_during_interpolation': [StandardFieldNames.USER_ID, 'age', 'recommended_split']
    }
    
    result, stats = preprocessor.interpolator.interpolate_missing_values(df, field_categories)
    
    # Should still have 3 rows
    assert len(result) == 3
    
    # Check middle row
    middle_row = result.filter(pl.col(StandardFieldNames.TIMESTAMP) == base_time + timedelta(minutes=5))
    assert middle_row[StandardFieldNames.USER_ID][0] == "user1"
    assert middle_row["age"][0] == 30
    assert middle_row["recommended_split"][0] == "train"

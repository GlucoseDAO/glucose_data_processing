import pytest
import polars as pl
from datetime import datetime, timedelta
from processing.steps.data_cleaning import DataCleaner
from processing.core.fields import StandardFieldNames

def test_data_cleaner_small_gaps():
    """Test that data in small gaps is kept."""
    cleaner = DataCleaner(small_gap_max_minutes=15)
    
    ts = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        StandardFieldNames.TIMESTAMP: [
            ts, 
            ts + timedelta(minutes=5), # In small gap (5 min)
            ts + timedelta(minutes=10)
        ],
        StandardFieldNames.GLUCOSE_VALUE: [100.0, None, 110.0],
        "covariate": [1, 2, 3]
    }
    df = pl.DataFrame(data)
    
    cleaned_df, stats = cleaner.clean_remote_data(df)
    
    assert len(cleaned_df) == 3
    assert stats["removed_records"] == 0

def test_data_cleaner_large_gaps():
    """Test that data in large gaps is removed."""
    cleaner = DataCleaner(small_gap_max_minutes=15)
    
    ts = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        StandardFieldNames.TIMESTAMP: [
            ts, 
            ts + timedelta(minutes=10), # In large gap (20 min between glucose)
            ts + timedelta(minutes=20)
        ],
        StandardFieldNames.GLUCOSE_VALUE: [100.0, None, 110.0],
        "covariate": [1, 2, 3]
    }
    df = pl.DataFrame(data)
    
    cleaned_df, stats = cleaner.clean_remote_data(df)
    
    # Only the first and last rows should remain (they have glucose)
    assert len(cleaned_df) == 2
    assert stats["removed_records"] == 1
    assert None not in cleaned_df[StandardFieldNames.GLUCOSE_VALUE].to_list()

def test_data_cleaner_no_gaps():
    """Test that data with no gaps is kept."""
    cleaner = DataCleaner(small_gap_max_minutes=15)
    
    ts = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        StandardFieldNames.TIMESTAMP: [ts, ts + timedelta(minutes=5)],
        StandardFieldNames.GLUCOSE_VALUE: [100.0, 110.0],
        "covariate": [1, 2]
    }
    df = pl.DataFrame(data)
    
    cleaned_df, stats = cleaner.clean_remote_data(df)
    
    assert len(cleaned_df) == 2
    assert stats["removed_records"] == 0

def test_data_cleaner_remote_data_removal():
    """Test that data before/after all glucose is removed."""
    cleaner = DataCleaner(small_gap_max_minutes=15)
    
    ts = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        StandardFieldNames.TIMESTAMP: [
            ts - timedelta(minutes=5), # Before first glucose
            ts, 
            ts + timedelta(minutes=5), # After last glucose
        ],
        StandardFieldNames.GLUCOSE_VALUE: [None, 100.0, None],
        "covariate": [1, 2, 3]
    }
    df = pl.DataFrame(data)
    
    cleaned_df, stats = cleaner.clean_remote_data(df)
    
    # Only the middle row should remain
    assert len(cleaned_df) == 1
    assert cleaned_df[StandardFieldNames.GLUCOSE_VALUE][0] == 100.0
    assert stats["removed_records"] == 2

def test_data_cleaner_multi_user():
    """Test cleaning with multiple users."""
    cleaner = DataCleaner(small_gap_max_minutes=15)
    
    ts = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        StandardFieldNames.USER_ID: ["user1", "user1", "user1", "user2", "user2", "user2"],
        StandardFieldNames.TIMESTAMP: [
            ts, ts + timedelta(minutes=20), ts + timedelta(minutes=40),
            ts, ts + timedelta(minutes=5), ts + timedelta(minutes=10)
        ],
        StandardFieldNames.GLUCOSE_VALUE: [
            100.0, None, 110.0, # Large gap for user1 (40 min)
            100.0, None, 110.0  # Small gap for user2 (10 min)
        ],
        "covariate": [1, 2, 3, 4, 5, 6]
    }
    df = pl.DataFrame(data)
    
    cleaned_df, stats = cleaner.clean_remote_data(df)
    
    # user1 should lose 1 row, user2 should lose 0
    assert len(cleaned_df) == 5
    assert stats["removed_records"] == 1
    
    user1_data = cleaned_df.filter(pl.col(StandardFieldNames.USER_ID) == "user1")
    user2_data = cleaned_df.filter(pl.col(StandardFieldNames.USER_ID) == "user2")
    
    assert len(user1_data) == 2
    assert len(user2_data) == 3

def test_data_cleaner_no_glucose_at_all():
    """Test case where a user has no glucose data."""
    cleaner = DataCleaner(small_gap_max_minutes=15)
    
    ts = datetime(2023, 1, 1, 12, 0, 0)
    data = {
        StandardFieldNames.TIMESTAMP: [ts, ts + timedelta(minutes=5)],
        StandardFieldNames.GLUCOSE_VALUE: [None, None],
        "covariate": [1, 2]
    }
    df = pl.DataFrame(data)
    
    cleaned_df, stats = cleaner.clean_remote_data(df)
    
    assert len(cleaned_df) == 0
    assert stats["removed_records"] == 2

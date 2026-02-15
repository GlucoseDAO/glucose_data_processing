"""Tests for SentinelCleaner — cleaning invalid/sentinel values before aggregation."""

import pytest
import polars as pl
from processing.steps.sentinel_cleaning import SentinelCleaner


@pytest.fixture
def cleaner():
    return SentinelCleaner()


def test_stress_negative_cleaned(cleaner):
    """Stress < 0 (Garmin sentinel codes -1, -2) should become NULL."""
    df = pl.DataFrame({
        "stress_level": [50.0, -1.0, -2.0, 30.0, 0.0],
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    vals = cleaned["stress_level"].to_list()
    assert vals[0] == 50.0
    assert vals[1] is None  # was -1
    assert vals[2] is None  # was -2
    assert vals[3] == 30.0
    assert vals[4] == 0.0   # zero is valid for stress
    assert stats["sentinel_cleaned"]["stress_level"]["cleaned"] == 2


def test_heart_rate_zero_cleaned(cleaner):
    """Heart rate <= 0 should become NULL (device offline)."""
    df = pl.DataFrame({
        "heart_rate": [72.0, 0.0, -1.0, 85.0, 1.0],
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    vals = cleaned["heart_rate"].to_list()
    assert vals[0] == 72.0
    assert vals[1] is None  # was 0
    assert vals[2] is None  # was -1
    assert vals[3] == 85.0
    assert vals[4] == 1.0   # 1 is valid
    assert stats["sentinel_cleaned"]["heart_rate"]["cleaned"] == 2


def test_respiratory_rate_negative_cleaned(cleaner):
    """Respiratory rate < 0 should become NULL."""
    df = pl.DataFrame({
        "respiratory_rate": [16.0, -1.0, 18.0],
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    vals = cleaned["respiratory_rate"].to_list()
    assert vals[0] == 16.0
    assert vals[1] is None
    assert vals[2] == 18.0
    assert stats["sentinel_cleaned"]["respiratory_rate"]["cleaned"] == 1


def test_valid_values_unchanged(cleaner):
    """Valid values should not be modified."""
    df = pl.DataFrame({
        "stress_level": [10.0, 50.0, 99.0],
        "heart_rate": [60.0, 80.0, 100.0],
        "respiratory_rate": [12.0, 16.0, 20.0],
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    assert cleaned["stress_level"].to_list() == [10.0, 50.0, 99.0]
    assert cleaned["heart_rate"].to_list() == [60.0, 80.0, 100.0]
    assert cleaned["respiratory_rate"].to_list() == [12.0, 16.0, 20.0]
    assert stats["total_cleaned"] == 0


def test_missing_columns_skipped(cleaner):
    """Fields not present in the DataFrame should be silently skipped."""
    df = pl.DataFrame({
        "glucose_value_mgdl": [100.0, 110.0],
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    assert cleaned["glucose_value_mgdl"].to_list() == [100.0, 110.0]
    assert stats["total_cleaned"] == 0


def test_empty_dataframe(cleaner):
    """Empty DataFrame should pass through without error."""
    df = pl.DataFrame({
        "stress_level": pl.Series([], dtype=pl.Float64),
        "heart_rate": pl.Series([], dtype=pl.Float64),
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    assert len(cleaned) == 0
    assert stats["total_cleaned"] == 0


def test_statistics_percentage_accuracy(cleaner):
    """Statistics should report correct percentages."""
    df = pl.DataFrame({
        "heart_rate": [72.0, 0.0, None, 85.0, 0.0],
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    hr_stats = stats["sentinel_cleaned"]["heart_rate"]
    assert hr_stats["cleaned"] == 2
    assert hr_stats["total_non_null"] == 4  # 5 rows, 1 null
    assert hr_stats["pct_of_non_null"] == 50.0


def test_all_fields_cleaned_together(cleaner):
    """All three fields cleaned in a single pass."""
    df = pl.DataFrame({
        "stress_level": [-1.0, 50.0],
        "heart_rate": [0.0, 80.0],
        "respiratory_rate": [-2.0, 16.0],
    })
    cleaned, stats = cleaner.clean_sentinel_values(df)

    assert cleaned["stress_level"].to_list() == [None, 50.0]
    assert cleaned["heart_rate"].to_list() == [None, 80.0]
    assert cleaned["respiratory_rate"].to_list() == [None, 16.0]
    assert stats["total_cleaned"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

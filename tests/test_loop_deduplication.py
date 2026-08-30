#!/usr/bin/env python3
"""
Test case for Loop dataset deduplication.

The raw Loop dataset contains duplicate records within files.
This test verifies that the LoopDatabaseConverter properly deduplicates them.
"""

import pytest
import polars as pl
from pathlib import Path


LOOP_DATA_PATH = Path("DATA/loop")


@pytest.fixture
def loop_data_available() -> bool:
    """Check if Loop data is available for testing."""
    return LOOP_DATA_PATH.exists() and (LOOP_DATA_PATH / "Data Tables").exists()


class TestLoopRawDataDuplicates:
    """Tests to verify raw Loop data contains duplicates (documenting the issue)."""
    
    def test_raw_cgm_files_contain_duplicates(self, loop_data_available: bool) -> None:
        """Verify that raw CGM files contain within-file duplicates."""
        if not loop_data_available:
            pytest.skip("Loop data not available")
        
        cgm_file = LOOP_DATA_PATH / "Data Tables" / "LOOPDeviceCGM1.txt"
        if not cgm_file.exists():
            pytest.skip("CGM1 file not found")
        
        # Read sample of CGM data
        df = pl.read_csv(
            cgm_file,
            separator="|",
            truncate_ragged_lines=True,
            n_rows=50000,
            schema_overrides={"PtID": pl.Utf8, "UTCDtTm": pl.Utf8, "CGMVal": pl.Utf8},
            ignore_errors=True
        )
        
        # Check for duplicates
        dups = df.group_by(["PtID", "UTCDtTm"]).len().filter(pl.col("len") > 1)
        
        # Document: raw data DOES contain duplicates
        assert len(dups) > 0, "Expected raw CGM data to contain duplicates (this is a data quality issue)"
        print(f"Found {len(dups)} duplicate groups in first 50k CGM records")


class TestLoopDatabaseConverterDeduplication:
    """Tests to verify LoopDatabaseConverter properly deduplicates data."""
    
    def test_consolidated_data_has_no_duplicates(self, loop_data_available: bool) -> None:
        """Verify that consolidated Loop data has no duplicate (user_id, timestamp, event_type) records."""
        if not loop_data_available:
            pytest.skip("Loop data not available")
        
        from formats.loop.loop_database_converter import LoopDatabaseConverter
        
        # Create converter with first_n_users=1 for speed
        config = {"first_n_users": 1}
        converter = LoopDatabaseConverter(config)
        
        # Consolidate
        df = converter.consolidate_data(str(LOOP_DATA_PATH))
        
        # Check for duplicates
        dups = df.group_by(["user_id", "timestamp", "event_type"]).len().filter(pl.col("len") > 1)
        
        assert len(dups) == 0, f"Found {len(dups)} duplicate groups after consolidation. First 5: {dups.head(5)}"
        print(f"Verified no duplicates in {len(df)} consolidated records")
    
    def test_deduplication_preserves_data_integrity(self, loop_data_available: bool) -> None:
        """Verify that deduplication doesn't drop valid unique records."""
        if not loop_data_available:
            pytest.skip("Loop data not available")
        
        from formats.loop.loop_database_converter import LoopDatabaseConverter
        
        config = {"first_n_users": 1}
        converter = LoopDatabaseConverter(config)
        
        df = converter.consolidate_data(str(LOOP_DATA_PATH))
        
        # Should have records from multiple event types
        event_types = df["event_type"].unique().to_list()
        assert "EGV" in event_types, "Should have CGM (EGV) records"
        
        # Each user should have data
        user_counts = df.group_by("user_id").len()
        assert len(user_counts) > 0, "Should have at least one user"
        
        for row in user_counts.iter_rows(named=True):
            assert row["len"] > 0, f"User {row['user_id']} should have records"
        
        print(f"Data integrity verified: {len(df)} records, {len(event_types)} event types")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

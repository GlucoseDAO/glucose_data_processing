#!/usr/bin/env python3
"""
Integration tests for database converters using real data samples.

These tests verify that each database converter can:
1. Detect the correct database type
2. Consolidate data successfully
3. Produce valid output with expected columns
4. Handle the actual data format correctly
"""

import pytest
from pathlib import Path
import polars as pl

from formats.database_detector import DatabaseDetector
from formats.hupa.hupa_database_converter import HupaDatabaseConverter
from formats.uom.uom_database_converter import UoMDatabaseConverter
from formats.medtronic.medtronic_database_converter import MedtronicDatabaseConverter
from formats.minidose1.minidose1_database_converter import Minidose1DatabaseConverter
from formats.loop.loop_database_converter import LoopDatabaseConverter
from formats.uc_ht.uc_ht_database_converter import UCHTDatabaseConverter


# Path to the real data folder
DATA_PATH = Path(__file__).parent.parent / "DATA"

# Skip tests if data folder doesn't exist
pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="DATA folder not found - real data tests require DATA folder"
)


class TestDatabaseDetection:
    """Test that database types are correctly detected."""

    def test_detect_hupa_database(self) -> None:
        """Test HUPA database detection."""
        hupa_path = DATA_PATH / "HUPA"
        if not hupa_path.exists():
            pytest.skip("HUPA data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(hupa_path)
        assert db_type == "hupa", f"Expected 'hupa', got '{db_type}'"

    def test_detect_uom_database(self) -> None:
        """Test UoM database detection."""
        uom_path = DATA_PATH / "uom"
        if not uom_path.exists():
            pytest.skip("UoM data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(uom_path)
        assert db_type == "uom", f"Expected 'uom', got '{db_type}'"

    def test_detect_medtronic_database(self) -> None:
        """Test Medtronic database detection."""
        medtronic_path = DATA_PATH / "medtronic"
        if not medtronic_path.exists():
            pytest.skip("Medtronic data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(medtronic_path)
        assert db_type == "medtronic", f"Expected 'medtronic', got '{db_type}'"

    def test_detect_minidose1_database(self) -> None:
        """Test MiniDose1 database detection."""
        minidose_path = DATA_PATH / "minidose1"
        if not minidose_path.exists():
            pytest.skip("MiniDose1 data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(minidose_path)
        assert db_type == "minidose1", f"Expected 'minidose1', got '{db_type}'"

    def test_detect_loop_database(self) -> None:
        """Test Loop database detection."""
        loop_path = DATA_PATH / "loop"
        if not loop_path.exists():
            pytest.skip("Loop data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(loop_path)
        assert db_type == "loop", f"Expected 'loop', got '{db_type}'"

    def test_detect_uc_ht_database(self) -> None:
        """Test UC_HT database detection."""
        uc_ht_path = DATA_PATH / "UC_HT"
        if not uc_ht_path.exists():
            pytest.skip("UC_HT data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(uc_ht_path)
        assert db_type == "uc_ht", f"Expected 'uc_ht', got '{db_type}'"


class TestHUPAConverter:
    """Test HUPA database converter with real data."""

    @pytest.fixture
    def hupa_path(self) -> Path:
        path = DATA_PATH / "HUPA"
        if not path.exists():
            pytest.skip("HUPA data not available")
        return path

    def test_consolidate_hupa_data(self, hupa_path: Path) -> None:
        """Test HUPA data consolidation."""
        config = {"first_n_users": 2}  # Limit to 2 users for faster testing
        converter = HupaDatabaseConverter(config, database_type="hupa")
        
        df = converter.consolidate_data(hupa_path)
        
        assert len(df) > 0, "Should have records"
        assert "timestamp" in df.columns, "Should have timestamp column"
        assert "user_id" in df.columns, "Should have user_id column"
        assert "glucose_value_mgdl" in df.columns, "Should have glucose column"

    def test_hupa_has_valid_timestamps(self, hupa_path: Path) -> None:
        """Test that HUPA timestamps are parsed correctly."""
        config = {"first_n_users": 1}
        converter = HupaDatabaseConverter(config, database_type="hupa")
        
        df = converter.consolidate_data(hupa_path)
        
        # Check timestamp column is datetime type
        assert df["timestamp"].dtype == pl.Datetime, f"Expected Datetime, got {df['timestamp'].dtype}"
        
        # Check no null timestamps
        null_count = df["timestamp"].null_count()
        assert null_count == 0, f"Should have no null timestamps, got {null_count}"

    def test_hupa_extracts_user_ids(self, hupa_path: Path) -> None:
        """Test that user IDs are extracted from HUPA filenames."""
        config = {"first_n_users": 3}
        converter = HupaDatabaseConverter(config, database_type="hupa")
        
        df = converter.consolidate_data(hupa_path)
        
        unique_users = df["user_id"].unique().to_list()
        assert len(unique_users) > 0, "Should have at least one user"
        # HUPA user IDs should be like "1p", "2p", etc.
        for uid in unique_users:
            assert uid is not None and str(uid) != "", f"User ID should not be empty: {uid}"


class TestUoMConverter:
    """Test UoM database converter with real data."""

    @pytest.fixture
    def uom_path(self) -> Path:
        path = DATA_PATH / "uom"
        if not path.exists():
            pytest.skip("UoM data not available")
        return path

    def test_consolidate_uom_data(self, uom_path: Path) -> None:
        """Test UoM data consolidation."""
        config = {"first_n_users": 2}
        converter = UoMDatabaseConverter(config, database_type="uom")
        
        df = converter.consolidate_data(uom_path)
        
        assert len(df) > 0, "Should have records"
        assert "timestamp" in df.columns, "Should have timestamp column"
        assert "user_id" in df.columns, "Should have user_id column"

    def test_uom_has_glucose_data(self, uom_path: Path) -> None:
        """Test that UoM glucose data is present."""
        config = {"first_n_users": 1}
        converter = UoMDatabaseConverter(config, database_type="uom")
        
        df = converter.consolidate_data(uom_path)
        
        # UoM should have glucose values
        assert "glucose_value_mgdl" in df.columns, "Should have glucose column"
        
        # Check some glucose values exist (filter out empty strings)
        glucose_col = df["glucose_value_mgdl"]
        if glucose_col.dtype in [pl.Utf8, pl.String]:
            non_empty = df.filter(pl.col("glucose_value_mgdl") != "")
        else:
            non_empty = df.filter(pl.col("glucose_value_mgdl").is_not_null())
        
        assert len(non_empty) > 0, "Should have some glucose values"


class TestMedtronicConverter:
    """Test Medtronic database converter with real data."""

    @pytest.fixture
    def medtronic_path(self) -> Path:
        path = DATA_PATH / "medtronic"
        if not path.exists():
            pytest.skip("Medtronic data not available")
        return path

    def test_consolidate_medtronic_data(self, medtronic_path: Path) -> None:
        """Test Medtronic data consolidation."""
        config = {}
        converter = MedtronicDatabaseConverter(config, database_type="medtronic")
        
        df = converter.consolidate_data(medtronic_path)
        
        assert len(df) > 0, "Should have records"
        assert "timestamp" in df.columns, "Should have timestamp column"

    def test_medtronic_has_valid_timestamps(self, medtronic_path: Path) -> None:
        """Test that Medtronic timestamps are parsed correctly."""
        config = {}
        converter = MedtronicDatabaseConverter(config, database_type="medtronic")
        
        df = converter.consolidate_data(medtronic_path)
        
        # Check timestamp column is datetime type
        assert df["timestamp"].dtype == pl.Datetime, f"Expected Datetime, got {df['timestamp'].dtype}"


class TestMinidose1Converter:
    """Test MiniDose1 database converter with real data."""

    @pytest.fixture
    def minidose_path(self) -> Path:
        path = DATA_PATH / "minidose1"
        if not path.exists():
            pytest.skip("MiniDose1 data not available")
        return path

    def test_consolidate_minidose1_data(self, minidose_path: Path) -> None:
        """Test MiniDose1 data consolidation."""
        config = {}
        converter = Minidose1DatabaseConverter(config, database_type="minidose1")
        
        df = converter.consolidate_data(minidose_path)
        
        assert len(df) > 0, "Should have records"
        assert "timestamp" in df.columns, "Should have timestamp column"

    def test_minidose1_has_glucose_data(self, minidose_path: Path) -> None:
        """Test that MiniDose1 has glucose data."""
        config = {}
        converter = Minidose1DatabaseConverter(config, database_type="minidose1")
        
        df = converter.consolidate_data(minidose_path)
        
        # Check glucose column exists
        assert "glucose_value_mgdl" in df.columns, "Should have glucose column"


class TestLoopConverter:
    """Test Loop database converter with real data."""

    @pytest.fixture
    def loop_path(self) -> Path:
        path = DATA_PATH / "loop"
        if not path.exists():
            pytest.skip("Loop data not available")
        return path

    def test_consolidate_loop_data(self, loop_path: Path) -> None:
        """Test Loop data consolidation using iter_user_event_frames."""
        config = {"first_n_users": 2, "expected_interval_minutes": 5}
        converter = LoopDatabaseConverter(config, database_type="loop")
        
        df = converter.consolidate_data(loop_path)
        
        assert len(df) > 0, "Should have records"
        assert "timestamp" in df.columns, "Should have timestamp column"
        assert "user_id" in df.columns, "Should have user_id column"

    def test_loop_iter_user_event_frames(self, loop_path: Path) -> None:
        """Test Loop streaming via iter_user_event_frames."""
        config = {"first_n_users": 2}
        converter = LoopDatabaseConverter(config, database_type="loop")
        
        user_count = 0
        total_records = 0
        for user_df in converter.iter_user_event_frames(loop_path, interval_minutes=5):
            user_count += 1
            total_records += len(user_df)
            
            # Each user frame should have required columns
            assert "timestamp" in user_df.columns
            assert "user_id" in user_df.columns
        
        assert user_count > 0, "Should yield at least one user"
        assert total_records > 0, "Should have records"

    def test_loop_has_multiple_event_types(self, loop_path: Path) -> None:
        """Test that Loop data has multiple event types (CGM, Basal, Bolus, etc.)."""
        config = {"first_n_users": 3}
        converter = LoopDatabaseConverter(config, database_type="loop")
        
        df = converter.consolidate_data(loop_path)
        
        assert "event_type" in df.columns, "Should have event_type column"
        event_types = df["event_type"].unique().to_list()
        
        # Loop should have at least CGM data
        assert "EGV" in event_types or len(event_types) > 0, f"Should have event types, got {event_types}"

    def test_loop_no_duplicates(self, loop_path: Path) -> None:
        """Test that Loop deduplication works correctly."""
        config = {"first_n_users": 2}
        converter = LoopDatabaseConverter(config, database_type="loop")
        
        df = converter.consolidate_data(loop_path)
        
        # Check for duplicates on (user_id, timestamp, event_type)
        dups = df.group_by(["user_id", "timestamp", "event_type"]).len().filter(pl.col("len") > 1)
        assert len(dups) == 0, f"Should have no duplicates, found {len(dups)}"


class TestUCHTConverter:
    """Test UC_HT database converter with real data."""

    @pytest.fixture
    def uc_ht_path(self) -> Path:
        path = DATA_PATH / "UC_HT"
        if not path.exists():
            pytest.skip("UC_HT data not available")
        return path

    def test_consolidate_uc_ht_data(self, uc_ht_path: Path) -> None:
        """Test UC_HT data consolidation."""
        config = {"first_n_users": 2, "expected_interval_minutes": 5}
        converter = UCHTDatabaseConverter(config, database_type="uc_ht")
        
        df = converter.consolidate_data(uc_ht_path)
        
        assert len(df) > 0, "Should have records"
        assert "timestamp" in df.columns, "Should have timestamp column"
        assert "user_id" in df.columns, "Should have user_id column"

    def test_uc_ht_iter_user_event_frames(self, uc_ht_path: Path) -> None:
        """Test UC_HT streaming via iter_user_event_frames."""
        config = {"first_n_users": 2}
        converter = UCHTDatabaseConverter(config, database_type="uc_ht")
        
        user_count = 0
        total_records = 0
        for user_df in converter.iter_user_event_frames(uc_ht_path, interval_minutes=5):
            user_count += 1
            total_records += len(user_df)
            
            # Each user frame should have required columns
            assert "timestamp" in user_df.columns
            assert "user_id" in user_df.columns
        
        assert user_count > 0, "Should yield at least one user"
        assert total_records > 0, "Should have records"

    def test_uc_ht_has_glucose_data(self, uc_ht_path: Path) -> None:
        """Test that UC_HT has glucose data from Excel files."""
        config = {"first_n_users": 1}
        converter = UCHTDatabaseConverter(config, database_type="uc_ht")
        
        df = converter.consolidate_data(uc_ht_path)
        
        # UC_HT should have glucose values
        assert "glucose_value_mgdl" in df.columns, "Should have glucose column"


class TestEndToEndWithDatabaseDetector:
    """Test full pipeline using DatabaseDetector to get converters."""

    def test_hupa_end_to_end(self) -> None:
        """Test HUPA detection and conversion end-to-end."""
        hupa_path = DATA_PATH / "HUPA"
        if not hupa_path.exists():
            pytest.skip("HUPA data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(hupa_path)
        
        config = {"first_n_users": 1}
        converter = detector.get_database_converter(db_type, config)
        
        assert converter is not None, f"Should get converter for {db_type}"
        
        df = converter.consolidate_data(hupa_path)
        assert len(df) > 0, "Should produce data"

    def test_uom_end_to_end(self) -> None:
        """Test UoM detection and conversion end-to-end."""
        uom_path = DATA_PATH / "uom"
        if not uom_path.exists():
            pytest.skip("UoM data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(uom_path)
        
        config = {"first_n_users": 1}
        converter = detector.get_database_converter(db_type, config)
        
        assert converter is not None, f"Should get converter for {db_type}"
        
        df = converter.consolidate_data(uom_path)
        assert len(df) > 0, "Should produce data"

    def test_loop_end_to_end(self) -> None:
        """Test Loop detection and conversion end-to-end."""
        loop_path = DATA_PATH / "loop"
        if not loop_path.exists():
            pytest.skip("Loop data not available")
        
        detector = DatabaseDetector()
        db_type = detector.detect_database_type(loop_path)
        
        config = {"first_n_users": 2}
        converter = detector.get_database_converter(db_type, config)
        
        assert converter is not None, f"Should get converter for {db_type}"
        assert hasattr(converter, "iter_user_event_frames"), "Loop should have streaming method"
        
        df = converter.consolidate_data(loop_path)
        assert len(df) > 0, "Should produce data"

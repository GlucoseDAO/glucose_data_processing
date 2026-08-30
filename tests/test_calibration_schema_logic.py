import pytest
import polars as pl
from datetime import datetime, timedelta

from glucose_ml_preprocessor import GlucoseMLPreprocessor

class TestCalibrationSchemaLogic:
    """
    Test that calibration period removal is correctly controlled by schema flags.
    """

    @pytest.fixture
    def preprocessor(self):
        return GlucoseMLPreprocessor(
            expected_interval_minutes=5,
            small_gap_max_minutes=15,
            calibration_period_minutes=165,  # 2h 45m
            remove_after_calibration_hours=24
        )

    def _create_data_with_calibration_gap(self):
        # Create data with a calibration gap
        # 1. Pre-calibration: 10 points
        start = datetime(2023, 1, 1, 10, 0, 0)
        pre_calib_times = [start + timedelta(minutes=5*i) for i in range(10)]
        
        # 2. Calibration gap: 3 hours (> 165 mins)
        calibration_end = pre_calib_times[-1] + timedelta(hours=3)
        
        # 3. Post-calibration: 24 hours + 1 hour (total 25 hours)
        # 24 hours @ 5 min intervals = 288 points
        # + 12 points for the extra hour = 300 points
        post_calib_times = [calibration_end + timedelta(minutes=5*i) for i in range(1, 301)]
        
        all_times = pre_calib_times + post_calib_times
        
        return pl.DataFrame({
            "timestamp": all_times,
            "glucose_value_mgdl": [100.0] * len(all_times)
        })

    def test_calibration_removal_enabled(self, preprocessor):
        """Test that data IS removed when remove_after_calibration is True."""
        df = self._create_data_with_calibration_gap()
        
        # Define categories with removal enabled
        field_categories = {
            'continuous': ['glucose_value_mgdl'],
            'occasional': [],
            'service': [],
            'remove_after_calibration': True
        }
        
        df_seq, stats, _ = preprocessor.gap_detector.detect_gaps_and_sequences(df, field_categories_dict=field_categories)
        
        # Verify stats show calibration detected and records removed
        calib_stats = stats.get('calibration_period_analysis', {})
        assert calib_stats.get('calibration_periods_detected') == 1
        assert calib_stats.get('total_records_marked_for_removal') > 0
        
        # Verify records were actually removed from df_seq
        # Pre-calibration (10) + post-calibration outside 24h window (12 points) = 22 points
        assert len(df_seq) == 22

    def test_calibration_removal_disabled(self, preprocessor):
        """Test that data is NOT removed when remove_after_calibration is False."""
        df = self._create_data_with_calibration_gap()
        
        # Define categories with removal disabled
        field_categories = {
            'continuous': ['glucose_value_mgdl'],
            'occasional': [],
            'service': [],
            'remove_after_calibration': False
        }
        
        df_seq, stats, _ = preprocessor.gap_detector.detect_gaps_and_sequences(df, field_categories_dict=field_categories)
        
        # Verify stats show calibration detected but NO records removed
        calib_stats = stats.get('calibration_period_analysis', {})
        assert calib_stats.get('calibration_periods_detected') == 1
        assert calib_stats.get('total_records_marked_for_removal') == 0
        
        # Verify all records are kept
        assert len(df_seq) == len(df)

    def test_calibration_removal_default_behavior(self, preprocessor):
        """Test that data IS removed by default (when flag is missing)."""
        df = self._create_data_with_calibration_gap()
        
        # Define categories without the flag
        field_categories = {
            'continuous': ['glucose_value_mgdl'],
            'occasional': [],
            'service': []
        }
        
        df_seq, stats, _ = preprocessor.gap_detector.detect_gaps_and_sequences(df, field_categories_dict=field_categories)
        
        # Verify stats show calibration detected and records removed (default = True)
        calib_stats = stats.get('calibration_period_analysis', {})
        assert calib_stats.get('calibration_periods_detected') == 1
        assert calib_stats.get('total_records_marked_for_removal') > 0
        assert len(df_seq) == 22

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


import pytest
import polars as pl
from datetime import datetime, timedelta

from glucose_ml_preprocessor import GlucoseMLPreprocessor

class TestFixedFrequencyEdgeCases:
    """
    Deep dive tests for create_fixed_frequency_data method.
    Focuses on interpolation logic, event shifting, and edge cases.
    """
    
    @pytest.fixture
    def preprocessor(self):
        return GlucoseMLPreprocessor(
            expected_interval_minutes=5,
            create_fixed_frequency=True
        )

    def test_glucose_linear_interpolation_exact(self, preprocessor):
        """
        Verify glucose is linearly interpolated based on time.
        """
        # Data: 10:00 (100), 10:10 (200).
        # Fixed grid should include 10:05.
        # Expected at 10:05: 150.
        
        df = pl.DataFrame({
            "timestamp": [datetime(2023,1,1,10,0), datetime(2023,1,1,10,10)],
            "sequence_id": [0, 0],
            "glucose_value_mgdl": [100.0, 200.0]
        })
        
        fixed_df, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(df)
        
        # Find 10:05 entry
        target_time = datetime(2023,1,1,10,5)
        row = fixed_df.filter(pl.col("timestamp") == target_time)
        
        assert len(row) == 1
        val = row["glucose_value_mgdl"][0]
        assert val == 150.0, f"Expected linear interpolation 150.0, got {val}"

    def test_event_shifting_nearest_neighbor(self, preprocessor):
        """
        Verify that discrete events (Carbs) are shifted to the NEAREST grid point.
        Current implementation might prioritize 'Next' over 'Prev'.
        """
        # Grid: 10:00, 10:05.
        # Event at 10:01 (Carb=10). Should shift to 10:00 (dist 1).
        # Event at 10:04 (Carb=20). Should shift to 10:05 (dist 1).
        
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023,1,1,10,0), # Anchor
                datetime(2023,1,1,10,1), # Event 1
                datetime(2023,1,1,10,4), # Event 2
                datetime(2023,1,1,10,5)  # Anchor
            ],
            "sequence_id": [0, 0, 0, 0],
            "glucose_value_mgdl": [100.0, 100.0, 100.0, 100.0],
            "carb_grams": [None, 10.0, 20.0, None]
        })

        fixed_df, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(
            df,
            {
                "continuous": ["glucose_value_mgdl"],
                "occasional": ["carb_grams"],
                "occasional_sum": ["carb_grams"],
                "service": ["event_type"]
            },
        )
        
        # Check 10:00
        row_00 = fixed_df.filter(pl.col("timestamp") == datetime(2023,1,1,10,0))
        # Check 10:05
        row_05 = fixed_df.filter(pl.col("timestamp") == datetime(2023,1,1,10,5))
        
        # Expect 10:00 to have Carb=10 (from 10:01)
        assert row_00["carb_grams"][0] == 10.0, \
            f"Event at 10:01 should shift to 10:00. Got {row_00['carb_grams'][0]}"
            
        # Expect 10:05 to have Carb=20 (from 10:04)
        assert row_05["carb_grams"][0] == 20.0, \
            f"Event at 10:04 should shift to 10:05. Got {row_05['carb_grams'][0]}"

    def test_multiple_events_between_grid_points(self, preprocessor):
        """
        Verify behavior when multiple events occur between grid points.
        Grid: 10:00, 10:05.
        Events: 10:02 (Carb=10), 10:03 (Carb=20).
        Ideally:
         - 10:02 (dist 2 to 10:00) -> 10:00
         - 10:03 (dist 2 to 10:05) -> 10:05
        """
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023,1,1,10,0),
                datetime(2023,1,1,10,2),
                datetime(2023,1,1,10,3),
                datetime(2023,1,1,10,5)
            ],
            "sequence_id": [0, 0, 0, 0],
            "glucose_value_mgdl": [100.0, 100.0, 100.0, 100.0],
            "carb_grams": [None, 10.0, 20.0, None]
        })

        fixed_df, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(
            df,
            {
                "continuous": ["glucose_value_mgdl"],
                "occasional": ["carb_grams"],
                "occasional_sum": ["carb_grams"],
                "service": ["event_type"]
            },
        )
        
        row_00 = fixed_df.filter(pl.col("timestamp") == datetime(2023,1,1,10,0))
        row_05 = fixed_df.filter(pl.col("timestamp") == datetime(2023,1,1,10,5))
        
        assert row_00["carb_grams"][0] == 10.0
        assert row_05["carb_grams"][0] == 20.0

    def test_conflicting_shifts(self, preprocessor):
        """
        Verify behavior when different parameters shift from different directions.
        Grid: 10:05.
        Data: 
         - 10:04: Carbs=50
         - 10:06: Insulin=5
        Expectation: 10:05 should capture BOTH (Carbs from prev, Insulin from next) if nearest logic applies.
        """
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023,1,1,10,0),
                datetime(2023,1,1,10,4),
                datetime(2023,1,1,10,6),
                datetime(2023,1,1,10,10)
            ],
            "sequence_id": [0, 0, 0, 0],
            "glucose_value_mgdl": [100.0, 100.0, 100.0, 100.0],
            "carb_grams": [None, 50.0, None, None],
            "fast_acting_insulin_u": [None, None, 5.0, None]
        })

        fixed_df, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(
            df,
            {
                "continuous": ["glucose_value_mgdl"],
                "occasional": ["carb_grams", "fast_acting_insulin_u"],
                "occasional_sum": ["carb_grams", "fast_acting_insulin_u"],
                "service": ["event_type"]
            },
        )
        
        # Target 10:05
        target = fixed_df.filter(pl.col("timestamp") == datetime(2023,1,1,10,5))
        
        assert len(target) == 1
        # Should have Carbs from 10:04 (dist 1)
        assert target["carb_grams"][0] == 50.0
        # Should have Insulin from 10:06 (dist 1)
        assert target["fast_acting_insulin_u"][0] == 5.0

    def test_duplicate_events_on_shift(self, preprocessor):
        """
        Test if a single event is duplicated to multiple grid points.
        Grid: 10:00, 10:05.
        Event: 10:02 (Carb=10).
        10:00 sees it as Next (dist 2).
        10:05 sees it as Prev (dist 3).
        Should it go to 10:00 only (nearest)?
        Or duplicate?
        """
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023,1,1,10,0),
                datetime(2023,1,1,10,2),
                datetime(2023,1,1,10,5)
            ],
            "sequence_id": [0, 0, 0],
            "glucose_value_mgdl": [100.0, 100.0, 100.0],
            "carb_grams": [None, 10.0, None]
        })

        fixed_df, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(
            df,
            {
                "continuous": ["glucose_value_mgdl"],
                "occasional": ["carb_grams"],
                "occasional_sum": ["carb_grams"],
                "service": ["event_type"]
            },
        )
        
        row_00 = fixed_df.filter(pl.col("timestamp") == datetime(2023,1,1,10,0))
        row_05 = fixed_df.filter(pl.col("timestamp") == datetime(2023,1,1,10,5))
        
        # 10:02 is closer to 10:00.
        assert row_00["carb_grams"][0] == 10.0
        # 10:05 is farther. Should be None or at least distinct if not duplicating.
        # If duplicated, it would be 10.0. Ideal is None.
        assert row_05["carb_grams"][0] is None, "Event duplicated to farther grid point"



import pytest
import polars as pl
from datetime import datetime, timedelta
from glucose_ml_preprocessor import GlucoseMLPreprocessor

class TestFixedFrequencyRobustness:
    """
    Tests that demonstrate the robustness of Step 6.
    """

    @pytest.fixture
    def preprocessor(self):
        # Default gap limit is 15 minutes
        return GlucoseMLPreprocessor(
            expected_interval_minutes=5,
            small_gap_max_minutes=15,
            create_fixed_frequency=True
        )

    def test_glucose_bounding_with_margin(self, preprocessor):
        """
        Test that Step 6 bounds the output by glucose timestamps but allows a half-interval
        margin for covariates to be shifted onto the first/last grid points.
        """
        # Grid: 5 min intervals (expected grid points at 10:20, 10:25)
        # Event at 10:18 (2 mins before first glucose) -> Should be PRESERVED and shifted to 10:20
        # Event at 10:10 (10 mins before first glucose) -> Should be DROPPED
        
        first_glucose = datetime(2023, 1, 1, 10, 20)
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1, 10, 10), # Dropped (> 2.5m away)
                datetime(2023, 1, 1, 10, 18), # Preserved (2m away < 2.5m)
                first_glucose,                # First Glucose
                datetime(2023, 1, 1, 10, 25), # Last Glucose
                datetime(2023, 1, 1, 10, 27), # Preserved (2m away < 2.5m)
                datetime(2023, 1, 1, 10, 35), # Dropped (> 2.5m away)
            ],
            "sequence_id": [1, 1, 1, 1, 1, 1],
            "glucose_value_mgdl": [None, None, 100.0, 110.0, None, None],
            "carb_grams": [50.0, 10.0, None, None, 20.0, 30.0]
        })

        field_categories = {
            "continuous": ["glucose_value_mgdl"],
            "occasional": ["carb_grams"],
            "occasional_sum": ["carb_grams"],
            "service": []
        }

        df_fixed, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(df, field_categories)
        
        # Check first grid point (10:20)
        row_start = df_fixed.filter(pl.col("timestamp") == first_glucose)
        assert row_start["carb_grams"][0] == 10.0, f"Expected 10.0g from 10:18, got {row_start['carb_grams'][0]}"
        
        # Check last grid point (10:25)
        row_end = df_fixed.filter(pl.col("timestamp") == datetime(2023, 1, 1, 10, 25))
        assert row_end["carb_grams"][0] == 20.0, f"Expected 20.0g from 10:27, got {row_end['carb_grams'][0]}"
        
        # Verify no 50.0 or 30.0 made it in
        assert df_fixed["carb_grams"].sum() == 30.0 # 10 + 20

    def test_glucose_bounding_at_start(self, preprocessor):
        """
        Test that Step 6 bounds the output by glucose timestamps.
        Covariates outside the glucose range should be dropped.
        """
        # Sequence starts at 10:00 with Heart Rate, but Glucose starts at 10:20
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1, 10, 0),  # HR only -> Should be dropped
                datetime(2023, 1, 1, 10, 20), # First Glucose
                datetime(2023, 1, 1, 10, 25), 
            ],
            "sequence_id": [1, 1, 1],
            "glucose_value_mgdl": [None, 100.0, 110.0],
            "heart_rate": [70.0, None, None]
        })

        field_categories = {
            "continuous": ["glucose_value_mgdl", "heart_rate"],
            "occasional": [],
            "service": []
        }

        df_fixed, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(df, field_categories)
        
        # Grid should start at or after 10:20 (first glucose)
        assert df_fixed["timestamp"].min() >= datetime(2023, 1, 1, 10, 20)
        assert len(df_fixed.filter(pl.col("timestamp") < datetime(2023, 1, 1, 10, 20))) == 0

    def test_glucose_bounding_at_end(self, preprocessor):
        """
        Test that Step 6 bounds the output by glucose timestamps at the end.
        """
        # Glucose ends at 10:00, but sequence continues with Heart Rate until 10:20
        df = pl.DataFrame({
            "timestamp": [
                datetime(2023, 1, 1, 9, 55),
                datetime(2023, 1, 1, 10, 0),  # Last Glucose
                datetime(2023, 1, 1, 10, 20), # HR only -> Should be dropped
            ],
            "sequence_id": [1, 1, 1],
            "glucose_value_mgdl": [90.0, 100.0, None],
            "heart_rate": [None, None, 75.0]
        })

        field_categories = {
            "continuous": ["glucose_value_mgdl", "heart_rate"],
            "occasional": [],
            "service": []
        }

        df_fixed, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(df, field_categories)
        
        # Grid should end at or before 10:00 (last glucose)
        assert df_fixed["timestamp"].max() <= datetime(2023, 1, 1, 10, 0)
        assert len(df_fixed.filter(pl.col("timestamp") > datetime(2023, 1, 1, 10, 0))) == 0

    def test_full_pipeline_simulation(self, preprocessor):
        """
        Simulate data as it would come from previous steps and verify Step 6 logic.
        """
        # Step 3 (Gap Detection) would split at 30m gaps.
        # Step 4 (Interpolation) would fill 10m gaps.
        # So Step 6 only sees continuous sequences of glucose.
        
        base_time = datetime(2023, 1, 1, 10, 0)
        df = pl.DataFrame({
            "timestamp": [
                base_time, 
                base_time + timedelta(minutes=5),
                base_time + timedelta(minutes=10),
                base_time + timedelta(minutes=15)
            ],
            "sequence_id": [1, 1, 1, 1],
            "glucose_value_mgdl": [100.0, 105.0, 110.0, 115.0]
        })
        
        df_fixed, _ = preprocessor.fixed_freq_generator.create_fixed_frequency_data(df)
        
        assert len(df_fixed) == 4
        assert df_fixed["glucose_value_mgdl"].null_count() == 0
        assert df_fixed["timestamp"].to_list() == [
            base_time, 
            base_time + timedelta(minutes=5),
            base_time + timedelta(minutes=10),
            base_time + timedelta(minutes=15)
        ]

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

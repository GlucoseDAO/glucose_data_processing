from __future__ import annotations

from datetime import datetime

import polars as pl

from glucose_ml_preprocessor import GlucoseMLPreprocessor
from formats.base_converter import CSVFormatConverter


def test_new_field_only_in_output_fields_kept_as_field_name() -> None:
    """
    If a new field exists in config.output_fields but NOT in config.field_to_display_name_map:
    - it should still appear in output
    - it should keep its standard name as the CSV/header name
    """
    config = {
        "output_fields": [
            "timestamp",
            "event_type",
            "glucose_value_mgdl",
            "new_metric",
        ],
        "field_to_display_name_map": {
            "timestamp": "Timestamp (YYYY-MM-DDThh:mm:ss)",
            "event_type": "Event Type",
            "glucose_value_mgdl": "Glucose Value (mg/dL)",
        },
    }

    CSVFormatConverter.initialize_from_config(config)
    try:
        preprocessor = GlucoseMLPreprocessor()
        # Ensure glucose is treated as continuous (Float64 cast).
        preprocessor._field_categories_dict = {
            "continuous": ["glucose_value_mgdl"],
            "occasional": [],
            "service": ["timestamp", "event_type"],
        }

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1, 10, 0, 0)],
                "sequence_id": [1],
                "glucose_value_mgdl": ["100"],  # string -> should become float
                "new_metric": [7],  # int -> should become float (stable numeric)
            }
        )

        ml_df = preprocessor.prepare_ml_data(df)

        # Display names used for mapped fields
        assert "Glucose Value (mg/dL)" in ml_df.columns

        # New field present and kept as-is (no mapping)
        assert "new_metric" in ml_df.columns

        assert ml_df["Glucose Value (mg/dL)"].dtype == pl.Float64
        assert ml_df["new_metric"].dtype == pl.Float64
    finally:
        # Reset global converter config so we don't affect other tests
        CSVFormatConverter.initialize_from_config(None)


def test_new_field_in_output_fields_and_display_map_uses_display_name() -> None:
    """
    If a new field exists in config.output_fields AND config.field_to_display_name_map:
    - it should appear in output
    - it should be renamed to the display name in the final output
    """
    config = {
        "output_fields": [
            "timestamp",
            "event_type",
            "glucose_value_mgdl",
            "new_metric",
        ],
        "field_to_display_name_map": {
            "timestamp": "Timestamp (YYYY-MM-DDThh:mm:ss)",
            "event_type": "Event Type",
            "glucose_value_mgdl": "Glucose Value (mg/dL)",
            "new_metric": "New Metric (units)",
        },
    }

    CSVFormatConverter.initialize_from_config(config)
    try:
        preprocessor = GlucoseMLPreprocessor()
        preprocessor._field_categories_dict = {
            "continuous": ["glucose_value_mgdl"],
            "occasional": [],
            "service": ["timestamp", "event_type"],
        }

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1, 10, 0, 0)],
                "sequence_id": [1],
                "glucose_value_mgdl": ["100"],
                "new_metric": [7],
            }
        )

        ml_df = preprocessor.prepare_ml_data(df)

        assert "New Metric (units)" in ml_df.columns
        assert "new_metric" not in ml_df.columns
    finally:
        CSVFormatConverter.initialize_from_config(None)



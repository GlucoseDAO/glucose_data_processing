import polars as pl
import pytest
from processing.stats_manager import StatsManager
from processing.core.fields import StandardFieldNames

def test_stats_manager_dynamic_fields():
    # Create a dataframe with some standard and some non-standard fields
    df = pl.DataFrame({
        StandardFieldNames.TIMESTAMP: ["2023-01-01 00:00:00", "2023-01-01 00:05:00", "2023-01-01 00:10:00"],
        StandardFieldNames.SEQUENCE_ID: [1, 1, 1],
        StandardFieldNames.EVENT_TYPE: ["Regular", "Regular", "Regular"],
        StandardFieldNames.GLUCOSE_VALUE: [100, 110, None], # 66.6% completeness
        "custom_field_1": [1, None, None], # 33.3% completeness
        "custom_field_2": [None, None, None], # 0% completeness
    })
    
    stats_manager = StatsManager(original_record_count=3)
    
    # These stats are usually provided by other steps, we can mock them
    gap_stats = {"total_gaps": 0}
    interp_stats = {"total_interpolations": 0}
    
    stats = stats_manager.get_statistics(df, gap_stats, interp_stats)
    
    quality = stats.get('data_quality', {})
    
    # Current behavior check (will likely fail for custom fields)
    assert f'{StandardFieldNames.GLUCOSE_VALUE}_data_completeness' in quality
    assert abs(quality[f'{StandardFieldNames.GLUCOSE_VALUE}_data_completeness'] - 66.666) < 0.1
    
    # We want these to be present dynamically
    # Note: The names might be slightly different depending on how we implement it
    # but they should be there.
    assert 'custom_field_1_data_completeness' in quality
    assert abs(quality['custom_field_1_data_completeness'] - 33.333) < 0.1
    assert 'custom_field_2_data_completeness' in quality
    assert quality['custom_field_2_data_completeness'] == 0

def test_stats_manager_aggregate_dynamic():
    stats1 = {
        'dataset_overview': {'total_records': 100, 'total_sequences': 2, 'original_records': 100, 'date_range': {'start': '2023-01-01', 'end': '2023-01-02'}},
        'sequence_analysis': {'longest_sequence': 50, 'shortest_sequence': 50, 'all_lengths': [50, 50]},
        'gap_analysis': {'total_gaps': 1, 'total_sequences': 2},
        'interpolation_analysis': {'total_interpolations': 5, 'total_interpolated_data_points': 5},
        'data_quality': {
            'glucose_value_mgdl_data_completeness': 90.0,
            'custom_field_data_completeness': 50.0,
            'interpolated_records': 2,
            'inserted_records': 3
        }
    }
    
    stats2 = {
        'dataset_overview': {'total_records': 200, 'total_sequences': 1, 'original_records': 200, 'date_range': {'start': '2023-01-02', 'end': '2023-01-03'}},
        'sequence_analysis': {'longest_sequence': 200, 'shortest_sequence': 200, 'all_lengths': [200]},
        'gap_analysis': {'total_gaps': 0, 'total_sequences': 1},
        'interpolation_analysis': {'total_interpolations': 0, 'total_interpolated_data_points': 0},
        'data_quality': {
            'glucose_value_mgdl_data_completeness': 100.0,
            'custom_field_data_completeness': 80.0,
            'other_field_data_completeness': 100.0,
            'interpolated_records': 0,
            'inserted_records': 0
        }
    }
    
    stats_manager = StatsManager()
    aggregated = stats_manager.aggregate_statistics([stats1, stats2], ["db1", "db2"])
    
    quality = aggregated.get('data_quality', {})
    
    # Check weighted averages:
    # glucose: (90*100 + 100*200) / 300 = (9000 + 20000) / 300 = 29000 / 300 = 96.666
    assert abs(quality['glucose_value_mgdl_data_completeness'] - 96.666) < 0.1
    # custom: (50*100 + 80*200) / 300 = (5000 + 16000) / 300 = 21000 / 300 = 70.0
    assert abs(quality['custom_field_data_completeness'] - 70.0) < 0.1
    # other: (0*100 + 100*200) / 300 = 20000 / 300 = 66.666
    assert abs(quality['other_field_data_completeness'] - 66.666) < 0.1
    
    # Check sums
    assert quality['interpolated_records'] == 2
    assert quality['inserted_records'] == 3

def test_stats_manager_print_dynamic():
    from processing.stats_manager import print_statistics
    
    stats = {
        'dataset_overview': {'total_records': 100, 'total_sequences': 2, 'original_records': 100, 'date_range': {'start': '2023-01-01', 'end': '2023-01-02'}},
        'sequence_analysis': {'longest_sequence': 50, 'shortest_sequence': 50, 'sequence_lengths': {'mean': 50, '50%': 50}},
        'gap_analysis': {'total_gaps': 1, 'total_sequences': 2},
        'interpolation_analysis': {'total_interpolations': 5, 'total_interpolated_data_points': 5},
        'data_quality': {
            'glucose_value_mgdl_data_completeness': 90.0,
            'custom_field_data_completeness': 50.0,
            'interpolated_records': 2,
            'inserted_records': 3
        }
    }
    
    output = print_statistics(stats)
    
    assert "Glucose Value Mgdl Data Completeness: 90.0%" in output
    assert "Custom Field Data Completeness: 50.0%" in output
    assert "Interpolated Records (Existing rows): 2" in output
    assert "Inserted Records (New rows): 3" in output

if __name__ == "__main__":
    # If run directly, try to run the test
    try:
        test_stats_manager_dynamic_fields()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")

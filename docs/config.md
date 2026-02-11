# Pipeline Configuration

The `GlucoseMLPreprocessor` is governed by a YAML configuration file (typically `glucose_config.yaml`). Command-line arguments take precedence over these settings.

## Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_interval_minutes` | int | 5 | The target time resolution for the ML-ready dataset. |
| `small_gap_max_minutes` | int | 15 | Maximum gap size (in minutes) to be filled via linear interpolation. |
| `min_sequence_len` | int | 200 | Minimum number of contiguous records required for a sequence to be preserved. |
| `create_fixed_frequency` | bool | true | Whether to resample data to the `expected_interval_minutes`. |
| `glucose_only` | bool | false | If true, drops all non-glucose fields and non-glucose records. |
| `round_precision` | int | 3 | Number of digits after the decimal point to round numeric fields. Can be negative. |
| `save_intermediate_files` | bool | false | If true, exports CSVs at each stage of the pipeline for debugging. |

## Calibration Settings

- `calibration_period_minutes`: Duration (default: 165m) considered a "startup" or "calibration" period.
- `remove_after_calibration_hours`: Period of data following a calibration event that should be purged due to potential instability.

## Output Configuration

### `output_file`
The default path where the processed dataset will be saved. 
- **Type**: string (path)
- **Default**: `"OUTPUT/processed_dataset.csv"`

### `output_fields`
A list of standardized field names to include in the final CSV. Fields excluded from this list will be dropped during the final preparation step.

### `field_to_display_name_map`
Maps internal standardized names to user-friendly column headers in the output file.
Example: `glucose_value_mgdl: "Glucose Value (mg/dL)"`

## Result Naming Priority

The final output filename is resolved using the following priority:

1.  **Command Line**: Explicitly provided via `--output` or `-o`.
2.  **Configuration**: Defined by the `output_file` field in the YAML config.
3.  **Schema Default**: If neither of the above are set, and a **single** dataset is provided, the pipeline looks up the `database` field in the database's schema (e.g., `uom.csv`).
4.  **Generic Fallback**: If multiple datasets are combined or no schema name is found, it defaults to `OUTPUT/processed_dataset.csv`.

Note: Since the default `glucose_config.yaml` often contains an `output_file` entry, you may need to comment it out or remove it if you want to use the "Smart Default" schema-based naming.

## Database-Specific Overrides

Settings can be customized per database type under the `database_configs` section:

```yaml
database_configs:
  dexcom:
    high_glucose_value: 401
    low_glucose_value: 39
    remove_calibration: true
  ai_ready:
    start_with_user_id: "1023"
```


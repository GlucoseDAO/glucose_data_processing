# Pipeline Configuration

The `GlucoseMLPreprocessor` is governed by a YAML configuration file (typically `glucose_config.yaml`). Command-line arguments take precedence over these settings. If `glucose_config.yaml` exists in the current directory, it is loaded automatically even without `--config`.

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remove_calibration` | bool | true | Remove calibration events to create interpolatable gaps. |
| `calibration_period_minutes` | int | 165 | Duration (in minutes) of a startup/calibration period (≈ 2 h 45 m). |
| `remove_after_calibration_hours` | int | 24 | Hours of data following a calibration event to purge due to potential instability. |

## Output Configuration

### `output_file`
The default path where the processed dataset will be saved.
- **Type**: string (path)
- **Default**: not set – filename is generated from the input folder name (e.g., `OUTPUT/uom_ml_ready.csv`).

### `output_fields`
A list of standardized field names to include in the final CSV. Fields excluded from this list will be dropped during the final preparation step.

### `field_to_display_name_map`
Maps internal standardized names to user-friendly column headers in the output file.
Example: `glucose_value_mgdl: "Glucose Value (mg/dL)"`

## Result Naming Priority

The final output filename is resolved using the following priority:

1. **Command Line**: Explicitly provided via `--output` or `-o`.
2. **Configuration**: Defined by the `output_file` field in the YAML config.
3. **Folder-name-based**: Generated from the input folder/ZIP names joined with underscores and suffixed with `_ml_ready.csv` (e.g., `OUTPUT/uom_ml_ready.csv` for `DATA/uom`).

When multiple datasets are combined the names are joined: `OUTPUT/hupa_uom_ml_ready.csv`.

## Database-Specific Overrides

Settings can be customized per database type by adding a top-level key matching the database name in the YAML config:

```yaml
dexcom:
  high_glucose_value: 401
  low_glucose_value: 39
  remove_calibration: true

hupa:
  # HUPA dataset specific settings

uc_ht:
  # UC_HT dataset specific settings
```

The supported database keys are: `dexcom`, `libre3`, `uom`, `hupa`, `uc_ht`, `medtronic`, `minidose1`, `loop`, `ai_ready`.

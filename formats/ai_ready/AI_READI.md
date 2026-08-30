# AI-READI Dataset

## Source and Reference

- **Main Dataset Page**: [FAIRhub - Flagship Dataset of Type 2 Diabetes from the AI-READI Project](https://fairhub.io/datasets/2)
- **Data Repository**: [Zenodo (v1.0.0)](https://zenodo.org/records/14680135)
- **Project Website**: [AI-READI](https://aireadi.org/)

## Database Structure

The AI-READI database is a **multi-user** format stored as a ZIP file containing structured JSON data from multiple participants. It includes comprehensive health data from Garmin devices and Dexcom G6 CGM systems.

### File Structure

- **Format**: ZIP archive containing JSON files organized by participant
- **Structure**: `/dataset/participants.tsv` + participant-specific JSON files
- **Detection**: Automatically detected by presence of `participants.tsv` and `*_DEX.json` files

### Data Types

The AI-READI format includes multiple data modalities:

1. **CGM (Dexcom G6)**: `wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/*_DEX.json`
2. **Heart Rate**: `wearable_blood_glucose/physical_activity/garmin_heart_rate/*_heartrate.json`
3. **Activity**: `wearable_blood_glucose/physical_activity/garmin_activity/*_activity.json`
4. **Calories**: `wearable_blood_glucose/physical_activity/garmin_calorie/*_calorie.json`
5. **Stress**: `wearable_blood_glucose/physical_activity/garmin_stress/*_stress.json`
6. **Sleep**: `wearable_blood_glucose/physical_activity/garmin_sleep/*_sleep.json`
7. **Respiratory Rate**: `wearable_blood_glucose/physical_activity/garmin_respiratory_rate/*_breathing.json`
8. **Oxygen Saturation**: `wearable_blood_glucose/physical_activity/garmin_oxygen_saturation/*_breathing.json`

### Data Structure

Each JSON file follows a structured format:
```json
{
  "header": {
    "patient_id": "...",  // or "user_id" depending on data type
    ...
  },
  "body": {
    "cgm": [...],  // or "heart_rate", "activity", etc.
    ...
  }
}
```

Fields are extracted using path-based mappings defined in `ai_ready_schema.yaml`:

- **CGM**: `body.cgm` → `blood_glucose.value` → `glucose_value_mgdl`
- **Heart Rate**: `body.heart_rate` → `heart_rate.value` → `heart_rate`
- **Activity**: `body.activity` → `base_movement_quantity.value` → `step_count`
- **Calories**: `body.activity` → `calories_value.value` → `active_kcal`
- **Stress**: `body.stress` → `stress.value` → `stress_level`
- **Sleep**: `body.sleep` → `sleep_stage_state` → `sleep_level`
- **Respiratory Rate**: `body.breathing` → `respiratory_rate.value` → `respiratory_rate`
- **Oxygen Saturation**: `body.breathing` → `oxygen_saturation.value` → `oxygen_saturation_percent`

### Timestamp Format

AI-READI uses: `%Y-%m-%dT%H:%M:%SZ`, `%Y-%m-%dT%H:%M:%S%z`, or `%Y-%m-%dT%H:%M:%S` (ISO format)

### Special Features

- **Streaming Processing**: Supports large datasets with configurable buffer sizes
- **Multi-user**: Each participant processed separately with `user_id` tracking
- **Rich Metadata**: Includes clinical site, study group, recommended split, age, study visit date

## Adding or Removing Features in Post-Processing

### Method 1: Configuration File (Recommended)

Use `glucose_config_ai_ready.yaml` or modify your config:

#### Adding Fields

```yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "user_id"
  - "glucose_value_mgdl"
  - "heart_rate"
  - "step_count"
  - "active_kcal"
  - "stress_level"
  - "respiratory_rate"
  - "oxygen_saturation_percent"
  - "sleep_level"
  - "age"
  - "clinical_site"
  - "study_group"
  - "recommended_split"
  - "study_visit_date"
  - "your_new_field"  # Add new field

field_to_display_name_map:
  your_new_field: "Your New Field Display Name"
```

#### Removing Fields

Remove fields from the `output_fields` list. Use `restrict_output_to_config_fields: true` to strictly enforce the list:

```yaml
restrict_output_to_config_fields: true
service_fields_allowlist:
  - "timestamp"
  - "event_type"
  - "user_id"
  # Only these service fields will be kept
```

### Method 2: Schema Modification

To add support for new JSON fields, edit `formats/ai_ready_schema.yaml`:

#### Adding a New Data Type

1. **Add field category**:
```yaml
field_categories:
  your_new_field: continuous  # or 'occasional' or 'service'
```

2. **Add converter definition**:
```yaml
converters:
  garmin_your_data:
    format: json
    event_type: YourEventType
    user_id_source: header.user_id
    records_path: body.your_data_path
    timestamp_path: effective_time_frame.date_time
    field_paths:
      your_field.value: your_new_field
```

3. **Update converter code** in `ai_ready_database_converter.py` to extract the new data type

#### Removing a Data Type

Remove the converter definition from `ai_ready_schema.yaml` and remove extraction code from the converter.

### Method 3: Streaming Configuration

For large datasets, configure streaming behavior:

```yaml
# glucose_config_ai_ready.yaml
streaming_max_buffer_mb: 64        # Max memory buffer size
streaming_flush_max_users: 5       # Flush after N users
```

### Method 4: Post-Processing Script

After running the pipeline:

```python
import polars as pl

df = pl.read_csv("OUTPUT/ai_ready_ml_ready.csv")

# Remove columns
df = df.drop(["column_to_remove"])

# Filter by user or metadata
df = df.filter(pl.col("clinical_site") == "Site1")

# Add computed fields
df = df.with_columns([
    (pl.col("glucose_value_mgdl").cast(pl.Float64) / 18.0).alias("glucose_mmol")
])

df.write_csv("OUTPUT/ai_ready_ml_ready_modified.csv")
```

## Available Fields

Based on `ai_ready_schema.yaml`:

### Continuous Fields
- `glucose_value_mgdl`
- `heart_rate`
- `active_kcal`
- `step_count`
- `stress_level`
- `respiratory_rate`
- `oxygen_saturation_percent`

### Occasional Fields
- `sleep_level`

### Service Fields
- `timestamp`
- `event_type`
- `user_id`
- `clinical_site`
- `study_group`
- `recommended_split`
- `age`
- `study_visit_date`

## Example: Minimal Glucose-Only Output

```yaml
# glucose_config_ai_ready.yaml
restrict_output_to_config_fields: true
output_fields:
  - "timestamp"
  - "event_type"
  - "user_id"
  - "glucose_value_mgdl"

service_fields_allowlist:
  - "timestamp"
  - "event_type"
  - "user_id"
```

## Example: Full Feature Set

```yaml
# glucose_config_ai_ready.yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "user_id"
  - "glucose_value_mgdl"
  - "heart_rate"
  - "step_count"
  - "active_kcal"
  - "stress_level"
  - "respiratory_rate"
  - "oxygen_saturation_percent"
  - "sleep_level"
  - "age"
  - "clinical_site"
  - "study_group"
  - "recommended_split"
  - "study_visit_date"
```

## Notes

- AI-READI datasets can be very large; use streaming configuration for memory efficiency
- User IDs are extracted from JSON headers (varies by data type: `patient_id` vs `user_id`)
- Multiple data types are merged by timestamp during processing
- Clinical metadata (site, group, split) is preserved for ML training splits

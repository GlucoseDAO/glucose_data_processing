# University of Manchester (UoM) Dataset

## Source and Reference

- **Data Repository**: [Zenodo - T1D-UOM – A Longitudinal Multimodal Dataset of Type 1 Diabetes](https://zenodo.org/records/15806142)
- **Dataset Page**: [University of Manchester Research Explorer](https://research.manchester.ac.uk/en/datasets/t1d-uom-a-longitudinal-multimodal-dataset-of-type-1-diabetes)
- **Scientific Publication**: [A Longitudinal Multimodal Dataset of Type 1 Diabetes, Scientific Data, 2025](https://research.manchester.ac.uk/en/publications/a-longitudinal-multimodal-dataset-of-type-1-diabetes/)
- **DOI**: 10.5281/zenodo.15806142

## Database Structure

The UoM database is a **multi-user** T1D dataset containing data from multiple participants. Each participant has separate CSV files for different data types.

### File Naming Convention

Files follow the pattern: `UoM{DataType}{ParticipantID}.csv`

- **Participant ID**: Extracted from filename (e.g., `UoMGlucose2301.csv` → participant `2301`)
- **Data Types**:
  - `UoMGlucose*.csv` - Continuous glucose monitoring data
  - `UoMActivity*.csv` - Physical activity data (steps, calories, motion intensity)
  - `UoMBasal*.csv` - Long-acting insulin (basal) doses
  - `UoMBolus*.csv` - Fast-acting insulin (bolus) doses
  - `UoMNutrition*.csv` - Meal/nutrition data (carbs, protein, fat, fiber)
  - `UoMSleep*.csv` - Sleep data with heart rate and stress levels
  - `UoMSleeptime*.csv` - Sleep duration and stage data

### Data Structure

Each data type has specific CSV columns that are mapped to standardized field names via the schema (`uom_schema.yaml`):

- **Glucose**: `bg_ts`, `value` (mmol/L, converted to mg/dL)
- **Activity**: `activity_ts`, `activity_type`, `step_count`, `active_Kcal`, `distance_m`, `motion_intensity_mean/max`, etc.
- **Basal**: `basal_ts`, `basal_dose`, `insulin_kind`
- **Bolus**: `bolus_ts`, `bolus_dose`
- **Nutrition**: `meal_ts`, `meal_type`, `meal_tag`, `carbs_g`, `prot_g`, `fat_g`, `fibre_g`
- **Sleep**: `sleep_ts`, `sleep_level`, `heart_rate`, `stress_level_value`, `resting_heart_rate`
- **Sleeptime**: `start_date_ts`, `duration_in_sec`, sleep stage breakdowns

### Timestamp Format

UoM uses: `%d/%m/%Y %H:%M:%S` or `%d/%m/%Y %H:%M` (converted to ISO format `%Y-%m-%dT%H:%M:%S`)

## Adding or Removing Features in Post-Processing

### Method 1: Configuration File (Recommended)

Modify `glucose_config.yaml` to control which fields appear in the final output:

#### Adding Fields

1. **Add to `output_fields` list** (use standard field names):
```yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"
  - "step_count"          # Add activity fields
  - "heart_rate"          # Add from sleep data
  - "active_kcal"         # Add activity calories
  - "motion_intensity_max" # Add motion data
```

2. **Add display name mapping** (optional, for user-friendly column headers):
```yaml
field_to_display_name_map:
  step_count: "Step Count"
  heart_rate: "Heart Rate (bpm)"
  active_kcal: "Active Calories (kcal)"
  motion_intensity_max: "Motion Intensity (max)"
```

#### Removing Fields

Simply remove the field from the `output_fields` list. Fields not in this list will be filtered out during conversion.

**Note**: The `timestamp` and `event_type` fields are always included and cannot be removed.

### Method 2: Schema Modification

To add support for new source fields or modify field mappings, edit `formats/uom_schema.yaml`:

#### Adding a New Field Mapping

1. **Add field category** (determines interpolation behavior):
```yaml
field_categories:
  your_new_field: continuous  # or 'occasional' or 'service'
```

2. **Add to converter's field_mappings**:
```yaml
converters:
  activity:
    field_mappings:
      source_column_name: your_new_field
```

3. **Update converter code** if needed (see `uom_activity_converter.py`)

#### Removing a Field

Remove the field from:
- `field_categories` section
- Converter's `field_mappings` section

### Method 3: Post-Processing Script

After running the main pipeline, you can use Polars to manipulate the output CSV:

```python
import polars as pl

# Load output
df = pl.read_csv("OUTPUT/uom_ml_ready.csv")

# Remove columns
df = df.drop(["column_to_remove"])

# Add computed columns
df = df.with_columns([
    (pl.col("glucose_value_mgdl").cast(pl.Float64) * 0.0555).alias("glucose_mmol")
])

# Save
df.write_csv("OUTPUT/uom_ml_ready_modified.csv")
```

## Available Fields

Based on `uom_schema.yaml`, the following standardized fields are available:

### Continuous Fields (interpolated during resampling)
- `glucose_value_mgdl`
- `heart_rate`
- `motion_intensity_max`
- `active_time_s`
- `stress_level`

### Occasional Fields (preserved at nearest timestamp)
- `fast_acting_insulin_u` / `long_acting_insulin_u`
- `carb_grams`, `protein_grams`, `fat_grams`, `fiber_grams`
- `step_count`, `distance_m`, `active_kcal`
- `motion_intensity_mean`
- Sleep stages: `sleep_duration_seconds`, `deep_sleep_seconds`, etc.
- `activity_type`, `meal_type`, `meal_tag`

### Service Fields (metadata, not interpolated)
- `timestamp`
- `event_type`
- `user_id` (for multi-user databases)

## Example: Including Activity Data

To include step count and heart rate in your output:

```yaml
# glucose_config.yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"
  - "step_count"      # From UoMActivity*.csv
  - "heart_rate"      # From UoMSleep*.csv
  - "active_kcal"     # From UoMActivity*.csv

field_to_display_name_map:
  timestamp: "Timestamp (YYYY-MM-DDThh:mm:ss)"
  event_type: "Event Type"
  glucose_value_mgdl: "Glucose Value (mg/dL)"
  step_count: "Step Count"
  heart_rate: "Heart Rate (bpm)"
  active_kcal: "Active Calories (kcal)"
```

Then run:
```bash
uv run glucose_cli.py DATA/uom --config glucose_config.yaml
```

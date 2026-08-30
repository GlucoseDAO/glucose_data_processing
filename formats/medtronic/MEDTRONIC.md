# Medtronic Guardian Connect Dataset

## Database Structure

The Medtronic Guardian Connect database is a **mono-user** format containing continuous glucose monitoring (CGM) data from Medtronic Guardian Connect CGM systems paired with MiniMed insulin pumps.

### File Structure

- **Single-user datasets**: All CSV files in a folder belong to one user
- **File naming**: Export files typically include patient name and date range (e.g., `PatientName 06.09.2021 (1)01.05-25.07.2021.csv`)
- **Format**: Semicolon-delimited CSV with multiple sections (pump data and sensor data)
- **Encoding**: UTF-8 with possible BOM

### Data Structure

Medtronic Guardian Connect CSV files are semicolon-delimited (`;`) and contain metadata rows before the actual data headers. The file may contain multiple sections:

1. **Header metadata** (first 6 rows): Patient information, device serial numbers, date range
2. **Pump section**: Event data (boluses, meals, alarms, calibrations)
3. **Sensor section**: Continuous glucose monitoring data at 5-minute intervals

Key columns (mapped via `medtronic_schema.yaml`):

- `Date` + `Time` → `timestamp` (combined during conversion)
- `Sensor Glucose (mg/dL)` → `glucose_value_mgdl` (CGM readings)
- `BG Reading (mg/dL)` → `glucose_value_mgdl` (fingerstick calibrations, used if no sensor data)
- `Basal Rate (U/h)` → `basal_rate`
- `Bolus Volume Delivered (U)` → `fast_acting_insulin_u`
- `BWZ Carb Input (grams)` → `carb_grams`
- `Event Marker` → Parsed for meal/insulin events (e.g., "Meal: 60,00grams", "Insulin: 27,00")
- `Sensor Calibration BG (mg/dL)` → `sensor_calibration_bg`

### Timestamp Format

Medtronic uses separate Date and Time columns:
- **Date**: `YYYY/MM/DD` (e.g., `2021/07/25`)
- **Time**: `HH:MM:SS` (e.g., `09:35:28`)

These are combined and converted to ISO format: `%Y-%m-%dT%H:%M:%S`

### Special Handling

Medtronic data requires special processing:

1. **European Number Format**:
   - Uses comma as decimal separator (e.g., `27,00` instead of `27.00`)
   - Automatically converted during processing

2. **Multiple Data Sections**:
   - Files contain both pump events and sensor CGM data
   - Headers appear multiple times (once per section)
   - All sections are processed and merged

3. **Event Marker Parsing**:
   - Meal entries: "Meal: 60,00grams" → `carb_grams: 60.0`
   - Insulin entries: "Insulin: 27,00" → `fast_acting_insulin_u: 27.0`

4. **Calibration Events**:
   - BG readings are used for calibration
   - These can be removed in post-processing if needed

## Adding or Removing Features in Post-Processing

### Method 1: Configuration File (Recommended)

Modify `glucose_config.yaml`:

#### Adding Fields

```yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"
  - "fast_acting_insulin_u"
  - "long_acting_insulin_u"
  - "carb_grams"
  - "basal_rate"  # Add basal rate from pump data

field_to_display_name_map:
  basal_rate: "Basal Rate (U/h)"
```

#### Removing Fields

Remove fields from the `output_fields` list. Only fields in this list will appear in the final output.

### Method 2: Schema Modification

To modify field mappings, edit `formats/medtronic_schema.yaml`:

```yaml
converters:
  guardian:
    field_mappings:
      Date: date
      Time: time
      Sensor Glucose (mg/dL): glucose_value_mgdl
      Your New Column: your_new_field  # Add new mapping
```

Then add the field category:
```yaml
field_categories:
  your_new_field: occasional  # or 'continuous' or 'service'
```

### Method 3: Post-Processing Script

After running the pipeline, manipulate the output:

```python
import polars as pl

df = pl.read_csv("OUTPUT/medtronic_ml_ready.csv")

# Remove columns
df = df.drop(["column_to_remove"])

# Add computed fields
df = df.with_columns([
    (pl.col("Glucose Value (mg/dL)").cast(pl.Float64) / 18.0).alias("Glucose (mmol/L)")
])

# Filter rows
df = df.filter(pl.col("Glucose Value (mg/dL)").cast(pl.Float64) < 300)

df.write_csv("OUTPUT/medtronic_ml_ready_modified.csv")
```

## Available Fields

Based on `medtronic_schema.yaml`:

### Continuous Fields
- `glucose_value_mgdl` (from Sensor Glucose or BG Reading)
- `basal_rate`

### Occasional Fields
- `fast_acting_insulin_u` (bolus insulin)
- `carb_grams` (from BWZ Carb Input or Event Marker)
- `sensor_calibration_bg` (calibration readings)

### Service Fields
- `timestamp`
- `event_type` (set to "Medtronic")

## Example: Glucose-Only Output

To output only glucose data:

```yaml
# glucose_config.yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"

# Or use CLI flag:
# uv run glucose_cli.py DATA/medtronic --glucose-only
```

## Example: Including Insulin and Carb Data

```yaml
# glucose_config.yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"
  - "fast_acting_insulin_u"
  - "carb_grams"
  - "basal_rate"

field_to_display_name_map:
  timestamp: "Timestamp (YYYY-MM-DDThh:mm:ss)"
  event_type: "Event Type"
  glucose_value_mgdl: "Glucose Value (mg/dL)"
  fast_acting_insulin_u: "Fast-Acting Insulin Value (u)"
  carb_grams: "Carb Value (grams)"
  basal_rate: "Basal Rate (U/h)"
```

## Notes

- Medtronic files use semicolon (`;`) as the column delimiter
- Numbers use European format (comma as decimal separator)
- Files may contain multiple header lines - the converter automatically detects the correct data headers
- Both CGM sensor data and pump event data are extracted and combined
- The Event Marker column is parsed to extract meal and insulin information that may not be in dedicated columns

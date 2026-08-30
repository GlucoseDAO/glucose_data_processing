# FreeStyle Libre 3 Dataset

## Database Structure

The FreeStyle Libre 3 database is a **mono-user** format containing continuous glucose monitoring (CGM) data from FreeStyle Libre 3 devices.

### File Structure

- **Single-user datasets**: All CSV files in a folder belong to one user
- **File naming**: No specific pattern required; all `.csv` files in the folder are processed
- **Format**: FreeStyle Libre 3 export format with specific column headers

### Data Structure

FreeStyle Libre 3 CSV files contain the following columns (mapped via `freestyle_libre3_schema.yaml`):

- `Device Timestamp` → `timestamp`
- `Record Type` → `record_type` (metadata field)
- `Historic Glucose mg/dL` → `glucose_value_mgdl` (continuous readings)
- `Scan Glucose mg/dL` → `glucose_value_mgdl` (manual scan readings)
- `Rapid-Acting Insulin (units)` → `rapid_acting_insulin_u` (mapped to `fast_acting_insulin_u`)
- `Meal Insulin (units)` → `meal_insulin_u` (mapped to `fast_acting_insulin_u`)
- `Correction Insulin (units)` → `correction_insulin_u` (mapped to `fast_acting_insulin_u`)
- `Long-Acting Insulin (units)` → `long_acting_insulin_u`
- `Carbohydrates (grams)` → `carb_grams`

### Timestamp Format

FreeStyle Libre 3 uses: `%d-%m-%Y %H:%M` or `%d-%m-%Y %H:%M:%S` (converted to ISO format `%Y-%m-%dT%H:%M:%S`)

### Special Handling

Unlike Dexcom, Libre 3 data:
- **No High/Low replacement needed**: Libre 3 provides numeric glucose values directly
- **No calibration removal needed**: Data is already in numeric format
- **Multiple insulin types**: Rapid-acting, meal, and correction insulin are all mapped to `fast_acting_insulin_u`

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
  - "record_type"  # Add if you want to preserve record type metadata

field_to_display_name_map:
  record_type: "Record Type"
```

#### Removing Fields

Remove fields from the `output_fields` list. Only fields in this list will appear in the final output.

### Method 2: Schema Modification

To modify field mappings, edit `formats/freestyle_libre3_schema.yaml`:

```yaml
converters:
  libre3:
    field_mappings:
      Device Timestamp: timestamp
      Record Type: record_type
      Historic Glucose mg/dL: glucose_value_mgdl
      Scan Glucose mg/dL: glucose_value_mgdl
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

df = pl.read_csv("OUTPUT/libre3_ml_ready.csv")

# Remove columns
df = df.drop(["column_to_remove"])

# Add computed fields
df = df.with_columns([
    (pl.col("Glucose Value (mg/dL)").cast(pl.Float64) / 18.0).alias("Glucose (mmol/L)")
])

# Filter by record type if needed
df = df.filter(pl.col("Record Type") == "Historic")

df.write_csv("OUTPUT/libre3_ml_ready_modified.csv")
```

## Available Fields

Based on `freestyle_libre3_schema.yaml`:

### Continuous Fields
- `glucose_value_mgdl`

### Occasional Fields
- `fast_acting_insulin_u` (aggregated from rapid-acting, meal, and correction insulin)
- `long_acting_insulin_u`
- `carb_grams`

### Service Fields
- `timestamp`
- `event_type`
- `record_type` (if preserved)

## Example: Preserving Record Type

To distinguish between historic and scan readings:

```yaml
# glucose_config.yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"
  - "record_type"  # Preserve source record type

field_to_display_name_map:
  record_type: "Record Type"
```

## Example: Glucose-Only Output

To output only glucose data:

```yaml
# glucose_config.yaml
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"

# Or use CLI flag:
# uv run glucose_cli.py DATA/libre3 --glucose-only
```

## Notes

- Libre 3 data is typically cleaner than Dexcom (no High/Low values to replace)
- Multiple insulin columns are automatically aggregated into `fast_acting_insulin_u`
- Both historic and scan glucose readings are mapped to the same `glucose_value_mgdl` field

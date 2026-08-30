# Dexcom G6 Dataset


## Database Structure

The Dexcom database is a **mono-user** format containing continuous glucose monitoring (CGM) data from Dexcom G6 devices.

### File Structure

- **Single-user datasets**: All CSV files in a folder belong to one user
- **File naming**: No specific pattern required; all `.csv` files in the folder are processed
- **Format**: Standard Dexcom export format with specific column headers

### Data Structure

Dexcom G6 CSV files contain the following columns (mapped via `dexcom_schema.yaml`):

- `Timestamp (YYYY-MM-DDThh:mm:ss)` → `timestamp`
- `Event Type` → `event_type` (typically "EGV" for Estimated Glucose Value)
- `Glucose Value (mg/dL)` → `glucose_value_mgdl`
- `Insulin Value (u)` → `insulin_u` (mapped to `fast_acting_insulin_u` or `long_acting_insulin_u` based on context)
- `Event Subtype` → `event_subtype` (optional metadata)
- `Carb Value (grams)` → `carb_grams`

### Timestamp Format

Dexcom uses: `%Y-%m-%dT%H:%M:%S` or `%Y-%m-%d %H:%M:%S` (ISO-like format)

### Special Handling

Dexcom data requires special processing:

1. **High/Low Value Replacement**: 
   - "High" values (typically >400 mg/dL) are replaced with a configurable value (default: 401)
   - "Low" values (typically <40 mg/dL) are replaced with a configurable value (default: 39)

2. **Calibration Removal**: 
   - Calibration events can be removed to create interpolatable gaps
   - Data after calibration periods can be removed (configurable duration)

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
  - "event_subtype"  # Add if available in source data

field_to_display_name_map:
  event_subtype: "Event Subtype"
```

#### Removing Fields

Remove fields from the `output_fields` list. Only fields in this list will appear in the final output.

### Method 2: Schema Modification

To modify field mappings, edit `formats/dexcom_schema.yaml`:

```yaml
converters:
  g6:
    field_mappings:
      Timestamp (YYYY-MM-DDThh:mm:ss): timestamp
      Event Type: event_type
      Glucose Value (mg/dL): glucose_value_mgdl
      Your New Column: your_new_field  # Add new mapping
```

Then add the field category:
```yaml
field_categories:
  your_new_field: occasional  # or 'continuous' or 'service'
```

### Method 3: Database-Specific Configuration

Configure Dexcom-specific settings in `glucose_config.yaml`:

```yaml
dexcom:
  high_glucose_value: 401      # Value to replace "High"
  low_glucose_value: 39         # Value to replace "Low"
  remove_calibration: true       # Remove calibration events
```

### Method 4: Post-Processing Script

After running the pipeline, manipulate the output:

```python
import polars as pl

df = pl.read_csv("OUTPUT/dexcom_ml_ready.csv")

# Remove columns
df = df.drop(["column_to_remove"])

# Add computed fields
df = df.with_columns([
    (pl.col("Glucose Value (mg/dL)").cast(pl.Float64) / 18.0).alias("Glucose (mmol/L)")
])

# Filter rows
df = df.filter(pl.col("Glucose Value (mg/dL)").cast(pl.Float64) < 300)

df.write_csv("OUTPUT/dexcom_ml_ready_modified.csv")
```

## Available Fields

Based on `dexcom_schema.yaml`:

### Continuous Fields
- `glucose_value_mgdl`

### Occasional Fields
- `fast_acting_insulin_u`
- `long_acting_insulin_u`
- `carb_grams`

### Service Fields
- `timestamp`
- `event_type`
- `event_subtype` (if present in source)

## Example: Custom High/Low Values

```yaml
# glucose_config.yaml
dexcom:
  high_glucose_value: 450  # Custom high threshold
  low_glucose_value: 50    # Custom low threshold
  remove_calibration: true
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
# uv run glucose_cli.py DATA/dexcom --glucose-only
```

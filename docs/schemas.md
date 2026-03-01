# Database Schema Configuration

The preprocessing engine uses YAML schema files located in the `formats/` directory to define how raw database fields map to the standardized internal format.

## Schema Structure

Each schema file defines the following top-level attributes:

- `database`: Internal identifier for the database type (e.g., `uom`, `dexcom`).
- `timestamp_formats`: A list of `strptime` compatible strings used to parse dates in the raw data.
- `timestamp_output_format`: The standard ISO-like format used for all output files.
- `remove_after_calibration`: Boolean flag indicating if data following a calibration event should be purged.
- `field_categories`: Mapping of standardized field names to their processing behavior.
- `converters`: Definitions for individual data modalities (e.g., glucose, insulin, activity).

## Field Categories

Standardized fields are categorized to determine how they are handled during interpolation and resampling:

- `service`: Metadata fields used for processing logic (e.g., `timestamp`, `user_id`). Not subject to interpolation.
- `continuous`: Numeric values that can be safely interpolated and averaged (e.g., `glucose_value_mgdl`, `heart_rate`).
- `occasional`: Event-based data (e.g., `carb_grams`, `insulin_dose`). These are preserved during resampling by shifting them to the nearest valid time bucket.
- `remove_after_calibration`: Fields that should be cleared during calibration periods.

## Converters

Converters define the mapping from raw file columns to standardized fields:

```yaml
converters:
  glucose:
    timestamp_field: "raw_date_column"
    event_type: "EGV" # Standard event label
    field_mappings:
      raw_value_column: glucose_value_mgdl
```

### Advanced Mapping
For nested JSON structures (like in AI-READY), converters support path-based extraction:

```yaml
converters:
  cgm:
    format: json
    records_path: body.cgm
    timestamp_path: effective_time_frame.time_interval.start_date_time
    field_paths:
      blood_glucose.value: glucose_value_mgdl
```

## Supported Schemas

- `uom_schema.yaml`: University of Manchester T1D database.
- `dexcom_schema.yaml`: Dexcom G6 system.
- `freestyle_libre3_schema.yaml`: Abbott FreeStyle Libre 3.
- `ai_ready_schema.yaml`: AI-READI (BIDS-like) zip dataset.
- `hupa_schema.yaml`: HUPA dataset (CGM with heart rate, steps, meals, insulin).
- `uc_ht_schema.yaml`: UC_HT dataset (Type 1 + healthy controls with CGM, HR, steps).
- `loop_schema.yaml`: Loop automated insulin delivery system dataset.
- `medtronic_schema.yaml`: Medtronic pump/CGM export format.
- `minidose1_schema.yaml`: MiniDose1 clinical trial dataset.


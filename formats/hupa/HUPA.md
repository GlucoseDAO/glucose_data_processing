# HUPA Dataset

## Source and Reference

- **Data Repository**: [Mendeley Data - HUPA-UCM diabetes dataset](https://data.mendeley.com/datasets/3hbcscwz44/1)
- **Scientific Publication**: [HUPA-UCM diabetes dataset, Data in Brief, 2024](https://www.sciencedirect.com/science/article/pii/S2352340924005262)
- **DOI**: 10.17632/3hbcscwz44.1

## Database Structure

The HUPA dataset is a **multi-user** format containing multi-modal glucose monitoring and activity data.

### File Structure

- **Multi-user dataset**: Each CSV file represents a single participant.
- **File naming**: Files are named `HUPAxxxxP.csv`, where `xxxx` is the participant ID.
- **Format**: CSV with semicolon (`;`) separator.

### Data Structure

HUPA CSV files contain the following columns (mapped via `hupa_schema.yaml`):

- `time` → `timestamp`
- `glucose` → `glucose_value_mgdl`
- `calories` → `active_kcal`
- `heart_rate` → `heart_rate`
- `steps` → `step_count`
- `basal_rate` → `basal_rate` (Continuous basal insulin rate)
- `bolus_volume_delivered` → `fast_acting_insulin_u` (Bolus insulin)
- `carb_input` → `carb_grams`

## Field Categorization

Based on `hupa_schema.yaml`, fields are categorized as:

### Continuous Fields
- `glucose_value_mgdl`
- `heart_rate`
- `basal_rate`

These fields are subject to linear interpolation during gap filling and fixed-frequency resampling.

### Occasional Fields
- `active_kcal`
- `step_count`
- `fast_acting_insulin_u`
- `carb_grams`

These fields are event-based and are preserved during resampling by shifting them to the nearest valid time bucket.

### Service Fields
- `timestamp`
- `event_type`

## Configuration

### HUPA-Specific Settings

HUPA-specific settings can be configured in `glucose_config.yaml`:

```yaml
hupa:
  # HUPA specific settings
```

### Adding HUPA to Output

To include HUPA-specific fields like `basal_rate` or `heart_rate` in the output, update `output_fields` in your configuration:

```yaml
output_fields:
  - "timestamp"
  - "glucose_value_mgdl"
  - "heart_rate"
  - "basal_rate"
  - "active_kcal"
  - "step_count"
  - "fast_acting_insulin_u"
  - "carb_grams"
  - "interpolated"
```

## Implementation Details

- **Delimiter**: Semicolon (`;`). The converter handles this by splitting the row if the default CSV reader fails to do so.
- **Timestamp**: ISO 8601 format (`%Y-%m-%dT%H:%M:%S`).
- **User ID**: Extracted from the filename by stripping "HUPA" and leading zeros, and lowercasing the trailing letter (e.g., `HUPA0001P` becomes `1p`).
- **User Separation**: Data is strictly processed per user to prevent mixing records between different participants.

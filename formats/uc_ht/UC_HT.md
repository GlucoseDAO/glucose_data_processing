# UC_HT Dataset

## Source and Reference

- **Data Repository**: [Repo and dataset](https://github.com/fisiologiacuantitativauc/UC_HT_T1DM)

## Database Structure

The UC_HT dataset is a multi-user format where each user's data is stored in a separate directory named after the user (e.g., `HT_01`, `T1DM_02`). Each directory contains several Excel (.xlsx) files, one for each data modality.

### Directory Structure

```
UC_HT/
├── HT_01/
│   ├── Carbohidrates.xlsx
│   ├── Glucose.xlsx
│   ├── Heart Rate.xlsx
│   ├── IGAR.xlsx
│   └── Steps.xlsx
├── T1DM_02/
│   ├── Carbohidrates.xlsx
│   ├── Glucose.xlsx
│   ├── Heart Rate.xlsx
│   ├── IGAR.xlsx
│   ├── Insulin.xlsx
│   └── Steps.xlsx
└── ...
```

### Supported Modalities and Fields

The following Excel files are processed:

1.  **Glucose.xlsx**: Interstitial glucose (mg/dL)
    -   `Value (mg/dl)` -> `glucose_value_mgdl`
2.  **Heart Rate.xlsx**: Heart rate (bpm)
    -   `Value (bpm)` -> `heart_rate`
3.  **Steps.xlsx**: Step count
    -   `Value (-)` -> `step_count`
4.  **Carbohidrates.xlsx**: Carbohydrate intake (g)
    -   `Value (g)` -> `carb_grams`
5.  **Insulin.xlsx**: Basal and bolus insulin (Units)
    -   `Basal Rate (U/5min)` -> `basal_rate_u_h`
    -   `Bolus (U)` -> `bolus_u`
6.  **IGAR.xlsx**: Estimated rate of glucose absorption (g/min)
    -   `Value (g)` -> `igar_g_min`

### Timestamp Format

All Excel files use a timestamp in the first column (unnamed in Excel, usually parsed as `__UNNAMED__0`) with the format `%Y-%m-%d %H:%M:%S`.

## Configuration

To use the UC_HT dataset, ensure the `uc_ht` database type is correctly configured in your `glucose_config.yaml`.

### Example Configuration

```yaml
database_configs:
  uc_ht:
    # Optional: start processing from a specific user
    # start_with_user_id: "T1DM_02"
```

## Implementation Details

The `UCHTDatabaseConverter` class handles the consolidation of these Excel files. It iterates through each user directory, reads the available Excel files using `polars.read_excel`, normalizes the fields according to `uc_ht_schema.yaml`, and merges all modalities for each user based on the timestamp.

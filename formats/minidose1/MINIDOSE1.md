# MiniDose1 Dataset

## Source and Reference

- **Data Repository**: [T1D Exchange MiniDose Glucagon Study](https://t1dexchange.org/)
- **Description**: This dataset contains data from a clinical trial evaluating mini-dose glucagon for the prevention of exercise-induced hypoglycemia in adults with type 1 diabetes.

## Database Structure

The MiniDose1 database is a **multi-user** T1D dataset. Data is stored in pipe-separated text files (`.txt`) within the `Data Tables` subdirectory.

### File Structure

The dataset contains several tables, including:
- `IDataCGM.txt`: Continuous glucose monitoring data.
- `IDataBGM.txt`: Blood glucose meter readings.
- `IDataPump.txt`: Insulin pump data (basal, bolus, settings).
- `IPtRoster.txt`: Participant roster and basic demographics.
- `IVisitInfo.txt`: Visit dates and enrollment information.

### Timestamp Handling

Timestamps in MiniDose1 are relative to an enrollment date:
- `DeviceDtDaysFromEnroll`: Number of days since enrollment.
- `DeviceTm`: Time of day (HH:MM:SS).

For processing, a reference enrollment date of `2020-01-01` is assumed for all participants to create absolute timestamps.

### Data Mapping

- **Glucose (CGM)**: From `IDataCGM.txt`, using `Glucose` column.
- **Insulin (Pump)**: From `IDataPump.txt`.
  - `BolusVolDeliv` is mapped to fast-acting insulin.
  - `BasalRate` is mapped to basal insulin.
- **Carbs**: From `IDataPump.txt`, using `WizardCarbs` column.

## Configuration

To process this dataset, use the `minidose1` database type in your configuration.

### Example:

```yaml
# glucose_config.yaml
database_type: "minidose1"
output_fields:
  - "timestamp"
  - "event_type"
  - "glucose_value_mgdl"
  - "fast_acting_insulin_u"
  - "long_acting_insulin_u"
  - "carb_grams"
```

Then run:
```bash
uv run glucose_cli.py DATA/minidose1 --config glucose_config.yaml
```

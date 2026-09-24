# JAEB comma-separated device tables

## Source

Several JAEB Center for Health Research public datasets ship device data as comma-separated
tables. The converter was built against the Shah healthy non-diabetic cohort (CGMND,
`NonDiabDeviceCGM.csv`, `NonDiabDeviceBGM.csv`); `test_data/jaeb_small` holds three of its
participants.

## Layout

```
PtID,DeviceDtDaysFromEnroll,DeviceTm,RecordType,Value[,DeviceStorUnits]
```

All participants share one file. Any file with these five columns is recognised, whatever
its name. Pipe-separated JAEB tables with a `Glucose` column (MiniDose1) go through the
`minidose1` converter instead.

## Mapping

- `PtID` becomes `user_id`; one output user per participant.
- `DeviceDtDaysFromEnroll` + `DeviceTm` become an absolute timestamp relative to the same
  reference enrollment date MiniDose1 uses (2020-01-01). Dates are synthetic; only offsets
  within a participant are meaningful.
- `RecordType` `CGM` becomes `EGV` with `Value` as glucose (mg/dL). `Calibration` and `bgm`
  rows are counted in the log and dropped, because the pipeline would otherwise resample
  meter readings into the sensor trace (docs/DECISIONS.md D2).
- Shah's CGM values are already clipped by the sensor to 39–401 mg/dL; no High/Low strings
  occur.

## Usage

```
uv run glucose-process <folder containing NonDiabDeviceCGM.csv>
```

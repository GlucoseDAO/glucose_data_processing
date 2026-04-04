# Loop Dataset

## Database Structure

The Loop database is a **multi-user** format containing data from the Loop automated insulin delivery system. All data for all users is consolidated into a set of pipe-separated text files.

### File Structure

- **Multi-user datasets**: All users are present in the same set of files.
- **File naming**: Files are named by data type (e.g., `LOOPDeviceCGM1.txt`, `LOOPDeviceBolus.txt`, `LOOPDeviceBasal1.txt`, `LOOPDeviceFood.txt`).
- **Format**: Pipe-delimited (`|`) text files.
- **Encoding**: UTF-8.

### Data Structure

Loop text files use `PtID` to identify participants. The primary files processed are:

1. **CGM Data** (`LOOPDeviceCGM*.txt`):
   - `PtID`: Participant identifier
   - `UTCDtTm`: UTC timestamp
   - `CGMVal`: Glucose value in mg/dL
   - `RecordType`: Type of record (mapped to EGV)

2. **Bolus Data** (`LOOPDeviceBolus.txt`):
   - `PtID`: Participant identifier
   - `UTCDtTm`: UTC timestamp
   - `Normal`: Insulin dose in units
   - `BolusType`: Type of bolus

3. **Basal Data** (`LOOPDeviceBasal*.txt`):
   - `PtID`: Participant identifier
   - `UTCDtTm`: UTC timestamp
   - `Rate`: Basal rate in U/h

4. **Food Data** (`LOOPDeviceFood.txt`):
   - `PtID`: Participant identifier
   - `UTCDtTm`: UTC timestamp
   - `CarbsNet`: Carbohydrates in grams

### Timestamp Format

Loop uses UTC timestamps in the following format:
- `UTCDtTm`: `YYYY-MM-DD HH:MM:SS` (e.g., `2018-04-20 06:01:58`)

These are converted to standard ISO format: `%Y-%m-%dT%H:%M:%S`.

### Special Handling

1. **Multi-user Consolidation**:
   - The converter extracts `PtID` from each row and assigns it to the `user_id` field.
   - Data from multiple files (CGM, Basal, Bolus, Food) are merged and sorted by `user_id` and `timestamp`.

2. **Pipe Delimiter**:
   - The files use `|` as the delimiter, which is handled by the `LoopConverter`.

3. **Glucose bounds (same numeric limits as Dexcom config)**:
   - CGM and BGM values are **numeric** in Loop exports; they are clipped to `[low_glucose_value, high_glucose_value]` (defaults `39`–`401` mg/dL), matching Dexcom’s substituted min/max mg/dL.
   - Configure under `dexcom:` in the YAML (see `glucose_config_loop.yaml`).
   - Applied in DuckDB when building cache and again in Polars per user (idempotent).

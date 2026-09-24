# ShanghaiT1DM / ShanghaiT2DM

## Source

Zhao et al., "Chinese diabetes datasets for data-driven machine learning", *Scientific Data*
(2023); Figshare article 21600933 (`diabetes_datasets.zip`, row 18 of `docs/datasets.csv`).
16 T1DM and 109 T2DM recording periods from 112 patients. `test_data/shanghai_small` holds
patients 1001, 1002 and 2045 plus the T1DM summary workbook, unchanged.

## Layout

One Excel workbook per recording period, `<patient>_<period>_<YYYYMMDD>.xls` or `.xlsx`, for
example `1002_0_20210504.xls`, `1002_1_20210521.xls`. Pass either cohort folder or the
unpacked root; `Shanghai_T*DM_Summary.xlsx` does not match the name pattern and is ignored.

Columns: `Date`, `CGM (mg / dl)` (labelled `CGM ` in two T2DM workbooks, same mg/dL range),
`CBG (mg / dl)`, `Blood Ketone`, dietary intake (English and Chinese), free-text
`Insulin dose - s.c.`, `Non-insulin hypoglycemic agents`, CSII bolus/basal, `Insulin dose - i.v.`.

## Mapping

- The patient number becomes `user_id`; all periods of one patient form one user.
- `Date` → timestamp, CGM column → `glucose_value_mgdl`, event type `EGV`. Values are kept
  as recorded (39.6–475.2 mg/dL across the release); they are not clipped to the Dexcom
  bounds.
- Only CGM is read. `CBG` is a fingerstick (docs/DECISIONS.md D2); the insulin columns are
  partly free text and are not mapped yet.

## Usage

The sensor reports every 15 minutes. Process at that interval, otherwise two of every three
output rows are interpolated:

```
uv run glucose-process <Shanghai root> --interval 15
```

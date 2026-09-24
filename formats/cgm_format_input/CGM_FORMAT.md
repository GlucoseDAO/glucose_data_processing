# cgm_format input backend

Reads inputs through the [cgm-format](https://github.com/GlucoseDAO/cgm_format) library
(`FormatParser`) and maps its unified frames onto this pipeline's fields. The detector uses
it only when no native converter recognises the input (docs/DECISIONS.md D3).

## Inputs

- **Corpora** that `FormatParser.detect_path_format` recognises: CGMacros, BIG IDEAs, D1NAMO.
  One user per corpus subject. In practice only CGMacros reaches this backend, since BIG IDEAs
  and D1NAMO have native converters.
- **Folders of exports** cgm_format reads one file at a time: Dexcom, Libre, Medtronic,
  Nightscout, and the EU (mmol/L) Dexcom/Libre variants. Subjects are first-level subfolders,
  as for native folder formats. Records repeated across overlapping exports are dropped.

## Configuration

```yaml
cgm_format:
  track: libre   # CGMacros only: libre, dexcom or mean. Required for CGMacros.
```

`dexcom.high_glucose_value` / `low_glucose_value` apply to Dexcom `High`/`Low` readings here too.

## Mapping

| unified | this pipeline |
|---|---|
| `datetime` | `timestamp` (cast to `Datetime("us")`) |
| `glucose` | `glucose_value_mgdl` |
| `carbs` | `carb_grams` |
| `insulin_fast` / `insulin_slow` | `fast_acting_insulin_u` / `long_acting_insulin_u` |
| `heart_rate`, `steps` (extended schema) | `heart_rate`, `step_count` |
| `event_type` `EGV_READ`, `CALIBRAT`, `CARBS_IN`, `INS_*`, `XRCS_*` | `EGV`, `Calibration`, `Carbs`, `Insulin`, `Exercise`; anything else `Other` |

`CALIBRAT` rows (fingersticks) are dropped before the per-timestamp merge. Rows sharing a
timestamp are merged with doses and carbs summed (`merge_same_timestamp_rows`).

## Checking parity

```
uv run python scripts/cgm_format_parity.py test_data/dexcom_small test_data/d1namo_small/diabetes DATA/bigideas
```

rewrites `docs/CGM_FORMAT_PARITY.md`; `tests/test_cgm_format_parity.py` asserts the same numbers.

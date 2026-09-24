# Changelog

Newest first.

## Unreleased — cgm_format input backend (2026-09-24)

### Added
- `cgm_format` database type: inputs no native converter recognises are read through
  `cgm-format` 0.12.2 (CGMacros, Nightscout, EU Dexcom/Libre exports, and its corpora).
  CGMacros requires `cgm_format.track`.
- `docs/CGM_FORMAT_PARITY.md` ledger and `scripts/cgm_format_parity.py`; parity with native
  converters is asserted in `tests/test_cgm_format_parity.py` (docs/DECISIONS.md D3).

### Changed
- polars 1.34 -> 1.44 (required by cgm-format). Real-data outputs byte-identical; the
  `explode` deprecation is pinned.

## Unreleased — 2026-09-24

Faults reported by the MetaboNet data seat against commit 8f79f45.

### Fixed
- Folder-based databases no longer merge subjects. Each first-level subfolder is one user
  named after the folder; a flat input folder's user is named after the folder instead of
  `Subject 000`; MiniDose1 keeps `PtID` per participant. BIG IDEAs as one input now yields
  16 users (92% preservation, was one merged trace at 58%). See `docs/DECISIONS.md` D1.
- A run in which no file matched the detected format now fails with
  "N of M data files matched a converter" instead of
  `unsupported format string passed to NoneType.__format__`.
- The database detector no longer classifies files as Dexcom from `cgm`/`g6` in the file
  name (CGMacros, `NonDiabDeviceCGM.csv`); those folders now report "Could not detect
  database type" unless a converter recognises their header.
- AI-READI Dexcom `High`/`Low` readings map to the configured bounds (401/39 by default)
  instead of being dropped.
- Dexcom CSV `High`/`Low` readings map to 401/39, and calibration events are removed
  under `dexcom.remove_calibration`. Both steps checked display column names on frames
  that carry standard names, so neither ran: High/Low were nulled and calibration
  fingersticks stayed in the trace. `test_data/dexcom_small`: 31 readings at 401 and 4 at
  39 now survive, 6 calibration rows are gone.
- The streaming writer kept losing the CSV header when the first user's output was empty.
- Folder-based converters lost insulin and carb values when another row shared the
  timestamp: row converters write `""` for absent fields, and the same-timestamp merge took
  `""` as the first value. `test_data/dexcom_small` output: fast insulin 958.5 -> 1058.5 U,
  long-acting 905 -> 918 U; glucose unchanged.
- Doses and carbs recorded in separate rows at the same timestamp are now summed instead
  of keeping the first row's value (exact duplicate rows from overlapping exports are
  dropped first). D1NAMO subject 004 enters a fast and a slow dose as two rows at
  2014-10-03 20:00; long-acting insulin in the D1NAMO T1D output goes 173 -> 197 U.
  Absent insulin/carb cells are now written as empty (null) rather than `""`.

### Added
- `jaeb` format for comma-separated JAEB CGM device tables (Shah healthy non-diabetic).
- `d1namo` format (diabetes subset CGM + insulin).
- `shanghai` format for ShanghaiT1DM/T2DM workbooks (CGM only).
- `max_workers` config key and `--workers` CLI option for the per-user thread pool.

### Data
- `docs/datasets.csv`: Direct to Patient CGM (#30) and GMI (#32) marked `manual`, CGM=false;
  their public ZIPs hold a manuscript .docx and a per-participant summary table. BIG IDEAs
  (#17) relabelled normoglycemic/prediabetic.

# Changelog

Newest first.

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

### Added
- `jaeb` format for comma-separated JAEB CGM device tables (Shah healthy non-diabetic).
- `d1namo` format (diabetes subset CGM + insulin).
- `shanghai` format for ShanghaiT1DM/T2DM workbooks (CGM only).
- `max_workers` config key and `--workers` CLI option for the per-user thread pool.

### Data
- `docs/datasets.csv`: Direct to Patient CGM (#30) and GMI (#32) marked `manual`, CGM=false;
  their public ZIPs hold a manuscript .docx and a per-participant summary table. BIG IDEAs
  (#17) relabelled normoglycemic/prediabetic.

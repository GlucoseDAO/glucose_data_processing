# Design decisions

Calls made without a human in the loop, with the option rejected and why. Revisit any of
them by editing the entry, not by deleting it.

## D1 — Folder-based databases: one user per subject folder (2026-09-24)

**Context.** `MonoUserDatabaseConverter` (Dexcom, Libre3, Medtronic, MiniDose1) globbed every
file under the input root, stamped every row `user_id = "Subject 000"`, de-duplicated by
timestamp and resampled the result as one trace. A BIG IDEAs root (16 subject folders) came
out as a single person with 58% "preservation" and exit 0. Row converters that know the
subject (`PtID` in MiniDose1) had their `user_id` overwritten too.

**Chosen.** Users are resolved in this order: a `user_id` the row converter set; else the
first-level subfolder the file lives in; else, for files directly in the root, the root
folder's name. One frame is yielded per user.

**Rejected.**
- *Keep `"Subject 000"` for a flat folder.* Byte-identical checkpoints, but separately
  processed subjects stay indistinguishable except via `dataset_name`, which the reporter
  flagged as the second half of the same fault. The hardcoded value sat next to a commented
  `# csv_path.name`, so the folder name looks like the original intent.
- *Split only when two or more subfolders hold data.* Keeps one-subfolder inputs merged
  under the root name, but makes the user id depend on how many siblings a folder has.

**Cost.** A single export split across nested subfolders (`export/2019/`, `export/2020/`)
now becomes two users. Pass the subject folder itself in that case. Folder names become
`user_id` values, so do not name export folders after patients.

## D2 — CGM-only rows from JAEB, D1NAMO and Shanghai exports (2026-09-24)

The pipeline treats every row with a glucose value as part of the CGM trace; it does not
look at `event_type`. JAEB `Calibration`/`bgm` rows, D1NAMO `manual` fingersticks and Shanghai `CBG` would
therefore be resampled into the sensor trace. The converters emit CGM rows only and log
how many rows of each other kind they skipped. Rejected: emitting them as `BGM` events,
which is what MiniDose1 does, because nothing downstream separates them again.

## D3 — cgm_format as an additional input backend, native converters kept (2026-09-24)

**Context.** `cgm-format` (GlucoseDAO, PyPI) parses Dexcom, Libre, Medtronic and Nightscout
exports plus the BIG IDEAs, CGMacros and D1NAMO corpora into its unified schema. Several of
those overlap with native converters here.

**Chosen.** A `cgm_format` database type (`formats/cgm_format_input/`) that the detector
tries only when no native converter recognises the input. Native output for every input it
already handled is unchanged. Where both read an input, `tests/test_cgm_format_parity.py`
compares them per user and `docs/CGM_FORMAT_PARITY.md` is the ledger. On the committed
fixtures and full BIG IDEAs/D1NAMO releases they agree on users, EGV timestamps and insulin,
with three differences asserted exactly:

- D1NAMO glucose: cgm_format converts mmol/L with 18.0182, native with 18.0 (ratio 1.00101).
- Dexcom: where two readings share one wall-clock time (a DST fold), each keeps a different one.
- BIG IDEAs: only cgm_format reads the food log, so carbs exist only on its side.

Building the comparison exposed three native bugs, fixed on the fault branch: Dexcom
High/Low and calibration removal never ran, `""` overwrote doses in the same-timestamp
merge, and doses logged as separate rows in one minute were not summed.

**Rejected.**
- *Route the overlap through cgm_format now and delete native converters.* Moves existing
  checkpoints (D1NAMO glucose ×1.001, BIG IDEAs gains carbs) as a side effect of a plumbing
  change. The candidates are clear from the ledger (native D1NAMO duplicates cgm_format
  apart from the unit factor; BIG IDEAs via cgm_format is strictly richer), so retiring them
  is a separate, deliberate change.
- *Default CGMacros to one track.* Its two sensors are alternative measurements of one
  quantity; picking one silently is a fallback. `cgm_format.track` is required.

**Semantics the backend keeps from native.** Fingersticks (`CALIBRAT`) are dropped (D2).
Dexcom High/Low placeholders are remapped to `dexcom.high/low_glucose_value`, because the
public cgm_format API does not take bounds (filed upstream as S6). Exercise is not mapped.

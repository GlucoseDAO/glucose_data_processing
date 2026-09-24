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

## D2 — CGM-only rows from JAEB and D1NAMO exports (2026-09-24)

The pipeline treats every row with a glucose value as part of the CGM trace; it does not
look at `event_type`. JAEB `Calibration`/`bgm` rows and D1NAMO `manual` fingersticks would
therefore be resampled into the sensor trace. The converters emit CGM rows only and log
how many rows of each other kind they skipped. Rejected: emitting them as `BGM` events,
which is what MiniDose1 does, because nothing downstream separates them again.

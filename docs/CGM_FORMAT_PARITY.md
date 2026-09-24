# cgm_format parity ledger

Generated 2026-09-24T19:21:09+00:00 by `scripts/cgm_format_parity.py` with cgm-format 0.12.2 and glucose-dataset 0.1.0. Do not edit by hand; rerun the script.

Compares the per-user frames each backend yields before the processing pipeline. Expected differences are explained in docs/DECISIONS.md D3 and asserted in tests/test_cgm_format_parity.py.

### `test_data/dexcom_small` (native: `dexcom`)

Users: native 1, cgm_format 1, same set: yes.

| user | EGV native | EGV cgm_format | only native | only cgm_format | glucose mismatches | median ratio | fast_acting_insulin_u native / cgm_format | long_acting_insulin_u native / cgm_format | carb_grams native / cgm_format |
|---|---|---|---|---|---|---|---|---|---|
| dexcom_small | 11,208 | 11,208 | 0 | 0 | 1 | 1.000000 | 1,268.5 / 1,268.5 | 1,116.0 / 1,116.0 | 4,171.0 / 4,171.0 |

### `test_data/d1namo_small/diabetes` (native: `d1namo`)

Users: native 3, cgm_format 3, same set: yes.

| user | EGV native | EGV cgm_format | only native | only cgm_format | glucose mismatches | median ratio | fast_acting_insulin_u native / cgm_format | long_acting_insulin_u native / cgm_format | carb_grams native / cgm_format |
|---|---|---|---|---|---|---|---|---|---|
| 001 | 1,413 | 1,413 | 0 | 0 | 1413 | 1.001011 | 122.0 / 122.0 | 93.0 / 93.0 | — / — |
| 002 | 1,056 | 1,056 | 0 | 0 | 1056 | 1.001011 | 58.0 / 58.0 | 32.0 / 32.0 | — / — |
| 004 | 969 | 969 | 0 | 0 | 969 | 1.001011 | 56.0 / 56.0 | 96.0 / 96.0 | — / — |

### `DATA/bigideas` (native: `dexcom`)

Users: native 16, cgm_format 16, same set: yes.

| user | EGV native | EGV cgm_format | only native | only cgm_format | glucose mismatches | median ratio | fast_acting_insulin_u native / cgm_format | long_acting_insulin_u native / cgm_format | carb_grams native / cgm_format |
|---|---|---|---|---|---|---|---|---|---|
| 001 | 2,561 | 2,561 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,702.9 |
| 002 | 2,119 | 2,119 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 2,363.2 |
| 003 | 2,302 | 2,302 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,484.5 |
| 004 | 2,164 | 2,164 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,233.8 |
| 005 | 2,558 | 2,558 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,515.7 |
| 006 | 2,847 | 2,847 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 3,051.6 |
| 007 | 2,207 | 2,207 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,597.6 |
| 008 | 2,505 | 2,505 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,981.3 |
| 009 | 2,306 | 2,306 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,220.7 |
| 010 | 2,148 | 2,148 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,875.4 |
| 011 | 2,843 | 2,843 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 3,679.7 |
| 012 | 2,169 | 2,169 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,944.8 |
| 013 | 1,979 | 1,979 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 2,126.6 |
| 014 | 2,240 | 2,240 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 2,110.9 |
| 015 | 1,673 | 1,673 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,748.3 |
| 016 | 2,277 | 2,277 | 0 | 0 | 0 | 1.000000 | — / — | — / — | — / 1,768.7 |

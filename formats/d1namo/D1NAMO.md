# D1NAMO

## Source

D1NAMO (Dubosson et al., 2018, *Informatics in Medicine Unlocked*): a type 1 diabetes subset
(nine subjects, CGM plus insulin and food) and a healthy subset (twenty subjects, glucose
spot checks and food). `test_data/d1namo_small` holds diabetes subjects 001 and 002 and
healthy subject 004, unchanged.

## Layout

One folder per subject; pass the subset folder, whose first-level subfolders become users:

```
diabetes_subset_pictures-glucose-food-insulin/
  001/glucose.csv   date,time,glucose,type,comments   (glucose in mmol/L)
  001/insulin.csv   date,time,fast_insulin,slow_insulin,comment
  001/food.csv      not read (calories only, no carbohydrate grams)
```

`time` occurs as both `HH:MM:SS` and `HH:MM`.

## Mapping

- `glucose` × 18.0 (`MGDL_PER_MMOL`) → `glucose_value_mgdl`.
- `type == cgm` → `EGV`. Every other `type` (`manual` in the diabetes subset, meal-relative
  labels such as `BB`/`AL` in the healthy subset) is a fingerstick; those rows are counted in
  the log and dropped before de-duplication (docs/DECISIONS.md D2).
- `fast_insulin` → `fast_acting_insulin_u`, `slow_insulin` → `long_acting_insulin_u`.

The healthy subset has no CGM rows, so processing it refuses with "no rows left after
d1namo-specific filtering" instead of writing an empty file.

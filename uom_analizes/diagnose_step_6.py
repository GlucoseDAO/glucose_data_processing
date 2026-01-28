import polars as pl
from pathlib import Path

def diagnose_step_6():
    df = pl.read_csv("OUTPUT/uom_step_6.csv")
    print(f"Total records in Step 6: {len(df)}")
    
    # Check intervals
    if "Timestamp" in df.columns and "sequence_id" in df.columns:
        df = df.with_columns(pl.col("Timestamp").str.to_datetime())
        
        # Check if intervals are exactly 5 minutes within each sequence
        diffs = df.group_by("sequence_id").agg(
            pl.col("Timestamp").diff().dt.total_minutes().alias("diffs")
        )
        
        bad_intervals = []
        for row in diffs.to_dicts():
            d = [x for x in row['diffs'] if x is not None and x != 5.0]
            if d:
                bad_intervals.append((row['sequence_id'], d[:5]))
        
        if bad_intervals:
            print(f"Found {len(bad_intervals)} sequences with non-5-minute intervals!")
            print(f"Samples: {bad_intervals[:3]}")
        else:
            print("All intervals are exactly 5 minutes.")

    # Check for glucose completeness (should be 100% per stats)
    if "Glucose" in df.columns:
        null_count = df["Glucose"].null_count()
        print(f"Glucose null count: {null_count} ({null_count/len(df)*100:.2f}%)")

    # Check for occasional fields
    for col in ["Basal", "Bolus", "Carbs"]:
        if col in df.columns:
            non_null = df.filter(pl.col(col).is_not_null())
            print(f"Field {col}: {len(non_null)} non-null records")

diagnose_step_6()

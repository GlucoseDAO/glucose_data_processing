import polars as pl
from pathlib import Path
import sys
import io

# Set stdout to UTF-8 to avoid encoding issues on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

def analyze_ai_ready_dataset(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        return

    print(f"Analyzing Dataset: {path.name}")
    print("=" * 60)

    # Load dataset
    try:
        # 10M records might be a lot for read_csv depending on available RAM.
        df = pl.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    total_rows = len(df)
    if total_rows == 0:
        print("Dataset is empty.")
        return
        
    print(f"Total Records: {total_rows:,}")
    print(f"Total Columns: {len(df.columns)}")
    print("-" * 60)

    # Separate columns by type
    numeric_cols = []
    string_cols = []
    datetime_cols = []
    bool_cols = []

    for col in df.columns:
        dtype = df.schema[col]
        
        # Check if it's temporal
        is_temporal = dtype.is_temporal() or "timestamp" in col.lower() or "date" in col.lower()
        
        if is_temporal:
            datetime_cols.append(col)
        elif dtype.is_numeric():
            numeric_cols.append(col)
        elif dtype == pl.Boolean:
            bool_cols.append(col)
        elif dtype == pl.Utf8 or dtype == pl.String:
            string_cols.append(col)
        else:
            string_cols.append(col)

    # 1. Numeric Statistics
    if numeric_cols:
        print("\nNUMERIC FIELD STATISTICS")
        # Use ASCII_FULL to avoid box drawing characters that cause encoding issues
        with pl.Config(tbl_formatting="ASCII_FULL"):
            stats = df.select(numeric_cols).describe()
            print(stats)
        
        # Add data completeness info
        print("\nData Completeness (Numeric):")
        null_info = []
        for c in numeric_cols:
            null_count = df[c].null_count()
            null_pct = (null_count / total_rows) * 100
            null_info.append({
                "Field": c,
                "Nulls": null_count,
                "Completeness": f"{100 - null_pct:.1f}%"
            })
        with pl.Config(tbl_formatting="ASCII_FULL"):
            print(pl.DataFrame(null_info))

    # 2. Date/Timestamp Statistics
    if datetime_cols:
        print("\nTEMPORAL FIELD STATISTICS")
        for col in datetime_cols:
            temp_col = df[col]
            if temp_col.dtype == pl.Utf8 or temp_col.dtype == pl.String:
                try:
                    parsed = temp_col.str.to_datetime(strict=False)
                    if parsed.null_count() < total_rows:
                        temp_col = parsed
                except:
                    pass
            
            try:
                min_date = temp_col.min()
                max_date = temp_col.max()
                print(f"\nField: {col}")
                print(f"  Start: {min_date}")
                print(f"  End:   {max_date}")
                if min_date and max_date and hasattr(min_date, 'year'):
                    print(f"  Span:  {max_date - min_date}")
            except Exception as e:
                print(f"  Could not calculate date stats for {col}: {e}")

    # 3. String/Categorical Statistics (Distribution)
    if string_cols:
        print("\nSTRING/CATEGORICAL DISTRIBUTION (Top 5 Values)")
        for col in string_cols:
            counts = df[col].value_counts().sort("count", descending=True).head(5)
            print(f"\nField: {col}")
            for row in counts.iter_rows():
                val = str(row[0]) if row[0] is not None else "NULL"
                count = row[1]
                pct = (count / total_rows) * 100
                print(f"  - {val}: {count:,} ({pct:.1f}%)")

    # 4. Boolean Statistics
    if bool_cols:
        print("\nBOOLEAN FIELD DISTRIBUTION")
        for col in bool_cols:
            counts = df[col].value_counts().sort("count", descending=True)
            print(f"\nField: {col}")
            for row in counts.iter_rows():
                val = str(row[0])
                count = row[1]
                pct = (count / total_rows) * 100
                print(f"  - {val}: {count:,} ({pct:.1f}%)")

    # 5. Sequence Specific Analysis
    seq_col = next((c for c in df.columns if c.lower() == "sequence_id"), None)
    if seq_col:
        seq_stats = df.group_by(seq_col).len()
        print("\nSEQUENCE ANALYSIS")
        print(f"  Total Unique Sequences: {df[seq_col].n_unique():,}")
        print(f"  Avg Sequence Length:    {seq_stats['len'].mean():.1f}")
        print(f"  Max Sequence Length:    {seq_stats['len'].max():,}")
        print(f"  Min Sequence Length:    {seq_stats['len'].min():,}")

if __name__ == "__main__":
    file_to_analyze = "OUTPUT/ai_ready_processed_dataset.csv"
    if len(sys.argv) > 1:
        file_to_analyze = sys.argv[1]
    
    analyze_ai_ready_dataset(file_to_analyze)

#!/usr/bin/env python3
"""
Compare two checkpoint CSV files and provide detailed statistics.

This script compares:
- Field number, names, and order
- Sequence IDs and their row counts
- Row values (exact and approximate matches)
- Overall statistics
"""

from __future__ import annotations

import sys
import typer
import polars as pl
from loguru import logger
from pathlib import Path
from typing import Any, Optional
import re

app = typer.Typer()


def _scan_csv(path: Path) -> pl.LazyFrame:
    # Lazy scan prevents pulling the whole file into memory.
    # infer_schema_length is a tradeoff between dtype accuracy and upfront scan time.
    return pl.scan_csv(path, infer_schema_length=10_000, low_memory=True)


def _collect_scalar(lf: pl.LazyFrame, expr: pl.Expr, *, streaming: bool) -> Any:
    # Polars changed its streaming API: `streaming=` -> `engine=`.
    to_collect = lf.select(expr.alias("_x"))
    try:
        engine = "streaming" if streaming else "auto"
        df = to_collect.collect(engine=engine)  # type: ignore[call-arg]
    except TypeError:
        df = to_collect.collect(streaming=streaming)  # type: ignore[call-arg]
    return df["_x"][0]


def _collect_df(lf: pl.LazyFrame, *, streaming: bool) -> pl.DataFrame:
    """
    Collect a LazyFrame with best-effort streaming, and fall back if the streaming
    engine fails (Polars can panic/cancel tasks on some query shapes).
    """
    if not streaming:
        try:
            return lf.collect(engine="auto")  # type: ignore[call-arg]
        except TypeError:
            return lf.collect(streaming=False)  # type: ignore[call-arg]

    # First try streaming
    try:
        return lf.collect(engine="streaming")  # type: ignore[call-arg]
    except TypeError:
        # Older Polars
        try:
            return lf.collect(streaming=True)  # type: ignore[call-arg]
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: non-streaming collection
    try:
        return lf.collect(engine="auto")  # type: ignore[call-arg]
    except TypeError:
        return lf.collect(streaming=False)  # type: ignore[call-arg]


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    try:
        # Available in newer Polars.
        return bool(pl.datatypes.is_numeric(dtype))  # type: ignore[attr-defined]
    except Exception:
        return dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }


def _is_temporal_dtype(dtype: pl.DataType) -> bool:
    return dtype in {pl.Date, pl.Datetime, pl.Time, pl.Duration}


def _is_comparable_dtype(dtype: pl.DataType) -> bool:
    # Conservative: avoid nested / object-like types in equality comparisons.
    if dtype == pl.Object:
        return False
    try:
        if bool(dtype.is_nested()):  # type: ignore[attr-defined]
            return False
    except Exception:
        dt_str = str(dtype).lower()
        if "list" in dt_str or "struct" in dt_str or "array" in dt_str:
            return False
    return True


def compare_schemas(schema1: pl.Schema, schema2: pl.Schema, *, max_list: int) -> dict[str, Any]:
    """Compare schemas (names, order, and dtypes).

    Notes:
      - Column *order* is reported separately and does not imply a mismatch in "names match".
      - Type comparison is performed by column name (order-insensitive).
    """
    names1 = list(schema1.keys())
    names2 = list(schema2.keys())

    missing_in_2 = [c for c in names1 if c not in schema2]
    missing_in_1 = [c for c in names2 if c not in schema1]

    type_differences: dict[str, dict[str, str]] = {}
    common = set(names1) & set(names2)
    for col in common:
        if schema1[col] != schema2[col]:
            type_differences[col] = {"type_1": str(schema1[col]), "type_2": str(schema2[col])}

    # Order-insensitive comparisons for names/types.
    names_match = (set(names1) == set(names2))
    order_match = (names1 == names2)
    types_match = names_match and (len(type_differences) == 0)

    return {
        "field_count_match": len(schema1) == len(schema2),
        "field_count_1": len(schema1),
        "field_count_2": len(schema2),
        "field_names_match": names_match,
        "field_order_match": order_match,
        "field_types_match": types_match,
        "missing_in_2_count": len(missing_in_2),
        "missing_in_1_count": len(missing_in_1),
        "missing_in_2_sample": missing_in_2[:max_list],
        "missing_in_1_sample": missing_in_1[:max_list],
        "type_differences_count": len(type_differences),
        "type_differences_sample": dict(list(type_differences.items())[:max_list]),
    }


def compare_sequences(
    lf1: pl.LazyFrame,
    lf2: pl.LazyFrame,
    *,
    streaming: bool,
    sample_n: int,
) -> dict[str, Any]:
    """Compare sequence_id presence and per-sequence row counts (streaming-friendly)."""
    schema1 = lf1.collect_schema()
    schema2 = lf2.collect_schema()
    if "sequence_id" not in schema1 or "sequence_id" not in schema2:
        return {"error": "Missing required column 'sequence_id' in one or both files"}

    seq1 = lf1.group_by("sequence_id").agg(pl.len().alias("row_count_1"))
    seq2 = lf2.group_by("sequence_id").agg(pl.len().alias("row_count_2"))
    joined = seq1.join(seq2, on="sequence_id", how="full")

    total_sequences_1 = int(_collect_scalar(seq1, pl.len(), streaming=streaming))
    total_sequences_2 = int(_collect_scalar(seq2, pl.len(), streaming=streaming))
    common_sequences = int(
        _collect_scalar(
            joined.filter(pl.col("row_count_1").is_not_null() & pl.col("row_count_2").is_not_null()),
            pl.len(),
            streaming=streaming,
        )
    )
    only_in_1 = int(
        _collect_scalar(joined.filter(pl.col("row_count_2").is_null()), pl.len(), streaming=streaming)
    )
    only_in_2 = int(
        _collect_scalar(joined.filter(pl.col("row_count_1").is_null()), pl.len(), streaming=streaming)
    )
    row_count_matches = int(
        _collect_scalar(
            joined.filter(
                pl.col("row_count_1").is_not_null()
                & pl.col("row_count_2").is_not_null()
                & (pl.col("row_count_1") == pl.col("row_count_2"))
            ),
            pl.len(),
            streaming=streaming,
        )
    )
    row_count_differences = int(
        _collect_scalar(
            joined.filter(
                pl.col("row_count_1").is_not_null()
                & pl.col("row_count_2").is_not_null()
                & (pl.col("row_count_1") != pl.col("row_count_2"))
            ),
            pl.len(),
            streaming=streaming,
        )
    )

    diffs_sample_df = (
        joined.filter(
            pl.col("row_count_1").is_not_null()
            & pl.col("row_count_2").is_not_null()
            & (pl.col("row_count_1") != pl.col("row_count_2"))
        )
        .with_columns((pl.col("row_count_2") - pl.col("row_count_1")).alias("difference"))
        .select(["sequence_id", "row_count_1", "row_count_2", "difference"])
        .head(sample_n)
        .pipe(lambda x: _collect_df(x, streaming=streaming))
    )

    only1_ids = (
        joined.filter(pl.col("row_count_2").is_null())
        .select("sequence_id")
        .head(sample_n)
        .pipe(lambda x: _collect_df(x, streaming=streaming))["sequence_id"]
        .to_list()
    )
    only2_ids = (
        joined.filter(pl.col("row_count_1").is_null())
        .select("sequence_id")
        .head(sample_n)
        .pipe(lambda x: _collect_df(x, streaming=streaming))["sequence_id"]
        .to_list()
    )

    return {
        "total_sequences_1": total_sequences_1,
        "total_sequences_2": total_sequences_2,
        "common_sequences": common_sequences,
        "only_in_1": only_in_1,
        "only_in_2": only_in_2,
        "sequence_ids_match": (only_in_1 == 0 and only_in_2 == 0),
        "row_count_matches": row_count_matches,
        "row_count_differences": row_count_differences,
        "row_count_differences_details": diffs_sample_df.to_dicts(),
        "only_in_1_ids": only1_ids,
        "only_in_2_ids": only2_ids,
    }


def _choose_default_key_cols(schema1: pl.Schema, schema2: pl.Schema) -> list[str]:
    candidates = [
        "sequence_id",
        "Timestamp (YYYY-MM-DDThh:mm:ss)",
        "timestamp",
        "Timestamp",
        "time",
        "Time",
    ]
    return [c for c in candidates if c in schema1 and c in schema2]


def _parse_key_columns_spec(
    key_columns: str,
) -> tuple[list[str], dict[str, str]]:
    """
    Parse a key column specification.

    Supported formats (comma-separated):
      - "col": same column name in both files
      - "left=right": left column from file1, right column from file2 (file2 will be renamed to left)

    Returns:
      - join_cols: column names to join on (after applying rename_map to file2)
      - rename_map: mapping to apply to file2, {right_name: left_name}
    """
    join_cols: list[str] = []
    rename_map: dict[str, str] = {}
    for raw in key_columns.split(","):
        token = raw.strip()
        if not token:
            continue
        if "=" in token:
            left, right = (p.strip() for p in token.split("=", 1))
            if not left or not right:
                continue
            join_cols.append(left)
            rename_map[right] = left
        else:
            join_cols.append(token)
    # Deduplicate while preserving order
    seen: set[str] = set()
    join_cols = [c for c in join_cols if not (c in seen or seen.add(c))]
    return join_cols, rename_map


def _normalize_col_name(name: str) -> str:
    # Aggressive normalization for best-effort matching across naming styles:
    # "Event Type" <-> "event_type", "Glucose Value (mg/dL)" <-> "glucose_value_mgdl", etc.
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _infer_rename_map_for_file2(
    schema1: pl.Schema,
    schema2: pl.Schema,
    *,
    exclude_file1_cols: set[str],
    exclude_file2_cols: set[str],
) -> dict[str, str]:
    """
    Best-effort mapping of file2 -> file1 column names by normalized-name equality.
    Only creates one-to-one mappings; collisions are skipped.
    """
    norm_to_file1: dict[str, list[str]] = {}
    for c1 in schema1.keys():
        if c1 in exclude_file1_cols:
            continue
        norm_to_file1.setdefault(_normalize_col_name(c1), []).append(c1)

    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()
    for c2 in schema2.keys():
        if c2 in exclude_file2_cols:
            continue
        targets = norm_to_file1.get(_normalize_col_name(c2))
        if not targets or len(targets) != 1:
            continue
        target = targets[0]
        if target in used_targets:
            continue
        if c2 == target:
            continue
        rename_map[c2] = target
        used_targets.add(target)
    return rename_map


def _duplicate_key_group_count(
    lf: pl.LazyFrame, key_cols: list[str], *, streaming: bool
) -> int:
    return int(
        _collect_scalar(
            lf.group_by(key_cols)
            .agg(pl.len().alias("_n"))
            .filter(pl.col("_n") > 1),
            pl.len(),
            streaming=streaming,
        )
    )


def compare_values(
    lf1: pl.LazyFrame,
    lf2: pl.LazyFrame,
    *,
    key_cols: Optional[list[str]],
    tolerance: float,
    streaming: bool,
    max_columns: int,
    require_unique_keys: bool,
    compare_fields: bool,
) -> dict[str, Any]:
    """Compare values using joins on key columns without sorting or full outer joins."""
    schema1 = lf1.collect_schema()
    schema2 = lf2.collect_schema()

    if key_cols is None:
        key_cols = _choose_default_key_cols(schema1, schema2)
    else:
        key_cols = [c for c in key_cols if c in schema1 and c in schema2]

    if not key_cols:
        return {"error": "No common key columns found for comparison"}

    total_rows_1 = int(_collect_scalar(lf1, pl.len(), streaming=streaming))
    total_rows_2 = int(_collect_scalar(lf2, pl.len(), streaming=streaming))

    dup_groups_1 = _duplicate_key_group_count(lf1, key_cols, streaming=streaming)
    dup_groups_2 = _duplicate_key_group_count(lf2, key_cols, streaming=streaming)
    non_unique = (dup_groups_1 > 0) or (dup_groups_2 > 0)

    # If the key isn't unique, inner-join comparisons can explode in size (cartesian product per key).
    # In that case we only compare *key coverage* unless the user explicitly allows non-unique keys.
    if non_unique and require_unique_keys:
        keys1 = lf1.select(key_cols).unique()
        keys2 = lf2.select(key_cols).unique()
        total_keys_1 = int(_collect_scalar(keys1, pl.len(), streaming=streaming))
        total_keys_2 = int(_collect_scalar(keys2, pl.len(), streaming=streaming))
        matched_keys = int(
            _collect_scalar(keys1.join(keys2, on=key_cols, how="inner"), pl.len(), streaming=streaming)
        )
        only_in_1 = int(_collect_scalar(keys1.join(keys2, on=key_cols, how="anti"), pl.len(), streaming=streaming))
        only_in_2 = int(_collect_scalar(keys2.join(keys1, on=key_cols, how="anti"), pl.len(), streaming=streaming))
        return {
            "key_columns": key_cols,
            "total_rows_1": total_rows_1,
            "total_rows_2": total_rows_2,
            "unique_keys_1": total_keys_1,
            "unique_keys_2": total_keys_2,
            "duplicate_key_groups_1": dup_groups_1,
            "duplicate_key_groups_2": dup_groups_2,
            "matched_rows": matched_keys,
            "only_in_1": only_in_1,
            "only_in_2": only_in_2,
            "match_percentage": (matched_keys / max(total_keys_1, total_keys_2) * 100.0)
            if max(total_keys_1, total_keys_2) > 0
            else 0.0,
            "warning": "Key columns are not unique in at least one file; computed match counts on unique key combinations only. "
            "Provide more key columns (or pass --allow-non-unique-keys) to enable per-row field comparison safely.",
            "value_comparisons": {},
        }

    joined = lf1.join(lf2, on=key_cols, how="inner", suffix="_2")
    matched_rows = int(_collect_scalar(joined, pl.len(), streaming=streaming))
    only_in_1 = int(_collect_scalar(lf1.join(lf2.select(key_cols), on=key_cols, how="anti"), pl.len(), streaming=streaming))
    only_in_2 = int(_collect_scalar(lf2.join(lf1.select(key_cols), on=key_cols, how="anti"), pl.len(), streaming=streaming))

    if not compare_fields or matched_rows == 0:
        return {
            "key_columns": key_cols,
            "total_rows_1": total_rows_1,
            "total_rows_2": total_rows_2,
            "duplicate_key_groups_1": dup_groups_1,
            "duplicate_key_groups_2": dup_groups_2,
            "matched_rows": matched_rows,
            "only_in_1": only_in_1,
            "only_in_2": only_in_2,
            "match_percentage": (matched_rows / max(total_rows_1, total_rows_2) * 100.0)
            if max(total_rows_1, total_rows_2) > 0
            else 0.0,
            "value_comparisons": {},
        }

    common_cols = [c for c in schema1.keys() if c in schema2 and c not in key_cols]
    if not common_cols:
        return {
            "key_columns": key_cols,
            "total_rows_1": total_rows_1,
            "total_rows_2": total_rows_2,
            "duplicate_key_groups_1": dup_groups_1,
            "duplicate_key_groups_2": dup_groups_2,
            "matched_rows": matched_rows,
            "only_in_1": only_in_1,
            "only_in_2": only_in_2,
            "match_percentage": (matched_rows / max(total_rows_1, total_rows_2) * 100.0)
            if max(total_rows_1, total_rows_2) > 0
            else 0.0,
            "value_comparisons": {},
        }

    exprs: list[pl.Expr] = []
    incompatible_columns: dict[str, dict[str, str]] = {}
    comparable_cols: list[str] = []
    numeric_compare_cols: set[str] = set()
    for col in common_cols:
        dt1 = schema1[col]
        dt2 = schema2[col]

        if not _is_comparable_dtype(dt1) or not _is_comparable_dtype(dt2):
            incompatible_columns[col] = {
                "type_1": str(dt1),
                "type_2": str(dt2),
                "reason": "non-comparable dtype",
            }
            continue

        num1 = _is_numeric_dtype(dt1)
        num2 = _is_numeric_dtype(dt2)
        tmp1 = _is_temporal_dtype(dt1)
        tmp2 = _is_temporal_dtype(dt2)

        # If one side is temporal and the other is not, comparing can error.
        if tmp1 != tmp2:
            incompatible_columns[col] = {
                "type_1": str(dt1),
                "type_2": str(dt2),
                "reason": "incompatible dtypes for comparison (temporal mismatch)",
            }
            continue

        # If one side is numeric and the other is string, we can still compare by casting both to Float64
        # (common for CSVs where one file infers numbers and the other infers strings).
        if num1 != num2:
            if (num1 and dt2 == pl.Utf8) or (num2 and dt1 == pl.Utf8):
                num1 = True
                num2 = True
                numeric_compare_cols.add(col)
            else:
                incompatible_columns[col] = {
                    "type_1": str(dt1),
                    "type_2": str(dt2),
                    "reason": "incompatible dtypes for comparison",
                }
                continue

        comparable_cols.append(col)

        c1 = pl.col(col)
        c2 = pl.col(f"{col}_2")

        # Coerce numeric comparisons to a common float dtype to avoid int/float mismatch issues.
        if num1 and num2:
            c1 = c1.cast(pl.Float64, strict=False)
            c2 = c2.cast(pl.Float64, strict=False)
        both = c1.is_not_null() & c2.is_not_null()
        null_both = (c1.is_null() & c2.is_null())
        null_one = (c1.is_null() & c2.is_not_null()) | (c1.is_not_null() & c2.is_null())

        exprs.append(((both & (c1 == c2)).cast(pl.UInt64)).sum().alias(f"{col}__exact_matches"))
        exprs.append((null_both.cast(pl.UInt64)).sum().alias(f"{col}__null_both"))
        exprs.append((null_one.cast(pl.UInt64)).sum().alias(f"{col}__null_one"))

        if num1 and num2:
            abs_diff = (c1 - c2).abs()
            exprs.append(
                ((both & (abs_diff <= tolerance)).cast(pl.UInt64)).sum().alias(f"{col}__approx_matches")
            )
            exprs.append(((both & (c1 != c2)).cast(pl.UInt64)).sum().alias(f"{col}__diff_non_null"))
            exprs.append(
                ((both & (abs_diff > tolerance)).cast(pl.UInt64)).sum().alias(f"{col}__diff_over_tol")
            )
            exprs.append((pl.when(both).then(abs_diff).otherwise(None).max()).alias(f"{col}__max_diff"))
            exprs.append((pl.when(both).then(abs_diff).otherwise(None).mean()).alias(f"{col}__mean_diff"))
        else:
            exprs.append(((both & (c1 != c2)).cast(pl.UInt64)).sum().alias(f"{col}__diff_non_null"))

        # Total diffs (null mismatch + non-null mismatch)
        exprs.append(
            (
                (null_one.cast(pl.UInt64)).sum() + ((both & (c1 != c2)).cast(pl.UInt64)).sum()
            ).alias(f"{col}__diff_total")
        )

    try:
        stats = _collect_df(joined.select(exprs), streaming=streaming) if exprs else None
    except Exception as e:
        return {
            "key_columns": key_cols,
            "total_rows_1": total_rows_1,
            "total_rows_2": total_rows_2,
            "duplicate_key_groups_1": dup_groups_1,
            "duplicate_key_groups_2": dup_groups_2,
            "matched_rows": matched_rows,
            "only_in_1": only_in_1,
            "only_in_2": only_in_2,
            "match_percentage": (matched_rows / max(total_rows_1, total_rows_2) * 100.0)
            if max(total_rows_1, total_rows_2) > 0
            else 0.0,
            "value_comparisons": {},
            "tolerance": tolerance,
            "reported_columns": 0,
            "total_common_columns": len(common_cols),
            "incompatible_columns": incompatible_columns,
            "error": f"Value comparison failed to compute statistics: {e}",
        }
    value_rows: list[dict[str, Any]] = []
    for col in comparable_cols:
        if stats is None:
            break
        diff_total = int(stats[f"{col}__diff_total"][0])
        if diff_total <= 0:
            continue
        is_numeric_row = (_is_numeric_dtype(schema1[col]) and _is_numeric_dtype(schema2[col])) or (col in numeric_compare_cols)
        row: dict[str, Any] = {
            "type": "numeric" if is_numeric_row else "string",
            "total_rows": matched_rows,
            "diff_total": diff_total,
            "exact_matches": int(stats[f"{col}__exact_matches"][0]),
            "null_both": int(stats[f"{col}__null_both"][0]),
            "null_one": int(stats[f"{col}__null_one"][0]),
            "diff_non_null": int(stats[f"{col}__diff_non_null"][0]),
        }
        if row["type"] == "numeric":
            row["approx_matches"] = int(stats[f"{col}__approx_matches"][0])
            row["diff_over_tolerance"] = int(stats[f"{col}__diff_over_tol"][0])
            max_diff = stats.get_column(f"{col}__max_diff")[0]
            mean_diff = stats.get_column(f"{col}__mean_diff")[0]
            row["max_difference"] = float(max_diff) if max_diff is not None else None
            row["mean_difference"] = float(mean_diff) if mean_diff is not None else None

        # Match % treating both-null as match
        match_count = row["exact_matches"] + row["null_both"]
        row["match_percentage"] = (match_count / matched_rows * 100.0) if matched_rows else 0.0
        value_rows.append((col, row))

    value_rows.sort(key=lambda x: x[1]["diff_total"], reverse=True)
    value_rows = value_rows[:max_columns]
    value_comparisons = {col: comp for col, comp in value_rows}

    return {
        "key_columns": key_cols,
        "total_rows_1": total_rows_1,
        "total_rows_2": total_rows_2,
        "duplicate_key_groups_1": dup_groups_1,
        "duplicate_key_groups_2": dup_groups_2,
        "matched_rows": matched_rows,
        "only_in_1": only_in_1,
        "only_in_2": only_in_2,
        "match_percentage": (matched_rows / max(total_rows_1, total_rows_2) * 100.0)
        if max(total_rows_1, total_rows_2) > 0
        else 0.0,
        "value_comparisons": value_comparisons,
        "tolerance": tolerance,
        "reported_columns": len(value_comparisons),
        "total_common_columns": len(common_cols),
        "incompatible_columns": incompatible_columns,
    }


def print_comparison_report(
    file1_path: str,
    file2_path: str,
    schema_comp: dict[str, Any],
    seq_comp: dict[str, Any],
    value_comp: dict[str, Any],
):
    """Print a formatted comparison report."""
    logger.info("\n" + "="*80)
    logger.info("CHECKPOINT COMPARISON REPORT")
    logger.info("="*80)
    logger.info(f"\nFile 1: {file1_path}")
    logger.info(f"File 2: {file2_path}")
    
    # Schema comparison
    logger.info("\n" + "-"*80)
    logger.info("SCHEMA COMPARISON")
    logger.info("-"*80)
    logger.info(f"Field count - File 1: {schema_comp['field_count_1']}, File 2: {schema_comp['field_count_2']}")
    logger.info(f"Field count match: {'[OK]' if schema_comp['field_count_match'] else '[FAIL]'}")
    logger.info(f"Field names match: {'[OK]' if schema_comp['field_names_match'] else '[FAIL]'}")
    # Order is informational: we treat different ordering as non-fatal for "same schema".
    if "field_order_match" in schema_comp:
        logger.info(f"Field order match: {'[OK]' if schema_comp['field_order_match'] else '[WARN]'}")
    logger.info(f"Field types match: {'[OK]' if schema_comp['field_types_match'] else '[FAIL]'}")

    if schema_comp["missing_in_2_count"]:
        logger.info(
            f"\n[WARNING] Fields in File 1 but not in File 2: {schema_comp['missing_in_2_count']} "
            f"(sample: {schema_comp['missing_in_2_sample']})"
        )
    if schema_comp["missing_in_1_count"]:
        logger.info(
            f"[WARNING] Fields in File 2 but not in File 1: {schema_comp['missing_in_1_count']} "
            f"(sample: {schema_comp['missing_in_1_sample']})"
        )
    if schema_comp["type_differences_count"]:
        logger.info(f"\n[WARNING] Type differences: {schema_comp['type_differences_count']} (sample below)")
        for col, diff in schema_comp["type_differences_sample"].items():
            logger.info(f"   {col}: {diff['type_1']} vs {diff['type_2']}")
    
    # Sequence comparison
    logger.info("\n" + "-"*80)
    logger.info("SEQUENCE COMPARISON")
    logger.info("-"*80)
    if "error" in seq_comp:
        logger.info(f"[WARNING] Sequence comparison error: {seq_comp['error']}")
    else:
        logger.info(
            f"Total sequences - File 1: {seq_comp['total_sequences_1']:,}, File 2: {seq_comp['total_sequences_2']:,}"
        )
        logger.info(f"Common sequences: {seq_comp['common_sequences']:,}")
        logger.info(f"Sequence IDs match: {'[OK]' if seq_comp['sequence_ids_match'] else '[FAIL]'}")

        if seq_comp["only_in_1"] > 0:
            logger.info(f"\n[WARNING] Sequences only in File 1: {seq_comp['only_in_1']}")
            if seq_comp["only_in_1_ids"]:
                logger.info(f"   Sample IDs: {seq_comp['only_in_1_ids']}")

        if seq_comp["only_in_2"] > 0:
            logger.info(f"[WARNING] Sequences only in File 2: {seq_comp['only_in_2']}")
            if seq_comp["only_in_2_ids"]:
                logger.info(f"   Sample IDs: {seq_comp['only_in_2_ids']}")

        logger.info(f"\nRow count matches: {seq_comp['row_count_matches']:,}/{seq_comp['common_sequences']:,}")
        if seq_comp["row_count_differences"] > 0:
            logger.info(f"[WARNING] Sequences with different row counts: {seq_comp['row_count_differences']}")
            if seq_comp["row_count_differences_details"]:
                logger.info("   Sample differences:")
                for diff in seq_comp["row_count_differences_details"][:10]:
                    logger.info(
                        f"      Sequence {diff['sequence_id']}: {diff['row_count_1']} vs {diff['row_count_2']} "
                        f"(diff: {diff['difference']:+d})"
                    )
    
    # Value comparison
    if 'error' not in value_comp:
        logger.info("\n" + "-"*80)
        logger.info("VALUE COMPARISON")
        logger.info("-"*80)
        logger.info(f"Total rows - File 1: {value_comp['total_rows_1']:,}, File 2: {value_comp['total_rows_2']:,}")
        logger.info(f"Key columns: {value_comp.get('key_columns')}")
        if value_comp.get("duplicate_key_groups_1", 0) or value_comp.get("duplicate_key_groups_2", 0):
            logger.info(
                f"Duplicate key groups - File 1: {value_comp.get('duplicate_key_groups_1', 0):,}, "
                f"File 2: {value_comp.get('duplicate_key_groups_2', 0):,}"
            )
        logger.info(f"Matched rows: {value_comp['matched_rows']:,}")
        logger.info(f"Only in File 1: {value_comp['only_in_1']:,}")
        logger.info(f"Only in File 2: {value_comp['only_in_2']:,}")
        logger.info(f"Match percentage: {value_comp['match_percentage']:.2f}%")

        if value_comp.get("warning"):
            logger.info(f"\n[WARNING] {value_comp['warning']}")
        
        if value_comp.get("incompatible_columns"):
            inc = value_comp["incompatible_columns"]
            logger.info(f"\n[WARNING] Incompatible columns skipped: {len(inc)} (sample below)")
            for name, meta in list(inc.items())[:10]:
                reason = meta.get("reason", "incompatible")
                logger.info(f"   {name}: {meta.get('type_1')} vs {meta.get('type_2')} ({reason})")

        if value_comp['value_comparisons']:
            logger.info(
                f"\nField-by-field comparison (showing {len(value_comp['value_comparisons'])} columns"
                f"{' of ' + str(value_comp.get('total_common_columns')) if value_comp.get('total_common_columns') is not None else ''}"
                f"; numeric tolerance={value_comp.get('tolerance', 0.01)}):"
            )
            for col, comp in value_comp['value_comparisons'].items():
                logger.info(f"\n  {col} ({comp['type']}):")
                logger.info(f"    Total rows compared: {comp['total_rows']:,}")
                if comp['type'] == 'numeric':
                    logger.info(f"    Exact matches: {comp['exact_matches']:,} ({comp['match_percentage']:.2f}%)")
                    logger.info(f"    Approx matches (<= tolerance): {comp.get('approx_matches', 0):,}")
                    logger.info(f"    Null mismatch (one null): {comp.get('null_one', 0):,}")
                    logger.info(f"    Non-null differences: {comp.get('diff_non_null', 0):,}")
                    logger.info(f"    Total differences: {comp.get('diff_total', 0):,}")
                    if comp['max_difference'] is not None:
                        logger.info(f"    Max difference: {comp['max_difference']:.6f}")
                    if comp['mean_difference'] is not None:
                        logger.info(f"    Mean difference: {comp['mean_difference']:.6f}")
                else:
                    logger.info(f"    Exact matches: {comp['exact_matches']:,} ({comp['match_percentage']:.2f}%)")
                    logger.info(f"    Both null: {comp.get('null_both', 0):,}")
                    logger.info(f"    One null: {comp.get('null_one', 0):,}")
                    logger.info(f"    Non-null differences: {comp.get('diff_non_null', 0):,}")
                    logger.info(f"    Total differences: {comp.get('diff_total', 0):,}")
        else:
            total_common = value_comp.get("total_common_columns")
            if total_common is not None:
                logger.info(f"\n[OK] No value differences found in {total_common} common columns.")
    else:
        logger.info(f"\n[WARNING] Value comparison error: {value_comp['error']}")
    
    logger.info("\n" + "="*80)


@app.command()
def compare(
    file1: str = typer.Argument(..., help="Path to first checkpoint file"),
    file2: str = typer.Argument(..., help="Path to second checkpoint file"),
    key_columns: Optional[str] = typer.Option(
        None,
        "--key-columns",
        "-k",
        help=(
            "Comma-separated key columns for row matching. Supports mapping with '=': "
            '"sequence_id,Timestamp (YYYY-MM-DDThh:mm:ss)=timestamp" '
            "(default tries common columns like sequence_id + timestamp)"
        ),
    ),
    column_map: Optional[str] = typer.Option(
        None,
        "--column-map",
        help=(
            "Optional additional column rename mappings applied to file2 for value comparison. "
            'Format: "File1Name=file2_name,Other=file2_other".'
        ),
    ),
    auto_align_columns: bool = typer.Option(
        True,
        "--auto-align-columns/--no-auto-align-columns",
        help="Best-effort: auto-rename file2 columns to match file1 based on normalized names (safe, one-to-one only).",
    ),
    tolerance: float = typer.Option(
        0.01,
        "--tolerance",
        "-t",
        help="Numeric tolerance for approximate matches (abs(diff) <= tolerance)",
    ),
    max_report_columns: int = typer.Option(
        25,
        "--max-report-columns",
        help="Max number of columns to include in the field-by-field report (sorted by most differences)",
    ),
    max_schema_list: int = typer.Option(
        30,
        "--max-schema-list",
        help="Max number of schema diffs (missing fields / type diffs) to print as samples",
    ),
    max_sequence_samples: int = typer.Option(
        20,
        "--max-sequence-samples",
        help="Max number of sequence IDs / row-count differences to show as samples",
    ),
    streaming: bool = typer.Option(
        True,
        "--streaming/--no-streaming",
        help="Enable Polars streaming where possible (lower memory, often faster on big files)",
    ),
    compare_fields: bool = typer.Option(
        True,
        "--compare-fields/--no-compare-fields",
        help="Compute field-by-field stats (can be heavy; disable for fastest run)",
    ),
    require_unique_keys: bool = typer.Option(
        True,
        "--require-unique-keys/--allow-non-unique-keys",
        help="Safety guard: if key columns are not unique, skip per-row field comparison to avoid join explosion",
    ),
) -> None:
    """
    Compare two checkpoint CSV files and provide detailed statistics.
    
    Examples:
        compare_checkpoints.py checkpoint1.csv checkpoint2.csv
        compare_checkpoints.py checkpoint1.csv checkpoint2.csv --key-columns "sequence_id,timestamp"
    """
    # Configure loguru to show only the message
    logger.remove()
    logger.add(sys.stdout, format="{message}")

    # Validate file paths
    file1_path = Path(file1)
    file2_path = Path(file2)
    
    if not file1_path.exists():
        typer.echo(f"[ERROR] File 1 does not exist: {file1}", err=True)
        raise typer.Exit(1)
    
    if not file2_path.exists():
        typer.echo(f"[ERROR] File 2 does not exist: {file2}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Loading files...")
    typer.echo(f"   File 1: {file1_path}")
    typer.echo(f"   File 2: {file2_path}")
    
    try:
        lf1 = _scan_csv(file1_path)
        lf2 = _scan_csv(file2_path)

        schema1 = lf1.collect_schema()
        schema2 = lf2.collect_schema()

        # Parse key columns if provided (supports mapping: left=file1, right=file2).
        key_cols: Optional[list[str]] = None
        applied_renames: dict[str, str] = {}
        if key_columns:
            parsed_cols, rename_map = _parse_key_columns_spec(key_columns)
            if rename_map:
                # Apply renames to file2 only, so joins can use file1's column names.
                lf2 = lf2.rename(rename_map)
                applied_renames.update(rename_map)
                schema2 = lf2.collect_schema()

            key_cols = parsed_cols
            missing_in_1 = [c for c in key_cols if c not in schema1]
            missing_in_2 = [c for c in key_cols if c not in schema2]
            if missing_in_1:
                typer.echo(f"[WARNING] Key columns not in File 1: {missing_in_1}", err=True)
            if missing_in_2:
                typer.echo(f"[WARNING] Key columns not in File 2 (after mapping): {missing_in_2}", err=True)

        # Optional manual value-column mapping (file1=file2). Applied to file2.
        if column_map:
            _, extra_map = _parse_key_columns_spec(column_map)
            if extra_map:
                lf2 = lf2.rename(extra_map)
                applied_renames.update(extra_map)
                schema2 = lf2.collect_schema()

        # Optional auto-align of remaining file2 columns to match file1 names.
        if auto_align_columns:
            exclude_file1 = set(key_cols or [])
            # exclude original and renamed names to avoid accidental key/critical remaps
            exclude_file2 = set(applied_renames.keys()) | set(applied_renames.values())
            inferred = _infer_rename_map_for_file2(
                schema1,
                schema2,
                exclude_file1_cols=exclude_file1,
                exclude_file2_cols=exclude_file2,
            )
            if inferred:
                lf2 = lf2.rename(inferred)
                schema2 = lf2.collect_schema()

        total_rows_1 = int(_collect_scalar(lf1, pl.len(), streaming=streaming))
        total_rows_2 = int(_collect_scalar(lf2, pl.len(), streaming=streaming))
        typer.echo(f"[OK] Scanned File 1: {total_rows_1:,} rows, {len(schema1)} columns")
        typer.echo(f"[OK] Scanned File 2: {total_rows_2:,} rows, {len(schema2)} columns")

        typer.echo(f"\nComparing schemas...")
        schema_comp = compare_schemas(schema1, schema2, max_list=max_schema_list)
        
        typer.echo(f"Comparing sequences...")
        seq_comp = compare_sequences(lf1, lf2, streaming=streaming, sample_n=max_sequence_samples)
        
        typer.echo(f"Comparing values...")
        value_comp = compare_values(
            lf1,
            lf2,
            key_cols=key_cols,
            tolerance=tolerance,
            streaming=streaming,
            max_columns=max_report_columns,
            require_unique_keys=require_unique_keys,
            compare_fields=compare_fields,
        )
        
        # Print report
        print_comparison_report(str(file1_path), str(file2_path), schema_comp, seq_comp, value_comp)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        
        all_match = (
            schema_comp['field_count_match'] and
            schema_comp['field_names_match'] and
            schema_comp['field_types_match'] and
            (seq_comp.get('sequence_ids_match', True)) and
            (seq_comp.get('row_count_differences', 0) == 0) and
            (value_comp.get('match_percentage', 0.0) == 100.0)
        )
        
        if all_match:
            typer.echo("[OK] Files are identical!")
        else:
            typer.echo("[WARNING] Files differ - see details above")
            issues = []
            if not schema_comp['field_count_match'] or not schema_comp['field_names_match'] or not schema_comp['field_types_match']:
                issues.append("Schema differences")
            if not seq_comp['sequence_ids_match']:
                issues.append("Sequence ID differences")
            if seq_comp['row_count_differences'] > 0:
                issues.append(f"{seq_comp['row_count_differences']} sequences with different row counts")
            if value_comp.get('match_percentage', 100) < 100:
                issues.append(f"Value differences ({100 - value_comp.get('match_percentage', 0):.2f}% mismatch)")
            if ("field_order_match" in schema_comp) and (not schema_comp["field_order_match"]):
                issues.append("Column order differs (non-fatal)")
            typer.echo(f"   Issues found: {', '.join(issues)}")
        
    except Exception as e:
        typer.echo(f"[ERROR] Error during comparison: {e}", err=True)
        import traceback
        typer.echo(f"Traceback:\n{traceback.format_exc()}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


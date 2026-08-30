from __future__ import annotations

from pathlib import Path

import polars as pl

import compare_checkpoints


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_compare_schemas_ignores_field_order_for_name_match() -> None:
    schema1 = pl.Schema({"sequence_id": pl.Int64, "timestamp": pl.Utf8, "value": pl.Int64})
    schema2 = pl.Schema({"timestamp": pl.Utf8, "value": pl.Int64, "sequence_id": pl.Int64})

    comp = compare_checkpoints.compare_schemas(schema1, schema2, max_list=10)

    assert comp["field_names_match"] is True
    assert comp["field_order_match"] is False
    assert comp["field_types_match"] is True


def test_value_compare_matches_when_csv_column_order_differs(tmp_path: Path) -> None:
    file1 = tmp_path / "a.csv"
    file2 = tmp_path / "b.csv"

    # Same rows, different column order.
    _write_text(
        file1,
        "sequence_id,timestamp,value\n"
        "1,2025-01-01T00:00:00,10\n"
        "1,2025-01-01T00:05:00,11\n",
    )
    _write_text(
        file2,
        "timestamp,value,sequence_id\n"
        "2025-01-01T00:00:00,10,1\n"
        "2025-01-01T00:05:00,11,1\n",
    )

    lf1 = compare_checkpoints._scan_csv(file1)
    lf2 = compare_checkpoints._scan_csv(file2)

    schema_comp = compare_checkpoints.compare_schemas(
        lf1.collect_schema(), lf2.collect_schema(), max_list=10
    )
    assert schema_comp["field_names_match"] is True
    assert schema_comp["field_order_match"] is False

    value_comp = compare_checkpoints.compare_values(
        lf1,
        lf2,
        key_cols=["sequence_id", "timestamp"],
        tolerance=0.0,
        streaming=False,
        max_columns=50,
        require_unique_keys=True,
        compare_fields=True,
    )

    assert "error" not in value_comp
    assert value_comp["matched_rows"] == 2
    assert value_comp["only_in_1"] == 0
    assert value_comp["only_in_2"] == 0
    assert value_comp["match_percentage"] == 100.0


def test_schema_detects_missing_columns_even_if_order_differs() -> None:
    schema1 = pl.Schema({"a": pl.Int64, "b": pl.Utf8})
    schema2 = pl.Schema({"b": pl.Utf8})

    comp = compare_checkpoints.compare_schemas(schema1, schema2, max_list=10)

    assert comp["field_names_match"] is False
    assert comp["missing_in_2_count"] == 1
    assert comp["missing_in_2_sample"] == ["a"]



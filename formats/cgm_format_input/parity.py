#!/usr/bin/env python3
"""
Native-vs-cgm_format parity for inputs both backends can read.

For every user, compares the frames each backend yields *before* the processing pipeline
(gap detection, interpolation, resampling): which EGV timestamps exist, whether glucose
agrees at them, and the per-user totals of insulin and carbs. Used by
tests/test_cgm_format_parity.py and scripts/cgm_format_parity.py, which writes the ledger
in docs/CGM_FORMAT_PARITY.md.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from formats.cgm_format_input.cgm_format_database_converter import (
    CGM_FORMAT_DATABASE_TYPE,
    CgmFormatDatabaseConverter,
)
from formats.database_detector import DatabaseDetector

PARITY_SUM_FIELDS: tuple[str, ...] = ("fast_acting_insulin_u", "long_acting_insulin_u", "carb_grams")
GLUCOSE_TOLERANCE_MGDL = 1e-6


@dataclass(frozen=True)
class UserParity:
    """How one user's native and cgm_format frames compare."""

    user_id: str
    egv_native: int
    egv_adapter: int
    only_native: int
    only_adapter: int
    #: EGV timestamps present in both whose glucose differs by more than the tolerance.
    glucose_mismatches: tuple[Any, ...]
    #: Median of adapter/native glucose at shared timestamps; None if none are shared.
    glucose_ratio: Optional[float]
    #: field -> (native total, adapter total); a total is None when no row carries the field.
    sums: Dict[str, tuple[Optional[float], Optional[float]]] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusParity:
    root: Path
    native_type: str
    native_users: tuple[str, ...]
    adapter_users: tuple[str, ...]
    users: tuple[UserParity, ...]


def _egv(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("glucose_value_mgdl").is_not_null()).select(
        pl.col("timestamp").cast(pl.Datetime("us")),
        pl.col("glucose_value_mgdl").cast(pl.Float64, strict=False).alias("glucose"),
    )


def _total(frame: pl.DataFrame, column: str) -> Optional[float]:
    if column not in frame.columns:
        return None
    values = frame[column].cast(pl.Float64, strict=False).drop_nulls()
    return float(values.sum()) if len(values) else None


def compare_user(user_id: str, native: pl.DataFrame, adapter: pl.DataFrame) -> UserParity:
    egv_n, egv_a = _egv(native), _egv(adapter)
    ts_n, ts_a = set(egv_n["timestamp"].to_list()), set(egv_a["timestamp"].to_list())
    shared = egv_n.join(egv_a, on="timestamp", suffix="_adapter")
    mismatched = shared.filter((pl.col("glucose") - pl.col("glucose_adapter")).abs() > GLUCOSE_TOLERANCE_MGDL)
    ratio = (
        float((shared["glucose_adapter"] / shared["glucose"]).median()) if shared.height else None
    )
    return UserParity(
        user_id=user_id,
        egv_native=egv_n.height,
        egv_adapter=egv_a.height,
        only_native=len(ts_n - ts_a),
        only_adapter=len(ts_a - ts_n),
        glucose_mismatches=tuple(sorted(mismatched["timestamp"].to_list())),
        glucose_ratio=ratio,
        sums={c: (_total(native, c), _total(adapter, c)) for c in PARITY_SUM_FIELDS},
    )


def compare_corpus(root: Path, config: Dict[str, Any]) -> CorpusParity:
    """Run both backends over ``root`` and compare them user by user."""
    detector = DatabaseDetector()
    native_type = detector.detect_database_type(root)
    if native_type in ("unknown", CGM_FORMAT_DATABASE_TYPE):
        raise ValueError(f"{root} has no native converter ({native_type}); parity needs both backends")
    native_converter = detector.get_database_converter(native_type, config)
    native = {f["user_id"][0]: f for f in native_converter.iter_user_event_frames(root, interval_minutes=5)}
    adapter_converter = CgmFormatDatabaseConverter(config, database_type=CGM_FORMAT_DATABASE_TYPE)
    adapter = {f["user_id"][0]: f for f in adapter_converter.iter_user_event_frames(root, interval_minutes=5)}
    users: List[UserParity] = [
        compare_user(u, native[u], adapter[u]) for u in sorted(set(native) & set(adapter))
    ]
    return CorpusParity(
        root=root,
        native_type=native_type,
        native_users=tuple(sorted(native)),
        adapter_users=tuple(sorted(adapter)),
        users=tuple(users),
    )

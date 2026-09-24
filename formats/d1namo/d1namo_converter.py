#!/usr/bin/env python3
"""
D1NAMO converter.

Each subject folder holds ``glucose.csv`` (``date,time,glucose,type,comments``, glucose in
mmol/L) and, in the diabetes subset, ``insulin.csv``
(``date,time,fast_insulin,slow_insulin,comment``). Times occur as ``HH:MM:SS`` and ``HH:MM``.
"""

from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter
from formats.glucose_bounds import MGDL_PER_MMOL

D1NAMO_GLUCOSE_HEADERS: frozenset[str] = frozenset({"date", "time", "glucose", "type"})
D1NAMO_INSULIN_HEADERS: frozenset[str] = frozenset({"date", "time", "fast_insulin", "slow_insulin"})

# glucose.csv 'type' value that marks a sensor reading. Everything else ('manual' in the
# diabetes subset, meal-relative labels such as 'BB'/'AL' in the healthy subset) is a
# fingerstick and becomes BGM.
D1NAMO_CGM_TYPE = "cgm"
D1NAMO_EVENT_CGM = "EGV"
D1NAMO_EVENT_BGM = "BGM"
D1NAMO_EVENT_INSULIN = "Insulin"


def _clean(headers: List[str]) -> set[str]:
    return {h.strip().lstrip("﻿") for h in headers if h.strip()}


class D1namoConverter(CSVFormatConverter):
    """Converter for D1NAMO ``glucose.csv`` and ``insulin.csv`` files."""

    def can_handle(self, headers: List[str]) -> bool:
        clean = _clean(headers)
        return D1NAMO_GLUCOSE_HEADERS.issubset(clean) or D1NAMO_INSULIN_HEADERS.issubset(clean)

    @staticmethod
    def _timestamp(row: Dict[str, str]) -> Optional[str]:
        date = (row.get("date") or "").strip()
        time = (row.get("time") or "").strip()
        if not date or not time:
            return None
        if time.count(":") == 1:
            time = f"{time}:00"
        return f"{date}T{time}"

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        timestamp = self._timestamp(row)
        if timestamp is None:
            return None

        if "glucose" in row:
            value = (row.get("glucose") or "").strip()
            if not value:
                return None
            try:
                mgdl = float(value) * MGDL_PER_MMOL
            except ValueError:
                return None
            is_cgm = (row.get("type") or "").strip().lower() == D1NAMO_CGM_TYPE
            return {
                "timestamp": timestamp,
                "event_type": D1NAMO_EVENT_CGM if is_cgm else D1NAMO_EVENT_BGM,
                "glucose_value_mgdl": str(mgdl),
            }

        fast = (row.get("fast_insulin") or "").strip()
        slow = (row.get("slow_insulin") or "").strip()
        if not fast and not slow:
            return None
        return {
            "timestamp": timestamp,
            "event_type": D1NAMO_EVENT_INSULIN,
            "fast_acting_insulin_u": fast,
            "long_acting_insulin_u": slow,
        }

    def get_format_name(self) -> str:
        return "D1NAMO"

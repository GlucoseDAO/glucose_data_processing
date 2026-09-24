#!/usr/bin/env python3
"""
JAEB comma-separated device export converter.

Several JAEB public datasets (e.g. the Shah healthy non-diabetic cohort, ``NonDiabDeviceCGM.csv``)
ship device tables as comma-separated files with the header
``PtID,DeviceDtDaysFromEnroll,DeviceTm,RecordType,Value``. Timestamps are relative to enrollment,
exactly as in the pipe-separated MiniDose1 tables, so the same reference date is used.
"""

from typing import Dict, List, Optional

from formats.minidose1.minidose1_converter import Minidose1Converter

JAEB_REQUIRED_HEADERS: frozenset[str] = frozenset(
    {"PtID", "DeviceDtDaysFromEnroll", "DeviceTm", "RecordType", "Value"}
)

# RecordType (lower-cased) -> event_type written to the output.
JAEB_RECORD_TYPE_TO_EVENT: Dict[str, str] = {
    "cgm": "EGV",
    "calibration": "Calibration",
    "bgm": "BGM",
}


class JaebConverter(Minidose1Converter):
    """Converter for JAEB comma-separated ``RecordType,Value`` device tables."""

    CSV_DELIMITER: str = ","

    def can_handle(self, headers: List[str]) -> bool:
        clean_headers = {h.strip().lstrip("﻿") for h in headers if h.strip()}
        return JAEB_REQUIRED_HEADERS.issubset(clean_headers)

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert one row. Every known RecordType is kept with its own event_type so the
        database converter can count and drop non-CGM rows in one place.
        """
        timestamp = self._parse_timestamp(row.get("DeviceDtDaysFromEnroll"), row.get("DeviceTm"))
        value = (row.get("Value") or "").strip()
        record_type = (row.get("RecordType") or "").strip().lower()
        event_type = JAEB_RECORD_TYPE_TO_EVENT.get(record_type)
        if not timestamp or not value or event_type is None:
            return None

        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "glucose_value_mgdl": value,
            "user_id": (row.get("PtID") or "").strip(),
        }

    def set_context(self, file_path) -> None:
        """RecordType carries the data type, so no filename context is needed."""

    def get_format_name(self) -> str:
        return "JAEB device export (comma-separated)"

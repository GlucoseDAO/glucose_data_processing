#!/usr/bin/env python3
"""
University of Manchester (UoM) sleeptime data format converter.

This module provides the converter for UoM sleeptime data format (UoM*sleeptime.csv).
"""

from datetime import datetime
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class UoMSleeptimeConverter(CSVFormatConverter):
    """Converter for University of Manchester sleeptime data format (UoM*sleeptime.csv)."""

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "uom"):
        """
        Initialize the UoM sleeptime converter.

        Args:
            output_fields: List of standard field names to include in output.
                          Uses standard field names (e.g., 'timestamp', 'duration_seconds').
                          If None, uses default fields.
            database_type: Database type identifier (default: "uom")
        """
        super().__init__(output_fields, database_type)
        # Load database schema
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["sleeptime"]

    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.

        Args:
            headers: List of column headers from the CSV file

        Returns:
            True if this converter can handle the format, False otherwise
        """
        # Clean headers (remove empty strings and BOM characters)
        clean_headers = [h.strip().lstrip("\ufeff") for h in headers if h.strip()]

        # Check for UoM sleeptime format headers (UoM*sleeptime.csv format)
        uom_sleeptime_headers = ["calendar_date", "start_date_ts", "duration_in_sec"]

        return all(header in clean_headers for header in uom_sleeptime_headers)

    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """
        Parse timestamp using schema-defined formats.
        Business logic: if no time component, default to midnight.

        Args:
            timestamp_str: Timestamp string from source

        Returns:
            Timestamp string in standard format or None if parsing fails
        """
        if not timestamp_str or timestamp_str.strip() == "":
            return None

        timestamp_str = timestamp_str.strip()

        # Use timestamp formats from schema
        formats = self.db_schema["timestamp_formats"]
        output_format = self.db_schema["timestamp_output_format"]

        # Add date-only formats for calendar_date field
        date_only_formats = ["%d/%m/%Y", "%m/%d/%Y"]

        for fmt in formats + date_only_formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                # Business logic: if no time component, default to midnight
                if fmt.endswith("%Y"):
                    return dt.strftime("%Y-%m-%dT00:00:00")
                return dt.strftime(output_format)
            except ValueError:
                continue

        return None

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.

        Maps source fields to standard fields using schema and outputs only requested fields.
        Always includes timestamp first, then other requested fields.

        Uses start_date_ts as the primary timestamp for the sleep event (business logic in code).

        Args:
            row: Dictionary representing a single CSV row

        Returns:
            Dictionary with standard field names, filtered to requested fields
        """
        # Parse timestamp - required for all rows (use start_date_ts)
        timestamp_field = self.converter_schema["timestamp_field"]
        timestamp = self._parse_timestamp(self._get_clean_value(row, timestamp_field))
        if not timestamp:
            return None

        # Build result dictionary with standard field names
        result: Dict[str, str] = {}

        # Always add timestamp first (using standard name)
        result["timestamp"] = timestamp

        # Add event type if requested
        if "event_type" in self.output_fields_standard:
            result["event_type"] = self.converter_schema["event_type"]

        # Map source fields to standard fields using schema
        field_mappings = self.converter_schema["field_mappings"]
        source_fields = self._get_source_fields(row)

        # Map source fields to standard fields (skip timestamp as it's already handled)
        for source_field, standard_field in field_mappings.items():
            if source_field == self.converter_schema["timestamp_field"]:
                continue  # Skip timestamp field, already handled
            if source_field in source_fields and standard_field in self.output_fields_standard:
                result[standard_field] = self._get_clean_value(row, source_field)

        # Skip records that only have timestamp and event_type (no meaningful data for default fields)
        meaningful_fields = set(result.keys()) - {"timestamp", "event_type"}
        if not meaningful_fields:
            return None

        # Filter and convert to display names
        return self._filter_output(result)

    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.

        Returns:
            String name of the format
        """
        return "UoM SleepTime Data"



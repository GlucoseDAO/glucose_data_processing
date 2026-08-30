#!/usr/bin/env python3
"""
FreeStyle Libre 3 format converter.

This module provides the converter for FreeStyle Libre 3 CSV format.
"""

from datetime import datetime
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class FreeStyleLibre3Converter(CSVFormatConverter):
    """Converter for FreeStyle Libre 3 format."""

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "libre3"):
        """
        Initialize the FreeStyle Libre 3 converter.

        Args:
            output_fields: List of standard field names to include in output.
                          Uses standard field names (e.g., 'timestamp', 'glucose_value_mgdl').
                          If None, uses default fields.
            database_type: Database type identifier (default: "libre3")
        """
        super().__init__(output_fields, database_type)
        # Load database schema
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["libre3"]

    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.

        Args:
            headers: List of column headers from the CSV file

        Returns:
            True if this converter can handle the format, False otherwise
        """
        return "Device Timestamp" in headers and "Historic Glucose mg/dL" in headers

    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """
        Parse timestamp using schema-defined formats.

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

        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt.strftime(output_format)
            except ValueError:
                continue

        return None

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.

        Maps source fields to standard fields using schema and outputs only requested fields.
        Always includes timestamp first, then other requested fields.

        Business logic:
        - Event type mapping: Record Type '0' → 'EGV', '6' → 'Note', else → 'Unknown'
        - Glucose value: Prefer Historic over Scan
        - Fast-acting insulin: Sum of Rapid-Acting + Meal + Correction
        - Long-acting insulin: Direct mapping

        Args:
            row: Dictionary representing a single CSV row

        Returns:
            Dictionary with standard field names, filtered to requested fields
        """
        # Parse timestamp - required for all rows
        timestamp_field = self.converter_schema["timestamp_field"]
        timestamp = self._parse_timestamp(self._get_clean_value(row, timestamp_field))
        if not timestamp:
            return None

        # Business logic: Determine event type based on record type
        record_type = self._get_clean_value(row, "Record Type")
        if record_type == "0":
            event_type = "EGV"  # Estimated Glucose Value
        elif record_type == "6":
            event_type = "Note"  # Manual entry/note
        else:
            event_type = "Unknown"

        # Business logic: Get glucose value (prefer historic over scan)
        glucose_value = self._get_clean_value(row, "Historic Glucose mg/dL")
        if not glucose_value:
            glucose_value = self._get_clean_value(row, "Scan Glucose mg/dL")

        # Business logic: Combine insulin values (rapid-acting + meal + correction = all fast-acting)
        rapid_insulin = self._safe_float(self._get_clean_value(row, "Rapid-Acting Insulin (units)"))
        meal_insulin = self._safe_float(self._get_clean_value(row, "Meal Insulin (units)"))
        correction_insulin = self._safe_float(self._get_clean_value(row, "Correction Insulin (units)"))
        total_fast_acting = rapid_insulin + meal_insulin + correction_insulin

        # Business logic: Long-acting insulin (direct mapping)
        long_acting_insulin = self._safe_float(self._get_clean_value(row, "Long-Acting Insulin (units)"))

        # Build result dictionary with standard field names
        result: Dict[str, str] = {}

        # Always add timestamp first (using standard name)
        result["timestamp"] = timestamp

        # Add event type if requested (business logic: mapped from Record Type)
        if "event_type" in self.output_fields_standard:
            result["event_type"] = event_type

        # Map glucose value (business logic: prefer historic over scan)
        if "glucose_value_mgdl" in self.output_fields_standard:
            result["glucose_value_mgdl"] = glucose_value

        # Map insulin values (business logic: combine for fast-acting)
        if "fast_acting_insulin_u" in self.output_fields_standard:
            result["fast_acting_insulin_u"] = str(total_fast_acting) if total_fast_acting > 0 else ""

        if "long_acting_insulin_u" in self.output_fields_standard:
            result["long_acting_insulin_u"] = str(long_acting_insulin) if long_acting_insulin > 0 else ""

        # Map carb value
        if "carb_grams" in self.output_fields_standard:
            result["carb_grams"] = self._get_clean_value(row, "Carbohydrates (grams)")

        # Map source fields to standard fields using schema
        field_mappings = self.converter_schema["field_mappings"]
        source_fields = self._get_source_fields(row)

        # Map source fields to standard fields (skip fields already handled with business logic)
        handled_fields = {
            "Device Timestamp",
            "Record Type",
            "Historic Glucose mg/dL",
            "Scan Glucose mg/dL",
            "Rapid-Acting Insulin (units)",
            "Meal Insulin (units)",
            "Correction Insulin (units)",
            "Long-Acting Insulin (units)",
            "Carbohydrates (grams)",
        }

        for source_field, standard_field in field_mappings.items():
            if source_field in handled_fields:
                continue  # Skip fields already handled with business logic
            if source_field in source_fields and standard_field in self.output_fields_standard:
                result[standard_field] = self._get_clean_value(row, source_field)

        # Filter and convert to display names
        return self._filter_output(result)

    def _safe_float(self, value: str) -> float:
        """
        Safely convert string to float, returning 0 if conversion fails.

        Args:
            value: String value to convert

        Returns:
            Float value or 0.0 if conversion fails
        """
        try:
            return float(value.strip()) if value.strip() else 0.0
        except (ValueError, TypeError):
            return 0.0

    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.

        Returns:
            String name of the format
        """
        return "FreeStyle Libre 3"



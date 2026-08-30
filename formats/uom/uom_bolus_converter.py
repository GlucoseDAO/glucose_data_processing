#!/usr/bin/env python3
"""
University of Manchester (UoM) bolus insulin data format converter.

This module provides the converter for UoM bolus insulin data format.
"""

from datetime import datetime
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class UoMBolusConverter(CSVFormatConverter):
    """Converter for University of Manchester bolus insulin data format."""

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "uom"):
        """
        Initialize the UoM bolus converter.

        Args:
            output_fields: List of standard field names to include in output.
                          Uses standard field names (e.g., 'timestamp', 'fast_acting_insulin_u').
                          If None, uses default fields.
            database_type: Database type identifier (default: "uom")
        """
        super().__init__(output_fields, database_type)
        # Load database schema
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["bolus"]

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

        # Check for UoM bolus format headers
        uom_bolus_headers = ["bolus_ts", "bolus_dose"]

        return all(header in clean_headers for header in uom_bolus_headers)

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

    def _convert_insulin_value(self, value_str: str) -> Optional[str]:
        """
        Convert insulin value to standard format.

        Args:
            value_str: Insulin value string

        Returns:
            Insulin value string or None if conversion fails
        """
        if not value_str or value_str.strip() == "":
            return None

        try:
            value = float(value_str.strip())
            return str(round(value, 3))
        except (ValueError, TypeError):
            return None

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.

        Maps source fields to standard fields using schema and outputs only requested fields.
        Always includes timestamp first, then other requested fields.

        Bolus insulin is fast-acting insulin (business logic in code).

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

        # Convert insulin value (business logic: round to 3 decimal places)
        insulin_value = self._convert_insulin_value(self._get_clean_value(row, "bolus_dose"))
        if insulin_value is None:
            return None

        # Build result dictionary with standard field names
        result: Dict[str, str] = {}

        # Always add timestamp first (using standard name)
        result["timestamp"] = timestamp

        # Add event type if requested
        if "event_type" in self.output_fields_standard:
            result["event_type"] = self.converter_schema["event_type"]

        # Map insulin value (business logic: bolus = fast-acting)
        if "fast_acting_insulin_u" in self.output_fields_standard:
            result["fast_acting_insulin_u"] = insulin_value

        # Map source fields to standard fields using schema
        field_mappings = self.converter_schema["field_mappings"]
        source_fields = self._get_source_fields(row)

        # Map source fields to standard fields (skip timestamp and bolus_dose as they're already handled)
        for source_field, standard_field in field_mappings.items():
            if source_field in ["bolus_ts", "bolus_dose"]:
                continue  # Skip timestamp and insulin dose fields, already handled
            if source_field in source_fields and standard_field in self.output_fields_standard:
                result[standard_field] = self._get_clean_value(row, source_field)

        # Filter and convert to display names
        return self._filter_output(result)

    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.

        Returns:
            String name of the format
        """
        return "UoM Bolus Insulin Data"



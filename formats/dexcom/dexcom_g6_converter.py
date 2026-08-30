#!/usr/bin/env python3
"""
Dexcom G6 format converter.

This module provides the converter for Dexcom G6 CSV format, which is the standard
format used by the glucose data preprocessor.
"""

from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class DexcomG6Converter(CSVFormatConverter):
    """Converter for Dexcom G6 format (standard format)."""

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "dexcom"):
        """
        Initialize the Dexcom G6 converter.

        Args:
            output_fields: List of standard field names to include in output.
                          Uses standard field names (e.g., 'timestamp', 'glucose_value_mgdl').
                          If None, uses default fields.
            database_type: Database type identifier (default: "dexcom")
        """
        super().__init__(output_fields, database_type)
        # Load database schema
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["g6"]

    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.

        Args:
            headers: List of column headers from the CSV file

        Returns:
            True if this converter can handle the format, False otherwise
        """
        return "Timestamp (YYYY-MM-DDThh:mm:ss)" in headers and "Event Type" in headers

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.

        Maps source fields to standard fields using schema and outputs only requested fields.
        Always includes timestamp first, then other requested fields.

        Business logic: Insulin is split into two columns based on Event Subtype:
        - Fast-Acting → fast_acting_insulin_u
        - Long-Acting → long_acting_insulin_u
        - If no subtype specified, default to fast-acting

        Args:
            row: Dictionary representing a single CSV row

        Returns:
            Dictionary with standard field names, filtered to requested fields
        """
        # Skip rows without timestamp
        timestamp_field = self.converter_schema["timestamp_field"]
        timestamp = self._get_clean_value(row, timestamp_field)
        if not timestamp:
            return None

        # Business logic: Split insulin based on Event Subtype
        insulin_value = self._get_clean_value(row, "Insulin Value (u)")
        event_subtype = self._get_clean_value(row, "Event Subtype")

        fast_acting_insulin = ""
        long_acting_insulin = ""

        if insulin_value:
            if "Fast-Acting" in event_subtype:
                fast_acting_insulin = insulin_value
            elif "Long-Acting" in event_subtype:
                long_acting_insulin = insulin_value
            else:
                # Business logic: If no subtype specified, default to fast-acting
                fast_acting_insulin = insulin_value

        # Build result dictionary with standard field names
        result: Dict[str, str] = {}

        # Always add timestamp first (using standard name)
        result["timestamp"] = timestamp

        # Map insulin values (business logic: split by subtype)
        if "fast_acting_insulin_u" in self.output_fields_standard:
            result["fast_acting_insulin_u"] = fast_acting_insulin

        if "long_acting_insulin_u" in self.output_fields_standard:
            result["long_acting_insulin_u"] = long_acting_insulin

        # Map source fields to standard fields using schema
        field_mappings = self.converter_schema["field_mappings"]
        source_fields = self._get_source_fields(row)

        # Map source fields to standard fields (skip insulin fields as they're already handled)
        for source_field, standard_field in field_mappings.items():
            if source_field in ["Insulin Value (u)", "Event Subtype"]:
                continue  # Skip insulin fields, already handled with business logic
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
        return "Dexcom G6"



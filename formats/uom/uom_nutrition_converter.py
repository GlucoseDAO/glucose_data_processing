#!/usr/bin/env python3
"""
University of Manchester (UoM) nutrition data format converter.

This module provides the converter for UoM nutrition data format.
"""

from datetime import datetime
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class UoMNutritionConverter(CSVFormatConverter):
    """Converter for University of Manchester nutrition data format."""

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "uom"):
        """
        Initialize the UoM nutrition converter.

        Args:
            output_fields: List of standard field names to include in output.
                          Uses standard field names (e.g., 'timestamp', 'carb_grams').
                          If None, uses default fields.
            database_type: Database type identifier (default: "uom")
        """
        super().__init__(output_fields, database_type)
        # Load database schema
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["nutrition"]

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

        # Check for UoM nutrition format headers
        uom_nutrition_headers = ["meal_ts", "meal_type", "meal_tag", "carbs_g", "prot_g", "fat_g", "fibre_g"]

        return all(header in clean_headers for header in uom_nutrition_headers)

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

    def _convert_carb_value(self, value_str: str) -> Optional[str]:
        """
        Convert carb value to standard format.

        Args:
            value_str: Carb value string in grams

        Returns:
            Carb value string or None if conversion fails
        """
        if not value_str or value_str.strip() == "":
            return None

        try:
            value = float(value_str.strip())
            return str(round(value, 1))
        except (ValueError, TypeError):
            return None

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.

        Maps source fields to standard fields using schema and outputs only requested fields.
        Always includes timestamp first, then other requested fields.

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

        # Convert carb value (business logic: round to 1 decimal, default to '0' if empty)
        carb_value_raw = self._get_clean_value(row, "carbs_g")
        if carb_value_raw and carb_value_raw.strip():
            carb_value = self._convert_carb_value(carb_value_raw)
            if carb_value is None:
                carb_value = "0"
        else:
            carb_value = "0"

        # Build result dictionary with standard field names
        result: Dict[str, str] = {}

        # Always add timestamp first (using standard name)
        result["timestamp"] = timestamp

        # Add event type if requested
        if "event_type" in self.output_fields_standard:
            result["event_type"] = self.converter_schema["event_type"]

        # Map carb value (business logic: default to '0' if empty)
        if "carb_grams" in self.output_fields_standard:
            result["carb_grams"] = carb_value

        # Map source fields to standard fields using schema
        field_mappings = self.converter_schema["field_mappings"]
        source_fields = self._get_source_fields(row)

        # Map source fields to standard fields (skip timestamp and carbs_g as they're already handled)
        for source_field, standard_field in field_mappings.items():
            if source_field in ["meal_ts", "carbs_g"]:
                continue  # Skip timestamp and carb fields, already handled
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
        return "UoM Nutrition Data"



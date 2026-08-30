#!/usr/bin/env python3
"""
University of Manchester (UoM) glucose data format converter.

This module provides the converter for UoM glucose data format.
"""

from datetime import datetime
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class UoMGlucoseConverter(CSVFormatConverter):
    """Converter for University of Manchester glucose data format."""

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "uom"):
        """
        Initialize the UoM glucose converter.

        Args:
            output_fields: List of standard field names to include in output.
                          Uses standard field names (e.g., 'timestamp', 'glucose_value_mgdl').
                          If None, uses default fields.
            database_type: Database type identifier (default: "uom")
        """
        super().__init__(output_fields, database_type)
        # Load database schema
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["glucose"]

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

        # Check for UoM glucose format headers
        uom_glucose_headers = ["bg_ts", "value"]

        return all(header in clean_headers for header in uom_glucose_headers)

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

    def _convert_glucose_value(self, value_str: str) -> Optional[str]:
        """
        Convert glucose value from mmol/L to mg/dL.
        Business logic: conversion factor is 18.0 (mmol/L * 18 = mg/dL)

        Args:
            value_str: Glucose value string in mmol/L

        Returns:
            Glucose value string in mg/dL or None if conversion fails
        """
        if not value_str or value_str.strip() == "":
            return None

        try:
            value_mmol = float(value_str.strip())
            # Business logic: convert mmol/L to mg/dL (factor = 18.0)
            value_mgdl = value_mmol * 18.0
            return str(round(value_mgdl, 1))
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

        # Convert glucose value (mmol/L to mg/dL)
        glucose_value = self._convert_glucose_value(self._get_clean_value(row, "value"))
        if glucose_value is None:
            return None

        # Build result dictionary with standard field names
        result: Dict[str, str] = {}

        # Always add timestamp first (using standard name)
        result["timestamp"] = timestamp

        # Add event type if requested
        if "event_type" in self.output_fields_standard:
            result["event_type"] = self.converter_schema["event_type"]

        # Map glucose value (special transformation with unit conversion)
        if "glucose_value_mgdl" in self.output_fields_standard:
            result["glucose_value_mgdl"] = glucose_value

        # Map source fields to standard fields using schema
        field_mappings = self.converter_schema["field_mappings"]
        source_fields = self._get_source_fields(row)

        # Map source fields to standard fields (skip timestamp and value as they're already handled)
        for source_field, standard_field in field_mappings.items():
            if source_field in ["bg_ts", "value"]:
                continue  # Skip timestamp and value fields, already handled
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
        return "UoM Glucose Data"



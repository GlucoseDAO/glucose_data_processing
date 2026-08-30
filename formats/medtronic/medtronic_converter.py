#!/usr/bin/env python3
"""
Medtronic Guardian Connect format converter.

This module provides the converter for Medtronic Guardian Connect CGM data exports.
Handles semicolon-delimited CSV files with European decimal notation (comma as decimal separator).
"""

import re
from datetime import datetime
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class MedtronicConverter(CSVFormatConverter):
    """Converter for Medtronic Guardian Connect format."""

    CSV_DELIMITER: str = ";"

    # Required headers to identify Medtronic format
    REQUIRED_HEADERS = [
        "Index",
        "Date",
        "Time",
        "Sensor Glucose (mg/dL)",
    ]

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "medtronic"):
        """
        Initialize the Medtronic converter.

        Args:
            output_fields: List of standard field names to include in output.
                          Uses standard field names (e.g., 'timestamp', 'glucose_value_mgdl').
                          If None, uses default fields.
            database_type: Database type identifier (default: "medtronic")
        """
        super().__init__(output_fields, database_type)
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["guardian"]

    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.

        Args:
            headers: List of column headers from the CSV file

        Returns:
            True if this converter can handle the format, False otherwise
        """
        # Clean headers (remove BOM, strip whitespace)
        clean_headers = [h.strip().lstrip("\ufeff") for h in headers if h.strip()]

        # Handle case where headers come as single semicolon-separated string
        if len(clean_headers) == 1 and ";" in clean_headers[0]:
            clean_headers = [h.strip() for h in clean_headers[0].split(";")]

        # Check for required Medtronic headers
        return all(header in clean_headers for header in self.REQUIRED_HEADERS)

    def _parse_timestamp(self, date_str: str, time_str: str) -> Optional[str]:
        """
        Parse date and time strings into ISO format timestamp.

        Args:
            date_str: Date string (e.g., "2021/07/25")
            time_str: Time string (e.g., "09:35:28")

        Returns:
            ISO format timestamp string or None if parsing fails
        """
        if not date_str or not time_str:
            return None

        date_str = date_str.strip()
        time_str = time_str.strip()

        combined = f"{date_str} {time_str}"
        formats = self.db_schema["timestamp_formats"]
        output_format = self.db_schema["timestamp_output_format"]

        for fmt in formats:
            try:
                dt = datetime.strptime(combined, fmt)
                return dt.strftime(output_format)
            except ValueError:
                continue

        return None

    def _parse_european_number(self, value: str) -> str:
        """
        Parse European-format numbers (comma as decimal separator).

        Args:
            value: Number string potentially using comma as decimal separator

        Returns:
            Number string with period as decimal separator, or empty string if invalid
        """
        if not value or not value.strip():
            return ""

        value = value.strip()

        # Replace comma with period for decimal
        value = value.replace(",", ".")

        # Try to parse as float to validate
        try:
            float(value)
            return value
        except ValueError:
            return ""

    def _parse_event_marker(self, event_marker: str) -> Dict[str, str]:
        """
        Parse Event Marker field to extract meal and insulin information.

        Event markers can contain:
        - "Meal: 60,00grams" -> carb_grams: 60.0
        - "Insulin: 27,00" -> fast_acting_insulin_u: 27.0

        Args:
            event_marker: Event marker string from Medtronic data

        Returns:
            Dictionary with extracted values (carb_grams, fast_acting_insulin_u)
        """
        result: Dict[str, str] = {}

        if not event_marker or not event_marker.strip():
            return result

        event_marker = event_marker.strip()

        # Parse meal/carbs: "Meal: 60,00grams"
        meal_match = re.search(r"Meal:\s*([\d,\.]+)\s*grams?", event_marker, re.IGNORECASE)
        if meal_match:
            carbs = self._parse_european_number(meal_match.group(1))
            if carbs:
                result["carb_grams"] = carbs

        # Parse insulin: "Insulin: 27,00"
        insulin_match = re.search(r"Insulin:\s*([\d,\.]+)", event_marker, re.IGNORECASE)
        if insulin_match:
            insulin = self._parse_european_number(insulin_match.group(1))
            if insulin:
                result["fast_acting_insulin_u"] = insulin

        return result

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.

        Medtronic data has separate Date and Time columns that need to be combined.
        Sensor Glucose and BG Reading are different sources of glucose data:
        - Sensor Glucose (mg/dL): CGM continuous readings
        - BG Reading (mg/dL): Fingerstick meter readings

        Args:
            row: Dictionary representing a single CSV row

        Returns:
            Dictionary with standard field names, filtered to requested fields
        """
        # Handle case where DictReader didn't split due to delimiter mismatch
        if len(row) == 1 and ";" in list(row.keys())[0]:
            key = list(row.keys())[0]
            val = row[key]
            headers = [h.strip().lstrip("\ufeff") for h in key.split(";")]
            values = val.split(";") if val else [""] * len(headers)
            if len(headers) == len(values):
                row = dict(zip(headers, values))

        # Parse timestamp from Date and Time columns
        date_val = self._get_clean_value(row, "Date")
        time_val = self._get_clean_value(row, "Time")
        timestamp = self._parse_timestamp(date_val, time_val)

        if not timestamp:
            return None

        # Build result dictionary with standard field names
        result: Dict[str, str] = {}

        # Always add timestamp first
        result["timestamp"] = timestamp

        # Add event_type
        if "event_type" in self.output_fields_standard:
            result["event_type"] = self.converter_schema["event_type"]

        # Get sensor glucose (primary CGM data)
        sensor_glucose = self._get_clean_value(row, "Sensor Glucose (mg/dL)")
        if sensor_glucose and "glucose_value_mgdl" in self.output_fields_standard:
            parsed_glucose = self._parse_european_number(sensor_glucose)
            if parsed_glucose:
                result["glucose_value_mgdl"] = parsed_glucose

        # Get BG reading (fingerstick) - use as glucose if no sensor glucose
        bg_reading = self._get_clean_value(row, "BG Reading (mg/dL)")
        if bg_reading and "glucose_value_mgdl" in self.output_fields_standard:
            parsed_bg = self._parse_european_number(bg_reading)
            if parsed_bg and "glucose_value_mgdl" not in result:
                result["glucose_value_mgdl"] = parsed_bg

        # Get basal rate
        basal_rate = self._get_clean_value(row, "Basal Rate (U/h)")
        if basal_rate and "basal_rate" in self.output_fields_standard:
            parsed_basal = self._parse_european_number(basal_rate)
            if parsed_basal:
                result["basal_rate"] = parsed_basal

        # Get bolus (fast-acting insulin)
        bolus = self._get_clean_value(row, "Bolus Volume Delivered (U)")
        if bolus and "fast_acting_insulin_u" in self.output_fields_standard:
            parsed_bolus = self._parse_european_number(bolus)
            if parsed_bolus:
                result["fast_acting_insulin_u"] = parsed_bolus

        # Get BWZ Carb Input
        carb_input = self._get_clean_value(row, "BWZ Carb Input (grams)")
        if carb_input and "carb_grams" in self.output_fields_standard:
            parsed_carbs = self._parse_european_number(carb_input)
            if parsed_carbs:
                result["carb_grams"] = parsed_carbs

        # Parse Event Marker for additional meal/insulin data
        event_marker = self._get_clean_value(row, "Event Marker")
        if event_marker:
            marker_data = self._parse_event_marker(event_marker)

            # Add carbs from event marker if not already set
            if "carb_grams" in marker_data and "carb_grams" not in result:
                if "carb_grams" in self.output_fields_standard:
                    result["carb_grams"] = marker_data["carb_grams"]

            # Add insulin from event marker if not already set
            if "fast_acting_insulin_u" in marker_data and "fast_acting_insulin_u" not in result:
                if "fast_acting_insulin_u" in self.output_fields_standard:
                    result["fast_acting_insulin_u"] = marker_data["fast_acting_insulin_u"]

        # Skip rows that have no meaningful data (no glucose, no insulin, no carbs)
        has_data = any(
            key in result and result[key]
            for key in ["glucose_value_mgdl", "fast_acting_insulin_u", "carb_grams", "basal_rate"]
        )
        if not has_data:
            return None

        # Filter and convert to display names
        return self._filter_output(result)

    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.

        Returns:
            String name of the format
        """
        return "Medtronic Guardian Connect"

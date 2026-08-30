#!/usr/bin/env python3
"""
HUPA dataset format converter.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from formats.base_converter import CSVFormatConverter


class HupaConverter(CSVFormatConverter):
    """Converter for HUPA dataset format."""
    
    CSV_DELIMITER: str = ";"

    def __init__(self, output_fields: Optional[List[str]] = None, database_type: str = "hupa"):
        """
        Initialize the Hupa converter.

        Args:
            output_fields: List of standard field names to include in output.
            database_type: Database type identifier (default: "hupa")
        """
        super().__init__(output_fields, database_type)
        self.db_schema = self._load_schema(self.database_type)
        self.converter_schema = self.db_schema["converters"]["hupa"]

    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.
        """
        # Clean headers
        clean_headers = [h.strip().lstrip("\ufeff") for h in headers if h.strip()]
        
        # HUPA often uses ';' as separator. If headers is a single string with ';', split it.
        if len(clean_headers) == 1 and ';' in clean_headers[0]:
            clean_headers = [h.strip() for h in clean_headers[0].split(';')]

        hupa_headers = ["time", "glucose", "calories", "heart_rate", "steps"]
        return all(header in clean_headers for header in hupa_headers)

    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """Parse timestamp using schema-defined formats."""
        if not timestamp_str or timestamp_str.strip() == "":
            return None

        timestamp_str = timestamp_str.strip()
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
        """Convert a single row to the configured output format."""
        # Handle cases where DictReader didn't split the row because of ';' delimiter
        if len(row) == 1 and ';' in list(row.keys())[0]:
            key = list(row.keys())[0]
            val = row[key]
            headers = [h.strip().lstrip("\ufeff") for h in key.split(';')]
            values = val.split(';')
            if len(headers) == len(values):
                row = dict(zip(headers, values))

        # Parse timestamp
        timestamp_field = self.converter_schema["timestamp_field"]
        timestamp = self._parse_timestamp(self._get_clean_value(row, timestamp_field))
        if not timestamp:
            return None

        # Build result
        result: Dict[str, str] = {}
        result["timestamp"] = timestamp

        if "event_type" in self.output_fields_standard:
            result["event_type"] = self.converter_schema["event_type"]

        # Map source fields to standard fields using schema
        field_mappings = self.converter_schema["field_mappings"]
        source_fields = self._get_source_fields(row)

        for source_field, standard_field in field_mappings.items():
            if source_field == timestamp_field:
                continue
            if source_field in source_fields and standard_field in self.output_fields_standard:
                result[standard_field] = self._get_clean_value(row, source_field)

        return self._filter_output(result)

    def get_format_name(self) -> str:
        return "HUPA Dataset"

#!/usr/bin/env python3
"""
University of Manchester (UoM) format converter.

This module provides the converter for UoM T1D dataset format, which includes
glucose, insulin (basal/bolus), and nutrition data from multiple participants.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class UoMConverter(CSVFormatConverter):
    """Converter for University of Manchester T1D dataset format."""

    def __init__(self):
        """Initialize the UoM converter."""
        self.participant_data = {}  # Store data by participant ID
        self.current_participant_id = None
        self.data_type = None  # 'glucose', 'basal', 'bolus', 'nutrition'

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

        # Check for UoM format headers
        uom_glucose_headers = ["bg_ts", "value"]
        uom_basal_headers = ["basal_ts", "basal_dose", "insulin_kind"]
        uom_bolus_headers = ["bolus_ts", "bolus_dose"]
        uom_nutrition_headers = ["meal_ts", "meal_type", "meal_tag", "carbs_g", "prot_g", "fat_g", "fibre_g"]

        # Check if headers match any UoM format
        if all(header in clean_headers for header in uom_glucose_headers):
            return True
        if all(header in clean_headers for header in uom_basal_headers):
            return True
        if all(header in clean_headers for header in uom_bolus_headers):
            return True
        if all(header in clean_headers for header in uom_nutrition_headers):
            return True

        return False

    def _parse_participant_id_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Extract participant ID from filename.

        Args:
            file_path: Path to the CSV file

        Returns:
            Participant ID string or None if not found
        """
        filename = file_path.stem  # Get filename without extension

        # UoM format: UoMGlucose2301.csv -> participant ID: 2301
        if filename.startswith("UoM"):
            # Extract the ID part after the data type
            parts = filename[3:]  # Remove 'UoM' prefix
            # Find where the ID starts (after the data type name)
            for i, char in enumerate(parts):
                if char.isdigit():
                    return parts[i:]

        return None

    def _determine_data_type_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Determine data type from filename.

        Args:
            file_path: Path to the CSV file

        Returns:
            Data type string ('glucose', 'basal', 'bolus', 'nutrition') or None
        """
        filename = file_path.stem.lower()

        if "glucose" in filename:
            return "glucose"
        if "basal" in filename:
            return "basal"
        if "bolus" in filename:
            return "bolus"
        if "nutrition" in filename:
            return "nutrition"

        return None

    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """
        Parse UoM timestamp format to standard format.

        Args:
            timestamp_str: Timestamp string in UoM format (MM/DD/YYYY HH:MM or MM/DD/YYYY HH:MM:SS)

        Returns:
            Timestamp string in standard format (YYYY-MM-DDTHH:MM:SS) or None if parsing fails
        """
        if not timestamp_str or timestamp_str.strip() == "":
            return None

        timestamp_str = timestamp_str.strip()

        # Try different UoM timestamp formats
        formats = [
            "%m/%d/%Y %H:%M:%S",  # MM/DD/YYYY HH:MM:SS
            "%m/%d/%Y %H:%M",  # MM/DD/YYYY HH:MM
            "%d/%m/%Y %H:%M:%S",  # DD/MM/YYYY HH:MM:SS
            "%d/%m/%Y %H:%M",  # DD/MM/YYYY HH:MM
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue

        return None

    def _convert_glucose_value(self, value_str: str) -> Optional[str]:
        """
        Convert glucose value from mmol/L to mg/dL.

        Args:
            value_str: Glucose value string in mmol/L

        Returns:
            Glucose value string in mg/dL or None if conversion fails
        """
        if not value_str or value_str.strip() == "":
            return None

        try:
            value_mmol = float(value_str.strip())
            value_mgdl = value_mmol * 18  # Convert mmol/L to mg/dL
            return str(round(value_mgdl, 1))
        except (ValueError, TypeError):
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
        Convert a single row to the standard format.

        Args:
            row: Dictionary representing a single CSV row

        Returns:
            Dictionary in standard format, or None if row should be skipped
        """
        # Determine data type and participant ID from context
        if not self.data_type or not self.current_participant_id:
            return None

        # Initialize result with standard fields
        result = {
            "Timestamp (YYYY-MM-DDThh:mm:ss)": "",
            "Event Type": "",
            "Glucose Value (mg/dL)": "",
            "Fast-Acting Insulin Value (u)": "",
            "Long-Acting Insulin Value (u)": "",
            "Carb Value (grams)": "",
        }

        # Helper function to get value with BOM handling
        def get_clean_value(key: str) -> str:
            value = row.get(key, "")
            if not value:
                # Try with BOM prefix
                bom_key = "\ufeff" + key
                value = row.get(bom_key, "")
            return value

        # Process based on data type
        if self.data_type == "glucose":
            # Convert glucose data
            timestamp = self._parse_timestamp(get_clean_value("bg_ts"))
            if not timestamp:
                return None

            glucose_value = self._convert_glucose_value(get_clean_value("value"))
            if glucose_value is None:
                return None

            result["Timestamp (YYYY-MM-DDThh:mm:ss)"] = timestamp
            result["Event Type"] = "EGV"  # Estimated Glucose Value
            result["Glucose Value (mg/dL)"] = glucose_value

        elif self.data_type == "basal":
            # Convert basal insulin data (basal = long-acting)
            timestamp = self._parse_timestamp(get_clean_value("basal_ts"))
            if not timestamp:
                return None

            insulin_value = self._convert_insulin_value(get_clean_value("basal_dose"))
            if insulin_value is None:
                return None

            result["Timestamp (YYYY-MM-DDThh:mm:ss)"] = timestamp
            result["Event Type"] = "Basal"
            result["Long-Acting Insulin Value (u)"] = insulin_value

        elif self.data_type == "bolus":
            # Convert bolus insulin data (bolus = fast-acting)
            timestamp = self._parse_timestamp(get_clean_value("bolus_ts"))
            if not timestamp:
                return None

            insulin_value = self._convert_insulin_value(get_clean_value("bolus_dose"))
            if insulin_value is None:
                return None

            result["Timestamp (YYYY-MM-DDThh:mm:ss)"] = timestamp
            result["Event Type"] = "Bolus"
            result["Fast-Acting Insulin Value (u)"] = insulin_value

        elif self.data_type == "nutrition":
            # Convert nutrition data
            timestamp = self._parse_timestamp(get_clean_value("meal_ts"))
            if not timestamp:
                return None

            carb_value = self._convert_carb_value(get_clean_value("carbs_g"))
            if carb_value is None:
                # Carb is the only meaningful field for the standard pipeline; skip if missing
                return None

            result["Timestamp (YYYY-MM-DDThh:mm:ss)"] = timestamp
            result["Event Type"] = "Meal"
            result["Carb Value (grams)"] = carb_value

        else:
            return None

        return result

    def set_context(self, file_path: Path) -> None:
        """
        Set context for conversion based on file path.

        Args:
            file_path: Path to current file being processed
        """
        self.current_participant_id = self._parse_participant_id_from_filename(file_path)
        self.data_type = self._determine_data_type_from_filename(file_path)

    def get_format_name(self) -> str:
        """Get the name of the format this converter handles."""
        return "University of Manchester T1D Dataset"



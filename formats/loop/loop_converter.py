#!/usr/bin/env python3
"""
Loop format converter.

This module provides the converter for Loop dataset format.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter

MMOL_TO_MGDL: float = 18.018


class LoopConverter(CSVFormatConverter):
    """Converter for Loop dataset format."""

    CSV_DELIMITER: str = "|"

    def __init__(self, output_fields: Optional[List[str]] = None):
        """Initialize the Loop converter."""
        super().__init__(output_fields)
        self.data_type = None  # 'cgm', 'basal', 'bolus', 'food'

    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.

        Args:
            headers: List of column headers from the CSV file

        Returns:
            True if this converter can handle the format, False otherwise
        """
        # Clean headers
        clean_headers = [h.strip().lstrip("\ufeff") for h in headers if h.strip()]

        # Check for Loop format headers
        loop_cgm_headers = ["PtID", "UTCDtTm", "CGMVal"]
        loop_basal_headers = ["PtID", "UTCDtTm", "Rate"]
        loop_bolus_headers = ["PtID", "UTCDtTm", "BolusType", "Normal"]
        loop_food_headers = ["PtID", "UTCDtTm", "CarbsNet"]

        if all(header in clean_headers for header in loop_cgm_headers):
            return True
        if all(header in clean_headers for header in loop_basal_headers):
            return True
        if all(header in clean_headers for header in loop_bolus_headers):
            return True
        if all(header in clean_headers for header in loop_food_headers):
            return True

        return False

    def _determine_data_type_from_filename(self, file_path: Path) -> Optional[str]:
        """Determine data type from filename."""
        filename = file_path.name.lower()
        if "cgm" in filename:
            return "cgm"
        if "basal" in filename:
            return "basal"
        if "bolus" in filename:
            return "bolus"
        if "food" in filename:
            return "food"
        return None

    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the standard format.
        """
        # Initialize result with standard fields
        result = {
            "timestamp": "",
            "event_type": "",
            "glucose_value_mgdl": "",
            "fast_acting_insulin_u": "",
            "long_acting_insulin_u": "",
            "carb_grams": "",
            "basal_rate": "",
            "user_id": row.get("PtID", "")
        }

        utc_time = row.get("UTCDtTm")
        if not utc_time or utc_time.strip() == "":
            return None
        
        # Convert UTCDtTm to standard format (YYYY-MM-DDTHH:MM:SS)
        try:
            # Loop UTCDtTm is usually "YYYY-MM-DD HH:MM:SS"
            dt = datetime.strptime(utc_time.strip(), "%Y-%m-%d %H:%M:%S")
            result["timestamp"] = dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None

        if self.data_type == "cgm":
            glucose = row.get("CGMVal")
            if not glucose or glucose.strip() == "":
                return None
            units = row.get("Units", "").strip().lower()
            try:
                glucose_float = float(glucose.strip())
                if units == "mmol/l":
                    glucose_float *= MMOL_TO_MGDL
                result["glucose_value_mgdl"] = str(round(glucose_float, 3))
            except ValueError:
                return None
            result["event_type"] = "EGV"
        
        elif self.data_type == "basal":
            rate = row.get("Rate")
            if not rate or rate.strip() == "":
                return None
            result["event_type"] = "Basal"
            result["basal_rate"] = rate
            
        elif self.data_type == "bolus":
            normal = row.get("Normal")
            if not normal or normal.strip() == "":
                return None
            result["event_type"] = "Bolus"
            result["fast_acting_insulin_u"] = normal
            
        elif self.data_type == "food":
            carbs = row.get("CarbsNet")
            if not carbs or carbs.strip() == "":
                return None
            result["event_type"] = "Meal"
            result["carb_grams"] = carbs
        else:
            return None

        return result

    def set_context(self, file_path: Path) -> None:
        """Set context for conversion."""
        self.data_type = self._determine_data_type_from_filename(file_path)

    def get_format_name(self) -> str:
        """Get the name of the format."""
        return "Loop Dataset"

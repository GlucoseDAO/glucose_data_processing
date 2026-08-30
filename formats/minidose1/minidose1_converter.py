#!/usr/bin/env python3
"""
MiniDose1 format converter.

This module provides the converter for MiniDose1 clinical trial dataset format.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from formats.base_converter import CSVFormatConverter


class Minidose1Converter(CSVFormatConverter):
    """Converter for MiniDose1 dataset format."""

    CSV_DELIMITER: str = "|"
    
    # Reference enrollment date for calculating absolute timestamps
    REFERENCE_ENROLLMENT_DATE = datetime(2020, 1, 1)

    def __init__(self, output_fields: Optional[List[str]] = None):
        """Initialize the MiniDose1 converter."""
        super().__init__(output_fields)
        self.current_participant_id = None
        self.data_type = None  # 'cgm', 'bgm', 'pump'

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

        # Check for MiniDose1 format headers
        minidose_cgm_headers = ["PtID", "ICGMDeviceType", "DeviceDtDaysFromEnroll", "DeviceTm", "Glucose"]
        minidose_bgm_headers = ["PtID", "IBGMDeviceType", "DeviceDtDaysFromEnroll", "DeviceTm", "Glucose"]
        minidose_pump_headers = ["PtID", "DeviceType", "DeviceDtDaysFromEnroll", "DeviceTm", "BasalRate", "BolusVolDeliv"]

        if all(header in clean_headers for header in minidose_cgm_headers):
            return True
        if all(header in clean_headers for header in minidose_bgm_headers):
            return True
        if all(header in clean_headers for header in minidose_pump_headers):
            return True

        return False

    def _parse_participant_id_from_row(self, row: Dict[str, str]) -> Optional[str]:
        """Extract participant ID from row."""
        return row.get("PtID")

    def _determine_data_type_from_filename(self, file_path: Path) -> Optional[str]:
        """Determine data type from filename."""
        filename = file_path.name.lower()
        if "cgm" in filename:
            return "cgm"
        if "bgm" in filename:
            return "bgm"
        if "pump" in filename:
            return "pump"
        return None

    def _parse_timestamp(self, days_str: str, time_str: str) -> Optional[str]:
        """
        Convert MiniDose1 relative time to absolute timestamp.
        """
        if not days_str or not time_str:
            return None
            
        try:
            days = int(days_str)
            # time_str is HH:MM:SS
            time_parts = time_str.split(':')
            if len(time_parts) < 2:
                return None
            
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            second = int(time_parts[2]) if len(time_parts) > 2 else 0
            
            dt = self.REFERENCE_ENROLLMENT_DATE + timedelta(days=days, hours=hour, minutes=minute, seconds=second)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError):
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
            "user_id": row.get("PtID", "")
        }

        days_from_enroll = row.get("DeviceDtDaysFromEnroll")
        time_str = row.get("DeviceTm")
        
        timestamp = self._parse_timestamp(days_from_enroll, time_str)
        if not timestamp:
            return None
        
        result["timestamp"] = timestamp

        if self.data_type == "cgm":
            glucose = row.get("Glucose")
            if not glucose:
                return None
            result["event_type"] = "EGV"
            result["glucose_value_mgdl"] = glucose
        
        elif self.data_type == "bgm":
            glucose = row.get("Glucose")
            if not glucose:
                return None
            result["event_type"] = "BGM"
            result["glucose_value_mgdl"] = glucose
            
        elif self.data_type == "pump":
            # Bolus
            bolus = row.get("BolusVolDeliv")
            if bolus and bolus.strip() and float(bolus) > 0:
                result["event_type"] = "Bolus"
                result["fast_acting_insulin_u"] = bolus
            
            # Basal
            basal = row.get("BasalRate")
            if basal and basal.strip() and float(basal) > 0:
                if result["event_type"]: # If already set by bolus, we might have a problem with 1 row -> 2 events
                    # In this pipeline, we usually prefer one event per row.
                    # If both are present, we could return bolus and handle basal separately, 
                    # but let's see how others do it.
                    pass
                result["event_type"] = result["event_type"] or "Basal"
                result["long_acting_insulin_u"] = basal
                
            # Carbs
            carbs = row.get("WizardCarbs")
            if carbs and carbs.strip() and float(carbs) > 0:
                result["event_type"] = result["event_type"] or "Meal"
                result["carb_grams"] = carbs

            if not result["event_type"]:
                return None
        else:
            return None

        return result

    def set_context(self, file_path: Path) -> None:
        """Set context for conversion."""
        self.data_type = self._determine_data_type_from_filename(file_path)

    def get_format_name(self) -> str:
        """Get the name of the format."""
        return "MiniDose1 Dataset"

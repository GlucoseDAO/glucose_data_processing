#!/usr/bin/env python3
"""
Database detector for identifying database types.

This module provides the database detection system that automatically identifies
database types and returns the appropriate database converter.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import zipfile

from formats.database_converters import DatabaseConverter


class DatabaseDetector:
    """Detects database type and returns appropriate database converter."""
    
    def __init__(self):
        """Initialize the database detector with available converters."""
        # Use lazy imports inside the methods that need them to avoid circular imports
        self.database_converters = {}
    
    def _get_converter_class(self, database_type: str):
        """Lazy load converter classes to avoid circular imports."""
        if database_type == 'dexcom':
            from formats.dexcom.dexcom_database_converter import DexcomDatabaseConverter
            return DexcomDatabaseConverter
        elif database_type == 'libre3':
            from formats.libre3.libre3_database_converter import Libre3DatabaseConverter
            return Libre3DatabaseConverter
        elif database_type == 'uom':
            from formats.uom.uom_database_converter import UoMDatabaseConverter
            return UoMDatabaseConverter
        elif database_type == 'ai_ready':
            from formats.ai_ready.ai_ready_database_converter import AIReadyDatabaseConverter
            return AIReadyDatabaseConverter
        elif database_type == 'hupa':
            from formats.hupa.hupa_database_converter import HupaDatabaseConverter
            return HupaDatabaseConverter
        elif database_type == 'uc_ht':
            from formats.uc_ht.uc_ht_database_converter import UCHTDatabaseConverter
            return UCHTDatabaseConverter
        elif database_type == 'medtronic':
            from formats.medtronic.medtronic_database_converter import MedtronicDatabaseConverter
            return MedtronicDatabaseConverter
        elif database_type == 'minidose1':
            from formats.minidose1.minidose1_database_converter import Minidose1DatabaseConverter
            return Minidose1DatabaseConverter
        elif database_type == 'loop':
            from formats.loop.loop_database_converter import LoopDatabaseConverter
            return LoopDatabaseConverter
        return None

    def detect_database_type(self, data_folder: Union[str, Path]) -> str:
        """
        Detect the database type from the folder structure and file patterns.
        
        Args:
            data_folder: Path to the data folder to analyze
            
        Returns:
            Database type string ('dexcom', 'libre3', 'uom', 'uc_ht', 'medtronic', 'hupa', 'loop', 'minidose1', or 'unknown')
        """
        data_path = Path(data_folder)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        # AI-READI: zip-backed dataset
        if data_path.is_file() and data_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(data_path, "r") as z:
                    # Detect via required AI-READI members under /dataset/
                    has_participants = False
                    has_dexcom = False
                    for name in z.namelist():
                        # Keep checks cheap; stop early when both match.
                        if not has_participants and name.endswith("/dataset/participants.tsv"):
                            has_participants = True
                        if (not has_dexcom) and ("/dataset/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/" in name) and name.endswith("_DEX.json"):
                            has_dexcom = True
                        if has_participants and has_dexcom:
                            return "ai_ready"
            except Exception:
                return "unknown"
            return "unknown"
        
        # UC_HT: folder-based with .xlsx files
        xlsx_files = list(data_path.glob("**/*.xlsx"))
        if xlsx_files:
            # Check for UC_HT specific filenames
            uc_ht_files = ['Glucose.xlsx', 'Heart Rate.xlsx', 'Steps.xlsx', 'Carbohidrates.xlsx', 'Insulin.xlsx']
            count = 0
            for xlsx in xlsx_files:
                if xlsx.name in uc_ht_files:
                    count += 1
            if count >= 2: # At least two matching files suggests UC_HT
                return 'uc_ht'

        # Get all CSV and TXT files
        csv_files = list(data_path.glob("**/*.csv"))
        txt_files = list(data_path.glob("**/*.txt"))
        all_files = csv_files + txt_files
        
        if not all_files:
            return 'unknown'
        
        # Analyze file patterns to determine database type
        file_patterns = {
            'dexcom': 0,
            'libre3': 0,
            'uom': 0,
            'hupa': 0,
            'medtronic': 0,
            'minidose1': 0,
            'loop': 0
        }
        
        for data_file in all_files:
            filename = data_file.stem.lower()
            
            # Check for Dexcom patterns (standard format files)
            if any(pattern in filename for pattern in ['dexcom', 'g6', 'cgm']):
                file_patterns['dexcom'] += 1
            elif any(pattern in filename for pattern in ['libre', 'freestyle']):
                file_patterns['libre3'] += 1
            elif filename.startswith('uom'):
                file_patterns['uom'] += 1
            elif filename.startswith('hupa'):
                file_patterns['hupa'] += 1
            elif 'medtronic' in filename or 'zaharia' in filename:
                file_patterns['medtronic'] += 1
            elif filename.startswith('idata'):
                file_patterns['minidose1'] += 1
            elif filename.startswith('loop'):
                file_patterns['loop'] += 1
            else:
                # Check file content to determine format
                try:
                    with open(data_file, 'r', encoding='utf-8-sig') as file:  # utf-8-sig handles BOM
                        first_lines = [file.readline().strip() for _ in range(10)] # Medtronic might have header lines
                        
                        # Check for Dexcom format headers
                        for line in first_lines:
                            if 'Timestamp (YYYY-MM-DDThh:mm:ss)' in line and 'Event Type' in line:
                                file_patterns['dexcom'] += 1
                                break
                        
                        # Check for Libre3 format headers
                        for line in first_lines:
                            if 'Device Timestamp' in line and 'Historic Glucose mg/dL' in line:
                                file_patterns['libre3'] += 1
                                break
                        
                        # Check for UoM format headers
                        for line in first_lines:
                            if any(header in line for header in ['bg_ts', 'basal_ts', 'bolus_ts', 'meal_ts']):
                                file_patterns['uom'] += 1
                                break
                                
                        # Check for HUPA format headers
                        for line in first_lines:
                            if any(header in line for header in ['time;glucose;calories', 'time;glucose;heart_rate']):
                                file_patterns['hupa'] += 1
                                break

                        # Check for Medtronic format headers
                        for line in first_lines:
                            if 'Sensor Glucose (mg/dL)' in line and 'Event Marker' in line:
                                file_patterns['medtronic'] += 1
                                break
                                
                        # Check for MiniDose1 format headers
                        for line in first_lines:
                            if 'PtID|' in line and 'DeviceDtDaysFromEnroll|' in line:
                                file_patterns['minidose1'] += 1
                                break
                                
                        # Check for Loop format headers
                        for line in first_lines:
                            if 'PtID|' in line and 'UTCDtTm' in line and ('CGMVal' in line or 'Normal' in line or 'Rate' in line or 'CarbsNet' in line):
                                file_patterns['loop'] += 1
                                break
                                
                except Exception:
                    # If we can't read the file, skip it
                    continue
        
        # Determine the most likely database type
        if max(file_patterns.values()) == 0:
            # If no pattern matched, check directory name
            if 'medtronic' in str(data_folder).lower():
                return 'medtronic'
            return 'unknown'
        
        return max(file_patterns, key=file_patterns.get)
    
    def get_database_converter(self, database_type: str, config: Dict[str, Any], output_fields: Optional[List[str]] = None) -> Optional[DatabaseConverter]:
        """
        Get the appropriate database converter for the given database type.
        
        Args:
            database_type: Database type string
            config: Configuration dictionary
            output_fields: List of field names to include in output
            
        Returns:
            Database converter instance or None if type not supported
        """
        converter_class = self._get_converter_class(database_type)
        if converter_class:
            return converter_class(config, output_fields=output_fields, database_type=database_type)
        else:
            return None
    
    def get_supported_database_types(self) -> list[str]:
        """
        Get list of supported database types.
        
        Returns:
            List of database type names supported by this detector
        """
        return ['dexcom', 'libre3', 'uom', 'ai_ready', 'hupa', 'uc_ht', 'medtronic', 'minidose1', 'loop']

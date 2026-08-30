#!/usr/bin/env python3
"""
Base converter class for CSV format converters.

This module provides the abstract base class that all format converters must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set
import json
import yaml
from pathlib import Path


class CSVFormatConverter(ABC):
    """Abstract base class for CSV format converters."""
    
    # Default CSV delimiter for this format (can be overridden by subclasses)
    CSV_DELIMITER: str = ","
    
    # Default output fields (STANDARD NAMES).
    # These are fallback defaults if config is not provided.
    # NOTE: We intentionally use standard names so the pipeline can support arbitrary fields
    # without needing display-name mappings for every new field.
    OUTPUT_FIELDS: List[str] = [
        'timestamp',
        'event_type',
        'glucose_value_mgdl',
        'fast_acting_insulin_u',
        'long_acting_insulin_u',
        'carb_grams',
    ]
    
    # Field to display name mapping (standard_name -> display_name)
    # These are fallback defaults if config is not provided
    FIELD_TO_DISPLAY_NAME_MAP: Dict[str, str] = {
        "timestamp": "Timestamp (YYYY-MM-DDThh:mm:ss)",
        "event_type": "Event Type",
        "glucose_value_mgdl": "Glucose Value (mg/dL)",
        "fast_acting_insulin_u": "Fast-Acting Insulin Value (u)",
        "long_acting_insulin_u": "Long-Acting Insulin Value (u)",
        "carb_grams": "Carb Value (grams)",
        "dataset_name": "Dataset Name"
    }
    
    # Class-level config-based field mappings (initialized from config)
    _config_output_fields: Optional[List[str]] = None
    _config_field_to_display_name_map: Optional[Dict[str, str]] = None
    
    _schema_cache: Optional[Dict] = None
    
    @classmethod
    def initialize_from_config(cls, config: Optional[Dict] = None) -> None:
        """
        Initialize field mappings from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'output_fields' and 'field_to_display_name_map' keys.
                   If None or keys missing, uses default hardcoded values.
        """
        if config is None:
            cls._config_output_fields = None
            cls._config_field_to_display_name_map = None
            return
        
        # Load output_fields from config (backward compatibility: also check old name)
        if 'output_fields' in config and config['output_fields']:
            cls._config_output_fields = config['output_fields'].copy()
        elif 'default_output_fields' in config and config['default_output_fields']:
            # Backward compatibility: convert display names to standard names
            old_fields = config['default_output_fields'].copy()
            old_map = config.get('standard_fields', config.get('field_to_display_name_map', {}))
            display_to_standard = {v: k for k, v in old_map.items()}
            cls._config_output_fields = [display_to_standard.get(f, f) for f in old_fields]
        else:
            cls._config_output_fields = None
        
        # Load field_to_display_name_map from config (backward compatibility: also check old name)
        if 'field_to_display_name_map' in config and config['field_to_display_name_map']:
            cls._config_field_to_display_name_map = config['field_to_display_name_map'].copy()
        elif 'standard_fields' in config and config['standard_fields']:
            cls._config_field_to_display_name_map = config['standard_fields'].copy()
        else:
            cls._config_field_to_display_name_map = None
    
    @classmethod
    def get_output_fields(cls) -> List[str]:
        """
        Get output fields, using config if available, otherwise class defaults.
        
        Returns:
            List of output field names (standard names)
        """
        if cls._config_output_fields is not None:
            return cls._config_output_fields.copy()
        return cls.OUTPUT_FIELDS.copy()
    
    @classmethod
    def get_field_to_display_name_map(cls) -> Dict[str, str]:
        """
        Get field to display name mapping, using config if available, otherwise class defaults.
        
        Returns:
            Dictionary mapping standard field names to display names
        """
        if cls._config_field_to_display_name_map is not None:
            return cls._config_field_to_display_name_map.copy()
        return cls.FIELD_TO_DISPLAY_NAME_MAP.copy()
    
    @classmethod
    def get_display_name(cls, standard_name: str) -> str:
        """
        Get display name for a standard field name.
        If field is not in mapping, returns the standard name as-is.
        
        Args:
            standard_name: Standard field name (e.g., 'timestamp', 'heart_rate')
            
        Returns:
            Display name if mapping exists, otherwise standard name
        """
        field_map = cls.get_field_to_display_name_map()
        return field_map.get(standard_name, standard_name)
    
    def __init__(self, output_fields: Optional[List[str]] = None, database_type: Optional[str] = None):
        """
        Initialize the converter with optional output fields configuration.
        
        Args:
            output_fields: List of standard field names to include in output. 
                          Uses standard field names (e.g., 'timestamp', 'glucose_value_mgdl').
                          If None, uses get_default_output_fields() (which uses config if available).
                          Timestamp is always included.
            database_type: Optional database type identifier.
        """
        self.database_type = database_type
        if output_fields is None:
            # Get output fields from config or defaults (already standard names)
            self.output_fields_standard: Set[str] = set(self.get_output_fields())
        else:
            # Use provided standard field names
            self.output_fields_standard = set(output_fields)
        
        # Always include timestamp
        self.output_fields_standard.add('timestamp')
    
    @classmethod
    def _load_schema(cls, database_type_or_file: str) -> Dict:
        """
        Load conversion schema from YAML file.
        
        Args:
            database_type_or_file: Database type (e.g. 'uom') or schema filename (e.g., 'uom_schema.yaml')
            
        Returns:
            Dictionary containing conversion schema
        """
        if database_type_or_file.endswith('.yaml'):
            schema_file = database_type_or_file
        else:
            from processing.core.config import get_schema_file
            schema_file = get_schema_file(database_type_or_file)
            
        schema_path = Path(__file__).parent.parent / "formats" / schema_file
        with open(schema_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_standard_field_name(self, standard_name: str) -> str:
        """
        Return the standard field name as-is (for flexible field approach).
        
        Args:
            standard_name: Standard field name (e.g., 'timestamp', 'glucose_value_mgdl')
            
        Returns:
            The same standard field name (e.g., 'timestamp', 'glucose_value_mgdl')
        """
        # Return standard name directly - no conversion to display names
        # This supports flexible field approach where arbitrary fields can be added
        return standard_name
    
    def _filter_output(self, result: Dict[str, str]) -> Dict[str, str]:
        """
        Filter output dictionary to only include requested fields.
        
        Args:
            result: Dictionary with standard field names as keys
            
        Returns:
            Dictionary with display field names, filtered to requested fields
        """
        filtered = {}
        for standard_name, value in result.items():
            if standard_name in self.output_fields_standard:
                display_name = self._get_standard_field_name(standard_name)
                filtered[display_name] = value
        return filtered
    
    def _get_clean_value(self, row: Dict[str, str], key: str) -> Optional[str]:
        """
        Get value from row with BOM handling.
        
        Args:
            row: Dictionary representing a CSV row
            key: Field name to retrieve
            
        Returns:
            Cleaned value string or None if empty
        """
        value = row.get(key)
        if value is None or value.strip() == '':
            # Try with BOM prefix
            bom_key = '\ufeff' + key
            value = row.get(bom_key)
        
        if value is None or value.strip() == '':
            return None
            
        return value.strip()
    
    def _get_source_fields(self, row: Dict[str, str]) -> set:
        """
        Get all source field names from row, normalizing BOM keys.
        
        Args:
            row: Dictionary representing a CSV row
            
        Returns:
            Set of source field names
        """
        source_fields = set()
        for key in row.keys():
            if key.startswith('\ufeff'):
                source_fields.add(key[1:])  # Remove BOM
            else:
                source_fields.add(key)
        return source_fields

    def get_csv_delimiter(self) -> str:
        """
        Get the expected delimiter for this CSV format.

        Returns:
            Single-character delimiter (default ",")
        """
        delimiter = getattr(self, "CSV_DELIMITER", ",")
        return delimiter if isinstance(delimiter, str) and len(delimiter) == 1 else ","
    
    @abstractmethod
    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.
        
        Args:
            headers: List of column headers from the CSV file
            
        Returns:
            True if this converter can handle the format, False otherwise
        """
        pass
    
    @abstractmethod
    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.
        
        Args:
            row: Dictionary representing a single CSV row
            
        Returns:
            Dictionary with requested fields, or None if row should be skipped
        """
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.
        
        Returns:
            String name of the format
        """
        pass

"""Configuration utilities for glucose data processing."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from processing.core.fields import StandardFieldNames

def get_schema_file(database_type: str) -> str:
    """
    Get the schema filename for a given database type.
    """
    schema_files = {
        'uom': 'uom_schema.yaml',
        'dexcom': 'dexcom_schema.yaml',
        'libre3': 'freestyle_libre3_schema.yaml',
        'freestyle_libre3': 'freestyle_libre3_schema.yaml',
        'ai_ready': 'ai_ready_schema.yaml',
        'hupa': 'hupa_schema.yaml',
        'loop': 'loop_schema.yaml',
        'medtronic': 'medtronic_schema.yaml',
        'minidose1': 'minidose1_schema.yaml',
        'uc_ht': 'uc_ht_schema.yaml',
    }
    return schema_files.get(database_type, f"{database_type}_schema.yaml")

def load_schema(database_type: str) -> Optional[Dict[str, Any]]:
    """
    Find, verify existence, and load the schema file for a given database type.
    
    Args:
        database_type: Database type string
        
    Returns:
        Dictionary containing schema content if found and valid, else None.
    """
    schema_file = get_schema_file(database_type)
    root_dir = Path(__file__).parent.parent.parent
    schema_path = root_dir / 'formats' / schema_file
    
    if not schema_path.exists():
        return None
        
    with open(schema_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_schema_field(database_type: str, field_name: str) -> Any:
    """
    Get a specific field value from the schema file.
    
    Args:
        database_type: Database type string
        field_name: Name of the field to retrieve from the schema
        
    Returns:
        The value of the field if present, else None.
    """
    schema = load_schema(database_type)
    if schema is None:
        return None
        
    return schema.get(field_name)

def extract_field_categories(database_type: str) -> Dict[str, Any]:
    """
    Extract field categories and settings from schema file.
    
    Args:
        database_type: Database type (e.g., 'uom', 'dexcom', 'freestyle_libre3')
        
    Returns:
        Dictionary with categories ('continuous', 'occasional', 'service') 
        and settings (e.g., 'remove_after_calibration')
    """
    schema = load_schema(database_type)
    
    if schema is None:
        # Return default with only glucose as continuous
        return {
            'continuous': [StandardFieldNames.GLUCOSE_VALUE],
            'occasional': [],
            'service': [],
            'remove_after_calibration': True
        }
    
    # Get field_categories from schema
    field_categories = schema.get('field_categories', {})
    
    # Build result dictionary using standard field names directly
    result = {
        'continuous': [],
        'occasional': [],
        'service': [],
        'remove_after_calibration': schema.get('remove_after_calibration', True),
        'fill_during_interpolation': schema.get('fill_during_interpolation', [])
    }
    
    for standard_name, category in field_categories.items():
        if category in result:
            result[category].append(standard_name)
    
    # Always ensure glucose is in continuous (if it exists)
    glucose_col = StandardFieldNames.GLUCOSE_VALUE
    if glucose_col not in result['continuous']:
        result['continuous'].append(glucose_col)
    
    return result


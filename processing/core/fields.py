"""Standard field names and constants for glucose data processing."""

from formats.base_converter import CSVFormatConverter

# Constants for common field names and literal values
INTERPOLATED_EVENT_TYPE = 'Interpolated'
INSERTED_EVENT_TYPE = 'Inserted'

class StandardFieldNames:
    """
    Standard field names for flexible field approach.
    Uses universal standard names (not display names) to support arbitrary fields.
    """
    
    # Core standard field names that the preprocessor knows about
    TIMESTAMP = 'timestamp'
    EVENT_TYPE = 'event_type'
    GLUCOSE_VALUE = 'glucose_value_mgdl'
    FAST_ACTING_INSULIN = 'fast_acting_insulin_u'
    LONG_ACTING_INSULIN = 'long_acting_insulin_u'
    CARB_VALUE = 'carb_grams'
    USER_ID = 'user_id'
    SEQUENCE_ID = 'sequence_id'
    DATASET_NAME = 'dataset_name'
        
    def __init__(self) -> None:
        """Initialize standard field names."""
        # Get known standard fields from CSVFormatConverter
        self._known_fields = set(CSVFormatConverter.get_field_to_display_name_map().keys())
    
    def is_known_field(self, standard_name: str) -> bool:
        """
        Check if a standard field name is in the known fields set.
        
        Args:
            standard_name: Standard field name to check
            
        Returns:
            True if field is known
        """
        return standard_name in self._known_fields


#!/usr/bin/env python3
"""
Formats package for glucose data preprocessing.

This package provides format detection and conversion capabilities for various
glucose monitoring device CSV formats.

Available converters:
- DexcomG6Converter: For Dexcom G6 format (standard format)
- FreeStyleLibre3Converter: For FreeStyle Libre 3 format

Usage:
    from formats import CSVFormatDetector
    
    detector = CSVFormatDetector()
    converter = detector.detect_format(file_path)
    if converter:
        # Process file with detected converter
        pass
"""

from formats.base_converter import CSVFormatConverter
from formats.database_detector import DatabaseDetector
from formats.dexcom.dexcom_g6_converter import DexcomG6Converter
from formats.format_detector import CSVFormatDetector
from formats.libre3.freestyle_libre3_converter import FreeStyleLibre3Converter

__all__ = [
    'CSVFormatConverter',
    'DexcomG6Converter', 
    'FreeStyleLibre3Converter',
    'CSVFormatDetector',
    'DatabaseDetector'
]

#!/usr/bin/env python3
"""
FreeStyle Libre 3 database converter.

This module provides the converter for FreeStyle Libre 3 databases (mono-user).
"""

from typing import Any, Dict

import polars as pl
from loguru import logger

from formats.database_converters import MonoUserDatabaseConverter


class Libre3DatabaseConverter(MonoUserDatabaseConverter):
    """Converter for FreeStyle Libre 3 databases."""

    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Libre3-specific processing (currently none, but can be extended).

        Args:
            df: DataFrame to process

        Returns:
            Processed DataFrame
        """
        # Libre3 doesn't need High/Low replacement or calibration removal
        # as it already provides numeric glucose values
        logger.info("Applying Libre3-specific processing: None required")
        return df

    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "FreeStyle Libre 3 Database"



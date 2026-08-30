#!/usr/bin/env python3
"""
MiniDose1 database converter.

This module provides the converter for MiniDose1 multi-user clinical trial dataset.
"""

from pathlib import Path
from typing import Optional

from formats.database_converters import MonoUserDatabaseConverter


class Minidose1DatabaseConverter(MonoUserDatabaseConverter):
    """Converter for MiniDose1 clinical trial database."""

    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "MiniDose1 Clinical Trial Database"

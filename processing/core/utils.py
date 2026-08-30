#!/usr/bin/env python3
"""Utility functions for glucose data processing."""

from datetime import datetime
from typing import Optional

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse timestamp string into datetime object.
    Supports various formats common in glucose data.
    """
    if not timestamp_str or not isinstance(timestamp_str, str) or timestamp_str.strip() == "":
        return None
    
    timestamp_str = timestamp_str.strip()
    formats = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    return None


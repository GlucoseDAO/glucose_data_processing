"""Shared Dexcom-style glucose bounds (mg/dL) for database converters."""

from __future__ import annotations

from typing import Any, Dict, Tuple

DEFAULT_LOW_GLUCOSE_MGDL: float = 39.0
DEFAULT_HIGH_GLUCOSE_MGDL: float = 401.0


def dexcom_style_bounds(config: Dict[str, Any] | None) -> Tuple[float, float]:
    """
    Read low/high glucose values from config under ``dexcom`` (same keys as Dexcom converter).

    Dexcom CSVs use these as replacements for the literal strings ``Low`` / ``High``.
    Loop (and similar) use the same numeric bounds only to **clip** parsed CGM/BGM values.
    """
    dex: Dict[str, Any] = (config or {}).get("dexcom", {})
    low = float(dex.get("low_glucose_value", DEFAULT_LOW_GLUCOSE_MGDL))
    high = float(dex.get("high_glucose_value", DEFAULT_HIGH_GLUCOSE_MGDL))
    return low, high

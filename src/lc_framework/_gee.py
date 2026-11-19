"""Shared lazy Google Earth Engine loader and optical schema constants (ADR-0005)."""

from __future__ import annotations

from typing import Any

OPTICAL_BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")


def require_ee(feature: str = "GEE workflows") -> Any:
    """Import ``ee`` lazily with an actionable installation hint if absent."""
    try:
        import ee as ee_module
    except ImportError as exc:
        raise ImportError(
            f"Google Earth Engine (`earthengine-api`) is required for {feature}. "
            "Install it with: pip install 'lc-framework[gee]'"
        ) from exc
    return ee_module

"""Shared optical band schema and CCDC temporal segmentation defaults."""

from __future__ import annotations

OPTICAL_BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")

DEFAULT_CCDC_REFLECTANCE_SCALE: float = 10000.0
DEFAULT_REFLECTANCE_MIN: float = -0.05
DEFAULT_REFLECTANCE_MAX: float = 1.0
DEFAULT_TOL_YEARS: float = 1.0

CCDC_CORE_EXPORT_BANDS: tuple[str, ...] = (
    "tStart",
    "tEnd",
    "tBreak",
    "changeProb",
    ".*_coefs",
)

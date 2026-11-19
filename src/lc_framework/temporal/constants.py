"""Shared optical band schema and CCDC temporal segmentation defaults."""

from __future__ import annotations

from lc_framework._gee import OPTICAL_BANDS

__all__ = [
    "CCDC_CORE_EXPORT_BANDS",
    "DEFAULT_CCDC_REFLECTANCE_SCALE",
    "DEFAULT_REFLECTANCE_MAX",
    "DEFAULT_REFLECTANCE_MIN",
    "DEFAULT_TOL_YEARS",
    "OPTICAL_BANDS",
]

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

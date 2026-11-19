"""Stage 1: Harmonized optical inputs, composites, and CCDC Temporal Segments (GEE)."""

from __future__ import annotations

from lc_framework.temporal.ccdc import (
    BandCase,
    CCDCModel,
    CCDCParams,
    CCDCProfile,
    ReflectanceBounds,
    SegmentMatch,
    export_ccdc_tiles,
    fill_isolated,
    get_harmonics,
    get_multi_year_stacks,
    get_synthetic_stack,
    load_ccdc_mosaic,
    nearest_segment,
    prepare_ccdc_for_export,
    run_ccdc,
    stable_in_period,
)
from lc_framework.temporal.composite import (
    CompositeReducer,
    seasonal_composite,
    to_monthly_median,
)
from lc_framework.temporal.constants import (
    CCDC_CORE_EXPORT_BANDS,
    DEFAULT_CCDC_REFLECTANCE_SCALE,
    DEFAULT_REFLECTANCE_MAX,
    DEFAULT_REFLECTANCE_MIN,
    DEFAULT_TOL_YEARS,
    OPTICAL_BANDS,
)
from lc_framework.temporal.inputs_landsat import get_landsat_collection
from lc_framework.temporal.inputs_s2 import CloudScoreBand, get_s2_collection

__all__ = [
    "CCDC_CORE_EXPORT_BANDS",
    "DEFAULT_CCDC_REFLECTANCE_SCALE",
    "DEFAULT_REFLECTANCE_MAX",
    "DEFAULT_REFLECTANCE_MIN",
    "DEFAULT_TOL_YEARS",
    "OPTICAL_BANDS",
    "BandCase",
    "CCDCModel",
    "CCDCParams",
    "CCDCProfile",
    "CloudScoreBand",
    "CompositeReducer",
    "ReflectanceBounds",
    "SegmentMatch",
    "export_ccdc_tiles",
    "fill_isolated",
    "get_harmonics",
    "get_landsat_collection",
    "get_multi_year_stacks",
    "get_s2_collection",
    "get_synthetic_stack",
    "load_ccdc_mosaic",
    "nearest_segment",
    "prepare_ccdc_for_export",
    "run_ccdc",
    "seasonal_composite",
    "stable_in_period",
    "to_monthly_median",
]

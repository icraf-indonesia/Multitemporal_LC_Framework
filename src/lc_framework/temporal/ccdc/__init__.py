"""Sensor-agnostic CCDC Temporal Segment subpackage."""

from __future__ import annotations

from lc_framework.temporal.ccdc.fit import (
    load_ccdc_mosaic,
    prepare_ccdc_for_export,
    run_ccdc,
)
from lc_framework.temporal.ccdc.model import CCDCModel
from lc_framework.temporal.ccdc.reconstruct import (
    fill_isolated,
    get_harmonics,
    get_multi_year_stacks,
    get_synthetic_stack,
    nearest_segment,
)
from lc_framework.temporal.ccdc.specs import (
    BandCase,
    CCDCParams,
    CCDCProfile,
    ReflectanceBounds,
    SegmentMatch,
)
from lc_framework.temporal.ccdc.stability import stable_in_period

__all__ = [
    "BandCase",
    "CCDCModel",
    "CCDCParams",
    "CCDCProfile",
    "ReflectanceBounds",
    "SegmentMatch",
    "fill_isolated",
    "get_harmonics",
    "get_multi_year_stacks",
    "get_synthetic_stack",
    "load_ccdc_mosaic",
    "nearest_segment",
    "prepare_ccdc_for_export",
    "run_ccdc",
    "stable_in_period",
]

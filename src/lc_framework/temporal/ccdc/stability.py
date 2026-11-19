"""Multi-epoch training sample stability filtering using CCDC Temporal Segments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from lc_framework.temporal.ccdc.reconstruct import nearest_segment
from lc_framework.temporal.ccdc.specs import CCDCProfile

if TYPE_CHECKING:
    import ee


def stable_in_period(
    ccdc: ee.Image,
    start_year: float,
    end_year: float,
    *,
    profile: CCDCProfile | None = None,
    apply_slope_filter: bool = True,
    drift_bands: Sequence[str] = ("nir", "swir1"),
    max_drift: float = 0.12,
    midpoint_offset: float = 0.50,
) -> ee.Image:
    """Compute a binary stability mask for transferring samples across epochs."""
    if end_year < start_year:
        raise ValueError(
            f"end_year ({end_year}) must be >= start_year ({start_year})."
        )
    if max_drift < 0:
        raise ValueError(f"max_drift must be >= 0, got {max_drift!r}.")
    if not (0.0 <= midpoint_offset < 1.0):
        raise ValueError(
            f"midpoint_offset must lie in [0.0, 1.0), got {midpoint_offset!r}."
        )

    prof = profile if profile is not None else CCDCProfile.standard()

    t_break = ccdc.select("tBreak")
    change_prob = ccdc.select("changeProb")
    confirmed_break = t_break.multiply(change_prob.eq(1))
    breaks_in_window = confirmed_break.arrayMask(
        confirmed_break.gte(start_year).And(confirmed_break.lte(end_year))
    )
    no_break = breaks_in_window.arrayLength(0).eq(0)

    seg_start = nearest_segment(ccdc, math.floor(start_year) + midpoint_offset)
    seg_end = nearest_segment(ccdc, math.floor(end_year) + midpoint_offset)

    valid_coverage = (
        seg_start.gap.lte(prof.tol_years)
        .And(seg_end.gap.lte(prof.tol_years))
        .And(seg_start.idx.eq(seg_end.idx))
    )

    is_stable = no_break.And(valid_coverage)

    if apply_slope_filter and drift_bands:
        delta_years = float(end_year) - float(start_year)
        for band in drift_bands:
            coef_band = prof.coef_band_name(band)
            slope = (
                ccdc.select(coef_band)
                .arraySlice(0, seg_end.idx, seg_end.idx.add(1))
                .arrayProject([1])
                .arrayGet([1])
                .divide(prof.coef_scale)
                .abs()
            )
            is_stable = is_stable.And(slope.multiply(delta_years).lte(max_drift))

    return is_stable.unmask(0).toByte()

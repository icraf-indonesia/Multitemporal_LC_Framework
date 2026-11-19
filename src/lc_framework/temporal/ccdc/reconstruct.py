"""Sensor-agnostic CCDC synthetic reconstruction and harmonic extraction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from lc_framework.temporal.ccdc.specs import BandCase, CCDCProfile, SegmentMatch, _require_ee

if TYPE_CHECKING:
    import ee


def nearest_segment(
    ccdc: ee.Image,
    target_date: float | ee.Number,
) -> SegmentMatch:
    """Locate the temporally closest CCDC Temporal Segment per pixel at ``target_date``."""
    ee_mod = _require_ee()
    t = ee_mod.Number(target_date)
    t_start = ccdc.select("tStart")
    t_end = ccdc.select("tEnd")

    dist = t_start.subtract(t).max(t_end.multiply(-1).add(t)).max(0)

    idx = dist.multiply(-1).arrayArgmax().arrayGet([0])
    gap = dist.arrayReduce(ee_mod.Reducer.min(), [0]).arrayGet([0]).rename("gap_years")

    return SegmentMatch(idx=idx, gap=gap, t_start=t_start, t_end=t_end)


def fill_isolated(
    image: ee.Image,
    min_neighbors: int = 5,
) -> ee.Image:
    """Repair isolated masked pixels via a conditional 3x3 focal median."""
    if not (1 <= min_neighbors <= 8):
        raise ValueError(f"min_neighbors must be between 1 and 8, got {min_neighbors!r}.")

    ee_mod = _require_ee()
    valid_neighbors = (
        image.select(0)
        .mask()
        .reduceNeighborhood(
            reducer=ee_mod.Reducer.sum(),
            kernel=ee_mod.Kernel.square(1),
        )
    )
    median_img = image.focalMedian(1, "square", "pixels")
    return image.unmask(median_img.updateMask(valid_neighbors.gte(min_neighbors)))


def get_synthetic_stack(
    ccdc: ee.Image,
    target_date: float | ee.Number,
    *,
    profile: CCDCProfile | None = None,
    bands: Sequence[str] | None = None,
    fill_holes: bool = True,
    min_neighbors: int = 5,
    include_diagnostics: bool = True,
) -> ee.Image:
    """Reconstruct cloud-free synthetic surface reflectance at ``target_date``."""
    ee_mod = _require_ee()
    prof = profile if profile is not None else CCDCProfile.standard()
    target_bands = tuple(b.lower() for b in bands) if bands is not None else prof.bands
    if not target_bands:
        raise ValueError("bands must contain at least one band name.")

    t = ee_mod.Number(target_date)
    seg = nearest_segment(ccdc, t)

    tc = ee_mod.Image(t).max(seg.t_start.arrayGet(seg.idx)).min(seg.t_end.arrayGet(seg.idx))
    omega = 2.0 * math.pi

    terms = ee_mod.Image.cat(
        [
            ee_mod.Image.constant(1),
            tc,
            ee_mod.Image.constant(t.multiply(omega).cos()),
            ee_mod.Image.constant(t.multiply(omega).sin()),
            ee_mod.Image.constant(t.multiply(omega * 2.0).cos()),
            ee_mod.Image.constant(t.multiply(omega * 2.0).sin()),
            ee_mod.Image.constant(t.multiply(omega * 3.0).cos()),
            ee_mod.Image.constant(t.multiply(omega * 3.0).sin()),
        ]
    ).toArray()

    synthetic_bands: list[Any] = []
    for band in target_bands:
        coef_band_name = prof.coef_band_name(band)
        coefs = (
            ccdc.select(coef_band_name)
            .arraySlice(0, seg.idx, seg.idx.add(1))
            .arrayProject([1])
        )
        evaluated = (
            coefs.multiply(terms)
            .arrayReduce(ee_mod.Reducer.sum(), [0])
            .arrayGet([0])
            .divide(prof.coef_scale)
            .rename(band)
        )
        synthetic_bands.append(evaluated)

    stack = ee_mod.Image.cat(synthetic_bands)

    plausible = (
        stack.reduce(ee_mod.Reducer.min())
        .gt(prof.bounds.min)
        .And(stack.reduce(ee_mod.Reducer.max()).lt(prof.bounds.max))
    )
    within_tol = seg.gap.lte(prof.tol_years)
    synth_valid = plausible.And(within_tol).rename("synth_valid")

    masked_stack = stack.updateMask(synth_valid)
    if fill_holes:
        masked_stack = fill_isolated(masked_stack, min_neighbors=min_neighbors)

    if include_diagnostics:
        return masked_stack.addBands([seg.gap, synth_valid])
    return masked_stack


def get_harmonics(
    ccdc: ee.Image,
    target_date: float | ee.Number,
    *,
    profile: CCDCProfile | None = None,
    bands: Sequence[str] = ("nir", "red", "swir1"),
    output_case: BandCase | None = None,
) -> ee.Image:
    """Extract first-order annual harmonic amplitude and phase at ``target_date``."""
    ee_mod = _require_ee()
    prof = profile if profile is not None else CCDCProfile.standard()
    target_bands = tuple(b.lower() for b in bands)
    if not target_bands:
        raise ValueError("bands must contain at least one band name.")

    t = ee_mod.Number(target_date)
    seg = nearest_segment(ccdc, t)

    harmonic_images: list[Any] = []
    for band in target_bands:
        coef_band_name = prof.coef_band_name(band)
        coefficients = (
            ccdc.select(coef_band_name)
            .arraySlice(0, seg.idx, seg.idx.add(1))
            .arrayProject([1])
        )
        a1 = coefficients.arrayGet([2]).divide(prof.coef_scale)
        b1 = coefficients.arrayGet([3]).divide(prof.coef_scale)

        amp_name, phase_name = prof.harmonic_band_names(band, output_case=output_case)
        harmonic_images.append(
            ee_mod.Image.cat(
                [
                    a1.hypot(b1).rename(amp_name),
                    b1.atan2(a1).rename(phase_name),
                ]
            )
        )

    return ee_mod.Image.cat(harmonic_images).updateMask(seg.gap.lte(prof.tol_years))


def get_multi_year_stacks(
    ccdc: ee.Image,
    years: Sequence[int],
    season_fraction: float = 0.50,
    *,
    profile: CCDCProfile | None = None,
    **kwargs: Any,
) -> dict[int, ee.Image]:
    """Generate synthetic reflectance stacks across multiple target years."""
    if not (0.0 <= season_fraction < 1.0):
        raise ValueError(
            f"season_fraction must lie in [0.0, 1.0), got {season_fraction!r}."
        )
    return {
        int(yr): get_synthetic_stack(
            ccdc,
            float(yr) + float(season_fraction),
            profile=profile,
            **kwargs,
        )
        for yr in years
    }

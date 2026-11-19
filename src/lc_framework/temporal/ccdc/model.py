"""Lightweight facade wrapping a CCDC ``ee.Image`` with its :class:`CCDCProfile`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lc_framework.temporal.ccdc.fit import (
    load_ccdc_mosaic,
    prepare_ccdc_for_export,
    run_ccdc,
)
from lc_framework.temporal.ccdc.reconstruct import (
    get_harmonics,
    get_multi_year_stacks,
    get_synthetic_stack,
    nearest_segment,
)
from lc_framework.temporal.ccdc.specs import BandCase, CCDCParams, CCDCProfile, SegmentMatch
from lc_framework.temporal.ccdc.stability import stable_in_period
from lc_framework.temporal.constants import CCDC_CORE_EXPORT_BANDS

if TYPE_CHECKING:
    import ee


@dataclass(frozen=True)
class CCDCModel:
    """Ergonomic, immutable wrapper pairing a CCDC ``ee.Image`` with its metadata profile."""

    image: ee.Image
    profile: CCDCProfile = field(default_factory=CCDCProfile.standard)

    @classmethod
    def from_tiles(
        cls,
        source: str | Sequence[str] | ee.ImageCollection,
        aoi: ee.Geometry | None = None,
        *,
        profile: CCDCProfile | None = None,
        is_collection: bool = False,
    ) -> CCDCModel:
        """Load and mosaic pre-exported CCDC tile asset(s) into a :class:`CCDCModel`."""
        mosaic = load_ccdc_mosaic(source, aoi=aoi, is_collection=is_collection)
        return cls(
            image=mosaic,
            profile=profile if profile is not None else CCDCProfile.standard(),
        )

    @classmethod
    def fit(
        cls,
        collection: ee.ImageCollection,
        *,
        params: CCDCParams | None = None,
        profile: CCDCProfile | None = None,
    ) -> CCDCModel:
        """Fit CCDC on a harmonized optical collection and wrap the resulting array image."""
        ccdc_img = run_ccdc(collection, params=params)
        return cls(
            image=ccdc_img,
            profile=profile if profile is not None else CCDCProfile.standard(),
        )

    def nearest_segment(self, target_date: float | ee.Number) -> SegmentMatch:
        """Locate the nearest CCDC Temporal Segment per pixel."""
        return nearest_segment(self.image, target_date)

    def synthetic(
        self,
        target_date: float | ee.Number,
        *,
        bands: Sequence[str] | None = None,
        fill_holes: bool = True,
        min_neighbors: int = 5,
        include_diagnostics: bool = True,
    ) -> ee.Image:
        """Reconstruct a trend-clamped synthetic reflectance stack at ``target_date``."""
        return get_synthetic_stack(
            self.image,
            target_date,
            profile=self.profile,
            bands=bands,
            fill_holes=fill_holes,
            min_neighbors=min_neighbors,
            include_diagnostics=include_diagnostics,
        )

    def synthetic_series(
        self,
        years: Sequence[int],
        season_fraction: float = 0.50,
        **kwargs: Any,
    ) -> dict[int, ee.Image]:
        """Reconstruct synthetic reflectance stacks for multiple calendar years."""
        return get_multi_year_stacks(
            self.image,
            years,
            season_fraction=season_fraction,
            profile=self.profile,
            **kwargs,
        )

    def harmonics(
        self,
        target_date: float | ee.Number,
        *,
        bands: Sequence[str] = ("nir", "red", "swir1"),
        output_case: BandCase | None = None,
    ) -> ee.Image:
        """Extract first-order annual harmonic amplitude and phase at ``target_date``."""
        return get_harmonics(
            self.image,
            target_date,
            profile=self.profile,
            bands=bands,
            output_case=output_case,
        )

    def stable_in_period(
        self,
        start_year: float,
        end_year: float,
        *,
        apply_slope_filter: bool = True,
        drift_bands: Sequence[str] = ("nir", "swir1"),
        max_drift: float = 0.12,
        midpoint_offset: float = 0.50,
    ) -> ee.Image:
        """Compute the binary multi-epoch stability mask over ``[start_year, end_year]``."""
        return stable_in_period(
            self.image,
            start_year,
            end_year,
            profile=self.profile,
            apply_slope_filter=apply_slope_filter,
            drift_bands=drift_bands,
            max_drift=max_drift,
            midpoint_offset=midpoint_offset,
        )

    def prepare_for_export(
        self,
        *,
        bands: Sequence[str] = CCDC_CORE_EXPORT_BANDS,
    ) -> ee.Image:
        """Trim the underlying CCDC image to bands required for downstream export."""
        return prepare_ccdc_for_export(self.image, bands=bands)

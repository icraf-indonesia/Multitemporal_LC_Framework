"""Typed specifications and asset metadata profiles for CCDC operations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from lc_framework.temporal.constants import (
    DEFAULT_CCDC_REFLECTANCE_SCALE,
    DEFAULT_REFLECTANCE_MAX,
    DEFAULT_REFLECTANCE_MIN,
    DEFAULT_TOL_YEARS,
    OPTICAL_BANDS,
)

if TYPE_CHECKING:
    import ee

BandCase = Literal["lower", "upper"]


def _require_ee() -> Any:
    """Import Google Earth Engine lazily with an actionable error message."""
    try:
        import ee as ee_module
    except ImportError as exc:
        raise ImportError(
            "Google Earth Engine (`earthengine-api`) is required for CCDC operations. "
            "Install it with: pip install 'lc-framework[gee]'"
        ) from exc
    return ee_module


@dataclass(frozen=True)
class ReflectanceBounds:
    """Physical surface reflectance validity interval (0..1 scale)."""

    min: float = DEFAULT_REFLECTANCE_MIN
    max: float = DEFAULT_REFLECTANCE_MAX

    def __post_init__(self) -> None:
        if self.min >= self.max:
            raise ValueError(
                f"ReflectanceBounds.min ({self.min}) must be strictly less than max ({self.max})."
            )


@dataclass(frozen=True)
class CCDCParams:
    """Hyperparameters for ``ee.Algorithms.TemporalSegmentation.Ccdc``."""

    breakpoint_bands: tuple[str, ...] = ("green", "red", "nir", "swir1", "swir2")
    tmask_bands: tuple[str, ...] | None = None
    date_format: int = 1
    min_observations: int = 6
    chi_square_probability: float = 0.99
    min_num_of_years_scaler: float = 1.33
    lambda_: float = 20.0
    max_iterations: int = 25000

    def __post_init__(self) -> None:
        bp = tuple(self.breakpoint_bands)
        if not bp:
            raise ValueError("CCDCParams.breakpoint_bands must contain at least one band.")
        object.__setattr__(self, "breakpoint_bands", bp)

        if self.tmask_bands is not None:
            tm = tuple(self.tmask_bands)
            if len(tm) != 2:
                raise ValueError(
                    f"CCDCParams.tmask_bands must have exactly 2 bands when set, got {tm!r}."
                )
            object.__setattr__(self, "tmask_bands", tm)

        if self.date_format not in (0, 1, 2):
            raise ValueError(
                "CCDCParams.date_format must be 0 (Jdays), 1 (fractional years), "
                f"or 2 (unix ms); got {self.date_format!r}."
            )
        if self.min_observations < 1:
            raise ValueError(
                f"CCDCParams.min_observations must be >= 1, got {self.min_observations!r}."
            )
        if not (0.0 < self.chi_square_probability < 1.0):
            raise ValueError(
                "CCDCParams.chi_square_probability must lie strictly in (0, 1), "
                f"got {self.chi_square_probability!r}."
            )
        if self.min_num_of_years_scaler <= 0:
            raise ValueError(
                "CCDCParams.min_num_of_years_scaler must be > 0, "
                f"got {self.min_num_of_years_scaler!r}."
            )
        if self.lambda_ < 0:
            raise ValueError(f"CCDCParams.lambda_ must be >= 0, got {self.lambda_!r}.")
        if self.max_iterations < 1:
            raise ValueError(
                f"CCDCParams.max_iterations must be >= 1, got {self.max_iterations!r}."
            )

    @classmethod
    def landsat(cls, **overrides: Any) -> CCDCParams:
        """Preset calibrated for harmonized Landsat 5/7/8/9 time series."""
        base = cls(
            breakpoint_bands=("green", "red", "nir", "swir1", "swir2"),
            tmask_bands=("green", "swir1"),
        )
        return replace(base, **overrides) if overrides else base

    @classmethod
    def sentinel2(cls, **overrides: Any) -> CCDCParams:
        """Preset calibrated for Sentinel-2 L2A (Cloud Score+ masked) time series."""
        base = cls(
            breakpoint_bands=("green", "red", "nir", "swir1"),
            tmask_bands=None,
        )
        return replace(base, **overrides) if overrides else base

    def to_ee_params(self, collection: ee.ImageCollection) -> dict[str, Any]:
        """Build keyword arguments for ``ee.Algorithms.TemporalSegmentation.Ccdc``."""
        params: dict[str, Any] = {
            "collection": collection,
            "breakpointBands": list(self.breakpoint_bands),
            "dateFormat": self.date_format,
            "minObservations": self.min_observations,
            "chiSquareProbability": self.chi_square_probability,
            "minNumOfYearsScaler": self.min_num_of_years_scaler,
            "lambda": self.lambda_,
            "maxIterations": self.max_iterations,
        }
        if self.tmask_bands is not None:
            params["tmaskBands"] = list(self.tmask_bands)
        return params


@dataclass(frozen=True)
class CCDCProfile:
    """Asset metadata profile describing how a CCDC image encodes coefficients."""

    bands: tuple[str, ...] = OPTICAL_BANDS
    coef_scale: float = DEFAULT_CCDC_REFLECTANCE_SCALE
    coef_case: BandCase = "lower"
    harm_case: BandCase = "lower"
    tol_years: float = DEFAULT_TOL_YEARS
    bounds: ReflectanceBounds = field(default_factory=ReflectanceBounds)

    def __post_init__(self) -> None:
        normalized_bands = tuple(b.lower() for b in self.bands)
        if not normalized_bands:
            raise ValueError("CCDCProfile.bands must contain at least one band.")
        object.__setattr__(self, "bands", normalized_bands)

        if self.coef_scale <= 0:
            raise ValueError(f"CCDCProfile.coef_scale must be > 0, got {self.coef_scale!r}.")
        if self.coef_case not in ("lower", "upper"):
            raise ValueError(
                f"CCDCProfile.coef_case must be 'lower' or 'upper', got {self.coef_case!r}."
            )
        if self.harm_case not in ("lower", "upper"):
            raise ValueError(
                f"CCDCProfile.harm_case must be 'lower' or 'upper', got {self.harm_case!r}."
            )
        if self.tol_years < 0:
            raise ValueError(f"CCDCProfile.tol_years must be >= 0, got {self.tol_years!r}.")

    def coef_band_name(self, band: str) -> str:
        """Resolve the stored coefficient array band name for a canonical band."""
        prefix = band.upper() if self.coef_case == "upper" else band.lower()
        return f"{prefix}_coefs"

    def harmonic_band_names(
        self,
        band: str,
        output_case: BandCase | None = None,
    ) -> tuple[str, str]:
        """Resolve output ``(amplitude, phase)`` band names for a canonical band."""
        case = self.harm_case if output_case is None else output_case
        if case not in ("lower", "upper"):
            raise ValueError(f"output_case must be 'lower' or 'upper', got {case!r}.")
        prefix = band.upper() if case == "upper" else band.lower()
        return f"{prefix}_amp", f"{prefix}_phase"

    @classmethod
    def standard(cls, **overrides: Any) -> CCDCProfile:
        """Preset for standard pipeline outputs (scale=10000, lowercase bands)."""
        base = cls(coef_scale=10000.0, coef_case="lower", harm_case="lower")
        return replace(base, **overrides) if overrides else base

    @classmethod
    def landsat_legacy(cls, **overrides: Any) -> CCDCProfile:
        """Preset for legacy Landsat CCDC assets (scale=10000, uppercase bands)."""
        base = cls(coef_scale=10000.0, coef_case="upper", harm_case="upper")
        return replace(base, **overrides) if overrides else base

    @classmethod
    def google_global(cls, **overrides: Any) -> CCDCProfile:
        """Preset for ``GOOGLE/GLOBAL_CCDC/V1`` (scale=1.0, uppercase coefs)."""
        base = cls(coef_scale=1.0, coef_case="upper", harm_case="lower")
        return replace(base, **overrides) if overrides else base


@dataclass(frozen=True)
class SegmentMatch:
    """Result of per-pixel nearest CCDC Temporal Segment lookup at a target date."""

    idx: ee.Image
    gap: ee.Image
    t_start: ee.Image
    t_end: ee.Image

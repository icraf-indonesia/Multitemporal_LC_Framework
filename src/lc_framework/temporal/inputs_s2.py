"""Harmonized Sentinel-2 L2A + Cloud Score+ surface reflectance ingestion (GEE)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lc_framework.temporal.ccdc import specs as ccdc_specs
from lc_framework.temporal.constants import DEFAULT_CCDC_REFLECTANCE_SCALE, OPTICAL_BANDS

if TYPE_CHECKING:
    import ee

CloudScoreBand = Literal["cs_cdf", "cs"]

_S2_SOURCE_BANDS: tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11", "B12")


def get_s2_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    *,
    clear_threshold: float = 0.60,
    qa_band: CloudScoreBand = "cs_cdf",
    reflectance_scale: float = DEFAULT_CCDC_REFLECTANCE_SCALE,
) -> ee.ImageCollection:
    """Prepare a Cloud Score+ masked Sentinel-2 SR Harmonized collection (6-band schema).

    Links ``COPERNICUS/S2_SR_HARMONIZED`` with ``GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED``
    and maps ``B2, B3, B4, B8, B11, B12`` to ``['blue', 'green', 'red', 'nir', 'swir1', 'swir2']``.
    """
    if qa_band not in ("cs_cdf", "cs"):
        raise ValueError(f"qa_band must be 'cs_cdf' or 'cs', got {qa_band!r}.")
    if not (0.0 <= clear_threshold <= 1.0):
        raise ValueError(f"clear_threshold must lie in [0, 1], got {clear_threshold!r}.")
    if reflectance_scale <= 0:
        raise ValueError(f"reflectance_scale must be > 0, got {reflectance_scale!r}.")
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date!r}) must be >= start_date ({start_date!r}).")

    ee_mod = ccdc_specs._require_ee()
    src_bands = list(_S2_SOURCE_BANDS)
    out_bands = list(OPTICAL_BANDS)
    multiplier = float(reflectance_scale) / 10000.0

    s2 = (
        ee_mod.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    cs_plus = (
        ee_mod.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    linked = s2.linkCollection(cs_plus, [qa_band])

    def _mask_and_scale(img: Any) -> Any:
        cloud_mask = img.select(qa_band).gte(clear_threshold)
        scaled = img.select(src_bands, out_bands)
        if multiplier != 1.0:
            scaled = scaled.multiply(multiplier)
        return scaled.updateMask(cloud_mask).copyProperties(img, ["system:time_start"])

    return linked.map(_mask_and_scale)

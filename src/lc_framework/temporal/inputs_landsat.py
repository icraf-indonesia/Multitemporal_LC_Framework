"""Harmonized Landsat 5/7/8/9 Collection 2 Level-2 surface reflectance ingestion (GEE)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from lc_framework.temporal.ccdc import specs as ccdc_specs
from lc_framework.temporal.constants import DEFAULT_CCDC_REFLECTANCE_SCALE, OPTICAL_BANDS

if TYPE_CHECKING:
    import ee

_TM_ETM_BANDS: tuple[str, ...] = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7")
_OLI_BANDS: tuple[str, ...] = ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")

_SENSOR_COLLECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "L5": ("LANDSAT/LT05/C02/T1_L2", _TM_ETM_BANDS),
    "L7": ("LANDSAT/LE07/C02/T1_L2", _TM_ETM_BANDS),
    "L8": ("LANDSAT/LC08/C02/T1_L2", _OLI_BANDS),
    "L9": ("LANDSAT/LC09/C02/T1_L2", _OLI_BANDS),
}


def get_landsat_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    *,
    reflectance_scale: float = DEFAULT_CCDC_REFLECTANCE_SCALE,
    mask_fill: bool = True,
    mask_cloud: bool = True,
    mask_shadow: bool = True,
    mask_snow: bool = True,
    sensors: Sequence[str] = ("L5", "L7", "L8", "L9"),
) -> ee.ImageCollection:
    """Prepare a merged, cloud-masked Landsat C2 L2 collection in the 6-band lowercase schema.

    Applies USGS Collection 2 linear surface reflectance scaling
    ``(0.0000275 * DN - 0.2) * reflectance_scale``. Use ``reflectance_scale=10000.0``
    for CCDC (calibrated for ``lambda=20``) and ``1.0`` for direct composites.
    """
    if reflectance_scale <= 0:
        raise ValueError(f"reflectance_scale must be > 0, got {reflectance_scale!r}.")
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date!r}) must be >= start_date ({start_date!r}).")

    normalized_sensors = [s.upper() for s in sensors]
    if not normalized_sensors:
        raise ValueError(
            "sensors must contain at least one Landsat sensor ('L5', 'L7', 'L8', 'L9')."
        )
    for sensor in normalized_sensors:
        if sensor not in _SENSOR_COLLECTIONS:
            raise ValueError(
                f"Unsupported Landsat sensor {sensor!r}; expected subset of "
                f"{tuple(_SENSOR_COLLECTIONS)}."
            )

    ee_mod = ccdc_specs._require_ee()
    output_bands = list(OPTICAL_BANDS)

    def _make_mapper(src_bands: tuple[str, ...]) -> Any:
        src_list = list(src_bands)

        def _mask_and_scale(img: Any) -> Any:
            qa = img.select("QA_PIXEL")
            mask = ee_mod.Image(1)
            if mask_fill:
                mask = mask.And(qa.bitwiseAnd(1 << 0).eq(0))
            if mask_cloud:
                mask = mask.And(qa.bitwiseAnd(1 << 1).eq(0)).And(qa.bitwiseAnd(1 << 3).eq(0))
            if mask_shadow:
                mask = mask.And(qa.bitwiseAnd(1 << 4).eq(0))
            if mask_snow:
                mask = mask.And(qa.bitwiseAnd(1 << 5).eq(0))

            sr = (
                img.select(src_list, output_bands)
                .multiply(0.0000275)
                .subtract(0.2)
                .multiply(float(reflectance_scale))
            )
            return sr.updateMask(mask).copyProperties(img, ["system:time_start"])

        return _mask_and_scale

    merged: Any = None
    for sensor in normalized_sensors:
        asset_id, src_bands = _SENSOR_COLLECTIONS[sensor]
        col = (
            ee_mod.ImageCollection(asset_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .map(_make_mapper(src_bands))
        )
        merged = col if merged is None else merged.merge(col)

    return merged

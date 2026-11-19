"""NASADEM topographic derivatives (elevation, slope, aspect, TPI, TWI) and GLCM texture (GEE)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from lc_framework import _gee

if TYPE_CHECKING:
    import ee

_DEG2RAD = math.pi / 180.0


def get_topography(
    aoi: ee.Geometry,
    *,
    dem_asset: str = "NASA/NASADEM_HGT/001",
    compute_tpi: bool = True,
    tpi_radius_m: float = 300.0,
    compute_twi: bool = True,
    flow_acc_asset: str = "MERIT/Hydro/v1_0_1",
    landform_image: ee.Image | None = None,
    twi_image: ee.Image | None = None,
) -> ee.Image:
    """Build topographic predictors (elevation, slope, aspect, TPI, TWI) clipped to ``aoi``.

    Computes all neighborhood/derivative layers before clipping to ``aoi`` to
    prevent border masking artifacts along study-area edges.
    """
    ee_mod = _gee.require_ee("topography features")
    elev = ee_mod.Image(dem_asset).select("elevation")
    terrain = ee_mod.Terrain.products(elev).select(["slope", "aspect"]).toFloat()
    stack = elev.int16().addBands(terrain)

    if compute_tpi:
        local_mean = elev.focalMean(radius=float(tpi_radius_m), units="meters")
        tpi = elev.subtract(local_mean).rename("TPI").toFloat()
        stack = stack.addBands(tpi)

    if twi_image is not None:
        stack = stack.addBands(twi_image.rename("TWI").toFloat())
    elif compute_twi:
        # TWI = ln(upslope_contributing_area / max(tan(slope_rad), 0.001))
        upa = ee_mod.Image(flow_acc_asset).select("upa").max(1e-3)
        tan_slope = terrain.select("slope").multiply(_DEG2RAD).tan().max(1e-3)
        twi = upa.multiply(1e6).divide(tan_slope).log().rename("TWI").toFloat()
        stack = stack.addBands(twi)

    if landform_image is not None:
        stack = stack.addBands(landform_image.rename("landform").toByte())

    return stack.clip(aoi)


def get_glcm_texture(
    image: ee.Image,
    *,
    band: str = "nir",
    size: int = 1,
    scale_factor: float = 1000.0,
    metrics: Sequence[str] = ("contrast", "ent", "asm", "idm"),
) -> ee.Image:
    """Compute GLCM texture metrics from a reflectance band (default dry-season ``nir``)."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size!r}.")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor!r}.")
    if not metrics:
        raise ValueError("metrics must contain at least one GLCM metric suffix.")

    _gee.require_ee("GLCM texture features")
    selected = image.select([band], [band]).multiply(float(scale_factor)).toInt32()
    glcm = selected.glcmTexture(size=int(size))
    out_bands = [f"{band}_{m}" for m in metrics]
    return glcm.select(out_bands)

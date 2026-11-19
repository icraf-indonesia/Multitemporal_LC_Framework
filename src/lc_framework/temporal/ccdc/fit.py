"""CCDC fitting, tiled mosaic loading, and export band preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lc_framework.temporal.ccdc.specs import CCDCParams, _require_ee
from lc_framework.temporal.constants import CCDC_CORE_EXPORT_BANDS

if TYPE_CHECKING:
    import ee


def run_ccdc(
    collection: ee.ImageCollection,
    params: CCDCParams | None = None,
) -> ee.Image:
    """Execute GEE Continuous Change Detection and Classification (CCDC)."""
    ee_mod = _require_ee()
    cfg = params if params is not None else CCDCParams()
    return ee_mod.Algorithms.TemporalSegmentation.Ccdc(**cfg.to_ee_params(collection))


def load_ccdc_mosaic(
    source: str | Sequence[str] | ee.ImageCollection,
    aoi: ee.Geometry | None = None,
    *,
    is_collection: bool = False,
) -> ee.Image:
    """Load and mosaic one or more pre-exported CCDC GEE assets."""
    ee_mod = _require_ee()
    if isinstance(source, str):
        if is_collection:
            col = ee_mod.ImageCollection(source)
            if aoi is not None:
                col = col.filterBounds(aoi)
            mosaic = col.mosaic()
        else:
            mosaic = ee_mod.Image(source)
    elif isinstance(source, ee_mod.ImageCollection):
        col = source.filterBounds(aoi) if aoi is not None else source
        mosaic = col.mosaic()
    elif isinstance(source, Sequence):
        if not source:
            raise ValueError("source must contain at least one CCDC tile asset ID.")
        images = [
            ee_mod.Image(tile_id) if isinstance(tile_id, str) else tile_id for tile_id in source
        ]
        mosaic = ee_mod.ImageCollection(images).mosaic()
    else:
        raise TypeError(
            "source must be a GEE asset ID string, a sequence of tile IDs, "
            f"or an ee.ImageCollection; got {type(source).__name__}."
        )

    return mosaic.clip(aoi) if aoi is not None else mosaic


def prepare_ccdc_for_export(
    ccdc_image: ee.Image,
    *,
    bands: Sequence[str] = CCDC_CORE_EXPORT_BANDS,
) -> ee.Image:
    """Trim a raw CCDC output image to the bands required downstream."""
    band_list = list(bands)
    if not band_list:
        raise ValueError("bands must contain at least one band name or regex selector.")
    return ccdc_image.select(band_list)

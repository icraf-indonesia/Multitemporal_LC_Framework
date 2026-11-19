"""Multi-sensor GEE feature stack composition and sample-feature CSV export."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lc_framework import _gee
from lc_framework._gee import OPTICAL_BANDS
from lc_framework.config import ClassSchema
from lc_framework.features.optical import compute_phenological_deltas, compute_seasonal_indices
from lc_framework.features.sar import get_palsar_backscatter, get_scansar_features, get_seasonal_s1
from lc_framework.features.terrain_texture import get_glcm_texture, get_topography

if TYPE_CHECKING:
    import ee


@dataclass(frozen=True)
class FeatureStackSpec:
    """Declarative sensor & feature toggles for :func:`build_feature_stack`."""

    indices: tuple[str, ...] = ("NDVI", "EVI", "MNDWI", "NDBI", "NBR", "BSI")
    include_optical_bands: bool = True
    include_deltas: bool = True
    include_s1: bool = True
    s1_include_ratio: bool = False
    include_scansar: bool = False
    include_palsar: bool = False
    include_topography: bool = True
    include_glcm: bool = True
    wet_months: tuple[int, int] = (12, 4)
    dry_months: tuple[int, int] = (5, 11)
    glcm_size: int = 1


def build_feature_stack(
    *,
    aoi: ee.Geometry,
    year: int,
    wet_image: ee.Image,
    dry_image: ee.Image,
    harmonics_image: ee.Image | None = None,
    spec: FeatureStackSpec | None = None,
) -> ee.Image:
    """Compose optical, phenological, harmonic, SAR, terrain, and GLCM predictors in GEE."""
    ee_mod = _gee.require_ee("feature stack engineering")
    cfg = spec if spec is not None else FeatureStackSpec()

    layers: list[Any] = []
    if cfg.include_optical_bands:
        opt = list(OPTICAL_BANDS)
        layers.append(wet_image.select(opt, [f"{b}_wet" for b in opt]))
        layers.append(dry_image.select(opt, [f"{b}_dry" for b in opt]))

    if cfg.indices:
        wet_idx = compute_seasonal_indices(wet_image, indices=cfg.indices, suffix="wet")
        dry_idx = compute_seasonal_indices(dry_image, indices=cfg.indices, suffix="dry")
        layers.extend([wet_idx, dry_idx])
        if cfg.include_deltas:
            layers.append(compute_phenological_deltas(wet_idx, dry_idx, indices=cfg.indices))

    if harmonics_image is not None:
        layers.append(harmonics_image)

    if cfg.include_s1:
        layers.append(
            get_seasonal_s1(
                year,
                aoi,
                wet_months=cfg.wet_months,
                dry_months=cfg.dry_months,
                include_ratio=cfg.s1_include_ratio,
            )
        )

    if cfg.include_scansar:
        layers.append(get_scansar_features(year, aoi))

    if cfg.include_palsar:
        layers.append(get_palsar_backscatter(year, aoi))

    if cfg.include_topography:
        layers.append(get_topography(aoi))

    if cfg.include_glcm:
        layers.append(get_glcm_texture(dry_image, band="nir", size=cfg.glcm_size))

    if not layers:
        raise ValueError("FeatureStackSpec enabled zero feature layers.")

    return ee_mod.Image.cat(layers).clip(aoi)


def extract_sample_features(
    feature_image: ee.Image,
    samples_fc: ee.FeatureCollection,
    *,
    class_schema: ClassSchema,
    class_property: str = "class_id",
    scale_m: float = 10.0,
    tile_scale: int = 4,
    geometries: bool = False,
) -> ee.FeatureCollection:
    """Sample ``feature_image`` at labeled points and attach canonical ``class_name``."""
    if scale_m <= 0:
        raise ValueError(f"scale_m must be > 0, got {scale_m!r}.")
    if tile_scale < 1:
        raise ValueError(f"tile_scale must be >= 1, got {tile_scale!r}.")

    ee_mod = _gee.require_ee("sample feature extraction")
    sampled = feature_image.sampleRegions(
        collection=samples_fc,
        properties=[class_property],
        scale=float(scale_m),
        tileScale=int(tile_scale),
        geometries=geometries,
    )

    name_dict = ee_mod.Dictionary({str(c.id): c.name for c in class_schema.classes})

    def _attach_name(feat: Any) -> Any:
        key = ee_mod.Number(feat.get(class_property)).int().format()
        return feat.set("class_name", name_dict.get(key, "Unknown"))

    return sampled.map(_attach_name)


def export_sample_features_to_drive(
    sampled_fc: ee.FeatureCollection,
    *,
    description: str,
    folder: str,
    file_name_prefix: str | None = None,
    selectors: Sequence[str] | None = None,
    dry_run: bool = False,
) -> Any | None:
    """Export extracted sample features to Google Drive as a CSV for local RFECV/ML."""
    ee_mod = _gee.require_ee("sample CSV export")
    if dry_run:
        return None

    kwargs: dict[str, Any] = {
        "collection": sampled_fc,
        "description": description,
        "folder": folder,
        "fileNamePrefix": file_name_prefix or description,
        "fileFormat": "CSV",
    }
    if selectors is not None:
        kwargs["selectors"] = list(selectors)

    task = ee_mod.batch.Export.table.toDrive(**kwargs)
    task.start()
    return task

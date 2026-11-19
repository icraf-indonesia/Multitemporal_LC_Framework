"""Google Earth Engine SNIC segmentation provider with overflow-safe tile striding."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lc_framework._gee import require_ee
from lc_framework.io import (
    GEE_INT32_CLAMP_VALUE,
    ExportTilePlan,
    SegmentRaster,
    generate_tile_grid,
    plan_tiled_gee_exports,
    read_segment_raster,
    submit_gee_export_tasks,
)

DEFAULT_SNIC_HASH_MODULUS: int = 200_000_000
DEFAULT_SNIC_ID_STRIDE: int = 300_000_000


@dataclass(frozen=True)
class SNICParams:
    """Parameters governing Google Earth Engine SNIC superpixel segmentation.

    Attributes:
        size: Seed spacing in pixels (larger values yield larger geo-objects).
        compactness: Spatial vs. spectral compactness weight (``>= 0``).
        connectivity: Pixel neighborhood connectivity (``4`` or ``8``).
        neighborhood_size: Computation neighborhood in pixels; defaults to ``2 * size``.
        hash_modulus: Optional positive modulus used to map GEE's 64-bit SNIC cluster
            IDs into ``[1, hash_modulus]`` before applying tile striding, preventing
            ``2^31 - 1`` signed 32-bit clamping overflow. Set to ``0`` to disable.
        id_stride: Deterministic per-tile integer offset (``tile_index * id_stride``)
            preventing segment ID collisions across exported tiles.
    """

    size: int = 15
    compactness: float = 0.8
    connectivity: Literal[4, 8] = 8
    neighborhood_size: int | None = None
    hash_modulus: int = DEFAULT_SNIC_HASH_MODULUS
    id_stride: int = DEFAULT_SNIC_ID_STRIDE

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"size must be > 0, got {self.size!r}.")
        if self.compactness < 0.0:
            raise ValueError(f"compactness must be >= 0, got {self.compactness!r}.")
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity!r}.")
        if self.neighborhood_size is not None and self.neighborhood_size <= 0:
            raise ValueError(
                f"neighborhood_size must be > 0 when provided, got {self.neighborhood_size!r}."
            )
        if self.hash_modulus < 0:
            raise ValueError(f"hash_modulus must be >= 0, got {self.hash_modulus!r}.")
        if self.hash_modulus >= GEE_INT32_CLAMP_VALUE:
            raise ValueError(
                f"hash_modulus ({self.hash_modulus}) must be strictly less than "
                f"GEE int32 limit ({GEE_INT32_CLAMP_VALUE})."
            )
        if self.id_stride < 0:
            raise ValueError(f"id_stride must be >= 0, got {self.id_stride!r}.")
        if self.hash_modulus > 0 and self.id_stride > 0 and self.id_stride < self.hash_modulus:
            raise ValueError(
                f"id_stride ({self.id_stride}) must be >= hash_modulus ({self.hash_modulus}) "
                "to guarantee collision-free segment IDs across tiles."
            )

    @property
    def effective_neighborhood_size(self) -> int:
        """Resolved neighborhood size passed to GEE SNIC."""
        return self.neighborhood_size if self.neighborhood_size is not None else 2 * self.size


class GEESnicSegmenter:
    """Google Earth Engine SNIC provider conforming to :class:`Segmenter`."""

    def __init__(self, params: SNICParams | None = None) -> None:
        self.params = params if params is not None else SNICParams()

    @property
    def provider_name(self) -> str:
        return "gee_snic"

    def tile_offset(self, tile_index: int) -> int:
        """Return deterministic integer ID offset for a zero-based tile index."""
        if tile_index < 0:
            raise ValueError(f"tile_index must be >= 0, got {tile_index!r}.")
        offset = tile_index * self.params.id_stride
        upper_bound = offset + self.params.hash_modulus
        if self.params.hash_modulus > 0 and upper_bound >= GEE_INT32_CLAMP_VALUE:
            warnings.warn(
                f"Tile {tile_index} segment IDs may reach {upper_bound}, exceeding GEE int32 "
                f"limit ({GEE_INT32_CLAMP_VALUE}). Reduce hash_modulus/id_stride or rely on "
                "load_exported_segment_raster() overflow repair on local read.",
                UserWarning,
                stacklevel=2,
            )
        elif self.params.hash_modulus == 0 and offset > 0:
            warnings.warn(
                f"Tile {tile_index} adds id_stride offset {offset} to unhashed SNIC cluster IDs; "
                "rely on load_exported_segment_raster() overflow repair on local read.",
                UserWarning,
                stacklevel=2,
            )
        return offset

    def segment(
        self,
        source: Any,
        *,
        bands: Sequence[str] | None = None,
        seeds: Any | None = None,
        band_min: Sequence[float] | None = None,
        band_max: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run GEE SNIC on ``source`` (``ee.Image``) and return a single-band ``segment`` image."""
        if kwargs:
            raise TypeError(
                f"GEESnicSegmenter.segment() got unexpected keyword arguments: {sorted(kwargs)}"
            )
        ee = require_ee("GEE SNIC segmentation")
        img = source.select(list(bands)) if bands is not None else source

        if (band_min is None) != (band_max is None):
            raise ValueError("band_min and band_max must either both be provided or both be None.")
        if band_min is not None and band_max is not None:
            if len(band_min) != len(band_max) or len(band_min) == 0:
                raise ValueError("band_min and band_max must be non-empty sequences of equal size.")
            if bands is not None and len(band_min) != len(bands):
                raise ValueError(
                    f"band_min/band_max length ({len(band_min)}) must match bands length "
                    f"({len(bands)})."
                )
            for lo_v, hi_v in zip(band_min, band_max, strict=True):
                if float(lo_v) >= float(hi_v):
                    raise ValueError(
                        f"Each band_min value must be strictly less than band_max, "
                        f"got {lo_v!r} >= {hi_v!r}."
                    )
            lo = ee.Image.constant(list(band_min))
            hi = ee.Image.constant(list(band_max))
            img = img.subtract(lo).divide(hi.subtract(lo)).clamp(0, 1)

        snic_kwargs: dict[str, Any] = {
            "image": img,
            "size": self.params.size,
            "compactness": self.params.compactness,
            "connectivity": self.params.connectivity,
            "neighborhoodSize": self.params.effective_neighborhood_size,
        }
        if seeds is not None:
            snic_kwargs["seeds"] = seeds

        snic_result = ee.Algorithms.Image.Segmentation.SNIC(**snic_kwargs)
        clusters = snic_result.select("clusters")

        if self.params.hash_modulus > 0:
            clusters = (
                clusters.toInt64()
                .abs()
                .mod(self.params.hash_modulus)
                .add(1)
                .toInt32()
                .rename("segment")
            )
        else:
            clusters = clusters.toInt32().rename("segment")

        return clusters

    def export_tiles(
        self,
        feature_image: Any,
        bounds: tuple[float, float, float, float],
        *,
        export_prefix: str,
        tile_size_deg: float = 0.5,
        scale_m: float = 10.0,
        crs: str = "EPSG:4326",
        bands: Sequence[str] | None = None,
        band_min: Sequence[float] | None = None,
        band_max: Sequence[float] | None = None,
        existing_assets: Iterable[str] = (),
        skip_existing: bool = True,
        dry_run: bool = False,
        destination: Literal["asset", "drive"] = "asset",
        drive_folder: str = "LC_Framework",
        pyramiding_policy: str = "mode",
        max_pixels: float = 1e13,
    ) -> tuple[list[ExportTilePlan], list[Any]]:
        """Schedule deterministic strided GEE SNIC Segment Label Raster exports over a tile grid."""
        if destination not in ("asset", "drive"):
            raise ValueError(f"destination must be 'asset' or 'drive', got {destination!r}.")

        tiles = generate_tile_grid(bounds, tile_size_deg)
        plans = plan_tiled_gee_exports(
            tiles,
            export_prefix=export_prefix,
            existing_assets=existing_assets,
            skip_existing=skip_existing,
        )
        if dry_run:
            return plans, []

        ee = require_ee("GEE SNIC tiled export")
        base_segments = self.segment(
            feature_image,
            bands=bands,
            band_min=band_min,
            band_max=band_max,
        )
        plan_index = {plan.asset_id: idx for idx, plan in enumerate(plans)}

        def _make_task(plan: ExportTilePlan) -> Any:
            idx = plan_index[plan.asset_id]
            offset = self.tile_offset(idx)
            region = ee.Geometry.Rectangle(list(plan.tile.bounds))
            tile_img = base_segments.add(offset).toInt32().rename("segment").clip(region)

            if destination == "asset":
                return ee.batch.Export.image.toAsset(
                    image=tile_img,
                    description=plan.description,
                    assetId=plan.asset_id,
                    region=region,
                    crs=crs,
                    scale=scale_m,
                    maxPixels=max_pixels,
                    pyramidingPolicy={".default": pyramiding_policy},
                )
            # Explicitly unmask to 0 and declare noData=0 so masked background never exports
            # as GDAL int32 sentinel -2147483648 (which valid_segment_mask treats as valid ID).
            drive_img = tile_img.unmask(0).toInt32()
            return ee.batch.Export.image.toDrive(
                image=drive_img,
                description=plan.description,
                folder=drive_folder,
                fileNamePrefix=plan.description,
                region=region,
                crs=crs,
                scale=scale_m,
                maxPixels=max_pixels,
                formatOptions={"cloudOptimized": True, "noData": 0},
            )

        tasks = submit_gee_export_tasks(plans, task_factory=_make_task, dry_run=False)
        return plans, tasks

    @staticmethod
    def load_exported_segment_raster(
        segment_path: str | Path,
        *,
        reference_profile: dict[str, Any] | None = None,
        align_to_reference: bool = False,
        max_clamp_cluster_px: int = 400,
    ) -> SegmentRaster:
        """Load a GEE-exported Segment Label Raster with ``int32`` overflow repair enabled."""
        return read_segment_raster(
            segment_path,
            reference_profile=reference_profile,
            align_to_reference=align_to_reference,
            repair_overflow=True,
            max_clamp_cluster_px=max_clamp_cluster_px,
        )

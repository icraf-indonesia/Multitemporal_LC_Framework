"""Shared pluggable Segmenter contract and Segment Label Raster helpers (ADR-0005)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import rasterio
from numpy.typing import NDArray

SEGMENT_NODATA: int = 0


@runtime_checkable
class Segmenter(Protocol):
    """Unified pluggable contract for GEE and local segmentation providers.

    Concrete providers include :class:`lc_framework.obia.GEESnicSegmenter` (cloud
    Earth Engine SNIC) and :class:`lc_framework.obia.LocalSlicSegmenter` (local
    scikit-image SLIC). Both emit standardized ``Segment Label Raster`` outputs
    compatible with pure Geo-Object Classification and Classification Assembly.
    """

    @property
    def provider_name(self) -> str:
        """Short identifier for this segmentation provider (e.g. ``'gee_snic'``)."""
        ...

    def segment(self, source: Any, **kwargs: Any) -> Any:
        """Segment ``source`` into a standardized Segment Label Raster representation."""
        ...


def valid_segment_mask(segments: np.ndarray) -> NDArray[np.bool_]:
    """Return a boolean mask of valid geo-object pixels.

    Excludes ``0`` (standard Segment Label Raster nodata) and ``-1`` (legacy GEE
    masked sentinel) while preserving valid wrapped negative IDs (``< -1``) that
    arise from GEE 32-bit signed integer wraparound or overflow repair.
    """
    arr = np.asarray(segments)
    return (arr != 0) & (arr != -1)


def write_segment_raster(
    output_path: str | Path,
    segments: np.ndarray,
    *,
    profile: dict[str, Any],
    compress: str = "deflate",
) -> Path:
    """Write a standardized single-band ``int32`` COG Segment Label Raster.

    Strips GeoTIFF profile keys incompatible with GDAL's ``COG`` driver and sets
    ``nodata=0`` with band description ``'segment'`` so downstream Geo-Object
    Classification and Classification Assembly can read the raster directly.
    """
    arr = np.asarray(segments, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"segments must be a 2D (height, width) array, got shape {arr.shape}.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = dict(profile)
    out_profile["driver"] = "COG"
    for incompatible_key in ("tiled", "blockxsize", "blockysize", "interleave"):
        out_profile.pop(incompatible_key, None)
    out_profile.update(
        count=1,
        dtype="int32",
        height=arr.shape[0],
        width=arr.shape[1],
        nodata=SEGMENT_NODATA,
        compress=compress,
    )

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "segment")

    return out_path

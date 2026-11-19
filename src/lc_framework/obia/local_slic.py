"""Local Python SLIC segmentation provider backed by ``scikit-image`` (``[obia]``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray

from lc_framework.io import SegmentRaster
from lc_framework.obia.base import SEGMENT_NODATA, write_segment_raster


def _require_skimage_slic() -> Any:
    """Import ``skimage.segmentation.slic`` lazily with an actionable install hint."""
    try:
        from skimage.segmentation import slic
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for local SLIC segmentation. "
            "Install it with: pip install 'lc-framework[obia]'"
        ) from exc
    return slic


@dataclass(frozen=True)
class SLICParams:
    """Parameters governing local Python SLIC superpixel segmentation."""

    n_segments: int = 200
    compactness: float = 10.0
    sigma: float = 1.0
    start_label: int = 1
    enforce_connectivity: bool = True
    min_size_factor: float = 0.5
    max_num_iter: int = 10
    normalize_bands: bool = True

    def __post_init__(self) -> None:
        if self.n_segments <= 0:
            raise ValueError(f"n_segments must be > 0, got {self.n_segments!r}.")
        if self.compactness <= 0.0:
            raise ValueError(f"compactness must be > 0, got {self.compactness!r}.")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma!r}.")
        if self.start_label < 1:
            raise ValueError(
                f"start_label must be >= 1 (0 is reserved for nodata), got {self.start_label!r}."
            )
        if not (0.0 < self.min_size_factor <= 1.0):
            raise ValueError(
                f"min_size_factor must lie in (0, 1], got {self.min_size_factor!r}."
            )
        if self.max_num_iter <= 0:
            raise ValueError(f"max_num_iter must be > 0, got {self.max_num_iter!r}.")


def _normalize_feature_cube(
    cube: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Robustly rescale each band to ``[0, 1]`` using 2nd-98th percentiles on valid pixels."""
    out = np.zeros_like(cube, dtype=np.float32)
    for b in range(cube.shape[0]):
        vals = cube[b, valid_mask]
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        if hi - lo > 1e-8:
            scaled = (vals - lo) / (hi - lo)
            out[b, valid_mask] = np.clip(scaled, 0.0, 1.0).astype(np.float32)
        else:
            out[b, valid_mask] = 0.0
    return out


def _read_feature_raster_with_mask(
    feature_raster_path: str | Path,
    *,
    band_indices: Sequence[int] | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
    """Read a multi-band feature GeoTIFF and construct a native-precision valid pixel mask."""
    with rasterio.open(feature_raster_path) as src:
        indexes = list(band_indices) if band_indices is not None else list(src.indexes)
        raw = src.read(indexes)
        profile = dict(src.profile)
        nodata = src.nodata

    data = raw.astype(np.float32)
    valid = np.all(np.isfinite(data), axis=0)
    if nodata is not None and not np.isnan(nodata):
        # Compare in native dtype before float32 cast to avoid float64 nodata rounding misses
        valid &= ~np.any(raw == nodata, axis=0)

    if valid_mask is not None:
        user_mask = np.asarray(valid_mask, dtype=bool)
        if user_mask.shape != valid.shape:
            raise ValueError(
                f"valid_mask shape {user_mask.shape} does not match spatial grid {valid.shape}."
            )
        valid &= user_mask

    return data, valid, profile


class LocalSlicSegmenter:
    """Local Python SLIC segmentation provider conforming to :class:`Segmenter`."""

    def __init__(self, params: SLICParams | None = None) -> None:
        self.params = params if params is not None else SLICParams()

    @property
    def provider_name(self) -> str:
        return "local_slic"

    def segment_array(
        self,
        feature_array: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> NDArray[np.int32]:
        """Segment a ``(bands, H, W)`` or ``(H, W)`` feature array into ``int32`` labels."""
        slic = _require_skimage_slic()
        arr = np.asarray(feature_array, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(
                f"feature_array must have shape (bands, H, W) or (H, W), got {arr.shape}."
            )

        _, height, width = arr.shape
        valid = np.all(np.isfinite(arr), axis=0)
        if valid_mask is not None:
            user_mask = np.asarray(valid_mask, dtype=bool)
            if user_mask.shape != (height, width):
                raise ValueError(
                    f"valid_mask shape {user_mask.shape} does not match spatial grid "
                    f"({height}, {width})."
                )
            valid &= user_mask

        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            return np.zeros((height, width), dtype=np.int32)
        if n_valid == 1:
            out = np.zeros((height, width), dtype=np.int32)
            out[valid] = self.params.start_label
            return out

        if self.params.normalize_bands:
            prepared = _normalize_feature_cube(arr, valid)
        else:
            prepared = np.where(valid[None, :, :], arr, 0.0).astype(np.float32)

        # skimage.segmentation.slic only accepts start_label in {0, 1}; shift afterwards if > 1.
        labels = slic(
            prepared,
            n_segments=min(self.params.n_segments, n_valid),
            compactness=self.params.compactness,
            max_num_iter=self.params.max_num_iter,
            sigma=self.params.sigma,
            convert2lab=False,
            enforce_connectivity=self.params.enforce_connectivity,
            min_size_factor=self.params.min_size_factor,
            start_label=1,
            mask=valid,
            channel_axis=0,
        )
        out = np.asarray(labels, dtype=np.int32)
        out[~valid] = SEGMENT_NODATA
        if self.params.start_label > 1:
            pos = out > 0
            out[pos] += self.params.start_label - 1
        return out

    def segment_raster(
        self,
        feature_raster_path: str | Path,
        output_segment_path: str | Path,
        *,
        band_indices: Sequence[int] | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> Path:
        """Segment a multi-band GeoTIFF and write a standardized COG Segment Label Raster."""
        data, valid, profile = _read_feature_raster_with_mask(
            feature_raster_path,
            band_indices=band_indices,
            valid_mask=valid_mask,
        )
        segments = self.segment_array(data, valid_mask=valid)
        return write_segment_raster(output_segment_path, segments, profile=profile)

    def segment(
        self,
        source: Any,
        *,
        output_path: str | Path | None = None,
        valid_mask: np.ndarray | None = None,
        band_indices: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> Path | SegmentRaster | NDArray[np.int32]:
        """Unified :class:`Segmenter` entry point for arrays or feature GeoTIFF paths."""
        if kwargs:
            raise TypeError(
                f"LocalSlicSegmenter.segment() got unexpected keyword arguments: "
                f"{sorted(kwargs)}"
            )
        if isinstance(source, np.ndarray):
            if output_path is not None:
                raise ValueError(
                    "output_path is only supported when source is a GeoTIFF path, not a "
                    "numpy ndarray (no geospatial profile is attached to an array)."
                )
            return self.segment_array(source, valid_mask=valid_mask)
        if not isinstance(source, str | Path):
            raise TypeError(
                f"LocalSlicSegmenter.segment() expects a GeoTIFF path or numpy array, "
                f"got {type(source).__name__}."
            )

        if output_path is not None:
            return self.segment_raster(
                source,
                output_path,
                band_indices=band_indices,
                valid_mask=valid_mask,
            )

        data, valid, profile = _read_feature_raster_with_mask(
            source,
            band_indices=band_indices,
            valid_mask=valid_mask,
        )
        segments = self.segment_array(data, valid_mask=valid)
        out_profile = dict(profile)
        out_profile.update(count=1, dtype="int32", nodata=SEGMENT_NODATA)
        return SegmentRaster(segments=segments, profile=out_profile)

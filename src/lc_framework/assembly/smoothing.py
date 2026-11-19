"""Logit-space empirical-Bayes spatial smoothing and tile seam healing.

Operates strictly on NumPy/SciPy arrays so Classification Assembly stays
completely standalone (ADR-0005).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import floor
from operator import index as op_index
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.special import expit

from lc_framework.config import ClassSchema
from lc_framework.io import (
    ProbabilityEncoding,
    read_probability_stack,
    write_probability_stack,
)

FloatArray = NDArray[np.float64]

_EPSILON = 1e-3


def _class_first(values: ArrayLike, class_axis: int) -> tuple[FloatArray, int, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (2, 3):
        raise ValueError("probabilities must be a 2D or 3D array")
    axis = op_index(class_axis)

    if array.ndim == 2:
        if axis not in (0, -1):
            raise ValueError("class_axis must be 0 or -1 for a 2D array")
        return array[np.newaxis, ...], 0, True

    if axis < -array.ndim or axis >= array.ndim:
        raise ValueError(f"class_axis {axis} is out of bounds for a 3D array")
    normalized_axis = axis % array.ndim
    return np.moveaxis(array, normalized_axis, 0), normalized_axis, False


def _restore_class_axis(values: FloatArray, original_axis: int, was_2d: bool) -> FloatArray:
    if was_2d:
        return values[0]
    return np.moveaxis(values, 0, original_axis)


def _logits_from_class_first(classes: FloatArray) -> FloatArray:
    if np.any(np.isinf(classes)):
        raise ValueError("probabilities must not contain infinite values")
    clipped = np.clip(classes, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def _validate_window(window_size: int, neigh_fraction: float) -> tuple[int, int]:
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not np.isfinite(neigh_fraction) or not (0.0 <= neigh_fraction <= 1.0):
        raise ValueError("neigh_fraction must be between 0 and 1")
    percentile = floor((1.0 - neigh_fraction) * 100.0 + 0.5)
    return window_size, percentile


def _finite_percentile(values: FloatArray, percentile: int) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.percentile(finite_values, percentile))


def probs_to_logits(probabilities: ArrayLike, class_axis: int = 0) -> FloatArray:
    """Convert probabilities to logits, clipping endpoints for numerical stability."""
    classes, original_axis, was_2d = _class_first(probabilities, class_axis)
    logits = _logits_from_class_first(classes)
    return _restore_class_axis(logits, original_axis, was_2d)


def logits_to_probs(
    logits: ArrayLike,
    class_axis: int = 0,
    normalize: bool = True,
) -> FloatArray:
    """Convert logits back to probabilities, optionally normalizing across classes."""
    classes, original_axis, was_2d = _class_first(logits, class_axis)
    probabilities = expit(classes)
    if normalize:
        totals = np.sum(probabilities, axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities = probabilities / totals
    return _restore_class_axis(probabilities, original_axis, was_2d)


def _smoothness_by_class(
    smoothness: float | Sequence[float] | Mapping[str, float],
    class_count: int,
    class_bands: Sequence[str] | None,
) -> FloatArray:
    if isinstance(smoothness, Mapping):
        if class_bands is None or len(class_bands) != class_count:
            raise ValueError(
                "class_bands matching the class dimension are required when "
                "smoothness is a mapping"
            )
        values = [float(smoothness.get(name, 1.0)) for name in class_bands]
    elif isinstance(smoothness, int | float):
        values = [float(smoothness)] * class_count
    else:
        values = [float(v) for v in smoothness]
        if len(values) != class_count:
            raise ValueError("smoothness sequence must match the class dimension")

    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("smoothness values must be finite and non-negative")
    return result[:, np.newaxis, np.newaxis]


def local_mean_variance(
    logits: ArrayLike,
    window_size: int = 7,
    neigh_fraction: float = 0.5,
    class_axis: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Compute local mean and variance of logits above a local neighborhood percentile."""
    size, percentile = _validate_window(window_size, neigh_fraction)
    classes, original_axis, was_2d = _class_first(logits, class_axis)
    kernel = np.ones((size, size), dtype=np.float64)
    means = np.empty_like(classes)
    variances = np.empty_like(classes)

    for idx, band in enumerate(classes):
        threshold = ndimage.generic_filter(
            band,
            _finite_percentile,
            size=size,
            mode="constant",
            cval=np.nan,
            extra_arguments=(percentile,),
        )
        selected = np.isfinite(band) & (band >= threshold)
        selected_values = np.where(selected, band, 0.0)
        counts = ndimage.convolve(selected.astype(np.float64), kernel, mode="constant", cval=0.0)
        totals = ndimage.convolve(selected_values, kernel, mode="constant", cval=0.0)
        squared_totals = ndimage.convolve(
            selected_values * selected_values, kernel, mode="constant", cval=0.0
        )

        mean = np.divide(totals, counts, out=band.copy(), where=counts > 0)
        variance = (
            np.divide(squared_totals, counts, out=np.ones_like(band), where=counts > 0)
            - np.square(mean)
        )
        variance = np.maximum(variance, 0.0)
        variance[(counts == 0) | (variance == 0)] = 1.0
        mean[~np.isfinite(band)] = np.nan
        variance[~np.isfinite(band)] = 1.0
        means[idx] = mean
        variances[idx] = variance

    return (
        _restore_class_axis(means, original_axis, was_2d),
        _restore_class_axis(variances, original_axis, was_2d),
    )


def bayesian_smooth(
    probabilities: ArrayLike,
    smoothness: float | Sequence[float] | Mapping[str, float] = 1.0,
    window_size: int = 7,
    neigh_fraction: float = 0.5,
    class_axis: int = 0,
    class_bands: Sequence[str] | None = None,
) -> FloatArray:
    """Apply empirical-Bayes spatial smoothing in logit space and renormalize to 1.0."""
    classes, original_axis, was_2d = _class_first(probabilities, class_axis)
    # Treat non-finite pixels and all-zero GEE background pixels as nodata
    invalid_pixels = ~np.all(np.isfinite(classes), axis=0)
    if not was_2d:
        invalid_pixels |= ~np.any(classes > 0.0, axis=0)
    masked_probs = classes.copy()
    masked_probs[:, invalid_pixels] = np.nan

    logits = _logits_from_class_first(masked_probs)
    mean, variance = local_mean_variance(
        logits,
        window_size=window_size,
        neigh_fraction=neigh_fraction,
        class_axis=0,
    )
    sigma2 = _smoothness_by_class(smoothness, logits.shape[0], class_bands)
    denominator = sigma2 + variance
    posterior_logits = logits * (variance / denominator) + mean * (sigma2 / denominator)
    posterior = expit(posterior_logits)
    totals = np.sum(posterior, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = posterior / totals
    posterior[:, invalid_pixels] = np.nan
    return _restore_class_axis(posterior, original_axis, was_2d)


def bayesian_smooth_raster(
    input_prob_path: str | Path,
    output_prob_path: str | Path,
    *,
    class_schema: ClassSchema | None = None,
    smoothness: float | Sequence[float] | Mapping[str, float] = 1.0,
    window_size: int = 7,
    neigh_fraction: float = 0.5,
    encoding: ProbabilityEncoding = "uint16",
) -> Path:
    """Apply logit-space empirical-Bayes smoothing to a Class Probability Stack GeoTIFF."""
    stack = read_probability_stack(input_prob_path, class_schema=class_schema)
    schema = stack.manifest.class_schema
    smoothed = bayesian_smooth(
        stack.probabilities,
        smoothness=smoothness,
        window_size=window_size,
        neigh_fraction=neigh_fraction,
        class_axis=0,
        class_bands=schema.class_names,
    ).astype(np.float32)
    return write_probability_stack(
        output_prob_path,
        smoothed,
        profile=stack.profile,
        class_schema=schema,
        encoding=encoding,
        epoch=stack.manifest.epoch,
        paradigm=stack.manifest.paradigm,
        source=f"bayesian_smooth:{Path(input_prob_path).name}",
    )


def heal_array_seams(
    data: np.ndarray,
    nodata: float | int = 0,
    max_seam_width: int = 4,
) -> np.ndarray:
    """Fill thin internal nodata seams (<= ``max_seam_width`` px) via nearest valid pixel.

    For 3D probability arrays ``(C, H, W)``, the entire probability vector is copied
    from the same nearest pixel so probabilities remain normalized.
    """
    if max_seam_width < 1:
        return data.copy()

    is_3d = data.ndim == 3
    if is_3d:
        if np.issubdtype(data.dtype, np.floating):
            valid_2d = np.any((data != nodata) & np.isfinite(data), axis=0)
        else:
            valid_2d = np.any(data != nodata, axis=0)
    else:
        if np.issubdtype(data.dtype, np.floating):
            valid_2d = (data != nodata) & np.isfinite(data)
        else:
            valid_2d = data != nodata

    if not np.any(valid_2d) or np.all(valid_2d):
        return data.copy()

    distances, indices = ndimage.distance_transform_edt(
        ~valid_2d,
        return_distances=True,
        return_indices=True,
    )
    fill_mask = (~valid_2d) & (distances <= max_seam_width)
    if not np.any(fill_mask):
        return data.copy()

    nearest_row = indices[0]
    nearest_col = indices[1]

    out = data.copy()
    if is_3d:
        for b in range(out.shape[0]):
            band = out[b]
            band[fill_mask] = band[nearest_row[fill_mask], nearest_col[fill_mask]]
    else:
        out[fill_mask] = out[nearest_row[fill_mask], nearest_col[fill_mask]]
    return out


def heal_probability_array_spikes(
    data: np.ndarray,
    *,
    spike_threshold: float | None = None,
    major_spike_threshold: float | None = None,
    min_bands_agreement: int = 3,
) -> np.ndarray:
    """Detect and heal isolated 1-pixel GEE tile computation seams in probability arrays."""
    is_2d = data.ndim == 2
    data_3d = data[np.newaxis, :, :] if is_2d else data
    count, height, width = data_3d.shape
    if width < 3 and height < 3:
        return data.copy()

    is_float_scale = bool(np.issubdtype(data.dtype, np.floating) and (np.nanmax(data) <= 1.05))
    s_thresh = (0.01 if is_float_scale else 100.0) if spike_threshold is None else spike_threshold
    m_thresh = (
        (0.06 if is_float_scale else 600.0)
        if major_spike_threshold is None
        else major_spike_threshold
    )

    out = data_3d.astype(np.float32, copy=True)

    # Horizontal pass (vertical seam columns)
    if width >= 3:
        diff_left = out[:, :, 1:-1] - out[:, :, :-2]
        diff_right = out[:, :, 1:-1] - out[:, :, 2:]
        is_peak = (diff_left > s_thresh) & (diff_right > s_thresh)
        is_valley = (diff_left < -s_thresh) & (diff_right < -s_thresh)
        has_big = np.any((diff_left > m_thresh) & (diff_right > m_thresh), axis=0)
        has_multi = (np.sum(is_peak, axis=0) + np.sum(is_valley, axis=0)) >= min_bands_agreement
        seam_cols = has_big | has_multi
        if np.any(seam_cols):
            for b in range(count):
                band = out[b]
                band[:, 1:-1] = np.where(
                    seam_cols,
                    0.5 * (band[:, :-2] + band[:, 2:]),
                    band[:, 1:-1],
                )

    # Vertical pass (horizontal seam rows)
    if height >= 3:
        diff_top = out[:, 1:-1, :] - out[:, :-2, :]
        diff_bottom = out[:, 1:-1, :] - out[:, 2:, :]
        is_peak_v = (diff_top > s_thresh) & (diff_bottom > s_thresh)
        is_valley_v = (diff_top < -s_thresh) & (diff_bottom < -s_thresh)
        has_big_v = np.any((diff_top > m_thresh) & (diff_bottom > m_thresh), axis=0)
        has_multi_v = (
            np.sum(is_peak_v, axis=0) + np.sum(is_valley_v, axis=0)
        ) >= min_bands_agreement
        seam_rows = has_big_v | has_multi_v
        if np.any(seam_rows):
            for b in range(count):
                band = out[b]
                band[1:-1, :] = np.where(
                    seam_rows,
                    0.5 * (band[:-2, :] + band[2:, :]),
                    band[1:-1, :],
                )

    result = (
        np.rint(out).astype(data.dtype)
        if np.issubdtype(data.dtype, np.integer)
        else out.astype(data.dtype)
    )
    return result[0] if is_2d else result


def heal_raster_seams(
    input_path: str | Path,
    output_path: str | Path,
    *,
    nodata: float | int | None = None,
    max_seam_width: int = 4,
) -> Path:
    """Heal thin internal nodata seams on a single-band or multi-band GeoTIFF."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    if in_path.resolve() == out_path.resolve():
        raise ValueError("output_path must differ from input_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        data = src.read()
        profile = dict(src.profile)
        profile["driver"] = "GTiff"
        profile["compress"] = "deflate"
        tags = src.tags()
        descriptions = src.descriptions
        effective_nodata = (
            nodata if nodata is not None else (src.nodata if src.nodata is not None else 0)
        )

    if data.shape[0] == 1:
        healed = heal_array_seams(
            data[0], nodata=effective_nodata, max_seam_width=max_seam_width
        )[np.newaxis, :, :]
    else:
        healed = heal_array_seams(
            data, nodata=effective_nodata, max_seam_width=max_seam_width
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(healed)
        if tags:
            dst.update_tags(**tags)
        for idx, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)
    return out_path


def heal_probability_raster_spikes(
    input_prob_path: str | Path,
    output_prob_path: str | Path,
    *,
    class_schema: ClassSchema | None = None,
    spike_threshold: float | None = None,
    major_spike_threshold: float | None = None,
    min_bands_agreement: int = 3,
    encoding: ProbabilityEncoding = "uint16",
) -> Path:
    """Heal 1-px GEE tile computation seams on a Class Probability Stack GeoTIFF."""
    stack = read_probability_stack(input_prob_path, class_schema=class_schema)
    healed = heal_probability_array_spikes(
        stack.probabilities,
        spike_threshold=spike_threshold,
        major_spike_threshold=major_spike_threshold,
        min_bands_agreement=min_bands_agreement,
    )
    # Renormalize valid pixels after linear interpolation across seams
    valid = np.all(np.isfinite(healed), axis=0) & np.any(healed > 0.0, axis=0)
    if np.any(valid):
        totals = np.sum(healed[:, valid], axis=0)
        pos = totals > 0.0
        if np.any(pos):
            vr, vc = np.nonzero(valid)
            healed[:, vr[pos], vc[pos]] /= totals[pos][None, :]
    return write_probability_stack(
        output_prob_path,
        healed,
        profile=stack.profile,
        class_schema=stack.manifest.class_schema,
        encoding=encoding,
        epoch=stack.manifest.epoch,
        paradigm=stack.manifest.paradigm,
        source=f"heal_spikes:{Path(input_prob_path).name}",
    )

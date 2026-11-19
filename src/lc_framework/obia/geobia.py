"""Pure Geo-Object Classification (GEOBIA) pipeline.

Implements the end-to-end object-based workflow:
``Segment -> Pool Features -> Assign Labels -> Classify Segments -> Project Probabilities``,
emitting a canonical ``Class Probability Stack`` (COG + ``*.manifest.json`` with
``paradigm='geobia'``) fully compatible with downstream accuracy evaluation and
trajectory analysis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform as transform_coords

from lc_framework.config import ClassSchema, ProjectConfig
from lc_framework.io import (
    ProbabilityEncoding,
    read_segment_raster,
    write_probability_stack,
)
from lc_framework.obia.base import Segmenter, valid_segment_mask
from lc_framework.obia.gee_snic import GEESnicSegmenter
from lc_framework.obia.local_slic import LocalSlicSegmenter


def _require_sklearn_rf() -> Any:
    """Import ``RandomForestClassifier`` lazily with an actionable install hint."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for Geo-Object Classification. "
            "Install it with: pip install 'lc-framework[obia]' (or 'lc-framework[ml]')"
        ) from exc
    return RandomForestClassifier


def _require_geopandas() -> Any:
    """Import ``geopandas`` lazily only when reading vector label files."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required to load vector geometry sample files. "
            "Install it with: pip install 'lc-framework[obia]' "
            "or pass a tabular DataFrame/CSV with (x, y, class_id) columns."
        ) from exc
    return gpd


@dataclass(frozen=True)
class GeobiaClassificationResult:
    """In-memory outputs from probabilistic classification of pooled geo-objects."""

    segment_ids: NDArray[np.int32]
    probabilities: NDArray[np.float32]
    predicted_class_ids: NDArray[np.int32]
    feature_names: tuple[str, ...]
    class_schema: ClassSchema
    estimator: Any
    n_labeled_segments: int


@dataclass(frozen=True)
class GeobiaPipelineOutputs:
    """Artifacts and tables emitted by :func:`run_geobia_classification`."""

    segment_raster: Path
    probabilities: Path
    segment_features: pd.DataFrame
    segment_labels: pd.DataFrame
    classification: GeobiaClassificationResult


def _resolve_band_names(
    n_bands: int,
    descriptions: Sequence[str | None],
    explicit_names: Sequence[str] | None,
) -> list[str]:
    if explicit_names is not None:
        if len(explicit_names) != n_bands:
            raise ValueError(
                f"band_names length ({len(explicit_names)}) must match raster band count "
                f"({n_bands})."
            )
        names = [str(n).strip() for n in explicit_names]
    else:
        names = []
        for idx in range(n_bands):
            raw = descriptions[idx] if idx < len(descriptions) else None
            cleaned = raw.strip() if raw and raw.strip() else f"band_{idx + 1}"
            names.append(cleaned)

    if len(set(names)) != len(names):
        raise ValueError(f"Feature band names must be unique, got {names!r}.")
    return names


def _compute_segment_morphology(
    segments: NDArray[np.int32],
    valid: NDArray[np.bool_],
    inv_flat: NDArray[np.intp],
    n_segments: int,
    counts: NDArray[np.intp],
) -> dict[str, NDArray[np.float64]]:
    """Compute vectorized per-segment morphological descriptors in O(H * W)."""
    height, width = segments.shape
    rows, cols = np.nonzero(valid)

    min_row = np.full(n_segments, height, dtype=np.int64)
    max_row = np.full(n_segments, -1, dtype=np.int64)
    min_col = np.full(n_segments, width, dtype=np.int64)
    max_col = np.full(n_segments, -1, dtype=np.int64)

    np.minimum.at(min_row, inv_flat, rows)
    np.maximum.at(max_row, inv_flat, rows)
    np.minimum.at(min_col, inv_flat, cols)
    np.maximum.at(max_col, inv_flat, cols)

    bbox_h = (max_row - min_row + 1).astype(np.float64)
    bbox_w = (max_col - min_col + 1).astype(np.float64)
    area = counts.astype(np.float64)
    bbox_aspect = np.minimum(bbox_h, bbox_w) / np.maximum(bbox_h, bbox_w)
    extent = area / (bbox_h * bbox_w)

    # Exact 4-neighbor pixel-edge perimeter per segment
    same_up = np.zeros((height, width), dtype=bool)
    same_up[1:, :] = valid[1:, :] & valid[:-1, :] & (segments[1:, :] == segments[:-1, :])
    same_down = np.zeros((height, width), dtype=bool)
    same_down[:-1, :] = same_up[1:, :]

    same_left = np.zeros((height, width), dtype=bool)
    same_left[:, 1:] = valid[:, 1:] & valid[:, :-1] & (segments[:, 1:] == segments[:, :-1])
    same_right = np.zeros((height, width), dtype=bool)
    same_right[:, :-1] = same_left[:, 1:]

    pixel_edges = (
        4
        - same_up.astype(np.int32)
        - same_down.astype(np.int32)
        - same_left.astype(np.int32)
        - same_right.astype(np.int32)
    )
    perimeter = np.bincount(
        inv_flat,
        weights=pixel_edges[valid].astype(np.float64),
        minlength=n_segments,
    ).astype(np.float64)
    compactness = np.clip(
        (4.0 * math.pi * area) / np.maximum(perimeter * perimeter, 1.0),
        0.0,
        1.0,
    )

    return {
        "area_px": area,
        "bbox_height": bbox_h,
        "bbox_width": bbox_w,
        "bbox_aspect": bbox_aspect,
        "extent": extent,
        "perimeter_px": perimeter,
        "compactness": compactness,
    }


def pool_features_by_segments(
    feature_raster_path: str | Path,
    segment_raster_path: str | Path,
    *,
    band_names: Sequence[str] | None = None,
    include_std: bool = True,
    include_morphology: bool = True,
    repair_overflow: bool = True,
    align_segments: bool = True,
    min_pixels: int = 1,
) -> pd.DataFrame:
    """Aggregate per-segment feature statistics (mean, optional std & morphology)."""
    if min_pixels < 1:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels!r}.")

    with rasterio.open(feature_raster_path) as src:
        raw_features = src.read()
        features = raw_features.astype(np.float64)
        feat_profile = dict(src.profile)
        feat_nodata = src.nodata
        resolved_names = _resolve_band_names(src.count, src.descriptions, band_names)

    seg_raster = read_segment_raster(
        segment_raster_path,
        reference_profile=feat_profile,
        align_to_reference=align_segments,
        repair_overflow=repair_overflow,
    )
    segments = seg_raster.segments

    feat_valid = np.all(np.isfinite(features), axis=0)
    if feat_nodata is not None and not np.isnan(feat_nodata):
        feat_valid &= ~np.any(raw_features == feat_nodata, axis=0)

    valid = valid_segment_mask(segments) & feat_valid
    if not np.any(valid):
        raise ValueError(
            "No overlapping valid pixels found between feature raster and Segment Label Raster."
        )

    seg_vals = segments[valid]
    unique_ids, inv_flat = np.unique(seg_vals, return_inverse=True)
    n_segments = len(unique_ids)
    counts = np.bincount(inv_flat, minlength=n_segments)

    columns: dict[str, Any] = {
        "segment_id": unique_ids.astype(np.int32),
        "pixel_count": counts.astype(np.int64),
    }

    counts_f = counts.astype(np.float64)
    for band_idx, name in enumerate(resolved_names):
        vals = features[band_idx, valid]
        sums = np.bincount(inv_flat, weights=vals, minlength=n_segments)
        means = sums / counts_f
        columns[f"{name}_mean"] = means.astype(np.float64)
        if include_std:
            # Centered two-pass variance avoids catastrophic cancellation on high-offset bands
            diffs = vals - means[inv_flat]
            var = np.bincount(inv_flat, weights=diffs * diffs, minlength=n_segments) / counts_f
            columns[f"{name}_std"] = np.sqrt(np.maximum(var, 0.0)).astype(np.float64)

    if include_morphology:
        columns.update(
            _compute_segment_morphology(segments, valid, inv_flat, n_segments, counts)
        )

    df = pd.DataFrame(columns)
    if min_pixels > 1:
        df = df.loc[df["pixel_count"] >= min_pixels].reset_index(drop=True)
    return df


def _is_geodataframe_like(obj: Any) -> bool:
    """Return True for GeoDataFrame-like vector tables without importing geopandas."""
    return hasattr(obj, "geometry") and hasattr(obj, "crs") and hasattr(obj, "to_crs")


def _extract_segment_class_pairs_from_points(
    segments: NDArray[np.int32],
    profile: dict[str, Any],
    df: pd.DataFrame,
    *,
    class_col: str,
    x_col: str,
    y_col: str,
    samples_crs: str | None,
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    for required in (x_col, y_col, class_col):
        if required not in df.columns:
            raise ValueError(
                f"Column {required!r} not found in sample table columns {list(df.columns)}."
            )

    xs_raw = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    ys_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    cls_raw = pd.to_numeric(df[class_col], errors="coerce").to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(xs_raw) & np.isfinite(ys_raw) & np.isfinite(cls_raw)
    if not np.any(finite_rows):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)

    xs = xs_raw[finite_rows]
    ys = ys_raw[finite_rows]
    cls = cls_raw[finite_rows].astype(np.int64)

    raster_crs = profile.get("crs")
    if (
        samples_crs is not None
        and raster_crs is not None
        and CRS.from_user_input(samples_crs) != CRS.from_user_input(raster_crs)
    ):
        tx, ty = transform_coords(samples_crs, raster_crs, xs.tolist(), ys.tolist())
        xs = np.asarray(tx, dtype=np.float64)
        ys = np.asarray(ty, dtype=np.float64)

    inv_transform = ~profile["transform"]
    col_f, row_f = inv_transform * (xs, ys)
    cols = np.floor(col_f).astype(np.int64)
    rows = np.floor(row_f).astype(np.int64)

    height, width = segments.shape
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    seg_ids = segments[rows[in_bounds], cols[in_bounds]]
    cls_in = cls[in_bounds]

    valid = valid_segment_mask(seg_ids)
    return seg_ids[valid].astype(np.int32), cls_in[valid]


def _extract_segment_class_pairs_from_vector(
    segments: NDArray[np.int32],
    profile: dict[str, Any],
    gdf: Any,
    *,
    class_col: str,
    samples_crs: str | None = None,
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    """Rasterize vector geometries onto the Segment Label Raster grid.

    When multiple vector geometries overlap the same pixel, later rows overwrite
    earlier rows during ``rasterio.features.rasterize``.
    """
    if class_col not in gdf.columns:
        raise ValueError(f"Column {class_col!r} not found in vector samples.")

    raster_crs = profile.get("crs")
    effective_crs = gdf.crs if gdf.crs is not None else samples_crs
    if (
        effective_crs is not None
        and raster_crs is not None
        and CRS.from_user_input(effective_crs) != CRS.from_user_input(raster_crs)
    ):
        if gdf.crs is None and hasattr(gdf, "set_crs"):
            gdf = gdf.set_crs(effective_crs)
        gdf = gdf.to_crs(raster_crs)

    shapes = [
        (geom, int(val))
        for geom, val in zip(gdf.geometry, gdf[class_col], strict=True)
        if geom is not None and not geom.is_empty and not pd.isna(val)
    ]
    if not shapes:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)

    # Use int32 (supported by all rasterio>=1.3 versions; int64 was added in rasterio 1.4)
    nodata_sentinel = -2_147_483_648
    burned = rasterize(
        shapes=shapes,
        out_shape=segments.shape,
        transform=profile["transform"],
        fill=nodata_sentinel,
        dtype="int32",
    )
    mask = (burned != nodata_sentinel) & valid_segment_mask(segments)
    return segments[mask].astype(np.int32), burned[mask].astype(np.int64)


def assign_segment_labels(
    segment_raster_path: str | Path,
    samples: pd.DataFrame | str | Path | Any,
    *,
    class_col: str = "class_id",
    x_col: str = "x",
    y_col: str = "y",
    samples_crs: str | None = None,
    min_samples_per_segment: int = 1,
    min_purity: float = 0.0,
    reference_profile: dict[str, Any] | None = None,
    align_to_reference: bool = False,
    repair_overflow: bool = True,
) -> pd.DataFrame:
    """Assign majority-vote reference class labels to geo-objects.

    Supports tabular ``DataFrame`` / ``.csv`` point tables without requiring
    ``geopandas``, as well as vector geometries (``GeoDataFrame``, ``.gpkg``,
    ``.geojson``, ``.shp``) via lazy ``geopandas`` + ``rasterio.features.rasterize``.
    """
    if min_samples_per_segment < 1:
        raise ValueError(
            f"min_samples_per_segment must be >= 1, got {min_samples_per_segment!r}."
        )
    if not (0.0 <= min_purity <= 1.0):
        raise ValueError(f"min_purity must lie in [0, 1], got {min_purity!r}.")

    seg_raster = read_segment_raster(
        segment_raster_path,
        reference_profile=reference_profile,
        align_to_reference=align_to_reference,
        repair_overflow=repair_overflow,
    )
    segments = seg_raster.segments
    profile = seg_raster.profile

    if isinstance(samples, str | Path):
        sample_path = Path(samples)
        if sample_path.suffix.lower() == ".csv":
            seg_ids, cls_ids = _extract_segment_class_pairs_from_points(
                segments,
                profile,
                pd.read_csv(sample_path),
                class_col=class_col,
                x_col=x_col,
                y_col=y_col,
                samples_crs=samples_crs,
            )
        else:
            gpd = _require_geopandas()
            gdf = gpd.read_file(sample_path)
            seg_ids, cls_ids = _extract_segment_class_pairs_from_vector(
                segments, profile, gdf, class_col=class_col, samples_crs=samples_crs
            )
    elif _is_geodataframe_like(samples):
        seg_ids, cls_ids = _extract_segment_class_pairs_from_vector(
            segments, profile, samples, class_col=class_col, samples_crs=samples_crs
        )
    elif isinstance(samples, pd.DataFrame):
        seg_ids, cls_ids = _extract_segment_class_pairs_from_points(
            segments,
            profile,
            samples,
            class_col=class_col,
            x_col=x_col,
            y_col=y_col,
            samples_crs=samples_crs,
        )
    else:
        raise TypeError(
            "samples must be a DataFrame, GeoDataFrame, CSV path, or vector file path."
        )

    if seg_ids.size == 0:
        return pd.DataFrame(
            {
                "segment_id": pd.Series(dtype=np.int32),
                class_col: pd.Series(dtype=np.int64),
                "sample_count": pd.Series(dtype=np.int64),
                "purity": pd.Series(dtype=np.float64),
            }
        )

    pairs = pd.DataFrame({"segment_id": seg_ids, class_col: cls_ids})
    counts_df = (
        pairs.groupby(["segment_id", class_col], as_index=False)
        .size()
        .rename(columns={"size": "vote_count"})
    )
    totals = pairs.groupby("segment_id").size().rename("sample_count")
    # Deterministic majority vote: highest vote_count, tie-break by smallest class_id
    winners = (
        counts_df.sort_values(
            by=["segment_id", "vote_count", class_col],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["segment_id"], keep="first")
        .merge(totals, on="segment_id", how="left")
    )
    winners["purity"] = winners["vote_count"] / winners["sample_count"]
    filtered = winners.loc[
        (winners["sample_count"] >= min_samples_per_segment)
        & (winners["purity"] >= min_purity),
        ["segment_id", class_col, "sample_count", "purity"],
    ].reset_index(drop=True)
    filtered["segment_id"] = filtered["segment_id"].astype(np.int32)
    filtered[class_col] = filtered[class_col].astype(np.int64)
    return filtered


def classify_segments(
    segment_features: pd.DataFrame,
    segment_labels: pd.DataFrame,
    *,
    class_schema: ClassSchema,
    class_col: str = "class_id",
    feature_cols: Sequence[str] | None = None,
    estimator: Any | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GeobiaClassificationResult:
    """Train a probabilistic classifier on labeled geo-objects and predict all segments."""
    if "segment_id" not in segment_features.columns:
        raise ValueError("segment_features must include a 'segment_id' column.")
    if "segment_id" not in segment_labels.columns or class_col not in segment_labels.columns:
        raise ValueError(f"segment_labels must include 'segment_id' and {class_col!r} columns.")
    if segment_features["segment_id"].duplicated().any():
        raise ValueError("segment_features must contain unique 'segment_id' rows.")
    if segment_labels["segment_id"].duplicated().any():
        raise ValueError("segment_labels must contain unique 'segment_id' rows.")

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in segment_features.columns]
        if missing:
            raise ValueError(f"Requested feature_cols missing from segment_features: {missing}")
        resolved_features = list(feature_cols)
    else:
        # Exclude metadata columns so pixel_count never leaks morphology when disabled
        # or duplicates area_px when morphology is enabled.
        metadata_cols = {"segment_id", "pixel_count"}
        resolved_features = [
            c
            for c in segment_features.columns
            if c not in metadata_cols and pd.api.types.is_numeric_dtype(segment_features[c])
        ]
    if not resolved_features:
        raise ValueError("No numeric feature columns available for geo-object classification.")

    train_df = segment_features.merge(
        segment_labels[["segment_id", class_col]],
        on="segment_id",
        how="inner",
    )
    if len(train_df) < 2:
        raise ValueError(
            f"At least 2 labeled segments are required for training, found {len(train_df)}."
        )

    y_train = train_df[class_col].to_numpy(dtype=np.int64)
    schema_id_set = set(class_schema.class_ids)
    unknown_labels = sorted({int(v) for v in y_train} - schema_id_set)
    if unknown_labels:
        raise ValueError(
            f"Training segment labels {unknown_labels} are not defined in ClassSchema "
            f"{class_schema.class_ids}."
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "At least 2 distinct land-cover classes are required to train a probabilistic "
            "geo-object classifier."
        )

    # Replace any non-finite values with training column medians (fallback 0.0)
    x_train_raw = train_df[resolved_features].to_numpy(dtype=np.float64)
    x_all_raw = segment_features[resolved_features].to_numpy(dtype=np.float64)
    medians = np.zeros(len(resolved_features), dtype=np.float64)
    for col_idx in range(len(resolved_features)):
        finite_vals = x_train_raw[:, col_idx][np.isfinite(x_train_raw[:, col_idx])]
        medians[col_idx] = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0
        bad_train = ~np.isfinite(x_train_raw[:, col_idx])
        if np.any(bad_train):
            x_train_raw[bad_train, col_idx] = medians[col_idx]
        bad_all = ~np.isfinite(x_all_raw[:, col_idx])
        if np.any(bad_all):
            x_all_raw[bad_all, col_idx] = medians[col_idx]

    if estimator is None:
        rf_cls = _require_sklearn_rf()
        clf = rf_cls(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
        )
    else:
        clf = estimator

    clf.fit(x_train_raw, y_train)
    raw_proba = np.asarray(clf.predict_proba(x_all_raw), dtype=np.float64)
    if not np.all(np.isfinite(raw_proba)):
        raise ValueError("Estimator predict_proba() returned non-finite (NaN/inf) probabilities.")
    model_classes = [int(c) for c in clf.classes_]

    n_segments = len(segment_features)
    n_schema_classes = len(class_schema)
    aligned = np.zeros((n_segments, n_schema_classes), dtype=np.float64)
    id_to_col = {cid: idx for idx, cid in enumerate(class_schema.class_ids)}
    for model_col, cid in enumerate(model_classes):
        if cid in id_to_col:
            aligned[:, id_to_col[cid]] = raw_proba[:, model_col]

    aligned = np.clip(aligned, 0.0, 1.0)
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(
            "Estimator predict_proba() produced zero total probability mass for one or more "
            "segments after alignment to ClassSchema."
        )
    aligned = aligned / row_sums
    probs_f32 = aligned.astype(np.float32)

    class_id_lookup = np.asarray(class_schema.class_ids, dtype=np.int32)
    predicted_ids = class_id_lookup[np.argmax(probs_f32, axis=1)]

    return GeobiaClassificationResult(
        segment_ids=segment_features["segment_id"].to_numpy(dtype=np.int32),
        probabilities=probs_f32,
        predicted_class_ids=predicted_ids,
        feature_names=tuple(resolved_features),
        class_schema=class_schema,
        estimator=clf,
        n_labeled_segments=len(train_df),
    )


def project_segment_probabilities(
    segment_raster_path: str | Path,
    segment_ids: Sequence[int] | np.ndarray,
    segment_probabilities: np.ndarray,
    output_path: str | Path,
    *,
    class_schema: ClassSchema,
    encoding: ProbabilityEncoding = "uint16",
    epoch: str | None = None,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
    reference_profile: dict[str, Any] | None = None,
    align_to_reference: bool = False,
    repair_overflow: bool = True,
) -> Path:
    """Project per-segment probability vectors onto a canonical Class Probability Stack COG."""
    ids_arr = np.asarray(segment_ids, dtype=np.int64)
    probs_arr = np.asarray(segment_probabilities, dtype=np.float32)
    if ids_arr.ndim != 1:
        raise ValueError(f"segment_ids must be 1D, got shape {ids_arr.shape}.")
    if probs_arr.ndim != 2 or probs_arr.shape != (ids_arr.size, len(class_schema)):
        raise ValueError(
            f"segment_probabilities must have shape ({ids_arr.size}, {len(class_schema)}), "
            f"got {probs_arr.shape}."
        )
    if np.unique(ids_arr).size != ids_arr.size:
        raise ValueError("segment_ids must contain unique segment identifiers.")

    seg_raster = read_segment_raster(
        segment_raster_path,
        reference_profile=reference_profile,
        align_to_reference=align_to_reference,
        repair_overflow=repair_overflow,
    )
    segments = seg_raster.segments
    height, width = segments.shape
    n_classes = len(class_schema)

    order = np.argsort(ids_arr)
    sorted_ids = ids_arr[order]
    sorted_probs = probs_arr[order]

    valid = valid_segment_mask(segments)
    out_probs = np.full((n_classes, height, width), np.nan, dtype=np.float32)

    if np.any(valid) and sorted_ids.size > 0:
        seg_valid = segments[valid].astype(np.int64)
        idx = np.searchsorted(sorted_ids, seg_valid)
        in_range = idx < sorted_ids.size
        matched = np.zeros_like(in_range, dtype=bool)
        matched[in_range] = sorted_ids[idx[in_range]] == seg_valid[in_range]

        if np.any(matched):
            valid_rows, valid_cols = np.nonzero(valid)
            m_rows = valid_rows[matched]
            m_cols = valid_cols[matched]
            out_probs[:, m_rows, m_cols] = sorted_probs[idx[matched]].T

    return write_probability_stack(
        output_path,
        out_probs,
        profile=seg_raster.profile,
        class_schema=class_schema,
        encoding=encoding,
        epoch=epoch,
        paradigm="geobia",
        source=source or Path(segment_raster_path).name,
        extra=extra,
    )


def run_geobia_classification(
    feature_raster_path: str | Path,
    samples: pd.DataFrame | str | Path | Any,
    output_dir: str | Path,
    *,
    class_schema: ClassSchema | None = None,
    config: ProjectConfig | None = None,
    segment_raster_path: str | Path | None = None,
    segmenter: Segmenter | None = None,
    band_names: Sequence[str] | None = None,
    include_std: bool = True,
    include_morphology: bool = True,
    class_col: str = "class_id",
    x_col: str = "x",
    y_col: str = "y",
    samples_crs: str | None = None,
    min_samples_per_segment: int = 1,
    min_purity: float = 0.0,
    estimator: Any | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
    encoding: ProbabilityEncoding = "uint16",
    epoch: str | None = None,
    repair_overflow: bool = True,
) -> GeobiaPipelineOutputs:
    """Run pure Geo-Object Classification end to end.

    Accepts either an explicit ``class_schema`` or a ``ProjectConfig`` (ADR-0005).
    When ``segment_raster_path`` is supplied (for example, a GEE SNIC export),
    segmentation is skipped and that Segment Label Raster is aligned consistently
    to ``feature_raster_path`` across pooling, label assignment, and projection.
    Otherwise, ``segmenter`` (defaulting to :class:`LocalSlicSegmenter`) is run
    on ``feature_raster_path`` to generate a standardized Segment Label Raster.
    """
    resolved_schema = (
        class_schema
        if class_schema is not None
        else (config.class_schema if config is not None else None)
    )
    if resolved_schema is None:
        raise ValueError("Provide either class_schema or config to run_geobia_classification().")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_raster_path) as src:
        feat_profile = dict(src.profile)

    if segment_raster_path is not None:
        resolved_seg_path = Path(segment_raster_path)
    else:
        active_segmenter: Segmenter = (
            segmenter if segmenter is not None else LocalSlicSegmenter()
        )
        if isinstance(active_segmenter, GEESnicSegmenter):
            raise TypeError(
                "GEESnicSegmenter runs in Google Earth Engine and cannot write a local Segment "
                "Label Raster directly. Export the SNIC raster with export_tiles() first, then "
                "pass its local path via segment_raster_path."
            )
        resolved_seg_path = out_root / "segments.tif"
        seg_result = active_segmenter.segment(
            feature_raster_path,
            output_path=resolved_seg_path,
        )
        if not isinstance(seg_result, Path):
            raise TypeError(
                f"Segmenter {active_segmenter.provider_name!r} did not write a local Segment "
                "Label Raster Path. Export the raster first and pass segment_raster_path."
            )
        resolved_seg_path = seg_result

    segment_features = pool_features_by_segments(
        feature_raster_path,
        resolved_seg_path,
        band_names=band_names,
        include_std=include_std,
        include_morphology=include_morphology,
        repair_overflow=repair_overflow,
        align_segments=True,
    )
    segment_labels = assign_segment_labels(
        resolved_seg_path,
        samples,
        class_col=class_col,
        x_col=x_col,
        y_col=y_col,
        samples_crs=samples_crs,
        min_samples_per_segment=min_samples_per_segment,
        min_purity=min_purity,
        reference_profile=feat_profile,
        align_to_reference=True,
        repair_overflow=repair_overflow,
    )
    classification = classify_segments(
        segment_features,
        segment_labels,
        class_schema=resolved_schema,
        class_col=class_col,
        estimator=estimator,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    prob_path = project_segment_probabilities(
        resolved_seg_path,
        classification.segment_ids,
        classification.probabilities,
        out_root / "geobia_probabilities.tif",
        class_schema=resolved_schema,
        encoding=encoding,
        epoch=epoch,
        source=Path(feature_raster_path).name,
        extra={
            "n_segments": int(len(classification.segment_ids)),
            "n_labeled_segments": int(classification.n_labeled_segments),
            "n_features": int(len(classification.feature_names)),
        },
        reference_profile=feat_profile,
        align_to_reference=True,
        repair_overflow=repair_overflow,
    )

    return GeobiaPipelineOutputs(
        segment_raster=resolved_seg_path,
        probabilities=prob_path,
        segment_features=segment_features,
        segment_labels=segment_labels,
        classification=classification,
    )

"""I/O contracts for Class Probability Stacks, Segment Label Rasters, CSVs, and tiled GEE exports.

Implements ADR-0003 (artifact-decoupled GEE <-> Python boundary) and ADR-0005
(standalone modularity without requiring ``earthengine-api``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio
from pydantic import BaseModel, ConfigDict, Field
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.ndimage import label as connected_components

from lc_framework.config import ClassDefinition, ClassSchema

GEE_INT32_CLAMP_VALUE: int = 2_147_483_647
UINT16_PROB_SCALE: float = 10000.0
UINT16_PROB_NODATA: int = 65535

ProbabilityEncoding = Literal["uint16", "float32"]
ClassificationParadigm = Literal["pixel", "geobia", "hybrid", "external"]


class ProbabilityManifest(BaseModel):
    """Sidecar JSON manifest paired with every Class Probability Stack raster."""

    model_config = ConfigDict(frozen=True)

    class_schema: ClassSchema
    dtype: ProbabilityEncoding = "uint16"
    scale_factor: float = Field(default=UINT16_PROB_SCALE, gt=0)
    nodata: float | int | None = UINT16_PROB_NODATA
    crs: str = ""
    epoch: str | None = None
    paradigm: ClassificationParadigm = "pixel"
    source: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> ProbabilityManifest:
        """Load and validate a sidecar manifest from JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def to_json(self, path: str | Path) -> Path:
        """Write this manifest as pretty-printed JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out


@dataclass(frozen=True)
class ProbabilityStack:
    """In-memory representation of a Class Probability Stack in ``float32`` ``[0, 1]``."""

    probabilities: np.ndarray
    valid_mask: np.ndarray
    profile: dict[str, Any]
    manifest: ProbabilityManifest


@dataclass(frozen=True)
class SegmentRaster:
    """In-memory representation of a Segment Label Raster aligned to a spatial grid."""

    segments: np.ndarray
    profile: dict[str, Any]


@dataclass(frozen=True)
class TileSpec:
    """Geographic bounding box for a single tile in an auto-generated export grid."""

    row: int
    col: int
    bounds: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)

    @property
    def tile_id(self) -> str:
        return f"r{self.row}_c{self.col}"


@dataclass(frozen=True)
class ExportTilePlan:
    """Planned GEE export task for one tile, supporting dry-run and skip-existing."""

    tile: TileSpec
    asset_id: str
    description: str
    skip: bool = False


def manifest_path_for(raster_path: str | Path) -> Path:
    """Return the canonical sidecar manifest path ``<stem>.manifest.json``."""
    p = Path(raster_path)
    return p.with_suffix(".manifest.json")


def _validate_probabilities(
    probabilities: np.ndarray,
    class_schema: ClassSchema,
    *,
    sum_tol: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shape, ``[0, 1]`` range, and sum-to-one invariant over valid pixels."""
    arr = np.asarray(probabilities, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            f"probabilities must have shape (n_classes, height, width), got {arr.shape}."
        )
    if arr.shape[0] != len(class_schema):
        raise ValueError(
            f"Probability band count ({arr.shape[0]}) does not match "
            f"ClassSchema length ({len(class_schema)})."
        )

    valid_mask = np.all(np.isfinite(arr), axis=0)
    if np.any(valid_mask):
        valid_vals = arr[:, valid_mask]
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        if min_val < -1e-4 or max_val > 1.0 + 1e-4:
            raise ValueError(
                f"Probabilities must lie in [0, 1]; observed [{min_val:.4f}, {max_val:.4f}]."
            )

        sums = np.sum(valid_vals, axis=0)
        max_dev = float(np.max(np.abs(sums - 1.0)))
        if max_dev > sum_tol:
            raise ValueError(
                f"Valid pixel probabilities must sum to 1.0 within {sum_tol}; "
                f"max deviation was {max_dev:.4f}."
            )

    return arr, valid_mask


def write_probability_stack(
    output_path: str | Path,
    probabilities: np.ndarray,
    *,
    profile: dict[str, Any],
    class_schema: ClassSchema,
    encoding: ProbabilityEncoding = "uint16",
    epoch: str | None = None,
    paradigm: ClassificationParadigm = "pixel",
    source: str | None = None,
    extra: dict[str, Any] | None = None,
    sum_tol: float = 1e-2,
) -> Path:
    """Write a validated Class Probability Stack GeoTIFF and sidecar JSON manifest."""
    if encoding not in ("uint16", "float32"):
        raise ValueError(f"encoding must be 'uint16' or 'float32', got {encoding!r}.")

    arr, valid_mask = _validate_probabilities(probabilities, class_schema, sum_tol=sum_tol)
    n_classes, height, width = arr.shape

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = dict(profile)
    out_profile["driver"] = "COG"
    for incompatible_key in ("tiled", "blockxsize", "blockysize", "interleave"):
        out_profile.pop(incompatible_key, None)
    out_profile.update(
        count=n_classes,
        height=height,
        width=width,
        compress="deflate",
    )

    if encoding == "uint16":
        scale_factor = UINT16_PROB_SCALE
        manifest_nodata: float | int | None = UINT16_PROB_NODATA
        data_to_write = np.full((n_classes, height, width), UINT16_PROB_NODATA, dtype=np.uint16)
        if np.any(valid_mask):
            scaled = np.rint(np.clip(arr[:, valid_mask], 0.0, 1.0) * UINT16_PROB_SCALE)
            data_to_write[:, valid_mask] = scaled.astype(np.uint16)
        out_profile.update(dtype="uint16", nodata=UINT16_PROB_NODATA)
    else:
        scale_factor = 1.0
        manifest_nodata = None
        data_to_write = np.where(valid_mask[None, :, :], arr, np.nan).astype(np.float32)
        out_profile.update(dtype="float32", nodata=np.nan)

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(data_to_write)
        for band_idx, desc in enumerate(class_schema.geotiff_band_descriptions, start=1):
            dst.set_band_description(band_idx, desc)

    crs_str = str(out_profile.get("crs", ""))
    manifest = ProbabilityManifest(
        class_schema=class_schema,
        dtype=encoding,
        scale_factor=scale_factor,
        nodata=manifest_nodata,
        crs=crs_str,
        epoch=epoch,
        paradigm=paradigm,
        source=source,
        extra=dict(extra) if extra else {},
    )
    manifest.to_json(manifest_path_for(out_path))
    return out_path


def _fallback_class_schema(n_bands: int, descriptions: Sequence[str | None]) -> ClassSchema:
    classes: list[ClassDefinition] = []
    for idx in range(1, n_bands + 1):
        desc = descriptions[idx - 1] if idx - 1 < len(descriptions) else None
        if desc and ":" in desc:
            raw_id, raw_name = desc.split(":", 1)
            try:
                cid = int(raw_id.strip())
                cname = raw_name.strip() or f"Class_{cid}"
                classes.append(ClassDefinition(id=cid, name=cname))
                continue
            except ValueError:
                pass
        classes.append(ClassDefinition(id=idx, name=desc.strip() if desc else f"Class_{idx}"))
    return ClassSchema(classes=tuple(classes))


def read_probability_stack(
    raster_path: str | Path,
    *,
    class_schema: ClassSchema | None = None,
    n_classes: int | None = None,
    scale_factor: float | None = None,
    manifest_path: str | Path | None = None,
    validate: bool = True,
    sum_tol: float = 1e-2,
) -> ProbabilityStack:
    """Read a Class Probability Stack GeoTIFF into ``float32`` ``[0, 1]`` arrays.

    Auto-discovers ``<stem>.manifest.json`` when present, and supports standalone
    external GeoTIFFs without a manifest using either an explicit ``class_schema``
    or ``n_classes`` (ADR-0005).
    """
    rpath = Path(raster_path)
    mpath = Path(manifest_path) if manifest_path is not None else manifest_path_for(rpath)

    with rasterio.open(rpath) as src:
        raw = src.read()
        profile = dict(src.profile)
        src_dtype = src.dtypes[0]
        src_nodata = src.nodata
        descriptions = src.descriptions

    if n_classes is not None and raw.shape[0] != n_classes:
        raise ValueError(
            f"Raster band count ({raw.shape[0]}) does not match expected n_classes ({n_classes})."
        )

    if mpath.exists():
        manifest = ProbabilityManifest.from_json(mpath)
        effective_scale = scale_factor if scale_factor is not None else manifest.scale_factor
        if class_schema is not None and class_schema != manifest.class_schema:
            raise ValueError("Provided class_schema does not match sidecar manifest.")
    else:
        inferred_dtype: ProbabilityEncoding = "uint16" if src_dtype == "uint16" else "float32"
        inferred_scale = UINT16_PROB_SCALE if inferred_dtype == "uint16" else 1.0
        effective_scale = scale_factor if scale_factor is not None else inferred_scale
        resolved_schema = (
            class_schema
            if class_schema is not None
            else _fallback_class_schema(raw.shape[0], descriptions)
        )
        manifest = ProbabilityManifest(
            class_schema=resolved_schema,
            dtype=inferred_dtype,
            scale_factor=effective_scale,
            nodata=src_nodata if src_nodata is not None and not np.isnan(src_nodata) else None,
            crs=str(profile.get("crs", "")),
            paradigm="external",
            source=str(rpath.name),
        )

    if raw.shape[0] != len(manifest.class_schema):
        raise ValueError(
            f"Raster band count ({raw.shape[0]}) does not match "
            f"ClassSchema length ({len(manifest.class_schema)})."
        )

    raw_f32 = raw.astype(np.float32)
    valid_mask = np.all(np.isfinite(raw_f32), axis=0)
    if src_nodata is not None and not np.isnan(src_nodata):
        valid_mask &= ~np.any(raw == src_nodata, axis=0)

    # GEE exports often write all-zero background over masked ocean/outside-AOI pixels
    valid_mask &= np.any(raw_f32 > 0.0, axis=0)

    probs = np.full_like(raw_f32, np.nan, dtype=np.float32)
    if np.any(valid_mask):
        probs[:, valid_mask] = raw_f32[:, valid_mask] / float(effective_scale)

    if validate:
        _validate_probabilities(probs, manifest.class_schema, sum_tol=sum_tol)

    return ProbabilityStack(
        probabilities=probs,
        valid_mask=valid_mask,
        profile=profile,
        manifest=manifest,
    )


def repair_gee_int32_overflow(
    segments: np.ndarray,
    *,
    clamp_value: int = GEE_INT32_CLAMP_VALUE,
    max_clamp_cluster_px: int | None = 400,
    id_base: int = -2_000_000_000,
) -> np.ndarray:
    """Repair GEE 32-bit signed overflow in a Segment Label Raster.

    - Valid wrapped negative IDs (``< -1``) and nodata (``-1``, ``0``) are preserved.
    - Pixels clamped to ``2^31 - 1`` are separated via connected-component labeling.
    - Small connected components receive unique negative IDs guaranteed not to
      collide with pre-existing wrapped negative IDs.
    - When ``max_clamp_cluster_px`` is set, oversized fused mega-clusters fall back
      to ``0`` so pure pixel probabilities are used instead of flooding boundaries.
    """
    repaired = np.asarray(segments, dtype=np.int32).copy()
    is_clamp = repaired == clamp_value
    if not np.any(is_clamp):
        return repaired

    labeled, n_comp = connected_components(is_clamp)
    if n_comp > 0:
        if max_clamp_cluster_px is not None:
            comp_sizes = np.bincount(labeled.ravel())
            oversized = comp_sizes[labeled] > max_clamp_cluster_px
            repaired[is_clamp & oversized] = 0
            valid_clamp = is_clamp & (~oversized)
        else:
            valid_clamp = is_clamp

        if np.any(valid_clamp):
            existing_non_clamp = repaired[~is_clamp]
            min_existing = int(existing_non_clamp.min()) if existing_non_clamp.size > 0 else 0
            # Choose a starting negative base that avoids colliding with existing IDs
            safe_base = min(id_base, min_existing - 1)
            if safe_base - n_comp < np.iinfo(np.int32).min:
                # Fall back to finding a clean contiguous negative range below -2
                existing_set = set(existing_non_clamp[existing_non_clamp < -1].tolist())
                assigned: dict[int, int] = {}
                candidate = -2
                for comp_id in np.unique(labeled[valid_clamp]):
                    while candidate in existing_set:
                        candidate -= 1
                    assigned[int(comp_id)] = candidate
                    existing_set.add(candidate)
                    candidate -= 1
                for comp_id, new_id in assigned.items():
                    repaired[valid_clamp & (labeled == comp_id)] = new_id
            else:
                repaired[valid_clamp] = (safe_base - labeled[valid_clamp]).astype(np.int32)
    return repaired


def _grids_match(profile_a: dict[str, Any], profile_b: dict[str, Any]) -> bool:
    crs_a = profile_a.get("crs")
    crs_b = profile_b.get("crs")
    if crs_a is None or crs_b is None:
        crs_equal = crs_a == crs_b
    else:
        crs_equal = bool(CRS.from_user_input(crs_a) == CRS.from_user_input(crs_b))
    return (
        profile_a.get("width") == profile_b.get("width")
        and profile_a.get("height") == profile_b.get("height")
        and crs_equal
        and profile_a.get("transform") == profile_b.get("transform")
    )


def read_segment_raster(
    segment_path: str | Path,
    *,
    reference_profile: dict[str, Any] | None = None,
    align_to_reference: bool = False,
    repair_overflow: bool = True,
    max_clamp_cluster_px: int = 400,
) -> SegmentRaster:
    """Read a single-band Segment Label Raster with optional grid alignment and overflow fix."""
    with rasterio.open(segment_path) as src:
        src_profile = dict(src.profile)
        if reference_profile is not None and not _grids_match(src_profile, reference_profile):
            if not align_to_reference:
                raise ValueError(
                    "Segment Label Raster grid does not match reference_profile. "
                    "Pass align_to_reference=True to resample with nearest-neighbor WarpedVRT."
                )
            with WarpedVRT(
                src,
                crs=reference_profile["crs"],
                transform=reference_profile["transform"],
                width=reference_profile["width"],
                height=reference_profile["height"],
                resampling=Resampling.nearest,
            ) as vrt:
                seg = vrt.read(1)
                out_profile = dict(reference_profile)
                out_profile.update(count=1, dtype=str(seg.dtype))
        else:
            seg = src.read(1)
            out_profile = src_profile

    if repair_overflow:
        seg = repair_gee_int32_overflow(seg, max_clamp_cluster_px=max_clamp_cluster_px)

    return SegmentRaster(segments=seg, profile=out_profile)


def write_sample_features_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    class_col: str = "class_id",
    class_schema: ClassSchema | None = None,
) -> Path:
    """Write extracted sample features to CSV, optionally attaching canonical class names."""
    if class_col not in df.columns:
        raise ValueError(f"Column {class_col!r} not found in DataFrame columns {list(df.columns)}.")

    out_df = df.copy()
    if class_schema is not None and "class_name" not in out_df.columns:
        out_df["class_name"] = [class_schema.name_for(int(cid)) for cid in out_df[class_col]]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path


def read_sample_features_csv(
    csv_path: str | Path,
    *,
    class_col: str = "class_id",
    feature_cols: Sequence[str] | None = None,
    ignore_cols: Sequence[str] = ("system:index", ".geo", "sample_id", "class_name"),
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Read an extracted sample-features CSV into ``(X, y, feature_names)`` for RFECV / local ML."""
    df = pd.read_csv(csv_path)
    if class_col not in df.columns:
        raise ValueError(f"Target column {class_col!r} not found in {csv_path}.")

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from CSV: {missing}")
        selected = list(feature_cols)
    else:
        excluded = set(ignore_cols) | {class_col}
        selected = [c for c in df.columns if c not in excluded]

    if not selected:
        raise ValueError("No feature columns remained after filtering ignore_cols.")

    return df[selected].copy(), df[class_col].copy(), selected


def generate_tile_grid(
    bounds: tuple[float, float, float, float],
    tile_size_deg: float,
) -> list[TileSpec]:
    """Subdivide ``(minx, miny, maxx, maxy)`` into a deterministic rectangular tile grid."""
    if tile_size_deg <= 0:
        raise ValueError(f"tile_size_deg must be > 0, got {tile_size_deg!r}.")
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid bounds {bounds!r}: min must be strictly less than max.")

    n_cols = math.ceil((maxx - minx) / tile_size_deg)
    n_rows = math.ceil((maxy - miny) / tile_size_deg)
    tiles: list[TileSpec] = []
    for r in range(n_rows):
        y0 = miny + r * tile_size_deg
        y1 = min(maxy, y0 + tile_size_deg)
        for c in range(n_cols):
            x0 = minx + c * tile_size_deg
            x1 = min(maxx, x0 + tile_size_deg)
            tiles.append(TileSpec(row=r, col=c, bounds=(x0, y0, x1, y1)))
    return tiles


def plan_tiled_gee_exports(
    tiles: Sequence[TileSpec],
    *,
    export_prefix: str,
    existing_assets: Iterable[str] = (),
    skip_existing: bool = True,
) -> list[ExportTilePlan]:
    """Build an idempotent tiled GEE export plan supporting ``skip_existing`` resumability."""
    prefix = export_prefix.rstrip("/")
    existing_set = set(existing_assets)
    plans: list[ExportTilePlan] = []
    for tile in tiles:
        asset_id = f"{prefix}_{tile.tile_id}"
        desc = f"{Path(prefix).name}_{tile.tile_id}"
        plans.append(
            ExportTilePlan(
                tile=tile,
                asset_id=asset_id,
                description=desc,
                skip=skip_existing and (asset_id in existing_set),
            )
        )
    return plans


def submit_gee_export_tasks(
    plans: Sequence[ExportTilePlan],
    *,
    task_factory: Callable[[ExportTilePlan], Any],
    dry_run: bool = False,
) -> list[Any]:
    """Start non-skipped GEE export tasks (or return ``[]`` when ``dry_run=True``)."""
    if dry_run:
        return []
    started: list[Any] = []
    for plan in plans:
        if plan.skip:
            continue
        task = task_factory(plan)
        task.start()
        started.append(task)
    return started

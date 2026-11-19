"""Object-Refined Pixel Probabilities (Hybrid Classification Assembly).

Implements adaptive convex blending of pixel-level Class Probability Stacks with
geo-object mean probabilities (Malinverni & Sanita 2026), Confusion Index (S / W),
and binary stability mapping, runnable completely standalone without GEE or ML
dependencies (ADR-0005).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray

from lc_framework.assembly.postprocess import (
    MaskRule,
    apply_mask_rules,
    apply_mmu,
    derive_hard_labels,
)
from lc_framework.assembly.smoothing import heal_array_seams
from lc_framework.config import ClassSchema, ProjectConfig
from lc_framework.io import (
    ProbabilityEncoding,
    read_probability_stack,
    read_segment_raster,
    repair_gee_int32_overflow,
    write_probability_stack,
)

DEFAULT_STABILITY_TAU: float = 0.65
DEFAULT_ADAPTIVE_SENSITIVITY: float = 0.60
STABILITY_NODATA: int = 255


@dataclass(frozen=True)
class HybridBlendResult:
    """In-memory outputs of Object-Refined Pixel Probabilities."""

    probabilities: NDArray[np.float32]
    confusion_index: NDArray[np.float32]
    stability: NDArray[np.uint8]
    hard_labels: NDArray[np.int32]
    valid_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class HybridAssemblyOutputs:
    """On-disk artifacts emitted by :func:`run_hybrid_assembly`."""

    probabilities: Path
    lulc: Path
    confusion_index: Path
    stability: Path


def _resolve_base_object_weights(
    n_classes: int,
    class_schema: ClassSchema | None,
    smoothness_weights: Mapping[str, float] | Sequence[float] | None,
) -> NDArray[np.float32]:
    """Map per-class smoothness weights to bounded base object weights in [0.2, 0.9]."""
    if smoothness_weights is None:
        return np.full(n_classes, 0.5, dtype=np.float32)

    if isinstance(smoothness_weights, Mapping):
        if class_schema is None:
            raise ValueError(
                "class_schema is required when smoothness_weights is provided as a name mapping."
            )
        raw = [float(smoothness_weights.get(c.name, 1.0)) for c in class_schema.classes]
    else:
        raw = [float(v) for v in smoothness_weights]
        if len(raw) != n_classes:
            raise ValueError(
                f"smoothness_weights length ({len(raw)}) must match n_classes ({n_classes})."
            )

    return np.clip(np.asarray(raw, dtype=np.float32) / 5.0, 0.2, 0.9).astype(np.float32)


def compute_confusion_index(
    probabilities: np.ndarray,
    *,
    tau: float = DEFAULT_STABILITY_TAU,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Compute per-pixel Confusion Index (``S / W``) and binary stability (``CI <= tau``)."""
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must lie in [0, 1], got {tau!r}.")

    arr = np.asarray(probabilities, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError("probabilities must have shape (C, H, W) with C >= 2.")

    valid = np.all(np.isfinite(arr), axis=0) & np.any(arr > 0.0, axis=0)
    safe = np.where(np.isfinite(arr), arr, -1.0)
    part = np.partition(safe, -2, axis=0)
    s_max = part[-2]
    w_max = part[-1]

    ci = np.full(arr.shape[1:], np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(w_max > 1e-4, s_max / w_max, 0.0)
    ci[valid] = np.clip(ratio[valid], 0.0, 1.0).astype(np.float32)

    stability = np.full(arr.shape[1:], STABILITY_NODATA, dtype=np.uint8)
    stability[valid] = (ci[valid] <= tau).astype(np.uint8)
    return ci, stability


def blend_pixel_and_object_probabilities(
    probabilities: np.ndarray,
    segments: np.ndarray,
    *,
    class_schema: ClassSchema | None = None,
    smoothness_weights: Mapping[str, float] | Sequence[float] | None = None,
    adaptive_sensitivity: float = DEFAULT_ADAPTIVE_SENSITIVITY,
    tau: float = DEFAULT_STABILITY_TAU,
    repair_overflow: bool = True,
    max_clamp_cluster_px: int | None = 400,
    heal_segment_seams: bool = True,
    max_seam_width: int = 4,
    nodata_label: int = 0,
) -> HybridBlendResult:
    """Blend pixel probabilities with per-segment mean probabilities adaptively.

    Returns a normalized ``Class Probability Stack`` summing to 1.0 on all valid
    pixels alongside ``Confusion Index`` (``S / W``), binary stability map, and
    derived hard class labels.
    """
    if not (0.0 <= adaptive_sensitivity <= 1.0):
        raise ValueError(
            f"adaptive_sensitivity must lie in [0, 1], got {adaptive_sensitivity!r}."
        )

    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 3:
        raise ValueError(f"probabilities must have shape (C, H, W), got {probs.shape}.")
    n_classes, height, width = probs.shape

    seg = np.asarray(segments, dtype=np.int32)
    if seg.shape != (height, width):
        raise ValueError(
            f"segments shape {seg.shape} does not match probabilities spatial shape "
            f"{(height, width)}."
        )

    if repair_overflow:
        seg = repair_gee_int32_overflow(seg, max_clamp_cluster_px=max_clamp_cluster_px)
    else:
        seg = seg.copy()

    if heal_segment_seams:
        seg = heal_array_seams(seg, nodata=0, max_seam_width=max_seam_width)

    valid_mask = np.all(np.isfinite(probs), axis=0) & np.any(probs > 0.0, axis=0)
    has_seg = (seg != -1) & (seg != 0) & valid_mask
    has_any_seg = bool(np.any(has_seg))

    if has_any_seg:
        _, inverse = np.unique(seg[has_seg], return_inverse=True)
        n_segs = int(inverse.max()) + 1
        seg_compact = np.full((height, width), -1, dtype=np.int32)
        seg_compact[has_seg] = inverse.astype(np.int32)
        counts = np.bincount(seg_compact[has_seg], minlength=n_segs)
    else:
        n_segs = 0
        seg_compact = np.full((height, width), -1, dtype=np.int32)
        counts = np.array([], dtype=np.int64)

    w_base = _resolve_base_object_weights(n_classes, class_schema, smoothness_weights)
    blended = np.full((n_classes, height, width), np.nan, dtype=np.float32)

    for c in range(n_classes):
        raw = probs[c]
        if has_any_seg:
            sums = np.bincount(
                seg_compact[has_seg],
                weights=raw[has_seg].astype(np.float64),
                minlength=n_segs,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_lookup = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)

            obj = np.zeros((height, width), dtype=np.float32)
            obj[has_seg] = mean_lookup[seg_compact[has_seg]]

            disagreement = np.clip(np.abs(obj - raw), 0.0, 1.0)
            w_eff = np.where(
                has_seg,
                w_base[c] * (1.0 - adaptive_sensitivity * disagreement),
                0.0,
            ).astype(np.float32)
            band_out = (obj * w_eff) + (raw * (1.0 - w_eff))
        else:
            band_out = raw.copy()

        blended[c, valid_mask] = band_out[valid_mask]

    # Renormalize blended probabilities so every valid pixel sums strictly to 1.0
    if np.any(valid_mask):
        totals = np.sum(blended[:, valid_mask], axis=0)
        positive = totals > 0.0
        if np.any(positive):
            valid_rows, valid_cols = np.nonzero(valid_mask)
            pos_r = valid_rows[positive]
            pos_c = valid_cols[positive]
            blended[:, pos_r, pos_c] /= totals[positive][None, :]

    ci, stability = compute_confusion_index(blended, tau=tau)
    hard_labels = derive_hard_labels(blended, class_schema=class_schema, nodata=nodata_label)

    return HybridBlendResult(
        probabilities=blended,
        confusion_index=ci,
        stability=stability,
        hard_labels=hard_labels,
        valid_mask=valid_mask,
    )


def _write_single_band_geotiff(
    output_path: Path,
    array: np.ndarray,
    *,
    profile: dict,
    dtype: str,
    nodata: float | int,
    tags: Mapping[str, str] | None = None,
) -> Path:
    out_profile = dict(profile)
    out_profile["driver"] = "GTiff"
    for incompatible_key in ("tiled", "blockxsize", "blockysize", "interleave"):
        out_profile.pop(incompatible_key, None)
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)
        if tags:
            dst.update_tags(**tags)
    return output_path


def run_hybrid_assembly(
    prob_path: str | Path,
    segment_path: str | Path,
    output_dir: str | Path,
    *,
    config: ProjectConfig | None = None,
    class_schema: ClassSchema | None = None,
    smoothness_weights: Mapping[str, float] | Sequence[float] | None = None,
    tau: float = DEFAULT_STABILITY_TAU,
    adaptive_sensitivity: float = DEFAULT_ADAPTIVE_SENSITIVITY,
    encoding: ProbabilityEncoding = "uint16",
    epoch: str | None = None,
    mask_rules: Sequence[MaskRule] | None = None,
    mmu_table: Mapping[int, int] | None = None,
    mmu_iterations: int = 6,
    align_segments: bool = True,
    repair_overflow: bool = True,
    max_clamp_cluster_px: int | None = 400,
    heal_segment_seams: bool = True,
    output_stem: str | None = None,
) -> HybridAssemblyOutputs:
    """Run end-to-end standalone Object-Refined Pixel Probabilities assembly on GeoTIFFs.

    Works either with plain GeoTIFF paths alone or paired with a :class:`ProjectConfig`.
    """
    resolved_schema = class_schema or (config.class_schema if config is not None else None)
    default_smoothness = (
        config.smoothness_weights
        if config is not None and config.smoothness_weights
        else None
    )
    resolved_smoothness = (
        smoothness_weights if smoothness_weights is not None else default_smoothness
    )

    prob_stack = read_probability_stack(prob_path, class_schema=resolved_schema)
    active_schema = prob_stack.manifest.class_schema
    active_epoch = epoch if epoch is not None else prob_stack.manifest.epoch

    seg_raster = read_segment_raster(
        segment_path,
        reference_profile=prob_stack.profile,
        align_to_reference=align_segments,
        repair_overflow=repair_overflow,
        max_clamp_cluster_px=max_clamp_cluster_px if max_clamp_cluster_px is not None else 10**9,
    )

    blended = blend_pixel_and_object_probabilities(
        prob_stack.probabilities,
        seg_raster.segments,
        class_schema=active_schema,
        smoothness_weights=resolved_smoothness,
        adaptive_sensitivity=adaptive_sensitivity,
        tau=tau,
        repair_overflow=False,  # already repaired by read_segment_raster
        heal_segment_seams=heal_segment_seams,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or Path(prob_path).stem

    prob_out = write_probability_stack(
        out_dir / f"{stem}_hybrid_probs.tif",
        blended.probabilities,
        profile=prob_stack.profile,
        class_schema=active_schema,
        encoding=encoding,
        epoch=active_epoch,
        paradigm="hybrid",
        source=f"hybrid_assembly:{Path(prob_path).name}",
    )

    schema_tags = {"lc_framework_class_schema": active_schema.model_dump_json()}
    rule_targets = [r.to_class for r in (mask_rules or ())]
    max_cid = max([*active_schema.class_ids, *rule_targets])
    lulc_dtype = "uint8" if max_cid <= 255 else "uint16"
    lulc_raw_path = out_dir / f"{stem}_lulc.tif"
    _write_single_band_geotiff(
        lulc_raw_path,
        blended.hard_labels,
        profile=prob_stack.profile,
        dtype=lulc_dtype,
        nodata=0,
        tags=schema_tags,
    )

    current_lulc = lulc_raw_path
    if mask_rules:
        masked_path = out_dir / f"{stem}_lulc_masked.tif"
        apply_mask_rules(current_lulc, masked_path, mask_rules)
        masked_path.replace(lulc_raw_path)
        current_lulc = lulc_raw_path

    if mmu_table:
        mmu_path = out_dir / f"{stem}_lulc_mmu.tif"
        apply_mmu(
            current_lulc,
            mmu_path,
            min_patch_sizes=mmu_table,
            iterations=mmu_iterations,
        )
        mmu_path.replace(lulc_raw_path)
        current_lulc = lulc_raw_path

    ci_out = _write_single_band_geotiff(
        out_dir / f"{stem}_confusion_index.tif",
        blended.confusion_index,
        profile=prob_stack.profile,
        dtype="float32",
        nodata=np.nan,
    )
    stab_out = _write_single_band_geotiff(
        out_dir / f"{stem}_stability.tif",
        blended.stability,
        profile=prob_stack.profile,
        dtype="uint8",
        nodata=STABILITY_NODATA,
    )

    return HybridAssemblyOutputs(
        probabilities=prob_out,
        lulc=current_lulc,
        confusion_index=ci_out,
        stability=stab_out,
    )

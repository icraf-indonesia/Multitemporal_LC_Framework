"""Hard-label derivation, multi-resolution mask rules, and MMU patch filtering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage

from lc_framework.config import ClassSchema
from lc_framework.io import read_probability_stack

_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


@dataclass
class MaskRule:
    """Generic reclassification rule driven by auxiliary raster mask layers.

    Masks of arbitrary CRS or pixel resolution are aligned on-the-fly via
    nearest-neighbor ``WarpedVRT``.
    """

    from_classes: int | Sequence[int]
    to_class: int
    require_mask_path: str | Path | None = None
    exclude_mask_path: str | Path | None = None
    require_value: int | float = 1
    exclude_value: int | float = 1
    name: str = "Mask Rule"

    def __post_init__(self) -> None:
        if isinstance(self.from_classes, int):
            self.from_classes = (self.from_classes,)
        else:
            self.from_classes = tuple(int(c) for c in self.from_classes)
        self.to_class = int(self.to_class)
        if self.require_mask_path is None and self.exclude_mask_path is None:
            raise ValueError("MaskRule requires at least require_mask_path or exclude_mask_path.")


def derive_hard_labels(
    probabilities: np.ndarray,
    *,
    class_schema: ClassSchema | None = None,
    nodata: int = 0,
) -> NDArray[np.int32]:
    """Derive hard class labels from a ``(C, H, W)`` Class Probability Stack."""
    arr = np.asarray(probabilities, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"probabilities must have shape (C, H, W), got {arr.shape}.")

    n_classes = arr.shape[0]
    if class_schema is not None:
        if len(class_schema) != n_classes:
            raise ValueError(
                f"probabilities band count ({n_classes}) does not match "
                f"ClassSchema length ({len(class_schema)})."
            )
        class_ids = np.asarray(class_schema.class_ids, dtype=np.int32)
    else:
        class_ids = np.arange(1, n_classes + 1, dtype=np.int32)

    valid_mask = np.all(np.isfinite(arr), axis=0) & np.any(arr > 0.0, axis=0)
    safe = np.where(np.isfinite(arr), arr, -np.inf)
    winners = np.argmax(safe, axis=0)

    labels = np.full(arr.shape[1:], nodata, dtype=np.int32)
    labels[valid_mask] = class_ids[winners[valid_mask]]
    return labels


def derive_hard_labels_raster(
    input_prob_path: str | Path,
    output_lulc_path: str | Path,
    *,
    class_schema: ClassSchema | None = None,
    nodata: int = 0,
) -> Path:
    """Derive a hard LULC GeoTIFF from a Class Probability Stack, preserving ClassSchema tags."""
    stack = read_probability_stack(input_prob_path, class_schema=class_schema)
    schema = stack.manifest.class_schema
    labels = derive_hard_labels(stack.probabilities, class_schema=schema, nodata=nodata)

    out_path = Path(output_lulc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_cid = max(schema.class_ids)
    out_dtype = "uint8" if max_cid <= 255 else "uint16"

    profile = dict(stack.profile)
    profile["driver"] = "GTiff"
    for incompatible_key in ("tiled", "blockxsize", "blockysize", "interleave"):
        profile.pop(incompatible_key, None)
    profile.update(count=1, dtype=out_dtype, nodata=nodata, compress="deflate")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(labels.astype(out_dtype), 1)
        dst.update_tags(lc_framework_class_schema=schema.model_dump_json())
    return out_path


def apply_mask_rules(
    input_lulc_path: str | Path,
    output_lulc_path: str | Path,
    rules: Sequence[MaskRule],
    *,
    block_size: int = 2048,
) -> Path:
    """Apply multi-resolution auxiliary mask rules sequentially to a hard LULC raster."""
    in_path = Path(input_lulc_path)
    out_path = Path(output_lulc_path)
    if in_path.resolve() == out_path.resolve():
        raise ValueError("output_lulc_path must differ from input_lulc_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unique_mask_paths: set[Path] = set()
    for rule in rules:
        if rule.require_mask_path is not None:
            unique_mask_paths.add(Path(rule.require_mask_path))
        if rule.exclude_mask_path is not None:
            unique_mask_paths.add(Path(rule.exclude_mask_path))

    with rasterio.open(in_path) as src:
        profile = dict(src.profile)
        profile["driver"] = "GTiff"
        profile["compress"] = "deflate"
        tags = src.tags()
        height, width = src.height, src.width

        mask_sources: dict[Path, Any] = {}
        vrts: dict[Path, WarpedVRT] = {}
        try:
            for mpath in unique_mask_paths:
                msrc = rasterio.open(mpath)
                mask_sources[mpath] = msrc
                vrts[mpath] = WarpedVRT(
                    msrc,
                    crs=src.crs,
                    transform=src.transform,
                    width=width,
                    height=height,
                    resampling=Resampling.nearest,
                )

            with rasterio.open(out_path, "w", **profile) as dst:
                if tags:
                    dst.update_tags(**tags)
                for r_off in range(0, height, block_size):
                    win_h = min(block_size, height - r_off)
                    for c_off in range(0, width, block_size):
                        win_w = min(block_size, width - c_off)
                        win = Window(c_off, r_off, win_w, win_h)
                        block = src.read(1, window=win)

                        if not rules:
                            dst.write(block, 1, window=win)
                            continue

                        cached_masks = {p: vrt.read(1, window=win) for p, vrt in vrts.items()}
                        for rule in rules:
                            candidate = np.isin(block, rule.from_classes)
                            if not np.any(candidate):
                                continue
                            rule_mask = candidate.copy()
                            if rule.require_mask_path is not None:
                                req = cached_masks[Path(rule.require_mask_path)]
                                rule_mask &= req == rule.require_value
                            if rule.exclude_mask_path is not None:
                                exc = cached_masks[Path(rule.exclude_mask_path)]
                                rule_mask &= exc != rule.exclude_value
                            if np.any(rule_mask):
                                block[rule_mask] = rule.to_class

                        dst.write(block, 1, window=win)
        finally:
            for vrt in vrts.values():
                vrt.close()
            for msrc in mask_sources.values():
                msrc.close()

    return out_path


def _find_small_patches(
    labels: np.ndarray,
    valid: np.ndarray,
    min_patch_sizes: Mapping[int, int],
) -> NDArray[np.bool_]:
    small = np.zeros(labels.shape, dtype=bool)
    for class_id, min_size in min_patch_sizes.items():
        if min_size <= 1:
            continue
        class_mask = valid & (labels == class_id)
        if not np.any(class_mask):
            continue
        components, n_comp = ndimage.label(class_mask, structure=_EIGHT_CONNECTED)
        if n_comp == 0:
            continue
        sizes = np.bincount(components.ravel(), minlength=n_comp + 1)
        small |= (components != 0) & (sizes[components] < min_size)
    return small


def apply_mmu_array(
    labels: np.ndarray,
    *,
    min_patch_sizes: Mapping[int, int],
    nodata: int = 0,
    iterations: int = 6,
) -> np.ndarray:
    """Remove 8-connected patches smaller than ``min_patch_sizes[class_id]`` via focal mode."""
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    for cid, msize in min_patch_sizes.items():
        if isinstance(cid, bool) or not isinstance(cid, Integral):
            raise ValueError(f"Class ID must be an integer, got {cid!r}")
        if isinstance(msize, bool) or not isinstance(msize, Integral) or msize < 0:
            raise ValueError(f"Minimum patch size must be >= 0, got {msize!r}")

    arr = np.asarray(labels)
    valid = arr != nodata
    if not np.any(valid):
        return arr.copy()

    small = _find_small_patches(arr, valid, min_patch_sizes)
    if not np.any(small):
        return arr.copy()

    active = valid & ~small
    sentinel = int(np.iinfo(np.int64).min)
    current = np.full(arr.shape, sentinel, dtype=np.int64)
    current[active] = arr[active].astype(np.int64)

    present_classes = np.unique(current[active])
    present_classes = present_classes[present_classes != sentinel]

    for _ in range(int(iterations)):
        remaining = valid & small & ~active
        if not np.any(remaining):
            break

        modes = np.full(arr.shape, sentinel, dtype=np.int64)
        best_counts = np.zeros(arr.shape, dtype=np.uint8)

        for class_id in present_classes:
            class_pixels = (current == class_id) & active
            if not np.any(class_pixels):
                continue
            counts = ndimage.convolve(
                class_pixels.astype(np.uint8),
                _EIGHT_CONNECTED,
                mode="constant",
                cval=0,
            )
            wins = counts > best_counts
            modes[wins] = class_id
            best_counts[wins] = counts[wins]

        can_fill = remaining & (best_counts > 0)
        if not np.any(can_fill):
            break

        current[can_fill] = modes[can_fill]
        active |= can_fill

    result = arr.copy()
    filled = valid & small & active
    result[filled] = current[filled].astype(arr.dtype)
    return result


def apply_mmu(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_patch_sizes: Mapping[int, int],
    iterations: int = 6,
) -> Path:
    """Apply Minimum Mapping Unit (MMU) filtering to a single-band classification GeoTIFF."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    if in_path.resolve() == out_path.resolve():
        raise ValueError("output_path must differ from input_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        if src.count != 1:
            raise ValueError("Input classification raster must have exactly 1 band.")
        labels = src.read(1)
        profile = dict(src.profile)
        profile["driver"] = "GTiff"
        profile["compress"] = "deflate"
        tags = src.tags()
        nodata_val = int(src.nodata) if src.nodata is not None and np.isfinite(src.nodata) else 0

    filtered = apply_mmu_array(
        labels,
        min_patch_sizes=min_patch_sizes,
        nodata=nodata_val,
        iterations=iterations,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)
        if tags:
            dst.update_tags(**tags)
    return out_path

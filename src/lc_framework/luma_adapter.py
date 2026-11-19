"""Thin, zero-hard-dependency adapter for external ``luma-stack`` (``luma_ge``).

Implements ADR-0001 (keep ``luma-stack`` external via adapters) and ADR-0004
(delegate direct multitemporal orchestration to ``luma-stack`` while owning
CCDC temporal segmentation natively in ``lc_framework.temporal``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from lc_framework.config import ClassDefinition, ClassSchema, ProjectConfig
from lc_framework.io import UINT16_PROB_SCALE

_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")


def _import_luma_module(submodule: str) -> Any:
    """Import a ``luma_ge`` submodule lazily with an actionable error message."""
    try:
        return __import__(f"luma_ge.{submodule}", fromlist=[submodule])
    except ImportError as exc:
        raise ImportError(
            "External package `luma-stack` (`luma_ge`) is required for this adapter. "
            "Install it from your organization repository, e.g.:\n"
            "  pip install git+https://github.com/<your-org>/luma-stack.git"
        ) from exc


def _sanitize_luma_class_name(name: str) -> str:
    """Mirror ``luma-stack`` sanitized probability band suffix for a class name."""
    cleaned = _NON_ALNUM_RE.sub("_", name.strip()).strip("_")
    return cleaned or "class"


def schema_to_luma_classes(class_schema: ClassSchema) -> list[dict[str, Any]]:
    """Convert a :class:`ClassSchema` into ``luma_ge`` class records."""
    return [
        {"ID": c.id, "Class Name": c.name, "Color Code": c.color}
        for c in class_schema.classes
    ]


def schema_to_luma_manager(class_schema: ClassSchema) -> Any:
    """Populate a ``luma_ge.classification_scheme.LULC_Scheme_Manager`` from ``ClassSchema``."""
    mod = _import_luma_module("classification_scheme")
    manager = mod.LULC_Scheme_Manager()
    for entry in class_schema.classes:
        ok, msg = manager.add_class(entry.id, entry.name, entry.color)
        if not ok:
            raise ValueError(f"luma-stack rejected class {entry}: {msg}")
    return manager


def schema_from_luma_manager(manager: Any) -> ClassSchema:
    """Construct a canonical :class:`ClassSchema` from a ``luma-stack`` scheme manager."""
    raw_classes = getattr(manager, "classes", None)
    if not raw_classes:
        raise ValueError("luma-stack scheme manager contains no classes.")

    definitions = [
        ClassDefinition(
            id=int(item["ID"]),
            name=str(item["Class Name"]),
            color=str(item.get("Color Code", "#808080")),
        )
        for item in raw_classes
    ]
    return ClassSchema(classes=tuple(definitions))


def normalize_luma_probability_image(
    luma_image: Any,
    class_schema: ClassSchema,
    *,
    prob_band_map: Mapping[int, str] | None = None,
    band_naming: Literal["id", "name"] = "id",
    source_scale: float = 100.0,
    as_uint16: bool = True,
) -> Any:
    """Normalize a ``luma-stack`` soft-classification ``ee.Image`` to canonical bands.

    Selects only the probability bands ordered by ``class_schema`` (supporting
    explicit ``prob_band_map``, ``prob_<id>``, or sanitized ``prob_<name>``),
    rescales from ``luma-stack``'s ``probability_scale`` (default ``100``),
    clamps to ``[0, 1]``, and renames bands to canonical ``prob_<id>``. When
    ``as_uint16=True`` (default), scales to ``uint16`` ``[0, 10000]`` to reduce
    GEE EECU and export storage overhead.
    """
    if source_scale <= 0:
        raise ValueError(f"source_scale must be > 0, got {source_scale!r}.")
    if band_naming not in ("id", "name"):
        raise ValueError(f"band_naming must be 'id' or 'name', got {band_naming!r}.")

    source_bands: list[str] = []
    for entry in class_schema.classes:
        if prob_band_map is not None:
            if entry.id not in prob_band_map:
                raise KeyError(
                    f"Class ID {entry.id} ({entry.name}) missing from prob_band_map."
                )
            source_bands.append(prob_band_map[entry.id])
        elif band_naming == "name":
            source_bands.append(f"prob_{_sanitize_luma_class_name(entry.name)}")
        else:
            source_bands.append(f"prob_{entry.id}")

    canonical_bands = list(class_schema.gee_prob_band_names)
    selected = luma_image.select(source_bands)
    normalized = selected.divide(float(source_scale)).clamp(0.0, 1.0)

    if as_uint16:
        return (
            normalized.multiply(UINT16_PROB_SCALE)
            .round()
            .toUint16()
            .rename(canonical_bands)
        )
    return normalized.toFloat().rename(canonical_bands)


def build_luma_multitemporal_analysis(
    config: ProjectConfig,
    *,
    aoi: Any = None,
    reference_period: str,
    acquire: bool = False,
    acquire_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Delegate direct multi-period GEE orchestration to ``luma_ge.MultitemporalAnalysis``.

    When ``aoi`` is omitted, resolves ``config.aoi_asset`` via Earth Engine.
    Set ``acquire=True`` to immediately invoke ``analysis.acquire_imagery(...)``.
    """
    if len(config.epochs) < 2:
        raise ValueError(
            "luma-stack MultitemporalAnalysis requires at least 2 epochs in ProjectConfig."
        )

    resolved_aoi = aoi
    if resolved_aoi is None:
        if not config.aoi_asset:
            raise ValueError("Either aoi or config.aoi_asset must be provided.")
        import ee

        resolved_aoi = ee.FeatureCollection(config.aoi_asset).geometry()

    mod = _import_luma_module("multitemporal")
    periods = [
        mod.PeriodConfig(
            label=ep.label,
            start_date=ep.start_date,
            end_date=ep.end_date,
            sensor=ep.sensor,
            cloud_cover=ep.cloud_cover,
            cs_threshold=ep.cs_threshold,
        )
        for ep in config.epochs
    ]
    analysis = mod.MultitemporalAnalysis(
        aoi=resolved_aoi,
        periods=periods,
        reference_period=reference_period,
        scale=int(round(config.scale_m)),
        **kwargs,
    )
    if acquire:
        analysis.acquire_imagery(**dict(acquire_kwargs or {}))
    return analysis

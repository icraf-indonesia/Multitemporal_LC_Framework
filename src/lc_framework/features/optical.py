"""Optical spectral indices and wet-minus-dry phenological deltas (GEE)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lc_framework import _gee

if TYPE_CHECKING:
    import ee

# Native GEE formulas using canonical 6-band lowercase optical schema:
# B=blue, G=green, R=red, N=nir, S1=swir1, S2=swir2 (0..1 surface reflectance)
SUPPORTED_SPECTRAL_INDICES: dict[str, str] = {
    "NDVI": "(N - R) / (N + R + 1e-6)",
    "EVI": "2.5 * ((N - R) / (N + 6.0 * R - 7.5 * B + 1.0 + 1e-6))",
    "SAVI": "1.5 * (N - R) / (N + R + 0.5 + 1e-6)",
    "NDWI": "(G - N) / (G + N + 1e-6)",
    "MNDWI": "(G - S1) / (G + S1 + 1e-6)",
    "NDBI": "(S1 - N) / (S1 + N + 1e-6)",
    "NBR": "(N - S2) / (N + S2 + 1e-6)",
    "NDMI": "(N - S1) / (N + S1 + 1e-6)",
    "BSI": "((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + 1e-6)",
    "GCVI": "(N / (G + 1e-6)) - 1.0",
}


def compute_seasonal_indices(
    image: ee.Image,
    *,
    indices: Sequence[str] = ("NDVI", "EVI", "MNDWI", "NDBI", "NBR", "BSI"),
    suffix: str = "dry",
    custom_expressions: Mapping[str, str] | None = None,
) -> ee.Image:
    """Compute named spectral indices on a 6-band lowercase reflectance ``ee.Image``.

    Appends ``_<suffix>`` (e.g. ``NDVI_wet``, ``MNDWI_dry``) to each output band.
    """
    if not indices:
        raise ValueError("indices must contain at least one spectral index name.")
    if not suffix or not suffix.strip():
        raise ValueError("suffix must be a non-empty string (e.g. 'wet' or 'dry').")

    ee_mod = _gee.require_ee("optical spectral indices")
    registry = dict(SUPPORTED_SPECTRAL_INDICES)
    if custom_expressions:
        registry.update({k.upper(): v for k, v in custom_expressions.items()})

    band_map = {
        "B": image.select("blue"),
        "G": image.select("green"),
        "R": image.select("red"),
        "N": image.select("nir"),
        "S1": image.select("swir1"),
        "S2": image.select("swir2"),
    }

    computed: list[Any] = []
    clean_suffix = suffix.strip()
    for raw_name in indices:
        key = raw_name.strip().upper()
        if key not in registry:
            raise ValueError(
                f"Unsupported spectral index {raw_name!r}; supported indices are "
                f"{sorted(registry)}."
            )
        expr = registry[key]
        out_name = f"{raw_name.strip()}_{clean_suffix}"
        computed.append(image.expression(expr, band_map).rename(out_name))

    return ee_mod.Image.cat(computed)


def compute_phenological_deltas(
    wet_indices: ee.Image,
    dry_indices: ee.Image,
    *,
    indices: Sequence[str],
    wet_suffix: str = "wet",
    dry_suffix: str = "dry",
    delta_suffix: str = "delta",
) -> ee.Image:
    """Compute seasonal phenological differences (``<Index>_wet - <Index>_dry``)."""
    if not indices:
        raise ValueError("indices must contain at least one index name.")
    ee_mod = _gee.require_ee("phenological deltas")

    deltas = [
        wet_indices.select(f"{idx.strip()}_{wet_suffix}")
        .subtract(dry_indices.select(f"{idx.strip()}_{dry_suffix}"))
        .rename(f"{idx.strip()}_{delta_suffix}")
        for idx in indices
    ]
    return ee_mod.Image.cat(deltas)

"""Project configuration and Class Schema contracts for LC_Framework.

Enforces strict separation of study-area settings from library code:
all AOI assets, CRS, class IDs/names/colors, epochs, and seasonal fractions
are loaded from validated YAML or JSON files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class ClassDefinition(BaseModel):
    """A single land cover class entry within a :class:`ClassSchema`."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., ge=1, description="Unique positive integer class identifier.")
    name: str = Field(..., min_length=1, description="Canonical human-readable class name.")
    color: str = Field(
        default="#808080",
        description="Hex RGB color string for visualization (e.g. '#1b5e20').",
    )

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Class name cannot be empty or whitespace.")
        return cleaned

    @field_validator("color")
    @classmethod
    def _validate_hex_color(cls, value: str) -> str:
        cleaned = value.strip()
        if not _HEX_COLOR_RE.match(cleaned):
            raise ValueError(
                f"Invalid hex color {value!r}; expected '#RRGGBB' (e.g. '#1b5e20')."
            )
        return cleaned.lower()


class ClassSchema(BaseModel):
    """Ordered catalog of land cover classes shared across every workflow stage."""

    model_config = ConfigDict(frozen=True)

    classes: tuple[ClassDefinition, ...] = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce_list_shorthand(cls, data: Any) -> Any:
        if isinstance(data, list | tuple):
            return {"classes": list(data)}
        return data

    @model_validator(mode="after")
    def _validate_and_sort(self) -> ClassSchema:
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        for entry in self.classes:
            if entry.id in seen_ids:
                raise ValueError(f"Duplicate class id detected in ClassSchema: {entry.id}")
            seen_ids.add(entry.id)

            norm_name = entry.name.casefold()
            if norm_name in seen_names:
                raise ValueError(f"Duplicate class name detected in ClassSchema: {entry.name!r}")
            seen_names.add(norm_name)

        sorted_classes = tuple(sorted(self.classes, key=lambda c: c.id))
        object.__setattr__(self, "classes", sorted_classes)
        return self

    @property
    def class_ids(self) -> tuple[int, ...]:
        """Ordered integer class IDs."""
        return tuple(c.id for c in self.classes)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Ordered canonical class names."""
        return tuple(c.name for c in self.classes)

    @property
    def palette(self) -> tuple[str, ...]:
        """Ordered hex visualization colors."""
        return tuple(c.color for c in self.classes)

    @property
    def gee_prob_band_names(self) -> tuple[str, ...]:
        """Canonical GEE probability band names (``prob_<id>``)."""
        return tuple(f"prob_{c.id}" for c in self.classes)

    @property
    def geotiff_band_descriptions(self) -> tuple[str, ...]:
        """Canonical GeoTIFF band descriptions (``<id>:<name>``)."""
        return tuple(f"{c.id}:{c.name}" for c in self.classes)

    def name_for(self, class_id: int) -> str:
        """Look up the canonical name for ``class_id``."""
        for c in self.classes:
            if c.id == class_id:
                return c.name
        raise KeyError(f"Class ID {class_id} not found in ClassSchema.")

    def id_for(self, class_name: str) -> int:
        """Look up the integer ID for ``class_name`` (case-insensitive)."""
        target = class_name.strip().casefold()
        for c in self.classes:
            if c.name.casefold() == target:
                return c.id
        raise KeyError(f"Class name {class_name!r} not found in ClassSchema.")

    def __len__(self) -> int:
        return len(self.classes)


class EpochConfig(BaseModel):
    """Definition of a single mapping period / epoch."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., min_length=1)
    start_date: str = Field(..., min_length=4)
    end_date: str = Field(..., min_length=4)
    sensor: str = Field(default="S2", min_length=1)
    cloud_cover: int = Field(default=30, ge=0, le=100)
    cs_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_date_order(self) -> EpochConfig:
        if self.end_date < self.start_date:
            raise ValueError(
                f"Epoch {self.label!r} end_date ({self.end_date!r}) must be >= "
                f"start_date ({self.start_date!r})."
            )
        return self


class ProjectConfig(BaseModel):
    """Study-area-agnostic project configuration loaded from YAML or JSON."""

    model_config = ConfigDict(frozen=True)

    project_name: str = Field(..., min_length=1)
    crs: str = Field(..., min_length=3, description="Target CRS, e.g. 'EPSG:32748'.")
    scale_m: float = Field(..., gt=0, description="Target spatial resolution in meters.")
    class_schema: ClassSchema
    aoi_asset: str | None = Field(
        default=None,
        description="Optional GEE FeatureCollection asset ID for the study area.",
    )
    export_prefix: str | None = Field(
        default=None,
        description="Optional GEE asset or Drive folder prefix for exported artifacts.",
    )
    seasons: dict[str, float] = Field(
        default_factory=lambda: {"wet": 0.16, "dry": 0.65, "harmonic_mid": 0.50},
        description="Named fractional-year offsets in [0.0, 1.0).",
    )
    epochs: tuple[EpochConfig, ...] = Field(default=())
    smoothness_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Optional per-class smoothness weights keyed by canonical class name.",
    )

    @field_validator("seasons")
    @classmethod
    def _validate_seasons(cls, value: dict[str, float]) -> dict[str, float]:
        for name, frac in value.items():
            if not (0.0 <= float(frac) < 1.0):
                raise ValueError(
                    f"Invalid season fraction for {name!r}: {frac!r}. Must lie in [0.0, 1.0)."
                )
        return {k: float(v) for k, v in value.items()}

    @model_validator(mode="after")
    def _validate_smoothness_keys(self) -> ProjectConfig:
        valid_names = {c.name for c in self.class_schema.classes}
        unknown = set(self.smoothness_weights) - valid_names
        if unknown:
            raise ValueError(
                f"smoothness_weights contains unknown class names: {sorted(unknown)}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
        """Load and validate a :class:`ProjectConfig` from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}.")
        return cls.model_validate(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> ProjectConfig:
        """Load and validate a :class:`ProjectConfig` from a JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a JSON object in {path}, got {type(raw).__name__}.")
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> Path:
        """Serialize this configuration cleanly to a YAML file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return out

    def to_json(self, path: str | Path) -> Path:
        """Serialize this configuration cleanly to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out

"""Shared internal helpers for ``lc_framework.classification`` (ADR-0005)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from lc_framework.config import ClassSchema, ProjectConfig


def resolve_class_schema(
    class_schema: ClassSchema | None,
    config: ProjectConfig | None,
    *,
    caller: str,
) -> ClassSchema:
    """Resolve a :class:`ClassSchema` from explicit argument or :class:`ProjectConfig`."""
    if class_schema is not None:
        if config is not None and config.class_schema != class_schema:
            raise ValueError(
                f"{caller}: explicit class_schema does not match config.class_schema."
            )
        return class_schema
    if config is not None:
        return config.class_schema
    raise ValueError(f"{caller}: provide either class_schema or config.")


def resolve_selected_features(
    selected_features: Any | None,
) -> tuple[str, ...] | None:
    """Resolve an ordered feature list from a sequence, RFECV artifact, or JSON/YAML file.

    Parses JSON/YAML RFECV artifacts directly so ``lc_framework.classification``
    does not eagerly import ``lc_framework.features`` or ``scikit-learn`` (ADR-0005).
    """
    if selected_features is None:
        return None

    raw_list: Sequence[Any]
    if isinstance(selected_features, str | Path):
        path = Path(selected_features)
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = json.loads(text)
        elif suffix in (".yaml", ".yml"):
            payload = yaml.safe_load(text)
        else:
            raise ValueError(
                f"Unsupported selected_features file extension {path.suffix!r}; "
                "expected '.json' or '.yaml'."
            )
        if not isinstance(payload, dict) or "selected_features" not in payload:
            raise ValueError(
                f"Artifact {path} does not contain a valid 'selected_features' field."
            )
        raw_list = payload["selected_features"]
    elif hasattr(selected_features, "selected_features"):
        raw_list = selected_features.selected_features
    elif isinstance(selected_features, Sequence) and not isinstance(
        selected_features, bytes | bytearray
    ):
        raw_list = selected_features
    else:
        raise TypeError(
            "selected_features must be a Sequence[str], SelectedFeaturesArtifact, "
            "or path to a JSON/YAML artifact."
        )

    cleaned = tuple(str(name).strip() for name in raw_list if str(name).strip())
    if not cleaned:
        raise ValueError("selected_features must contain at least one feature name.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"Duplicate feature names in selected_features: {cleaned!r}.")
    return cleaned


def align_probabilities_to_schema(
    raw_proba: np.ndarray,
    model_classes: Sequence[int],
    class_schema: ClassSchema,
) -> NDArray[np.float32]:
    """Align model probability columns to ``ClassSchema.class_ids`` and renormalize."""
    proba_arr = np.asarray(raw_proba, dtype=np.float64)
    if proba_arr.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape {proba_arr.shape}.")
    if proba_arr.shape[1] != len(model_classes):
        raise ValueError(
            f"Probability column count ({proba_arr.shape[1]}) does not match "
            f"model class count ({len(model_classes)})."
        )
    if not np.all(np.isfinite(proba_arr)):
        raise ValueError("Model predict_proba() returned non-finite (NaN/inf) probabilities.")

    n_rows = proba_arr.shape[0]
    n_schema = len(class_schema)
    aligned = np.zeros((n_rows, n_schema), dtype=np.float64)
    id_to_col = {cid: idx for idx, cid in enumerate(class_schema.class_ids)}

    for model_col, raw_cid in enumerate(model_classes):
        cid = int(raw_cid)
        if cid in id_to_col:
            aligned[:, id_to_col[cid]] = proba_arr[:, model_col]

    aligned = np.clip(aligned, 0.0, 1.0)
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(
            "Model predict_proba() produced zero total probability mass for one or more rows "
            "after alignment to ClassSchema."
        )
    aligned = aligned / row_sums
    return aligned.astype(np.float32)

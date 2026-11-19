"""Local CSV-based Recursive Feature Elimination with Cross-Validation (RFECV).

Runs completely outside Google Earth Engine (requiring only ``lc-framework[ml]``),
while keeping :class:`SelectedFeaturesArtifact` and :func:`load_selected_features`
usable on the lean core install so GEE and local classifiers can consume the
selected feature list directly (ADR-0005).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lc_framework.config import ProjectConfig

ArtifactFormat = Literal["json", "yaml"]


def _require_sklearn() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily import required scikit-learn RFECV components."""
    try:
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for local RFECV feature selection. "
            "Install it with: pip install 'lc-framework[ml]'"
        ) from exc
    return RandomForestClassifier, RFECV, StratifiedKFold, StratifiedGroupKFold, clone


class SelectedFeaturesArtifact(BaseModel):
    """Validated optimal feature subset artifact shared by GEE and local classifiers."""

    model_config = ConfigDict(frozen=True)

    selected_features: tuple[str, ...] = Field(..., min_length=1)
    all_candidate_features: tuple[str, ...] = Field(..., min_length=1)
    scoring: str = Field(..., min_length=1)
    best_score: float
    n_samples: int = Field(..., ge=1)
    class_col: str = "class_id"
    cv_folds: int = Field(default=5, ge=2)
    random_state: int = 42
    source_csv: str | None = None
    feature_rankings: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_features(self) -> SelectedFeaturesArtifact:
        if len(set(self.selected_features)) != len(self.selected_features):
            raise ValueError("Duplicate feature names detected in selected_features.")
        if len(set(self.all_candidate_features)) != len(self.all_candidate_features):
            raise ValueError("Duplicate feature names detected in all_candidate_features.")
        missing = set(self.selected_features) - set(self.all_candidate_features)
        if missing:
            raise ValueError(
                f"selected_features contains names not in all_candidate_features: "
                f"{sorted(missing)}"
            )
        unknown_ranked = set(self.feature_rankings) - set(self.all_candidate_features)
        if unknown_ranked:
            raise ValueError(
                f"feature_rankings contains unknown features: {sorted(unknown_ranked)}"
            )
        bad_ranks = {k: v for k, v in self.feature_rankings.items() if v < 1}
        if bad_ranks:
            raise ValueError(f"feature_rankings values must be >= 1; got {bad_ranks}")
        return self

    def to_json(self, path: str | Path) -> Path:
        """Write this artifact to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out

    def to_yaml(self, path: str | Path) -> Path:
        """Write this artifact to YAML."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> SelectedFeaturesArtifact:
        """Load and validate an artifact from JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SelectedFeaturesArtifact:
        """Load and validate an artifact from YAML."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}.")
        return cls.model_validate(raw)


def load_selected_features(path: str | Path) -> SelectedFeaturesArtifact:
    """Load a :class:`SelectedFeaturesArtifact` from either JSON or YAML."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return SelectedFeaturesArtifact.from_yaml(p)
    if suffix == ".json":
        return SelectedFeaturesArtifact.from_json(p)
    raise ValueError(f"Unsupported artifact extension {p.suffix!r}; expected '.json' or '.yaml'.")


@dataclass(frozen=True)
class RFECVResult:
    """Complete result container from local RFECV feature selection."""

    artifact: SelectedFeaturesArtifact
    selected_features: list[str]
    cv_scores: pd.DataFrame
    feature_rankings: pd.DataFrame
    selector: Any
    dropped_rows: int = 0

    def save_outputs(
        self,
        output_dir: str | Path,
        *,
        prefix: str = "rfecv",
        fmt: ArtifactFormat = "json",
    ) -> dict[str, Path]:
        """Save selected feature artifact, CV score curve CSV, and feature rankings CSV."""
        if fmt not in ("json", "yaml"):
            raise ValueError(f"fmt must be 'json' or 'yaml', got {fmt!r}.")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            art_path = self.artifact.to_json(out_dir / f"{prefix}_selected_features.json")
        else:
            art_path = self.artifact.to_yaml(out_dir / f"{prefix}_selected_features.yaml")

        scores_path = out_dir / f"{prefix}_cv_scores.csv"
        self.cv_scores.to_csv(scores_path, index=False)

        rankings_path = out_dir / f"{prefix}_feature_rankings.csv"
        self.feature_rankings.to_csv(rankings_path, index=False)

        return {
            "selected_features_artifact": art_path,
            "cv_scores_csv": scores_path,
            "feature_rankings_csv": rankings_path,
        }


def run_rfecv_selection(
    x_data: pd.DataFrame | np.ndarray,
    y_data: pd.Series | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    groups: pd.Series | np.ndarray | None = None,
    estimator: Any | None = None,
    cv: int = 5,
    scoring: str = "balanced_accuracy",
    step: int | float = 1,
    min_features_to_select: int = 1,
    random_state: int = 42,
    n_jobs: int | None = -1,
    n_estimators: int = 200,
    class_col: str = "class_id",
    source_csv: str | None = None,
) -> RFECVResult:
    """Run reproducible stratified (or group-stratified) RFECV on feature matrix ``x_data``."""
    if cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv!r}.")
    if min_features_to_select < 1:
        raise ValueError(
            f"min_features_to_select must be >= 1, got {min_features_to_select!r}."
        )

    rf_cls, rfecv_cls, stratified_kfold_cls, stratified_group_kfold_cls, clone_fn = (
        _require_sklearn()
    )

    if isinstance(x_data, pd.DataFrame):
        names = list(x_data.columns) if feature_names is None else list(feature_names)
        x_arr = x_data.to_numpy(dtype=np.float64, copy=False)
    else:
        x_arr = np.asarray(x_data, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"x_data must be 2D, got shape {x_arr.shape}.")
        names = (
            [f"f_{i}" for i in range(x_arr.shape[1])]
            if feature_names is None
            else list(feature_names)
        )

    if len(names) != x_arr.shape[1]:
        raise ValueError(
            f"feature_names length ({len(names)}) must match feature columns ({x_arr.shape[1]})."
        )
    if x_arr.shape[1] == 0:
        raise ValueError("x_data must contain at least one feature column.")
    if min_features_to_select > x_arr.shape[1]:
        raise ValueError(
            f"min_features_to_select ({min_features_to_select}) cannot exceed "
            f"available feature count ({x_arr.shape[1]})."
        )

    y_series = pd.Series(y_data).reset_index(drop=True)
    if len(y_series) != x_arr.shape[0]:
        raise ValueError(
            f"x_data rows ({x_arr.shape[0]}) and y_data length ({len(y_series)}) must match."
        )

    valid_mask = np.all(np.isfinite(x_arr), axis=1) & y_series.notna().to_numpy()
    groups_arr: np.ndarray | None = None
    if groups is not None:
        g_series = pd.Series(groups).reset_index(drop=True)
        if len(g_series) != x_arr.shape[0]:
            raise ValueError("groups length must match x_data row count.")
        valid_mask &= g_series.notna().to_numpy()
        groups_arr = g_series.to_numpy()[valid_mask]

    dropped_rows = int(x_arr.shape[0] - np.sum(valid_mask))
    x_clean = x_arr[valid_mask]
    y_clean = y_series.to_numpy()[valid_mask]

    if x_clean.shape[0] < cv:
        raise ValueError(
            f"Too few valid samples ({x_clean.shape[0]}) for {cv}-fold cross-validation."
        )

    if estimator is None:
        base_estimator = rf_cls(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=n_jobs,
        )
    else:
        base_estimator = estimator

    if groups_arr is not None:
        splitter = stratified_group_kfold_cls(n_splits=int(cv))
    else:
        splitter = stratified_kfold_cls(
            n_splits=int(cv), shuffle=True, random_state=int(random_state)
        )

    selector = rfecv_cls(
        estimator=base_estimator,
        step=step,
        min_features_to_select=int(min_features_to_select),
        cv=splitter,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    if groups_arr is not None:
        selector.fit(x_clean, y_clean, groups=groups_arr)
    else:
        selector.fit(x_clean, y_clean)

    support = np.asarray(selector.support_, dtype=bool)
    rankings = np.asarray(selector.ranking_, dtype=int)
    selected_features = [names[i] for i in np.flatnonzero(support).tolist()]

    cv_res = selector.cv_results_
    n_feat_grid = np.asarray(cv_res["n_features"], dtype=int)
    mean_scores = np.asarray(cv_res["mean_test_score"], dtype=np.float64)
    std_scores = np.asarray(cv_res["std_test_score"], dtype=np.float64)
    cv_scores_df = pd.DataFrame(
        {
            "n_features": n_feat_grid,
            "mean_test_score": mean_scores,
            "std_test_score": std_scores,
        }
    )

    def _extract_importance(est: Any) -> np.ndarray | None:
        if est is None:
            return None
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=np.float64)
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=np.float64)
            return np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        return None

    # Full-feature baseline importances so every candidate (including eliminated ones) is ranked
    full_est = clone_fn(base_estimator)
    full_est.fit(x_clean, y_clean)
    baseline_imp = _extract_importance(full_est)
    if baseline_imp is None:
        baseline_imp = np.full(len(names), np.nan, dtype=np.float64)

    selected_imp = np.full(len(names), np.nan, dtype=np.float64)
    fitted_imp = _extract_importance(getattr(selector, "estimator_", None))
    if fitted_imp is not None:
        selected_imp[support] = fitted_imp

    rankings_df = (
        pd.DataFrame(
            {
                "feature": names,
                "ranking": rankings,
                "selected": support,
                "importance": selected_imp,
                "baseline_importance": baseline_imp,
            }
        )
        .sort_values(
            ["ranking", "baseline_importance"],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    best_idx = int(np.argmax(mean_scores))
    artifact = SelectedFeaturesArtifact(
        selected_features=tuple(selected_features),
        all_candidate_features=tuple(names),
        scoring=scoring,
        best_score=float(mean_scores[best_idx]),
        n_samples=int(x_clean.shape[0]),
        class_col=class_col,
        cv_folds=int(cv),
        random_state=int(random_state),
        source_csv=source_csv,
        feature_rankings={names[i]: int(rankings[i]) for i in range(len(names))},
    )

    return RFECVResult(
        artifact=artifact,
        selected_features=selected_features,
        cv_scores=cv_scores_df,
        feature_rankings=rankings_df,
        selector=selector,
        dropped_rows=dropped_rows,
    )


def run_rfecv_from_csv(
    csv_path: str | Path | None = None,
    *,
    class_col: str = "class_id",
    group_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    ignore_cols: Sequence[str] = ("system:index", ".geo", "sample_id", "class_name"),
    config: ProjectConfig | None = None,
    estimator: Any | None = None,
    cv: int = 5,
    scoring: str = "balanced_accuracy",
    step: int | float = 1,
    min_features_to_select: int = 1,
    random_state: int = 42,
    n_jobs: int | None = -1,
    n_estimators: int = 200,
    output_dir: str | Path | None = None,
    artifact_prefix: str = "rfecv",
    artifact_format: ArtifactFormat = "json",
) -> RFECVResult:
    """Load an exported sample-features CSV (by path or via ``ProjectConfig``) and run RFECV."""
    resolved_csv = csv_path or (config.sample_features_csv if config is not None else None)
    if resolved_csv is None:
        raise ValueError("Either csv_path or config.sample_features_csv must be provided.")
    path = Path(resolved_csv)

    raw_df = pd.read_csv(path)
    if class_col not in raw_df.columns:
        raise ValueError(f"Target column {class_col!r} not found in {path}.")

    groups: pd.Series | None = None
    if group_col is not None:
        if group_col not in raw_df.columns:
            raise ValueError(f"group_col {group_col!r} not found in {path}.")
        groups = raw_df[group_col]

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from CSV: {missing}")
        if group_col is not None and group_col in feature_cols:
            raise ValueError(f"group_col {group_col!r} must not also appear in feature_cols.")
        candidate_names = list(feature_cols)
    else:
        excluded = set(ignore_cols) | {class_col}
        if group_col is not None:
            excluded.add(group_col)
        candidate_names = [c for c in raw_df.columns if c not in excluded]

    # Coerce candidate columns to numeric; fail loudly for explicit feature_cols, warn otherwise
    numeric_cols: list[str] = []
    coerced_data: dict[str, pd.Series] = {}
    dropped_non_numeric: list[str] = []
    for col in candidate_names:
        series = pd.to_numeric(raw_df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
            coerced_data[col] = series
        else:
            dropped_non_numeric.append(col)

    if dropped_non_numeric:
        if feature_cols is not None:
            raise ValueError(
                f"Explicitly requested feature_cols contain no numeric values: "
                f"{dropped_non_numeric}"
            )
        warnings.warn(
            f"Ignored non-numeric columns during RFECV: {dropped_non_numeric}",
            UserWarning,
            stacklevel=2,
        )
    if not numeric_cols:
        raise ValueError("No numeric feature columns remained in CSV after filtering metadata.")

    y_series = raw_df[class_col]
    if config is not None:
        valid_ids = set(config.class_schema.class_ids)
        observed_ids: set[int] = set()
        for val in y_series.dropna().unique().tolist():
            if isinstance(val, bool) or not isinstance(val, int | float | np.integer | np.floating):
                raise ValueError(
                    f"CSV column {class_col!r} contains non-integer class ID {val!r}."
                )
            fval = float(val)
            if not fval.is_integer():
                raise ValueError(
                    f"CSV column {class_col!r} contains non-integer class ID {val!r}."
                )
            observed_ids.add(int(fval))
        unknown = observed_ids - valid_ids
        if unknown:
            raise ValueError(
                f"CSV contains class IDs {sorted(unknown)} not defined in "
                f"ProjectConfig.class_schema ({sorted(valid_ids)})."
            )

    x_numeric = pd.DataFrame(coerced_data)
    result = run_rfecv_selection(
        x_numeric,
        y_series,
        feature_names=numeric_cols,
        groups=groups,
        estimator=estimator,
        cv=cv,
        scoring=scoring,
        step=step,
        min_features_to_select=min_features_to_select,
        random_state=random_state,
        n_jobs=n_jobs,
        n_estimators=n_estimators,
        class_col=class_col,
        source_csv=path.name,
    )

    if output_dir is not None:
        result.save_outputs(output_dir, prefix=artifact_prefix, fmt=artifact_format)

    return result

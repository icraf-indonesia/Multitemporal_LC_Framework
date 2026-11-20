"""Local Python Pixel-Based Classification backend (``lc-framework[ml]``).

Supports ``RandomForest``, ``ExtraTrees``, ``LightGBM``, and ``XGBoost`` (or any
scikit-learn-compatible probabilistic estimator), hyperparameter tuning via
``RandomizedSearchCV`` / ``GridSearchCV`` with spatial group-CV support, CV +
held-out evaluation metrics (including multi-class log-loss), and memory-safe
tiled Cloud-Optimized GeoTIFF probability inference emitting canonical
``Class Probability Stack`` artifacts (ADR-0003, ADR-0005).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window

from lc_framework.classification._common import (
    align_probabilities_to_schema,
    resolve_class_schema,
    resolve_selected_features,
)
from lc_framework.config import ClassSchema, ProjectConfig
from lc_framework.io import (
    UINT16_PROB_NODATA,
    UINT16_PROB_SCALE,
    ProbabilityEncoding,
    ProbabilityManifest,
    manifest_path_for,
)

ModelFamily = Literal["random_forest", "extra_trees", "lightgbm", "xgboost"]
SearchMethod = Literal["random", "grid"]

_CANONICAL_FAMILY_ALIASES: dict[str, ModelFamily] = {
    "randomforest": "random_forest",
    "rf": "random_forest",
    "extratrees": "extra_trees",
    "et": "extra_trees",
    "ert": "extra_trees",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "xgboost": "xgboost",
    "xgb": "xgboost",
}

_DEFAULT_PARAM_SPACES: dict[ModelFamily, dict[str, tuple[Any, ...]]] = {
    "random_forest": {
        "n_estimators": (100, 200, 400),
        "max_depth": (None, 12, 24),
        "min_samples_leaf": (1, 2, 5),
        "max_features": ("sqrt", 0.5),
    },
    "extra_trees": {
        "n_estimators": (100, 200, 400),
        "max_depth": (None, 12, 24),
        "min_samples_leaf": (1, 2, 5),
        "max_features": ("sqrt", 0.5),
    },
    "lightgbm": {
        "n_estimators": (100, 250, 400),
        "learning_rate": (0.03, 0.08, 0.15),
        "num_leaves": (15, 31, 63),
        "subsample": (0.8, 1.0),
    },
    "xgboost": {
        "n_estimators": (100, 250, 400),
        "learning_rate": (0.03, 0.08, 0.15),
        "max_depth": (4, 6, 8),
        "subsample": (0.8, 1.0),
    },
}


def _normalize_family(name: str) -> ModelFamily | None:
    key = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    return _CANONICAL_FAMILY_ALIASES.get(key)


def _require_sklearn_ml() -> dict[str, Any]:
    """Lazily import required scikit-learn components for training, CV, and tuning."""
    try:
        from sklearn.base import clone
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            log_loss,
        )
        from sklearn.model_selection import (
            GridSearchCV,
            RandomizedSearchCV,
            StratifiedGroupKFold,
            StratifiedKFold,
            cross_val_predict,
            train_test_split,
        )
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for Local Python Pixel-Based Classification. "
            "Install it with: pip install 'lc-framework[ml]'"
        ) from exc

    return {
        "clone": clone,
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "GridSearchCV": GridSearchCV,
        "RandomizedSearchCV": RandomizedSearchCV,
        "StratifiedKFold": StratifiedKFold,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "cross_val_predict": cross_val_predict,
        "train_test_split": train_test_split,
        "LabelEncoder": LabelEncoder,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "log_loss": log_loss,
    }


def default_param_space(classifier: str) -> dict[str, list[Any]]:
    """Return a copy of the default hyperparameter search space for ``classifier``."""
    family = _normalize_family(classifier)
    if family is None:
        raise ValueError(
            f"Unsupported classifier {classifier!r}. Choose one of: "
            "random_forest, extra_trees, lightgbm, xgboost."
        )
    return {k: list(v) for k, v in _DEFAULT_PARAM_SPACES[family].items()}


def build_pixel_classifier(
    classifier: str = "random_forest",
    *,
    n_estimators: int = 200,
    n_jobs: int | None = -1,
    random_state: int = 42,
    class_weight: str | Mapping[int, float] | None = "balanced_subsample",
) -> Any:
    """Construct a supported scikit-learn-compatible classifier lazily."""
    family = _normalize_family(classifier)
    if family is None:
        raise ValueError(
            f"Unsupported classifier {classifier!r}. Choose 'random_forest', "
            "'extra_trees', 'lightgbm', 'xgboost', or pass an estimator instance."
        )

    if family in ("random_forest", "extra_trees"):
        sk = _require_sklearn_ml()
        cls_type = (
            sk["RandomForestClassifier"]
            if family == "random_forest"
            else sk["ExtraTreesClassifier"]
        )
        return cls_type(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=n_jobs,
            class_weight=class_weight,
        )

    if family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for classifier='lightgbm'. "
                "Install it with: pip install 'lc-framework[ml]'"
            ) from exc
        return LGBMClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=n_jobs,
            class_weight="balanced" if class_weight is not None else None,
            verbosity=-1,
        )

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for classifier='xgboost'. "
            "Install it with: pip install 'lc-framework[ml]'"
        ) from exc
    return XGBClassifier(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=n_jobs,
        eval_metric="mlogloss",
    )


@dataclass(frozen=True)
class ModelEvaluationMetrics:
    """Classification metrics on held-out or out-of-fold validation predictions."""

    metrics: dict[str, float]
    confusion_matrix: NDArray[np.int64]
    report: dict[str, Any]


@dataclass(frozen=True)
class LocalPixelModel:
    """Fitted local pixel classifier bound to ordered feature names and ``ClassSchema``."""

    estimator: Any
    feature_names: tuple[str, ...]
    class_schema: ClassSchema
    trained_class_ids: tuple[int, ...]
    model_family: str
    class_col: str = "class_id"

    @property
    def classes_(self) -> tuple[int, ...]:
        """Ordered trained class IDs (in ``ClassSchema`` ID space)."""
        return self.trained_class_ids

    def _prepare_matrix(self, X: pd.DataFrame | np.ndarray) -> NDArray[np.float64]:
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame missing required features: {missing}")
            arr = X[list(self.feature_names)].to_numpy(dtype=np.float64, copy=False)
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Input array must have shape (n_samples, {len(self.feature_names)}), "
                    f"got {arr.shape}."
                )
        return arr

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> NDArray[np.float32]:
        """Predict probabilities aligned strictly to ``class_schema.class_ids``."""
        arr = self._prepare_matrix(X)
        raw = self.estimator.predict_proba(arr)
        return align_probabilities_to_schema(raw, self.trained_class_ids, self.class_schema)

    def predict(self, X: pd.DataFrame | np.ndarray) -> NDArray[np.int32]:
        """Predict top-1 class IDs in ``class_schema.class_ids``."""
        probs = self.predict_proba(X)
        lookup = np.asarray(self.class_schema.class_ids, dtype=np.int32)
        return lookup[np.argmax(probs, axis=1)]

    def predict_raster(
        self,
        feature_raster_path: str | Path,
        output_path: str | Path,
        *,
        band_names: Sequence[str] | None = None,
        encoding: ProbabilityEncoding = "uint16",
        epoch: str | None = None,
        block_size: int = 512,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Stream tiled probability predictions via :func:`predict_raster_probabilities`."""
        return predict_raster_probabilities(
            feature_raster_path,
            self,
            output_path,
            band_names=band_names,
            encoding=encoding,
            epoch=epoch,
            block_size=block_size,
            extra=extra,
        )


@dataclass(frozen=True)
class LocalPixelTrainingResult:
    """Complete result from local pixel classifier training / tuning & CV evaluation."""

    model: LocalPixelModel
    cv_metrics: dict[str, float]
    best_params: dict[str, Any]
    best_cv_score: float
    evaluation: ModelEvaluationMetrics | None = None
    dropped_rows: int = 0


def _contains_xgb_estimator(estimator: Any, _seen: set[int] | None = None) -> bool:
    """Inspect an estimator or meta-estimator tree (Pipeline, etc.) for XGBoost."""
    if estimator is None:
        return False
    seen = set() if _seen is None else _seen
    if id(estimator) in seen:
        return False
    seen.add(id(estimator))

    if "xgb" in type(estimator).__name__.lower():
        return True
    if hasattr(estimator, "get_params"):
        try:
            for val in estimator.get_params(deep=True).values():
                if val is not estimator and "xgb" in type(val).__name__.lower():
                    return True
        except Exception:
            pass
    if hasattr(estimator, "steps") and isinstance(estimator.steps, list | tuple):
        for _, step in estimator.steps:
            if _contains_xgb_estimator(step, seen):
                return True
    for attr in ("estimator", "base_estimator", "classifier"):
        if hasattr(estimator, attr) and _contains_xgb_estimator(getattr(estimator, attr), seen):
            return True
    if hasattr(estimator, "estimators") and isinstance(estimator.estimators, list | tuple):
        for item in estimator.estimators:
            sub = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if _contains_xgb_estimator(sub, seen):
                return True
    return False


def _requires_zero_based_labels(estimator: Any, classifier_name: str) -> bool:
    family = _normalize_family(classifier_name)
    if family == "xgboost":
        return True
    return _contains_xgb_estimator(estimator)


def _infer_family_from_estimator(estimator: Any) -> ModelFamily | None:
    cls_name = type(estimator).__name__.lower()
    if "randomforest" in cls_name:
        return "random_forest"
    if "extratrees" in cls_name:
        return "extra_trees"
    if "lgbm" in cls_name or "lightgbm" in cls_name:
        return "lightgbm"
    if "xgb" in cls_name:
        return "xgboost"
    return None


def _prepare_training_inputs(
    x_data: pd.DataFrame | np.ndarray,
    y_data: pd.Series | np.ndarray,
    *,
    class_schema: ClassSchema,
    selected_features: Any | None,
    feature_names: Sequence[str] | None,
    groups: pd.Series | np.ndarray | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    tuple[str, ...],
    NDArray[Any] | None,
    int,
]:
    resolved_subset = resolve_selected_features(selected_features)

    if isinstance(x_data, pd.DataFrame):
        if resolved_subset is not None:
            missing = [f for f in resolved_subset if f not in x_data.columns]
            if missing:
                raise ValueError(f"Selected features missing from input DataFrame: {missing}")
            active_names = resolved_subset
        elif feature_names is not None:
            missing = [f for f in feature_names if f not in x_data.columns]
            if missing:
                raise ValueError(f"feature_names missing from input DataFrame: {missing}")
            active_names = tuple(feature_names)
        else:
            active_names = tuple(str(c) for c in x_data.columns)
        x_arr = x_data[list(active_names)].to_numpy(dtype=np.float64, copy=False)
    else:
        full_arr = np.asarray(x_data, dtype=np.float64)
        if full_arr.ndim != 2 or full_arr.shape[1] == 0:
            raise ValueError(f"x_data must be non-empty 2D array, got shape {full_arr.shape}.")
        all_names = (
            tuple(f"f_{i}" for i in range(full_arr.shape[1]))
            if feature_names is None
            else tuple(feature_names)
        )
        if len(all_names) != full_arr.shape[1]:
            raise ValueError(
                f"feature_names length ({len(all_names)}) must match columns ({full_arr.shape[1]})."
            )
        if resolved_subset is not None:
            name_to_idx = {n: i for i, n in enumerate(all_names)}
            missing = [f for f in resolved_subset if f not in name_to_idx]
            if missing:
                raise ValueError(f"Selected features missing from feature_names: {missing}")
            cols_idx = [name_to_idx[f] for f in resolved_subset]
            x_arr = full_arr[:, cols_idx]
            active_names = resolved_subset
        else:
            x_arr = full_arr
            active_names = all_names

    if not active_names:
        raise ValueError("No feature columns available for local pixel classification.")

    y_series = pd.Series(y_data).reset_index(drop=True)
    if len(y_series) != x_arr.shape[0]:
        raise ValueError(
            f"x_data rows ({x_arr.shape[0]}) and y_data length ({len(y_series)}) must match."
        )

    valid_mask = np.all(np.isfinite(x_arr), axis=1) & y_series.notna().to_numpy()
    groups_clean: NDArray[Any] | None = None
    if groups is not None:
        g_series = pd.Series(groups).reset_index(drop=True)
        if len(g_series) != x_arr.shape[0]:
            raise ValueError("groups length must match x_data row count.")
        valid_mask &= g_series.notna().to_numpy()
        groups_clean = g_series.to_numpy()[valid_mask]

    dropped_rows = int(x_arr.shape[0] - np.count_nonzero(valid_mask))
    x_clean = x_arr[valid_mask]
    y_vals = y_series.to_numpy()[valid_mask]
    y_num = pd.to_numeric(pd.Series(y_vals), errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y_num)) or not np.all(np.mod(y_num, 1.0) == 0.0):
        raise ValueError("Target labels must be integer ClassSchema IDs (got non-integer values).")
    y_clean = y_num.astype(np.int64)

    schema_ids = set(class_schema.class_ids)
    unknown = sorted({int(v) for v in y_clean} - schema_ids)
    if unknown:
        raise ValueError(
            f"Training labels {unknown} are not defined in ClassSchema {class_schema.class_ids}."
        )
    if np.unique(y_clean).size < 2:
        raise ValueError("At least 2 distinct land-cover classes are required to train.")

    return x_clean, y_clean, active_names, groups_clean, dropped_rows


def _align_eval_probabilities(
    proba: NDArray[np.float64],
    trained_ids: Sequence[int],
    eval_labels: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Expand/align probability columns to cover all ``trained_ids | eval_labels``."""
    union_ids = sorted({int(c) for c in trained_ids} | {int(c) for c in eval_labels})
    out = np.zeros((proba.shape[0], len(union_ids)), dtype=np.float64)
    id_to_col = {cid: idx for idx, cid in enumerate(union_ids)}
    for src_col, cid in enumerate(trained_ids):
        out[:, id_to_col[int(cid)]] = proba[:, src_col]
    return out, np.asarray(union_ids, dtype=np.int64)


def _compute_metrics(
    sk: dict[str, Any],
    y_true: NDArray[np.int64],
    proba: NDArray[np.float64],
    trained_ids: Sequence[int],
) -> ModelEvaluationMetrics:
    aligned_proba, labels_arr = _align_eval_probabilities(proba, trained_ids, y_true.tolist())
    preds = labels_arr[np.argmax(aligned_proba, axis=1)]
    metrics = {
        "accuracy": float(sk["accuracy_score"](y_true, preds)),
        "balanced_accuracy": float(sk["balanced_accuracy_score"](y_true, preds)),
        "f1_macro": float(sk["f1_score"](y_true, preds, average="macro", zero_division=0)),
        "f1_weighted": float(sk["f1_score"](y_true, preds, average="weighted", zero_division=0)),
        "log_loss": float(sk["log_loss"](y_true, aligned_proba, labels=labels_arr)),
    }
    report = dict(
        sk["classification_report"](
            y_true,
            preds,
            labels=labels_arr,
            output_dict=True,
            zero_division=0,
        )
    )
    matrix = np.asarray(sk["confusion_matrix"](y_true, preds, labels=labels_arr), dtype=np.int64)
    return ModelEvaluationMetrics(metrics=metrics, confusion_matrix=matrix, report=report)


def _train_or_tune_core(
    x_data: pd.DataFrame | np.ndarray,
    y_data: pd.Series | np.ndarray,
    *,
    class_schema: ClassSchema,
    classifier: str | Any,
    selected_features: Any | None,
    feature_names: Sequence[str] | None,
    groups: pd.Series | np.ndarray | None,
    cv: int,
    scoring: str,
    n_estimators: int,
    n_jobs: int | None,
    random_state: int,
    val_fraction: float,
    param_space: Mapping[str, Any] | None,
    search_method: SearchMethod,
    n_iter: int,
    run_tuning: bool,
) -> LocalPixelTrainingResult:
    if cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv!r}.")
    if not (0.0 <= val_fraction < 0.9):
        raise ValueError(f"val_fraction must lie in [0.0, 0.9), got {val_fraction!r}.")

    sk = _require_sklearn_ml()
    x_clean, y_clean, active_names, groups_clean, dropped_rows = _prepare_training_inputs(
        x_data,
        y_data,
        class_schema=class_schema,
        selected_features=selected_features,
        feature_names=feature_names,
        groups=groups,
    )

    if val_fraction > 0.0:
        if groups_clean is not None:
            n_unique_groups = int(np.unique(groups_clean).size)
            if n_unique_groups < 2:
                raise ValueError(
                    "At least 2 distinct spatial/sample groups are required for grouped holdout."
                )
            n_splits_holdout = min(max(2, round(1.0 / val_fraction)), n_unique_groups)
            hold_splitter = sk["StratifiedGroupKFold"](n_splits=n_splits_holdout)
            train_idx, val_idx = next(hold_splitter.split(x_clean, y_clean, groups_clean))
            x_tr, x_val = x_clean[train_idx], x_clean[val_idx]
            y_tr, y_val = y_clean[train_idx], y_clean[val_idx]
            groups_tr: NDArray[Any] | None = groups_clean[train_idx]
        else:
            x_tr, x_val, y_tr, y_val = sk["train_test_split"](
                x_clean,
                y_clean,
                test_size=val_fraction,
                stratify=y_clean,
                random_state=int(random_state),
            )
            groups_tr = None
    else:
        x_tr, y_tr, groups_tr = x_clean, y_clean, groups_clean
        x_val = np.empty((0, x_clean.shape[1]), dtype=np.float64)
        y_val = np.empty(0, dtype=np.int64)

    if groups_tr is not None:
        n_groups_tr = int(np.unique(groups_tr).size)
        if cv > n_groups_tr:
            raise ValueError(
                f"Requested cv={cv} folds exceeds unique training groups ({n_groups_tr})."
            )

    if isinstance(classifier, str):
        estimator = build_pixel_classifier(
            classifier,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        family_label = _normalize_family(classifier) or classifier
    else:
        if not hasattr(classifier, "fit") or not hasattr(classifier, "predict_proba"):
            raise TypeError(
                "Custom estimator must implement fit() and predict_proba()."
            )
        estimator = sk["clone"](classifier)
        family_label = type(classifier).__name__

    need_encoding = _requires_zero_based_labels(
        estimator, classifier if isinstance(classifier, str) else ""
    )
    if need_encoding:
        encoder = sk["LabelEncoder"]()
        y_fit = encoder.fit_transform(y_tr)
        trained_ids = tuple(int(c) for c in encoder.classes_)
    else:
        y_fit = y_tr
        trained_ids = tuple(sorted(int(c) for c in np.unique(y_tr)))

    if np.unique(y_tr).size < 2:
        raise ValueError(
            "Training split contains fewer than 2 classes after holdout splitting; "
            "decrease val_fraction or verify class distribution across groups."
        )

    splitter = (
        sk["StratifiedGroupKFold"](n_splits=int(cv))
        if groups_tr is not None
        else sk["StratifiedKFold"](n_splits=int(cv), shuffle=True, random_state=int(random_state))
    )

    # Ensure every CV training fold sees all classes in y_tr (prevents OOF column mismatches)
    expected_n_classes = len(trained_ids)
    for fold_idx, (f_tr, _) in enumerate(splitter.split(x_tr, y_fit, groups_tr)):
        if np.unique(y_fit[f_tr]).size < expected_n_classes:
            raise ValueError(
                f"CV fold {fold_idx} omits one or more classes from its training split. "
                "Reduce cv folds or ensure each class spans at least cv samples/groups."
            )

    best_params: dict[str, Any] = {}
    best_cv_score: float | None = None

    if run_tuning:
        if param_space is not None:
            resolved_space = dict(param_space)
        elif isinstance(classifier, str):
            resolved_space = default_param_space(classifier)
        else:
            inferred_family = _infer_family_from_estimator(estimator)
            if inferred_family is None:
                raise ValueError(
                    f"param_space is required when tuning custom estimator {family_label!r}."
                )
            resolved_space = default_param_space(inferred_family)
        if search_method == "grid":
            search = sk["GridSearchCV"](
                estimator=estimator,
                param_grid=resolved_space,
                cv=splitter,
                scoring=scoring,
                n_jobs=n_jobs,
            )
        elif search_method == "random":
            search = sk["RandomizedSearchCV"](
                estimator=estimator,
                param_distributions=resolved_space,
                n_iter=int(n_iter),
                cv=splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=int(random_state),
            )
        else:
            raise ValueError(f"search_method must be 'random' or 'grid', got {search_method!r}.")

        if groups_tr is not None:
            search.fit(x_tr, y_fit, groups=groups_tr)
        else:
            search.fit(x_tr, y_fit)
        fitted_estimator = search.best_estimator_
        best_params = dict(search.best_params_)
        best_cv_score = float(search.best_score_)
    else:
        fitted_estimator = estimator
        fitted_estimator.fit(x_tr, y_fit)
        best_params = (
            dict(fitted_estimator.get_params()) if hasattr(fitted_estimator, "get_params") else {}
        )

    if not need_encoding and hasattr(fitted_estimator, "classes_"):
        trained_ids = tuple(int(c) for c in fitted_estimator.classes_)

    oof_proba = sk["cross_val_predict"](
        fitted_estimator,
        x_tr,
        y_fit,
        cv=splitter,
        groups=groups_tr,
        method="predict_proba",
        n_jobs=n_jobs,
    )
    cv_eval = _compute_metrics(sk, y_tr, np.asarray(oof_proba, dtype=np.float64), trained_ids)

    model = LocalPixelModel(
        estimator=fitted_estimator,
        feature_names=active_names,
        class_schema=class_schema,
        trained_class_ids=trained_ids,
        model_family=family_label,
    )

    heldout_eval: ModelEvaluationMetrics | None = None
    if val_fraction > 0.0 and y_val.size > 0:
        unseen_in_val = set(int(c) for c in y_val) - set(trained_ids)
        if unseen_in_val:
            warnings.warn(
                f"Validation split contains classes {sorted(unseen_in_val)} absent from training.",
                UserWarning,
                stacklevel=2,
            )
        val_raw_proba = fitted_estimator.predict_proba(x_val)
        heldout_eval = _compute_metrics(
            sk, y_val, np.asarray(val_raw_proba, dtype=np.float64), trained_ids
        )

    resolved_best_score = (
        best_cv_score if best_cv_score is not None else cv_eval.metrics["balanced_accuracy"]
    )
    return LocalPixelTrainingResult(
        model=model,
        cv_metrics=cv_eval.metrics,
        best_params=best_params,
        best_cv_score=resolved_best_score,
        evaluation=heldout_eval,
        dropped_rows=dropped_rows,
    )


def train_pixel_classifier(
    x_data: pd.DataFrame | np.ndarray,
    y_data: pd.Series | np.ndarray,
    *,
    class_schema: ClassSchema | None = None,
    config: ProjectConfig | None = None,
    classifier: str | Any = "random_forest",
    selected_features: Any | None = None,
    feature_names: Sequence[str] | None = None,
    groups: pd.Series | np.ndarray | None = None,
    cv: int = 5,
    n_estimators: int = 200,
    n_jobs: int | None = -1,
    random_state: int = 42,
    val_fraction: float = 0.0,
) -> LocalPixelTrainingResult:
    """Train a local probabilistic pixel classifier on an RFECV feature subset."""
    schema = resolve_class_schema(class_schema, config, caller="train_pixel_classifier")
    return _train_or_tune_core(
        x_data,
        y_data,
        class_schema=schema,
        classifier=classifier,
        selected_features=selected_features,
        feature_names=feature_names,
        groups=groups,
        cv=cv,
        scoring="balanced_accuracy",
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        val_fraction=val_fraction,
        param_space=None,
        search_method="random",
        n_iter=1,
        run_tuning=False,
    )


def tune_pixel_classifier(
    x_data: pd.DataFrame | np.ndarray,
    y_data: pd.Series | np.ndarray,
    *,
    class_schema: ClassSchema | None = None,
    config: ProjectConfig | None = None,
    classifier: str | Any = "random_forest",
    selected_features: Any | None = None,
    feature_names: Sequence[str] | None = None,
    groups: pd.Series | np.ndarray | None = None,
    param_space: Mapping[str, Any] | None = None,
    search_method: SearchMethod = "random",
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "balanced_accuracy",
    n_jobs: int | None = -1,
    random_state: int = 42,
    val_fraction: float = 0.0,
) -> LocalPixelTrainingResult:
    """Tune hyperparameters with CV (stratified or group-stratified) and return a fitted model."""
    schema = resolve_class_schema(class_schema, config, caller="tune_pixel_classifier")
    return _train_or_tune_core(
        x_data,
        y_data,
        class_schema=schema,
        classifier=classifier,
        selected_features=selected_features,
        feature_names=feature_names,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_estimators=200,
        n_jobs=n_jobs,
        random_state=random_state,
        val_fraction=val_fraction,
        param_space=param_space,
        search_method=search_method,
        n_iter=n_iter,
        run_tuning=True,
    )


def train_pixel_classifier_from_csv(
    csv_path: str | Path | None = None,
    *,
    class_schema: ClassSchema | None = None,
    config: ProjectConfig | None = None,
    classifier: str | Any = "random_forest",
    selected_features: Any | None = None,
    class_col: str = "class_id",
    group_col: str | None = None,
    ignore_cols: Sequence[str] = ("system:index", ".geo", "sample_id", "class_name"),
    tune: bool = False,
    param_space: Mapping[str, Any] | None = None,
    search_method: SearchMethod = "random",
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "balanced_accuracy",
    n_estimators: int = 200,
    n_jobs: int | None = -1,
    random_state: int = 42,
    val_fraction: float = 0.2,
) -> LocalPixelTrainingResult:
    """Load extracted sample-feature CSV (or ``config.sample_features_csv``) and train/tune."""
    schema = resolve_class_schema(class_schema, config, caller="train_pixel_classifier_from_csv")
    resolved_csv = (
        Path(csv_path)
        if csv_path is not None
        else (
            Path(config.sample_features_csv)
            if config is not None and config.sample_features_csv is not None
            else None
        )
    )
    if resolved_csv is None:
        raise ValueError(
            "Provide csv_path or set ProjectConfig.sample_features_csv."
        )

    df = pd.read_csv(resolved_csv)
    if class_col not in df.columns:
        raise ValueError(f"Target column {class_col!r} not found in {resolved_csv}.")

    groups: pd.Series | None = None
    excluded = set(ignore_cols) | {class_col}
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(f"Group column {group_col!r} not found in {resolved_csv}.")
        groups = df[group_col]
        excluded.add(group_col)

    resolved_subset = resolve_selected_features(selected_features)
    if resolved_subset is not None:
        feature_cols = list(resolved_subset)
    else:
        feature_cols = [
            c
            for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Requested features missing from {resolved_csv}: {missing}")

    return _train_or_tune_core(
        df[feature_cols],
        df[class_col],
        class_schema=schema,
        classifier=classifier,
        selected_features=feature_cols,
        feature_names=feature_cols,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        val_fraction=val_fraction,
        param_space=param_space,
        search_method=search_method,
        n_iter=n_iter,
        run_tuning=tune,
    )


def _resolve_raster_band_indexes(
    src_count: int,
    src_descriptions: Sequence[str | None],
    explicit_band_names: Sequence[str] | None,
    required_features: Sequence[str],
) -> list[int]:
    """Resolve 1-based GeoTIFF band indexes matching ``required_features`` in exact order."""
    if explicit_band_names is not None:
        if len(explicit_band_names) != src_count:
            raise ValueError(
                f"band_names length ({len(explicit_band_names)}) must match raster band count "
                f"({src_count})."
            )
        available = [str(n).strip() for n in explicit_band_names]
    elif any(d and d.strip() for d in src_descriptions):
        available = [
            d.strip() if d and d.strip() else f"band_{i + 1}"
            for i, d in enumerate(src_descriptions)
        ]
    else:
        if src_count != len(required_features):
            raise ValueError(
                f"Raster has no band descriptions and its band count ({src_count}) does not match "
                f"model feature count ({len(required_features)}). Pass band_names explicitly."
            )
        return list(range(1, src_count + 1))

    name_to_idx: dict[str, int] = {}
    duplicates: list[str] = []
    for idx, name in enumerate(available, start=1):
        if name in name_to_idx:
            duplicates.append(name)
        else:
            name_to_idx[name] = idx
    if duplicates:
        raise ValueError(f"Duplicate raster band names detected: {sorted(set(duplicates))}.")

    missing = [f for f in required_features if f not in name_to_idx]
    if missing:
        raise ValueError(
            f"Feature raster is missing model features {missing}. "
            f"Available raster bands: {available}"
        )
    return [name_to_idx[f] for f in required_features]


def predict_raster_probabilities(
    feature_raster_path: str | Path,
    model: LocalPixelModel,
    output_path: str | Path,
    *,
    band_names: Sequence[str] | None = None,
    encoding: ProbabilityEncoding = "uint16",
    epoch: str | None = None,
    block_size: int = 512,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Stream tiled COG probability predictions aligned to ``model.class_schema``.

    Reads only the bands matching ``model.feature_names`` in spatial blocks of
    ``block_size x block_size`` pixels so full-scene arrays never have to fit in RAM,
    and writes a canonical ``Class Probability Stack`` COG paired with its
    ``*.manifest.json`` sidecar.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size!r}.")
    if encoding not in ("uint16", "float32"):
        raise ValueError(f"encoding must be 'uint16' or 'float32', got {encoding!r}.")

    in_path = Path(feature_raster_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = model.class_schema
    n_classes = len(schema)

    with rasterio.open(in_path) as src:
        band_indexes = _resolve_raster_band_indexes(
            src.count,
            src.descriptions,
            band_names,
            model.feature_names,
        )
        out_profile = dict(src.profile)
        height, width = src.height, src.width

        out_profile["driver"] = "COG"
        for incompatible_key in ("tiled", "blockxsize", "blockysize", "interleave"):
            out_profile.pop(incompatible_key, None)

        if encoding == "uint16":
            scale_factor = UINT16_PROB_SCALE
            manifest_nodata: float | int | None = UINT16_PROB_NODATA
            fill_value: float | int = UINT16_PROB_NODATA
            out_dtype: np.dtype[Any] = np.dtype(np.uint16)
            out_profile.update(
                count=n_classes,
                dtype="uint16",
                nodata=UINT16_PROB_NODATA,
                compress="deflate",
            )
        else:
            scale_factor = 1.0
            manifest_nodata = None
            fill_value = np.nan
            out_dtype = np.dtype(np.float32)
            out_profile.update(
                count=n_classes,
                dtype="float32",
                nodata=np.nan,
                compress="deflate",
            )

        with rasterio.open(out_path, "w", **out_profile) as dst:
            for band_idx, desc in enumerate(schema.geotiff_band_descriptions, start=1):
                dst.set_band_description(band_idx, desc)

            for row_start in range(0, height, block_size):
                bh = min(block_size, height - row_start)
                for col_start in range(0, width, block_size):
                    bw = min(block_size, width - col_start)
                    window = Window(col_start, row_start, bw, bh)  # type: ignore[call-arg]
                    masked_block = src.read(band_indexes, window=window, masked=True)
                    block_f64 = np.ma.getdata(masked_block).astype(np.float64)
                    band_mask = np.ma.getmaskarray(masked_block)
                    valid = (~np.any(band_mask, axis=0)) & np.all(np.isfinite(block_f64), axis=0)

                    n_px = bh * bw
                    out_block = np.full((n_classes, n_px), fill_value, dtype=out_dtype)
                    valid_flat = valid.ravel()
                    if np.any(valid_flat):
                        feats = block_f64.reshape(len(band_indexes), n_px).T[valid_flat]
                        probs = model.predict_proba(feats)  # (N_valid, n_classes), sum == 1.0
                        if encoding == "uint16":
                            scaled = np.rint(np.clip(probs, 0.0, 1.0) * UINT16_PROB_SCALE)
                            out_block[:, valid_flat] = scaled.astype(np.uint16).T
                        else:
                            out_block[:, valid_flat] = probs.T

                    dst.write(out_block.reshape(n_classes, bh, bw), window=window)

    merged_extra = {
        "model_family": model.model_family,
        "features": list(model.feature_names),
    }
    if extra:
        merged_extra.update(extra)

    crs_val = out_profile.get("crs")
    manifest = ProbabilityManifest(
        class_schema=schema,
        dtype=encoding,
        scale_factor=scale_factor,
        nodata=manifest_nodata,
        crs=str(crs_val) if crs_val is not None else "",
        epoch=epoch,
        paradigm="pixel",
        source=in_path.name,
        extra=merged_extra,
    )
    manifest.to_json(manifest_path_for(out_path))
    return out_path

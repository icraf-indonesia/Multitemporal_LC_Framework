"""Offline unit tests for local CSV-based RFECV feature selection (Ticket #09)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lc_framework.config import ClassDefinition, ClassSchema, ProjectConfig
from lc_framework.features import (
    SelectedFeaturesArtifact,
    load_selected_features,
    run_rfecv_from_csv,
    run_rfecv_selection,
)


def _synthetic_feature_csv(tmp_path: Path, *, with_groups: bool = False) -> Path:
    """Create a synthetic CSV with 3 informative features and 4 pure-noise features."""
    rng = np.random.default_rng(123)
    n_per_class = 30
    classes = [1, 2, 3]
    rows: list[dict] = []
    # Each class is unique in one dimension so NO single feature separates all 3 classes
    centers = {
        1: (2.2, 0.0, 0.0),
        2: (0.0, 2.2, 0.0),
        3: (0.0, 0.0, 2.2),
    }
    names = {1: "Forest", 2: "Cropland", 3: "Water"}

    idx = 0
    for cid in classes:
        c0, c1, c2 = centers[cid]
        for k in range(n_per_class):
            row: dict = {
                "system:index": f"pt_{idx}",
                ".geo": '{"type":"Point"}',
                "sample_id": f"s_{idx}",
                "class_id": cid,
                "class_name": names[cid],
                "ndvi_dry": float(c0 + rng.normal(0, 0.55)),
                "mndwi_dry": float(c1 + rng.normal(0, 0.55)),
                "swir1_wet": float(c2 + rng.normal(0, 0.55)),
                "noise_1": float(rng.normal(0, 1.0)),
                "noise_2": float(rng.normal(0, 1.0)),
                "noise_3": float(rng.normal(0, 1.0)),
                "noise_4": float(rng.normal(0, 1.0)),
            }
            if with_groups:
                row["spatial_block"] = f"tile_{k % 4}"
            rows.append(row)
            idx += 1

    df = pd.DataFrame(rows)
    # Inject one row with NaN to verify missing-value row handling
    df.loc[0, "ndvi_dry"] = np.nan
    csv_path = tmp_path / "sample_features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_run_rfecv_from_csv_selects_informative_features_and_saves_artifacts(
    tmp_path: Path,
) -> None:
    csv_path = _synthetic_feature_csv(tmp_path)
    out_dir = tmp_path / "rfecv_out"

    res = run_rfecv_from_csv(
        csv_path,
        class_col="class_id",
        cv=3,
        n_estimators=40,
        random_state=42,
        n_jobs=1,
        output_dir=out_dir,
        artifact_format="json",
    )

    assert res.dropped_rows == 1
    assert res.artifact.n_samples == 89
    assert len(res.artifact.all_candidate_features) == 7
    # Informative features must be selected
    for informative in ("ndvi_dry", "mndwi_dry", "swir1_wet"):
        assert informative in res.selected_features

    # CV score curve covers 1..7 features with valid mean/std columns
    assert list(res.cv_scores.columns) == ["n_features", "mean_test_score", "std_test_score"]
    assert res.cv_scores["n_features"].min() == 1
    assert res.cv_scores["n_features"].max() == 7
    assert res.artifact.best_score > 0.85

    # Feature rankings table contains every candidate feature + full baseline importances
    assert len(res.feature_rankings) == 7
    assert set(res.feature_rankings["feature"]) == set(res.artifact.all_candidate_features)
    assert res.feature_rankings.iloc[0]["ranking"] == 1
    assert res.feature_rankings["baseline_importance"].notna().all()

    # Saved JSON artifact round-trips cleanly via load_selected_features
    json_path = out_dir / "rfecv_selected_features.json"
    assert json_path.exists()
    assert (out_dir / "rfecv_cv_scores.csv").exists()
    assert (out_dir / "rfecv_feature_rankings.csv").exists()

    reloaded = load_selected_features(json_path)
    assert reloaded == res.artifact
    assert list(reloaded.selected_features) == res.selected_features

    # YAML artifact round-trip also works
    yaml_path = out_dir / "selected.yaml"
    res.artifact.to_yaml(yaml_path)
    assert load_selected_features(yaml_path) == res.artifact


def test_run_rfecv_from_csv_supports_group_cv_and_project_config_validation(
    tmp_path: Path,
) -> None:
    csv_path = _synthetic_feature_csv(tmp_path, with_groups=True)
    cfg = ProjectConfig(
        project_name="demo_rfecv",
        crs="EPSG:32748",
        scale_m=10,
        sample_features_csv=str(csv_path),
        class_schema=ClassSchema(
            classes=[
                ClassDefinition(id=1, name="Forest", color="#1b5e20"),
                ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
                ClassDefinition(id=3, name="Water", color="#0288d1"),
            ]
        ),
    )

    # Reads CSV path directly from ProjectConfig.sample_features_csv when csv_path is omitted
    res = run_rfecv_from_csv(
        class_col="class_id",
        group_col="spatial_block",
        config=cfg,
        cv=3,
        n_estimators=30,
        random_state=7,
        n_jobs=1,
    )
    assert "spatial_block" not in res.artifact.all_candidate_features
    assert res.artifact.cv_folds == 3
    assert len(res.selected_features) >= 1

    # Reproducibility: identical random_state produces identical selected features
    res_repeat = run_rfecv_from_csv(
        csv_path,
        class_col="class_id",
        group_col="spatial_block",
        config=cfg,
        cv=3,
        n_estimators=30,
        random_state=7,
        n_jobs=1,
    )
    assert res_repeat.selected_features == res.selected_features


def test_run_rfecv_validation_rejects_bad_inputs_and_missing_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = pd.Series([1, 2])
    with pytest.raises(ValueError, match="cv"):
        run_rfecv_selection(x, y, cv=1)
    with pytest.raises(ValueError, match="min_features_to_select"):
        run_rfecv_selection(x, y, cv=2, min_features_to_select=5)

    with pytest.raises(ValueError, match="Duplicate"):
        SelectedFeaturesArtifact(
            selected_features=("ndvi", "ndvi"),
            all_candidate_features=("ndvi",),
            scoring="balanced_accuracy",
            best_score=0.9,
            n_samples=10,
        )

    # Helpful error when scikit-learn is unavailable
    import builtins
    import sys

    real_import = builtins.__import__

    def _block_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_sklearn)
    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ImportError, match=r"lc-framework\[ml\]"):
        run_rfecv_selection(x, y, cv=2)

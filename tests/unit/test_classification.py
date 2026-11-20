"""Offline tests for GEE MULTIPROBABILITY RF (#10) and Local ML Pixel Classification (#11)."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from lc_framework.classification import (
    GEERFModel,
    GEERFParams,
    LocalPixelModel,
    build_gee_probability_manifest,
    build_pixel_classifier,
    default_param_space,
    evaluate_gee_probabilities,
    export_probability_tiles,
    get_gee_feature_importance,
    predict_gee_probabilities,
    predict_raster_probabilities,
    scale_probability_image,
    train_gee_rf,
    train_pixel_classifier,
    train_pixel_classifier_from_csv,
    tune_pixel_classifier,
)
from lc_framework.config import ClassDefinition, ClassSchema, ProjectConfig
from lc_framework.evaluation import evaluate_probability_stack
from lc_framework.features import SelectedFeaturesArtifact
from lc_framework.io import (
    UINT16_PROB_NODATA,
    UINT16_PROB_SCALE,
    ProbabilityManifest,
    manifest_path_for,
    read_probability_stack,
)


def _schema(ids: tuple[int, ...] = (1, 2, 3)) -> ClassSchema:
    names = ("Forest", "Cropland", "Water")
    colors = ("#1b5e20", "#fbc02d", "#0288d1")
    return ClassSchema(
        classes=tuple(
            ClassDefinition(id=cid, name=name, color=color)
            for cid, name, color in zip(ids, names, colors, strict=True)
        )
    )


class _FakeEEImage:
    """Minimal mock recording chained GEE Image operations."""

    def __init__(self, name: str = "img", ops: list[tuple[str, Any]] | None = None) -> None:
        self.name = name
        self.ops: list[tuple[str, Any]] = list(ops) if ops else []

    def _chain(self, op: str, arg: Any = None) -> _FakeEEImage:
        return _FakeEEImage(self.name, [*self.ops, (op, arg)])

    def bandNames(self) -> list[str]:  # noqa: N802
        return ["ndvi_dry", "mndwi_dry", "swir1_wet"]

    def select(self, bands: Any) -> _FakeEEImage:
        return self._chain("select", list(bands) if not isinstance(bands, str) else bands)

    def classify(self, classifier: Any) -> _FakeEEImage:
        return self._chain("classify", classifier)

    def rename(self, names: Any) -> _FakeEEImage:
        return self._chain("rename", names)

    def arrayFlatten(self, axes: Any) -> _FakeEEImage:  # noqa: N802
        return self._chain("arrayFlatten", axes)

    def addBands(self, bands: Any) -> _FakeEEImage:  # noqa: N802
        return self._chain("addBands", bands)

    def multiply(self, val: Any) -> _FakeEEImage:
        return self._chain("multiply", val)

    def round(self) -> _FakeEEImage:
        return self._chain("round")

    def clamp(self, lo: Any, hi: Any) -> _FakeEEImage:
        return self._chain("clamp", (lo, hi))

    def toUint16(self) -> _FakeEEImage:  # noqa: N802
        return self._chain("toUint16")

    def toFloat(self) -> _FakeEEImage:  # noqa: N802
        return self._chain("toFloat")

    def float(self) -> _FakeEEImage:
        return self._chain("float")

    def unmask(self, val: Any) -> _FakeEEImage:
        return self._chain("unmask", val)

    def clip(self, region: Any) -> _FakeEEImage:
        return self._chain("clip", region)


def _make_fake_ee() -> MagicMock:
    fake_ee = MagicMock()

    def _make_classifier(**kwargs: Any) -> MagicMock:
        clf = MagicMock()
        clf.init_kwargs = kwargs
        clf.output_mode = None

        def _set_mode(mode: str) -> MagicMock:
            clf.output_mode = mode
            return clf

        def _train(**train_kwargs: Any) -> MagicMock:
            trained = MagicMock()
            trained.train_kwargs = train_kwargs
            trained.explain.return_value = MagicMock(
                get=lambda k: {"importance": {"ndvi_dry": 12.5, "mndwi_dry": 8.0}}.get(k)
            )
            return trained

        clf.setOutputMode.side_effect = _set_mode
        clf.train.side_effect = _train
        return clf

    fake_ee.Classifier.smileRandomForest.side_effect = _make_classifier
    fake_ee.Filter.notNull.side_effect = lambda props: ("notNull", tuple(props))
    fake_ee.Filter.inList.side_effect = lambda prop, vals: ("inList", prop, tuple(vals))
    fake_ee.Filter.eq.side_effect = lambda prop, val: ("eq", prop, val)
    fake_ee.Geometry.Rectangle.side_effect = lambda bounds: ("rect", tuple(bounds))
    fake_ee.Image.constant.side_effect = lambda v: _FakeEEImage("const", [("constant", v)])
    fake_ee.batch.Export.image.toAsset.side_effect = lambda **kwargs: MagicMock(
        spec=["start", "kwargs"], kwargs=kwargs
    )
    fake_ee.batch.Export.image.toDrive.side_effect = lambda **kwargs: MagicMock(
        spec=["start", "kwargs"], kwargs=kwargs
    )
    return fake_ee


# ---------------------------------------------------------------------------
# Ticket #10: GEE Pixel-Based Classification (MULTIPROBABILITY RF)
# ---------------------------------------------------------------------------


def test_gee_rf_trains_on_rfecv_subset_and_emits_schema_ordered_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.classification.gee_rf.require_ee", lambda *_, **__: fake_ee)

    schema = _schema((10, 20, 50))
    artifact = SelectedFeaturesArtifact(
        selected_features=("mndwi_dry", "ndvi_dry"),
        all_candidate_features=("mndwi_dry", "ndvi_dry", "noise_1"),
        scoring="balanced_accuracy",
        best_score=0.95,
        n_samples=120,
    )
    artifact_path = artifact.to_json(tmp_path / "rfecv.json")

    training_fc = MagicMock()
    training_fc.filter.return_value = training_fc
    feat_img = _FakeEEImage("features")

    model = train_gee_rf(
        training_fc,
        class_schema=schema,
        selected_features=artifact_path,
        params=GEERFParams(number_of_trees=150, variables_per_split=2, min_leaf_population=2),
    )
    assert isinstance(model, GEERFModel)
    assert model.feature_names == ("mndwi_dry", "ndvi_dry")
    assert fake_ee.Classifier.smileRandomForest.call_args.kwargs == {
        "numberOfTrees": 150,
        "minLeafPopulation": 2,
        "bagFraction": 0.5,
        "seed": 42,
        "variablesPerSplit": 2,
    }
    assert model.classifier.train_kwargs["inputProperties"] == ["mndwi_dry", "ndvi_dry"]
    assert model.classifier.train_kwargs["classProperty"] == "class_id"

    # Predict probabilities -> arrayFlatten + select in exact ClassSchema prob_<id> order
    prob_img = predict_gee_probabilities(model, feat_img, scale_output=True, encoding="uint16")
    assert ("select", ["mndwi_dry", "ndvi_dry"]) in prob_img.ops
    assert ("arrayFlatten", [["prob_10", "prob_20", "prob_50"]]) in prob_img.ops
    assert ("select", ["prob_10", "prob_20", "prob_50"]) in prob_img.ops
    assert ("multiply", UINT16_PROB_SCALE) in prob_img.ops
    assert ("round", None) in prob_img.ops
    assert ("clamp", (0.0, UINT16_PROB_SCALE)) in prob_img.ops
    assert ("toUint16", None) in prob_img.ops


def test_gee_rf_absent_training_class_zero_fills_missing_prob_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.classification.gee_rf.require_ee", lambda *_, **__: fake_ee)
    schema = _schema((1, 2, 3))

    training_fc = MagicMock()
    training_fc.filter.return_value = training_fc
    model = train_gee_rf(
        training_fc,
        class_schema=schema,
        selected_features=("ndvi_dry", "mndwi_dry"),
        trained_class_ids=(1, 3),  # Class 2 missing from training set
    )
    assert model.trained_class_ids == (1, 3)

    prob_img = predict_gee_probabilities(model, _FakeEEImage("feat"))
    assert ("arrayFlatten", [["prob_1", "prob_3"]]) in prob_img.ops
    add_ops = [arg for op, arg in prob_img.ops if op == "addBands"]
    assert len(add_ops) == 1
    assert ("rename", "prob_2") in add_ops[0].ops
    assert ("select", ["prob_1", "prob_2", "prob_3"]) in prob_img.ops


def test_gee_rf_tiled_export_scaling_and_sidecar_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.classification.gee_rf.require_ee", lambda *_, **__: fake_ee)
    schema = _schema((1, 2, 3))
    prob_img = _FakeEEImage("probs")

    # Dry run emits plans and validated manifests without writing files or calling start()
    manifest_dir = tmp_path / "manifests"
    dry_res = export_probability_tiles(
        prob_img,
        bounds=(100.0, 0.0, 101.0, 1.0),
        export_prefix="projects/demo/assets/prob_2025",
        class_schema=schema,
        epoch="2025",
        tile_size_deg=0.5,
        scale_m=10.0,
        crs="EPSG:32648",
        existing_assets=["projects/demo/assets/prob_2025_r0_c0"],
        skip_existing=True,
        dry_run=True,
        manifest_dir=manifest_dir,
    )
    assert len(dry_res.plans) == 4
    assert dry_res.plans[0].skip is True
    assert dry_res.tasks == []
    assert dry_res.manifest_paths == {}
    assert len(dry_res.manifests) == 4
    first_manifest = next(iter(dry_res.manifests.values()))
    assert first_manifest.dtype == "uint16"
    assert first_manifest.scale_factor == UINT16_PROB_SCALE
    assert first_manifest.nodata == UINT16_PROB_NODATA
    assert first_manifest.paradigm == "pixel"
    assert first_manifest.epoch == "2025"

    # Real Drive export submits 3 tasks (skipping existing tile 0) with COG + noData=65535
    drive_res = export_probability_tiles(
        prob_img,
        bounds=(100.0, 0.0, 101.0, 1.0),
        export_prefix="projects/demo/assets/prob_2025",
        class_schema=schema,
        epoch="2025",
        tile_size_deg=0.5,
        scale_m=10.0,
        crs="EPSG:32648",
        existing_assets=["projects/demo/assets/prob_2025_r0_c0"],
        skip_existing=True,
        destination="drive",
        drive_folder="LC_Drive",
        dry_run=False,
        manifest_dir=manifest_dir,
    )
    assert len(drive_res.tasks) == 3
    assert len(drive_res.manifest_paths) == 3
    for task in drive_res.tasks:
        task.start.assert_called_once()
        kw = task.kwargs
        assert kw["folder"] == "LC_Drive"
        assert kw["formatOptions"] == {"cloudOptimized": True, "noData": UINT16_PROB_NODATA}
        assert ("unmask", UINT16_PROB_NODATA) in kw["image"].ops

    for mpath in drive_res.manifest_paths.values():
        assert mpath.exists()
        reloaded = ProbabilityManifest.from_json(mpath)
        assert reloaded.class_schema == schema
        assert reloaded.nodata == UINT16_PROB_NODATA

    # Asset export uses pyramidingPolicy={" .default": "mean"}
    asset_res = export_probability_tiles(
        prob_img,
        bounds=(100.0, 0.0, 100.5, 0.5),
        export_prefix="projects/demo/assets/prob_2025",
        class_schema=schema,
        encoding="float32",
        tile_size_deg=0.5,
        destination="asset",
        dry_run=False,
    )
    assert len(asset_res.tasks) == 1
    assert asset_res.tasks[0].kwargs["pyramidingPolicy"] == {".default": "mean"}
    assert ("toFloat", None) in scale_probability_image(prob_img, encoding="float32").ops


def test_gee_rf_evaluation_and_importance_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.classification.gee_rf.require_ee", lambda *_, **__: fake_ee)

    model = train_gee_rf(
        MagicMock(),
        class_schema=_schema(),
        selected_features=("ndvi_dry", "mndwi_dry"),
    )
    imp = get_gee_feature_importance(model)
    assert imp is not None

    test_fc = MagicMock()
    scored_fc = MagicMock()
    test_fc.classify.return_value = scored_fc
    scored_fc.map.return_value = scored_fc
    scored_fc.size.return_value = 30
    scored_fc.aggregate_mean.side_effect = [0.21, 0.90]
    scored_fc.filter.return_value = scored_fc

    metrics = evaluate_gee_probabilities(model, test_fc)
    assert set(metrics) == {
        "overall_log_loss",
        "overall_accuracy",
        "per_class_log_loss",
        "per_class_accuracy",
        "n_samples",
        "per_class_n_samples",
    }

    manifest = build_gee_probability_manifest(
        class_schema=_schema(), encoding="uint16", crs="EPSG:4326"
    )
    assert manifest.paradigm == "pixel"


# ---------------------------------------------------------------------------
# Ticket #11: Local Python Pixel-Based Classification Backend ([ml])
# ---------------------------------------------------------------------------


def _synthetic_pixel_dataset(
    tmp_path: Path,
    *,
    class_ids: tuple[int, int, int] = (1, 2, 3),
    with_groups: bool = False,
) -> tuple[Path, Path, Path, ClassSchema]:
    """Generate a synthetic CSV, RFECV artifact, and 4-band GeoTIFF with extra/shuffled bands."""
    schema = _schema(class_ids)
    rng = np.random.default_rng(17)

    centers = {
        class_ids[0]: (2.5, 0.0, 0.0),
        class_ids[1]: (0.0, 2.5, 0.0),
        class_ids[2]: (0.0, 0.0, 2.5),
    }
    rows: list[dict[str, Any]] = []
    idx = 0
    for cid in class_ids:
        c0, c1, c2 = centers[cid]
        for k in range(25):
            row: dict[str, Any] = {
                "sample_id": f"s_{idx}",
                "class_id": cid,
                "class_name": schema.name_for(cid),
                "ndvi_dry": float(c0 + rng.normal(0, 0.3)),
                "mndwi_dry": float(c1 + rng.normal(0, 0.3)),
                "swir1_wet": float(c2 + rng.normal(0, 0.3)),
                "noise_1": float(rng.normal(0, 1.0)),
            }
            if with_groups:
                row["tile_group"] = f"grp_{k % 3}"
            rows.append(row)
            idx += 1

    df = pd.DataFrame(rows)
    df.loc[0, "ndvi_dry"] = np.nan  # Ensure non-finite training sample row is dropped cleanly
    csv_path = tmp_path / "samples.csv"
    df.to_csv(csv_path, index=False)

    artifact = SelectedFeaturesArtifact(
        selected_features=("ndvi_dry", "mndwi_dry", "swir1_wet"),
        all_candidate_features=("ndvi_dry", "mndwi_dry", "swir1_wet", "noise_1"),
        scoring="balanced_accuracy",
        best_score=0.98,
        n_samples=len(df) - 1,
    )
    art_path = artifact.to_json(tmp_path / "selected_features.json")

    # Write a 16x16 GeoTIFF whose bands are SHUFFLED + include noise_1, plus a nodata corner
    h, w = 16, 16
    raster = np.zeros((4, h, w), dtype=np.float32)
    # Band order on disk: [noise_1, swir1_wet, ndvi_dry, mndwi_dry]
    raster[0] = rng.normal(0, 1.0, (h, w))
    # Top half: class_ids[0]
    raster[2, :8, :] = centers[class_ids[0]][0] + rng.normal(0, 0.15, (8, w))
    raster[3, :8, :] = centers[class_ids[0]][1] + rng.normal(0, 0.15, (8, w))
    raster[1, :8, :] = centers[class_ids[0]][2] + rng.normal(0, 0.15, (8, w))
    # Bottom-left: class_ids[1]
    raster[2, 8:, :8] = centers[class_ids[1]][0] + rng.normal(0, 0.15, (8, 8))
    raster[3, 8:, :8] = centers[class_ids[1]][1] + rng.normal(0, 0.15, (8, 8))
    raster[1, 8:, :8] = centers[class_ids[1]][2] + rng.normal(0, 0.15, (8, 8))
    # Bottom-right: class_ids[2]
    raster[2, 8:, 8:] = centers[class_ids[2]][0] + rng.normal(0, 0.15, (8, 8))
    raster[3, 8:, 8:] = centers[class_ids[2]][1] + rng.normal(0, 0.15, (8, 8))
    raster[1, 8:, 8:] = centers[class_ids[2]][2] + rng.normal(0, 0.15, (8, 8))
    # Nodata + NaN corner
    raster[:, 0:2, 0:2] = -9999.0
    raster[:, 0, 15] = np.nan

    tif_path = tmp_path / "feature_stack.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 4,
        "height": h,
        "width": w,
        "crs": "EPSG:32648",
        "transform": from_origin(500000.0, 1000000.0, 10.0, 10.0),
        "nodata": -9999.0,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(raster)
        for i, bname in enumerate(("noise_1", "swir1_wet", "ndvi_dry", "mndwi_dry"), start=1):
            dst.set_band_description(i, bname)

    return csv_path, art_path, tif_path, schema


@pytest.mark.parametrize("family", ["random_forest", "extra_trees", "lightgbm"])
def test_local_ml_backends_tune_and_predict_cog_probability_stack(
    tmp_path: Path,
    family: str,
) -> None:
    csv_path, art_path, tif_path, schema = _synthetic_pixel_dataset(
        tmp_path, class_ids=(10, 20, 50)
    )

    df = pd.read_csv(csv_path)
    X = df[["ndvi_dry", "mndwi_dry", "swir1_wet", "noise_1"]]
    y = df["class_id"]

    # Grid / random search tuning using ONLY the RFECV selected subset
    small_space = {
        "n_estimators": [15, 25],
        "max_depth": [3, 5],
    }
    tuned = tune_pixel_classifier(
        X,
        y,
        class_schema=schema,
        classifier=family,
        selected_features=art_path,
        param_space=small_space,
        search_method="grid" if family == "random_forest" else "random",
        n_iter=2,
        cv=3,
        n_jobs=1,
        random_state=42,
    )
    model = tuned.model
    assert isinstance(model, LocalPixelModel)
    assert model.feature_names == ("ndvi_dry", "mndwi_dry", "swir1_wet")
    assert tuned.best_cv_score > 0.90
    assert tuned.cv_metrics["balanced_accuracy"] > 0.90
    assert tuned.cv_metrics["log_loss"] < 0.50

    # Tiled COG probability inference (block_size=6 forces multiple blocks/tiles)
    out_cog = tmp_path / f"pixel_probs_{family}.tif"
    predict_raster_probabilities(
        tif_path,
        model,
        out_cog,
        encoding="uint16",
        epoch="2025",
        block_size=6,
    )
    assert out_cog.exists()
    assert manifest_path_for(out_cog).exists()
    with rasterio.open(out_cog) as cog_src:
        assert cog_src.driver == "GTiff" or "COG" in cog_src.driver
        assert cog_src.descriptions == ("10:Forest", "20:Cropland", "50:Water")
        assert cog_src.dtypes == ("uint16", "uint16", "uint16")
        assert cog_src.nodata == UINT16_PROB_NODATA

    stack = read_probability_stack(out_cog)
    assert stack.manifest.paradigm == "pixel"
    assert stack.manifest.epoch == "2025"
    assert stack.manifest.class_schema == schema
    assert stack.manifest.dtype == "uint16"
    assert not np.any(stack.valid_mask[0:2, 0:2])
    assert not bool(stack.valid_mask[0, 15])
    np.testing.assert_allclose(
        stack.probabilities[:, stack.valid_mask].sum(axis=0), 1.0, atol=1e-3
    )

    # Check spatial classification accuracy on the three synthetic quadrants
    hard_idx = np.argmax(stack.probabilities, axis=0)
    assert float(np.mean(hard_idx[2:8, 2:] == 0)) >= 0.95
    assert float(np.mean(hard_idx[8:, :8] == 1)) >= 0.95
    assert float(np.mean(hard_idx[8:, 8:] == 2)) >= 0.95

    # Interoperability check with lc_framework.evaluation
    eval_res = evaluate_probability_stack(
        out_cog,
        sample_coords_xy=[(500055.0, 999945.0), (500045.0, 999885.0), (500125.0, 999885.0)],
        y_true=[10, 20, 50],
    )
    assert eval_res.overall_metrics["log_loss"] < 0.35


def test_local_ml_from_csv_group_cv_float32_and_absent_class(tmp_path: Path) -> None:
    csv_path, art_path, tif_path, schema = _synthetic_pixel_dataset(
        tmp_path, class_ids=(1, 2, 3), with_groups=True
    )
    cfg = ProjectConfig(
        project_name="local_ml_demo",
        crs="EPSG:32648",
        scale_m=10.0,
        sample_features_csv=str(csv_path),
        class_schema=schema,
    )

    res = train_pixel_classifier_from_csv(
        config=cfg,
        classifier="random_forest",
        selected_features=art_path,
        group_col="tile_group",
        cv=2,
        val_fraction=0.34,
        n_estimators=25,
        n_jobs=1,
        random_state=7,
    )
    assert res.evaluation is not None
    assert res.evaluation.metrics["balanced_accuracy"] > 0.90
    assert res.dropped_rows == 1

    # Float32 COG emission
    out_f32 = tmp_path / "probs_f32.tif"
    res.model.predict_raster(tif_path, out_f32, encoding="float32", block_size=7)
    stack_f32 = read_probability_stack(out_f32)
    assert stack_f32.manifest.dtype == "float32"
    assert stack_f32.manifest.nodata is None

    # Schema class absent during training -> zero-filled column cleanly renormalized
    df = pd.read_csv(csv_path)
    df_two = df[df["class_id"].isin([1, 2])]
    res_two = train_pixel_classifier(
        df_two[["ndvi_dry", "mndwi_dry", "swir1_wet"]],
        df_two["class_id"],
        class_schema=schema,
        classifier="random_forest",
        cv=2,
        n_estimators=15,
        n_jobs=1,
    )
    probs_df = res_two.model.predict_proba(df_two[["ndvi_dry", "mndwi_dry", "swir1_wet"]])
    assert probs_df.shape[1] == 3
    np.testing.assert_allclose(probs_df[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(probs_df.sum(axis=1), 1.0, atol=1e-5)


def test_local_ml_xgboost_label_encoding_and_missing_import_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verify default parameter spaces exist for every documented family
    for fam in ("random_forest", "extra_trees", "lightgbm", "xgboost"):
        assert len(default_param_space(fam)) >= 2

    from sklearn.base import BaseEstimator

    # Mock XGBoost expecting strict 0..K-1 labels, proving transparent schema ID decoding
    class _FakeXGBClassifier(BaseEstimator):
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.n_estimators = kwargs.get("n_estimators", 50)
            self.max_depth = kwargs.get("max_depth", 4)
            self.random_state = kwargs.get("random_state", 42)
            self.n_jobs = kwargs.get("n_jobs", -1)

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }

        def set_params(self, **params: Any) -> _FakeXGBClassifier:
            for k, v in params.items():
                setattr(self, k, v)
            return self

        def fit(self, X: np.ndarray, y: np.ndarray) -> _FakeXGBClassifier:
            unique = np.unique(y)
            # Enforce real XGBoost constraint: labels must be 0..K-1
            if not np.array_equal(unique, np.arange(len(unique))):
                raise ValueError(f"Expected 0..K-1, got {unique}")
            self.classes_ = unique
            self.n_features_in_ = X.shape[1]
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            probs = np.full((len(X), len(self.classes_)), 0.1, dtype=np.float64)
            probs[:, 0] = 0.8
            return probs / probs.sum(axis=1, keepdims=True)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    fake_xgb_mod = SimpleNamespace(XGBClassifier=_FakeXGBClassifier)
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb_mod)

    schema = _schema((10, 20, 50))
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 30), "f2": rng.normal(0, 1, 30)})
    y = pd.Series([10] * 10 + [20] * 10 + [50] * 10)

    res = train_pixel_classifier(
        X,
        y,
        class_schema=schema,
        classifier="xgboost",
        cv=3,
        n_jobs=1,
    )
    assert res.model.classes_ == (10, 20, 50)
    preds = res.model.predict(X.iloc[:3])
    assert set(preds.tolist()).issubset({10, 20, 50})

    # Verify actionable ImportError when optional boosting packages are missing
    real_import = builtins.__import__

    def _block_pkg(blocked_root: str) -> Any:
        def _imp(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.split(".")[0] == blocked_root:
                raise ImportError(f"No module named '{blocked_root}'")
            return real_import(name, *args, **kwargs)

        return _imp

    monkeypatch.delitem(sys.modules, "xgboost", raising=False)
    monkeypatch.setattr(builtins, "__import__", _block_pkg("xgboost"))
    with pytest.raises(ImportError, match=r"lc-framework\[ml\]"):
        build_pixel_classifier("xgboost")

    monkeypatch.setattr(builtins, "__import__", _block_pkg("lightgbm"))
    with pytest.raises(ImportError, match=r"lc-framework\[ml\]"):
        build_pixel_classifier("lightgbm")


def test_local_ml_xgboost_tuning_pipeline_custom_estimator_and_validation_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    class _FakeXGBForTuning(BaseEstimator):
        def __init__(self, n_estimators: int = 10, max_depth: int = 3) -> None:
            self.n_estimators = n_estimators
            self.max_depth = max_depth

        def fit(self, X: np.ndarray, y: np.ndarray) -> _FakeXGBForTuning:
            unique = np.unique(y)
            if not np.array_equal(unique, np.arange(len(unique))):
                raise ValueError(f"Expected 0..K-1, got {unique}")
            self.classes_ = unique
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            probs = np.full((len(X), len(self.classes_)), 0.1, dtype=np.float64)
            probs[:, 0] = 0.8
            return probs / probs.sum(axis=1, keepdims=True)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    schema = _schema((10, 20, 50))
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {"ndvi_dry": rng.normal(0, 1, 30), "mndwi_dry": rng.normal(0, 1, 30)}
    )
    y = pd.Series([10] * 10 + [20] * 10 + [50] * 10)

    # 1. Tuning + heldout validation via XGBoost inside a sklearn Pipeline
    pipe = Pipeline([("scaler", StandardScaler()), ("xgb", _FakeXGBForTuning())])
    tuned_pipe = tune_pixel_classifier(
        X,
        y,
        class_schema=schema,
        classifier=pipe,
        param_space={"xgb__n_estimators": [10, 20]},
        search_method="grid",
        cv=3,
        val_fraction=0.2,
        n_jobs=1,
    )
    assert tuned_pipe.model.classes_ == (10, 20, 50)
    assert tuned_pipe.evaluation is not None

    # 2. Custom LogisticRegression cloned cleanly without mutating caller instance
    lr = LogisticRegression(max_iter=200)
    lr_res = train_pixel_classifier(
        X, y, class_schema=schema, classifier=lr, cv=3, n_jobs=1
    )
    assert not hasattr(lr, "classes_")  # Original instance was cloned, not mutated
    assert lr_res.model.classes_ == (10, 20, 50)

    # Tuning custom estimator without param_space gives actionable error
    with pytest.raises(ValueError, match="param_space is required"):
        tune_pixel_classifier(X, y, class_schema=schema, classifier=lr, cv=2)

    # 3. Non-integer target label guard
    with pytest.raises(ValueError, match="integer ClassSchema IDs"):
        train_pixel_classifier(
            X, pd.Series([10.5] + [20] * 29), class_schema=schema, cv=2
        )

    # 4. Duplicate raster band description guard during inference
    dup_tif = tmp_path / "dup.tif"
    with rasterio.open(
        dup_tif,
        "w",
        driver="GTiff",
        dtype="float32",
        count=2,
        height=4,
        width=4,
        crs="EPSG:32648",
        transform=from_origin(0, 100, 10, 10),
    ) as dst:
        dst.write(np.ones((2, 4, 4), dtype=np.float32))
        dst.set_band_description(1, "ndvi_dry")
        dst.set_band_description(2, "ndvi_dry")

    with pytest.raises(ValueError, match="Duplicate raster band names"):
        predict_raster_probabilities(dup_tif, lr_res.model, tmp_path / "out.tif")

    # 5. GEE filter_to_schema=False + explicit trained_class_ids guard
    monkeypatch.setattr(
        "lc_framework.classification.gee_rf.require_ee",
        lambda *_, **__: _make_fake_ee(),
    )
    with pytest.raises(ValueError, match="filter_to_schema cannot be False"):
        train_gee_rf(
            MagicMock(),
            class_schema=schema,
            selected_features=("ndvi_dry",),
            trained_class_ids=(10, 20),
            filter_to_schema=False,
        )

"""Offline mocked-EE tests for GEE AlphaEarth Sample Augmentation (Ticket #08)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from lc_framework.config import ClassDefinition, ClassSchema, ProjectConfig
from lc_framework.samples import (
    ALPHAEARTH_COLLECTION,
    DiagnosticSoftmaxOverlay,
    SampleAugmentationParams,
    SampleAugmentationResult,
    build_class_similarity_stack,
    build_diagnostic_similarity_overlay,
    compute_class_cosine_similarity,
    export_augmented_samples_to_asset,
    export_augmented_samples_to_drive,
    extract_all_candidate_centroids,
    extract_class_candidate_centroids,
    get_alphaearth_mosaic,
    merge_augmented_samples,
    resolve_class_thresholds,
    run_sample_augmentation,
)


def _schema() -> ClassSchema:
    return ClassSchema(
        classes=(
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
            ClassDefinition(id=5, name="Built-up", color="#e53935"),
        )
    )


class _FakeEEImage:
    """Minimal chainable fake ee.Image for offline AlphaEarth similarity search tests."""

    def __init__(self, name: str = "img", ops: list[tuple[str, Any]] | None = None) -> None:
        self.name = name
        self.ops: list[tuple[str, Any]] = list(ops) if ops else []

    def _chain(self, op: str, arg: Any = None) -> _FakeEEImage:
        return _FakeEEImage(self.name, [*self.ops, (op, arg)])

    def bandNames(self) -> list[str]:  # noqa: N802
        return ["A00", "A01", "A02"]

    def select(self, bands: Any) -> _FakeEEImage:
        return self._chain("select", list(bands) if not isinstance(bands, str) else bands)

    def multiply(self, val: Any) -> _FakeEEImage:
        return self._chain("multiply", val)

    def subtract(self, val: Any) -> _FakeEEImage:
        return self._chain("subtract", val)

    def divide(self, val: Any) -> _FakeEEImage:
        return self._chain("divide", val)

    def exp(self) -> _FakeEEImage:
        return self._chain("exp")

    def reduce(self, reducer: Any) -> _FakeEEImage:
        return self._chain("reduce", reducer)

    def rename(self, names: Any) -> _FakeEEImage:
        return self._chain("rename", names)

    def clip(self, region: Any) -> _FakeEEImage:
        return self._chain("clip", region)

    def gt(self, val: Any) -> _FakeEEImage:
        return self._chain("gt", val)

    def selfMask(self) -> _FakeEEImage:  # noqa: N802
        return self._chain("selfMask")

    def updateMask(self, mask: Any) -> _FakeEEImage:  # noqa: N802
        return self._chain("updateMask", mask)

    def float(self) -> _FakeEEImage:
        return self._chain("float")

    def set(self, key_or_dict: Any, val: Any = None) -> _FakeEEImage:
        return self._chain("set", (key_or_dict, val) if val is not None else key_or_dict)

    def arrayFlatten(self, axes: Any) -> _FakeEEImage:  # noqa: N802
        return self._chain("arrayFlatten", axes)

    def sampleRegions(self, **kwargs: Any) -> MagicMock:  # noqa: N802
        fc = MagicMock(name="sampled_fc")
        fc.sample_kwargs = kwargs
        size_mock = MagicMock()
        size_mock.gt.return_value = "SIZE_GT_0"
        fc.size.return_value = size_mock

        def _map_to_list(fn: Any) -> list[Any]:
            dummy_feat = MagicMock(name="sampled_pt")
            dummy_feat.toArray.return_value = _FakeEEImage("arr")
            return [fn(dummy_feat)]

        fc.toList.return_value = MagicMock(map=_map_to_list)
        return fc

    def reduceToVectors(self, **kwargs: Any) -> MagicMock:  # noqa: N802
        polys = MagicMock(name="polys_fc")
        polys.reduce_kwargs = kwargs
        mapped = MagicMock(reduce_kwargs=kwargs)
        polys.map.side_effect = lambda fn: MagicMock(mapped_fn=fn, reduce_kwargs=kwargs)
        _ = mapped
        return polys

    def reduceRegion(self, **kwargs: Any) -> MagicMock:  # noqa: N802
        self._last_reduce_region_kwargs = kwargs
        return MagicMock(get=lambda k: 0.93, getInfo=lambda: {k: 0.93 for k in ("sim_1",)})


def _make_fake_ee() -> MagicMock:
    fake_ee = MagicMock()

    def _date(y: int, m: int, d: int) -> MagicMock:
        dt = MagicMock()
        dt.ymd = (y, m, d)
        dt.advance.side_effect = lambda n, unit: f"{y + n}-{m:02d}-{d:02d}"
        return dt

    fake_ee.Date.fromYMD.side_effect = _date

    mosaic_img = _FakeEEImage("mosaic")
    coll_mock = MagicMock()
    coll_mock.filterDate.return_value = coll_mock
    coll_mock.filterBounds.return_value = coll_mock
    coll_mock.mosaic.return_value = mosaic_img

    def _image_collection(arg: Any) -> Any:
        if isinstance(arg, str):
            return coll_mock
        return MagicMock(items=arg, mean=lambda: _FakeEEImage("mean"))

    fake_ee.ImageCollection.side_effect = _image_collection
    fake_ee.Image.side_effect = lambda val=None: (
        val if isinstance(val, _FakeEEImage) else _FakeEEImage("wrapped", [("init", val)])
    )
    fake_ee.Image.constant.side_effect = lambda v: _FakeEEImage("const", [("constant", v)])
    fake_ee.Image.cat.side_effect = lambda imgs: _FakeEEImage("cat", [("cat", list(imgs))])

    fake_ee.Algorithms.If.side_effect = lambda cond, t_val, f_val: t_val

    def _make_feature(geom: Any, props: dict[str, Any] | None = None) -> MagicMock:
        if props is None and isinstance(geom, MagicMock):
            return geom
        return MagicMock(name="clean_feature", geom=geom, props=props)

    fake_ee.Feature.side_effect = _make_feature
    fake_ee.FeatureCollection.side_effect = lambda items: MagicMock(
        flatten=lambda: MagicMock(flattened=items)
    )
    fake_ee.Filter.eq.side_effect = lambda prop, val: ("eq", prop, val)
    fake_ee.Reducer.sum.return_value = "SUM_REDUCER"
    fake_ee.Reducer.max.return_value = "MAX_REDUCER"
    fake_ee.Reducer.mean.return_value = "MEAN_REDUCER"
    fake_ee.Reducer.first.return_value = "FIRST_REDUCER"
    fake_ee.batch.Export.table.toAsset.side_effect = lambda **kwargs: MagicMock(
        spec=["start", "kwargs"], kwargs=kwargs
    )
    fake_ee.batch.Export.table.toDrive.side_effect = lambda **kwargs: MagicMock(
        spec=["start", "kwargs"], kwargs=kwargs
    )
    return fake_ee


def test_sample_augmentation_params_bool_nan_and_threshold_resolution() -> None:
    schema = _schema()
    resolved_by_name = resolve_class_thresholds(
        schema,
        thresholds={"Forest": 0.92, "Cropland": 0.88},
        default_threshold=0.85,
    )
    assert resolved_by_name == {1: 0.92, 2: 0.88, 5: 0.85}

    resolved_subset = resolve_class_thresholds(schema, class_ids=[5, 1], thresholds=0.91)
    assert resolved_subset == {1: 0.91, 5: 0.91}

    with pytest.raises(KeyError):
        resolve_class_thresholds(schema, thresholds={"Unknown": 0.9})
    with pytest.raises(ValueError, match="not defined in ClassSchema"):
        resolve_class_thresholds(schema, class_ids=[99])
    with pytest.raises(ValueError, match="Threshold"):
        resolve_class_thresholds(schema, thresholds=1.2)
    with pytest.raises(TypeError, match="bool"):
        resolve_class_thresholds(schema, thresholds=False)
    with pytest.raises(TypeError, match="bool"):
        resolve_class_thresholds(schema, thresholds={"Forest": True})
    with pytest.raises(TypeError, match="bool"):
        resolve_class_thresholds(schema, thresholds={True: 0.9})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        resolve_class_thresholds(schema, thresholds=float("nan"))
    with pytest.raises(ValueError, match="year"):
        SampleAugmentationParams(year=1800)


def test_alphaearth_mosaic_similarity_stack_and_centroid_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr(
        "lc_framework.samples.augmentation.require_ee",
        lambda *_, **__: fake_ee,
    )
    schema = _schema()
    region = MagicMock(name="aoi")

    mosaic = get_alphaearth_mosaic(2024, region)
    assert ("clip", region) in mosaic.ops
    fake_ee.ImageCollection.assert_any_call(ALPHAEARTH_COLLECTION)

    training_fc = MagicMock(name="training_fc")
    training_fc.filter.return_value = training_fc

    sim_one = compute_class_cosine_similarity(mosaic, training_fc, class_id=5, scale_m=10.0)
    assert ("rename", "sim_5") in sim_one.ops
    # Verify Algorithms.If empty-class fallback was constructed
    assert fake_ee.Algorithms.If.called

    sim_stack = build_class_similarity_stack(
        mosaic,
        training_fc,
        class_schema=schema,
        scale_m=10.0,
        materialize_per_class=True,
        wait_seconds=0.0,
    )
    cat_args = [arg for op, arg in sim_stack.ops if op == "cat"][0]
    assert len(cat_args) == 3
    assert [[arg for op, arg in b.ops if op == "rename"][0] for b in cat_args] == [
        "sim_1",
        "sim_2",
        "sim_5",
    ]

    # Thresholding + reduceToVectors + clean Feature(centroid, props) without label/count leak
    cand_fc = extract_class_candidate_centroids(
        sim_stack,
        class_id=5,
        class_schema=schema,
        region=region,
        threshold=0.91,
        scale_m=10.0,
        class_property="lulc_id",
    )
    to_point = cand_fc.mapped_fn
    poly_feat = MagicMock(name="poly")
    poly_geom = MagicMock(name="poly_geom")
    centroid_geom = MagicMock(name="centroid_geom")
    poly_feat.geometry.return_value = poly_geom
    poly_geom.centroid.return_value = centroid_geom

    clean_feat = to_point(poly_feat)
    assert clean_feat.geom is centroid_geom
    assert clean_feat.props == {
        "lulc_id": 5,
        "class_name": "Built-up",
        "confidence": 0.93,
        "sample_source": "alphaearth_similarity",
    }

    merged = extract_all_candidate_centroids(
        sim_stack,
        class_schema=schema,
        region=region,
        thresholds={"Forest": 0.92, "Cropland": 0.88, "Built-up": 0.90},
    )
    assert len(merged.flattened) == 3


def test_diagnostic_softmax_overlay_explicitly_labeled_uncalibrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr(
        "lc_framework.samples.augmentation.require_ee",
        lambda *_, **__: fake_ee,
    )
    schema = _schema()
    sim_stack = _FakeEEImage("sim_stack")

    overlay = build_diagnostic_similarity_overlay(
        sim_stack, class_schema=schema, temperature=10.0
    )
    assert isinstance(overlay, DiagnosticSoftmaxOverlay)
    assert overlay.calibrated is False
    assert "uncalibrated" in overlay.disclaimer.lower()
    assert overlay.band_names == ("qa_softmax_1", "qa_softmax_2", "qa_softmax_5")
    assert all(not b.startswith("prob_") for b in overlay.band_names)
    set_meta = [arg for op, arg in overlay.image.ops if op == "set"][0]
    assert set_meta["is_calibrated_probability"] is False
    assert set_meta["overlay_role"] == "diagnostic_qa_only"

    with pytest.raises(ValueError, match="temperature"):
        build_diagnostic_similarity_overlay(sim_stack, class_schema=schema, temperature=0.0)


def test_end_to_end_sample_augmentation_config_scale_merge_and_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr(
        "lc_framework.samples.augmentation.require_ee",
        lambda *_, **__: fake_ee,
    )
    cfg = ProjectConfig(
        project_name="aug_demo",
        crs="EPSG:32648",
        scale_m=20.0,
        class_schema=_schema(),
    )
    training_fc = MagicMock(name="training_fc")
    training_fc.filter.return_value = training_fc
    region = MagicMock(name="aoi")

    res = run_sample_augmentation(
        training_fc,
        config=cfg,
        region=region,
        thresholds={1: 0.93},
        include_diagnostic_overlay=True,
    )
    assert isinstance(res, SampleAugmentationResult)
    assert res.class_ids == (1, 2, 5)
    assert res.thresholds == {1: 0.93, 2: 0.90, 5: 0.90}
    assert res.diagnostic_overlay is not None and res.diagnostic_overlay.calibrated is False
    # Verify ProjectConfig.scale_m (20.0) propagated into reduceToVectors when params=None
    assert res.candidate_points.flattened[0].reduce_kwargs["scale"] == 20.0

    # Merge applies callbacks and merges collections cleanly
    orig_fc = MagicMock(name="orig_fc")
    aug_fc = MagicMock(name="aug_fc")
    orig_item = MagicMock()
    aug_item = MagicMock()
    orig_mapped = MagicMock(name="orig_mapped")
    aug_mapped = MagicMock(name="aug_mapped")

    def _map_orig(fn: Any) -> MagicMock:
        orig_mapped.mapped = fn(orig_item)
        return orig_mapped

    def _map_aug(fn: Any) -> MagicMock:
        aug_mapped.mapped = fn(aug_item)
        return aug_mapped

    orig_fc.map.side_effect = _map_orig
    aug_fc.map.side_effect = _map_aug
    merged_res = merge_augmented_samples(orig_fc, aug_fc)
    orig_item.set.assert_called_once_with("sample_source", "reference")
    aug_item.set.assert_called_once_with("sample_source", "alphaearth_similarity")
    orig_mapped.merge.assert_called_once_with(aug_mapped)
    assert merged_res is orig_mapped.merge.return_value

    # dry_run exports work without calling EE
    assert (
        export_augmented_samples_to_asset(
            res.candidate_points, asset_id="projects/demo/assets/aug", dry_run=True
        )
        is None
    )
    assert (
        export_augmented_samples_to_drive(
            res.candidate_points, description="aug_csv", dry_run=True
        )
        is None
    )

    task_asset = export_augmented_samples_to_asset(
        res.candidate_points, asset_id="projects/demo/assets/aug", dry_run=False
    )
    assert task_asset is not None
    task_asset.start.assert_called_once()
    assert task_asset.kwargs["assetId"] == "projects/demo/assets/aug"

    task_drive = export_augmented_samples_to_drive(
        res.candidate_points, description="aug_csv", folder="LC_Exports", dry_run=False
    )
    assert task_drive is not None
    task_drive.start.assert_called_once()
    assert task_drive.kwargs["fileFormat"] == "CSV"

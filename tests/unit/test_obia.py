"""Offline unit tests for pluggable segmentation (GEE SNIC & local SLIC) and GEOBIA."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from lc_framework.assembly import run_hybrid_assembly
from lc_framework.config import ClassDefinition, ClassSchema
from lc_framework.io import (
    GEE_INT32_CLAMP_VALUE,
    manifest_path_for,
    read_probability_stack,
    read_segment_raster,
    write_probability_stack,
)
from lc_framework.obia import (
    GEESnicSegmenter,
    LocalSlicSegmenter,
    Segmenter,
    SLICParams,
    SNICParams,
    assign_segment_labels,
    classify_segments,
    pool_features_by_segments,
    project_segment_probabilities,
    run_geobia_classification,
    valid_segment_mask,
    write_segment_raster,
)


def _three_class_schema() -> ClassSchema:
    return ClassSchema(
        classes=(
            ClassDefinition(id=1, name="Forest"),
            ClassDefinition(id=2, name="Cropland"),
            ClassDefinition(id=3, name="Water"),
        )
    )


def _write_synthetic_feature_raster(path: Path) -> tuple[Path, dict[str, Any]]:
    """Write a 3-band, 3-region synthetic GeoTIFF with a nodata strip."""
    height, width = 24, 24
    rng = np.random.default_rng(7)
    data = np.zeros((3, height, width), dtype=np.float32)

    # Region 1 (top half): Forest — high NIR (band 1), low red (band 0), low SWIR (band 2)
    data[0, :12, :] = 0.05 + rng.normal(0.0, 0.005, (12, width))
    data[1, :12, :] = 0.45 + rng.normal(0.0, 0.01, (12, width))
    data[2, :12, :] = 0.12 + rng.normal(0.0, 0.005, (12, width))

    # Region 2 (bottom-left): Cropland — medium red, medium NIR, high SWIR
    data[0, 12:, :12] = 0.18 + rng.normal(0.0, 0.005, (12, 12))
    data[1, 12:, :12] = 0.30 + rng.normal(0.0, 0.01, (12, 12))
    data[2, 12:, :12] = 0.35 + rng.normal(0.0, 0.01, (12, 12))

    # Region 3 (bottom-right): Water — low red, very low NIR, very low SWIR
    data[0, 12:, 12:] = 0.04 + rng.normal(0.0, 0.003, (12, 12))
    data[1, 12:, 12:] = 0.02 + rng.normal(0.0, 0.003, (12, 12))
    data[2, 12:, 12:] = 0.01 + rng.normal(0.0, 0.002, (12, 12))

    # Add a 2-pixel nodata corner to test nodata propagation
    data[:, 0:2, 0:2] = -9999.0

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 3,
        "height": height,
        "width": width,
        "crs": "EPSG:32648",
        "transform": from_origin(500000.0, 1000000.0, 10.0, 10.0),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "nir")
        dst.set_band_description(3, "swir1")
    return path, profile


def _make_training_points_for_synthetic_grid(profile: dict[str, Any]) -> pd.DataFrame:
    """Create training points inside each of the three synthetic land-cover regions."""
    transform = profile["transform"]
    rows_cols_labels = [
        # Forest (rows 2..10, cols 3..21)
        (3, 4, 1),
        (3, 12, 1),
        (3, 20, 1),
        (8, 4, 1),
        (8, 12, 1),
        (8, 20, 1),
        # Cropland (rows 14..21, cols 2..9)
        (14, 3, 2),
        (14, 9, 2),
        (19, 3, 2),
        (19, 9, 2),
        # Water (rows 14..21, cols 14..21)
        (14, 15, 3),
        (14, 21, 3),
        (19, 15, 3),
        (19, 21, 3),
    ]
    records: list[dict[str, float | int]] = []
    for r, c, cls_id in rows_cols_labels:
        x, y = transform * (c + 0.5, r + 0.5)
        records.append({"x": float(x), "y": float(y), "class_id": cls_id})
    return pd.DataFrame(records)


class _FakeEEImage:
    """Minimal mock mimicking chained Earth Engine Image calls for SNIC tests."""

    def __init__(self, name: str = "img", ops: list[tuple[str, Any]] | None = None) -> None:
        self.name = name
        self.ops: list[tuple[str, Any]] = list(ops) if ops else []

    def _chain(self, op: str, arg: Any = None) -> _FakeEEImage:
        return _FakeEEImage(self.name, [*self.ops, (op, arg)])

    def select(self, bands: Any) -> _FakeEEImage:
        return self._chain("select", bands)

    def subtract(self, other: Any) -> _FakeEEImage:
        return self._chain("subtract", other)

    def divide(self, other: Any) -> _FakeEEImage:
        return self._chain("divide", other)

    def clamp(self, lo: Any, hi: Any) -> _FakeEEImage:
        return self._chain("clamp", (lo, hi))

    def toInt64(self) -> _FakeEEImage:  # noqa: N802
        return self._chain("toInt64")

    def abs(self) -> _FakeEEImage:
        return self._chain("abs")

    def mod(self, val: Any) -> _FakeEEImage:
        return self._chain("mod", val)

    def add(self, val: Any) -> _FakeEEImage:
        return self._chain("add", val)

    def toInt32(self) -> _FakeEEImage:  # noqa: N802
        return self._chain("toInt32")

    def rename(self, name: Any) -> _FakeEEImage:
        return self._chain("rename", name)

    def clip(self, region: Any) -> _FakeEEImage:
        return self._chain("clip", region)

    def unmask(self, value: Any) -> _FakeEEImage:
        return self._chain("unmask", value)


def _make_fake_ee() -> MagicMock:
    fake_ee = MagicMock()
    fake_ee.Image.constant.side_effect = lambda vals: _FakeEEImage("const", [("constant", vals)])
    fake_ee.Geometry.Rectangle.side_effect = lambda bounds: ("rect", tuple(bounds))
    fake_ee.Algorithms.Image.Segmentation.SNIC.side_effect = lambda **kwargs: _FakeEEImage(
        "snic", [("SNIC", kwargs)]
    )
    fake_ee.batch.Export.image.toAsset.side_effect = lambda **kwargs: MagicMock(
        spec=["start", "kwargs"], kwargs=kwargs
    )
    fake_ee.batch.Export.image.toDrive.side_effect = lambda **kwargs: MagicMock(
        spec=["start", "kwargs"], kwargs=kwargs
    )
    return fake_ee


def test_segmenter_protocol_conformance() -> None:
    slic_seg = LocalSlicSegmenter(SLICParams(n_segments=12))
    snic_seg = GEESnicSegmenter(SNICParams(size=10))
    assert isinstance(slic_seg, Segmenter)
    assert isinstance(snic_seg, Segmenter)
    assert slic_seg.provider_name == "local_slic"
    assert snic_seg.provider_name == "gee_snic"


def test_valid_segment_mask_preserves_wrapped_negative_ids() -> None:
    arr = np.array([[0, -1, 1], [-500, 42, -2]], dtype=np.int32)
    mask = valid_segment_mask(arr)
    expected = np.array([[False, False, True], [True, True, True]])
    np.testing.assert_array_equal(mask, expected)


def test_local_slic_segment_raster_compatible_with_io_and_assembly(tmp_path: Path) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    seg_path = tmp_path / "segments_slic.tif"

    segmenter = LocalSlicSegmenter(SLICParams(n_segments=18, compactness=1.0, sigma=0.5))
    out = segmenter.segment_raster(feat_path, seg_path)
    assert out == seg_path
    assert seg_path.exists()

    # Standard Segment Label Raster read-back via lc_framework.io
    seg_raster = read_segment_raster(seg_path, reference_profile=profile)
    assert seg_raster.segments.dtype == np.int32
    assert seg_raster.segments.shape == (24, 24)
    # Nodata corner must remain 0
    assert np.all(seg_raster.segments[0:2, 0:2] == 0)
    # Valid interior pixels must have positive segment IDs and multiple segments
    valid_ids = seg_raster.segments[seg_raster.segments > 0]
    assert valid_ids.size == 24 * 24 - 4
    assert np.unique(valid_ids).size >= 6

    # Verify seamless compatibility with Classification Assembly (hybrid)
    schema = _three_class_schema()
    probs = np.full((3, 24, 24), np.nan, dtype=np.float32)
    probs[0, :12, :] = 0.8
    probs[1, :12, :] = 0.1
    probs[2, :12, :] = 0.1
    probs[0, 12:, :12] = 0.1
    probs[1, 12:, :12] = 0.8
    probs[2, 12:, :12] = 0.1
    probs[0, 12:, 12:] = 0.05
    probs[1, 12:, 12:] = 0.05
    probs[2, 12:, 12:] = 0.90
    probs[:, 0:2, 0:2] = np.nan

    pixel_prob_path = write_probability_stack(
        tmp_path / "pixel_probs.tif",
        probs,
        profile=profile,
        class_schema=schema,
        paradigm="pixel",
    )
    assembly_out = run_hybrid_assembly(
        pixel_prob_path,
        seg_path,
        tmp_path / "hybrid_out",
        class_schema=schema,
    )
    assert assembly_out.probabilities.exists()
    hybrid_stack = read_probability_stack(assembly_out.probabilities)
    assert hybrid_stack.manifest.paradigm == "hybrid"


def test_local_slic_raises_helpful_error_when_skimage_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _block_skimage(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "skimage":
            raise ImportError("No module named 'skimage'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_skimage)
    segmenter = LocalSlicSegmenter(SLICParams(n_segments=10))
    arr = np.ones((3, 8, 8), dtype=np.float32)
    with pytest.raises(ImportError, match=r"lc-framework\[obia\]"):
        segmenter.segment_array(arr)


def test_gee_snic_segment_and_tiled_export_striding(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.obia.gee_snic.require_ee", lambda *_, **__: fake_ee)

    params = SNICParams(
        size=15,
        compactness=0.8,
        connectivity=8,
        neighborhood_size=64,
        hash_modulus=200_000_000,
        id_stride=300_000_000,
    )
    segmenter = GEESnicSegmenter(params)
    feature_img = _FakeEEImage("features")

    seg_img = segmenter.segment(feature_img, bands=("red", "nir", "swir1"))
    assert ("select", "clusters") in seg_img.ops
    assert ("mod", 200_000_000) in seg_img.ops
    assert ("add", 1) in seg_img.ops
    assert ("rename", "segment") in seg_img.ops

    snic_kwargs = fake_ee.Algorithms.Image.Segmentation.SNIC.call_args.kwargs
    assert snic_kwargs["size"] == 15
    assert snic_kwargs["compactness"] == 0.8
    assert snic_kwargs["connectivity"] == 8
    assert snic_kwargs["neighborhoodSize"] == 64

    # 2x2 grid = 4 tiles, skip 1 existing asset
    plans, dry_tasks = segmenter.export_tiles(
        feature_img,
        bounds=(124.0, -9.5, 125.0, -8.5),
        export_prefix="projects/demo/assets/snic_2025",
        tile_size_deg=0.5,
        scale_m=20.0,
        crs="EPSG:32751",
        existing_assets=["projects/demo/assets/snic_2025_r0_c1"],
        skip_existing=True,
        dry_run=True,
    )
    assert len(plans) == 4
    assert dry_tasks == []
    assert plans[1].skip is True

    # Real submission submits 3 non-skipped tiles with deterministic tile_idx * id_stride offsets
    plans, tasks = segmenter.export_tiles(
        feature_img,
        bounds=(124.0, -9.5, 125.0, -8.5),
        export_prefix="projects/demo/assets/snic_2025",
        tile_size_deg=0.5,
        scale_m=20.0,
        crs="EPSG:32751",
        existing_assets=["projects/demo/assets/snic_2025_r0_c1"],
        skip_existing=True,
        dry_run=False,
    )
    assert len(tasks) == 3
    submitted_offsets = []
    for task in tasks:
        task.start.assert_called_once()
        kw = task.kwargs
        assert kw["pyramidingPolicy"] == {".default": "mode"}
        assert kw["scale"] == 20.0
        assert kw["crs"] == "EPSG:32751"
        add_ops = [arg for op, arg in kw["image"].ops if op == "add"]
        # Last add op is the tile stride offset (first add op is +1 from hash)
        submitted_offsets.append(add_ops[-1])
    # Tile 0 -> 0, Tile 1 skipped, Tile 2 -> 2 * 300_000_000, Tile 3 -> 3 * 300_000_000
    assert submitted_offsets == [0, 600_000_000, 900_000_000]


def test_gee_snic_overflow_repair_on_load(tmp_path: Path) -> None:
    seg = np.ones((10, 10), dtype=np.int32)
    seg[0:3, 0:3] = GEE_INT32_CLAMP_VALUE
    seg[7:10, 7:10] = GEE_INT32_CLAMP_VALUE
    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "count": 1,
        "height": 10,
        "width": 10,
        "crs": "EPSG:32648",
        "transform": from_origin(500000.0, 1000000.0, 10.0, 10.0),
        "nodata": 0,
    }
    seg_path = write_segment_raster(tmp_path / "snic_overflow.tif", seg, profile=profile)
    loaded = GEESnicSegmenter.load_exported_segment_raster(seg_path)
    assert not np.any(loaded.segments == GEE_INT32_CLAMP_VALUE)
    # Two disjoint clamped islands must become distinct negative segment IDs (< -1)
    id_a = int(loaded.segments[1, 1])
    id_b = int(loaded.segments[8, 8])
    assert id_a < -1 and id_b < -1 and id_a != id_b


def test_geobia_feature_pooling_label_assignment_classification_and_projection(
    tmp_path: Path,
) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    schema = _three_class_schema()
    samples_df = _make_training_points_for_synthetic_grid(profile)

    # Create deterministic 6-segment raster aligning cleanly with the 3 land-cover regions
    seg = np.zeros((24, 24), dtype=np.int32)
    seg[:12, :12] = 10  # Forest left
    seg[:12, 12:] = 11  # Forest right
    seg[12:18, :12] = 20  # Cropland upper
    seg[18:, :12] = 21  # Cropland lower
    seg[12:18, 12:] = 30  # Water upper
    seg[18:, 12:] = 31  # Water lower
    seg[0:2, 0:2] = 0  # Nodata corner
    seg_path = write_segment_raster(tmp_path / "segments.tif", seg, profile=profile)

    # 1. Feature pooling (mean + std + morphology)
    pooled = pool_features_by_segments(
        feat_path,
        seg_path,
        include_std=True,
        include_morphology=True,
    )
    assert list(pooled["segment_id"]) == [10, 11, 20, 21, 30, 31]
    for col in (
        "pixel_count",
        "red_mean",
        "nir_mean",
        "swir1_mean",
        "red_std",
        "nir_std",
        "swir1_std",
        "area_px",
        "bbox_aspect",
        "extent",
        "perimeter_px",
        "compactness",
    ):
        assert col in pooled.columns
    # Forest NIR mean (~0.45) > Cropland (~0.30) > Water (~0.02)
    nir_by_id = dict(zip(pooled["segment_id"], pooled["nir_mean"], strict=True))
    assert nir_by_id[10] > 0.40
    assert 0.25 < nir_by_id[20] < 0.35
    assert nir_by_id[30] < 0.05

    # 2. Label assignment (tabular DataFrame, zero geopandas needed)
    labels = assign_segment_labels(seg_path, samples_df)
    label_map = dict(zip(labels["segment_id"], labels["class_id"], strict=True))
    assert label_map == {10: 1, 11: 1, 20: 2, 21: 2, 30: 3, 31: 3}
    assert np.all(labels["purity"] == 1.0)

    # 3. Classify segments probabilistically
    cls_res = classify_segments(
        pooled,
        labels,
        class_schema=schema,
        n_estimators=40,
        random_state=42,
        n_jobs=1,
    )
    assert cls_res.probabilities.shape == (6, 3)
    np.testing.assert_allclose(cls_res.probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert list(cls_res.predicted_class_ids) == [1, 1, 2, 2, 3, 3]

    # 4. Project per-object probabilities onto canonical Class Probability Stack
    out_prob = project_segment_probabilities(
        seg_path,
        cls_res.segment_ids,
        cls_res.probabilities,
        tmp_path / "geobia_probs.tif",
        class_schema=schema,
        epoch="2025",
    )
    assert out_prob.exists()
    assert manifest_path_for(out_prob).exists()

    loaded = read_probability_stack(out_prob)
    assert loaded.manifest.paradigm == "geobia"
    assert loaded.manifest.epoch == "2025"
    assert loaded.manifest.class_schema == schema
    # Nodata corner remains invalid
    assert not np.any(loaded.valid_mask[0:2, 0:2])
    # Valid pixels sum to 1.0 within uint16 quantization tolerance
    valid_sums = loaded.probabilities[:, loaded.valid_mask].sum(axis=0)
    np.testing.assert_allclose(valid_sums, 1.0, atol=1e-3)
    # Predicted dominant classes match the three synthetic regions
    hard = np.argmax(loaded.probabilities, axis=0) + 1
    assert int(hard[6, 6]) == 1
    assert int(hard[15, 6]) == 2
    assert int(hard[15, 18]) == 3


def test_run_geobia_classification_end_to_end_with_local_slic(tmp_path: Path) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    schema = _three_class_schema()
    samples_csv = tmp_path / "samples.csv"
    _make_training_points_for_synthetic_grid(profile).to_csv(samples_csv, index=False)

    outputs = run_geobia_classification(
        feat_path,
        samples_csv,
        tmp_path / "geobia_run",
        class_schema=schema,
        segmenter=LocalSlicSegmenter(SLICParams(n_segments=18, compactness=1.0, sigma=0.5)),
        epoch="2025",
        n_estimators=40,
        random_state=42,
        n_jobs=1,
    )
    assert outputs.segment_raster.exists()
    assert outputs.probabilities.exists()
    assert manifest_path_for(outputs.probabilities).exists()

    stack = read_probability_stack(outputs.probabilities)
    assert stack.manifest.paradigm == "geobia"
    assert np.all(stack.valid_mask[2:, :])
    assert not np.any(stack.valid_mask[0:2, 0:2])
    # Check majority of each synthetic region is classified accurately
    hard = np.argmax(stack.probabilities, axis=0) + 1
    assert float(np.mean(hard[2:11, 2:22] == 1)) >= 0.85
    assert float(np.mean(hard[13:23, 1:11] == 2)) >= 0.85
    assert float(np.mean(hard[13:23, 13:23] == 3)) >= 0.85


def test_classify_segments_zero_fills_absent_schema_classes(tmp_path: Path) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    schema = _three_class_schema()
    seg = np.zeros((24, 24), dtype=np.int32)
    seg[:12, :12] = 1
    seg[:12, 12:] = 2
    seg[12:, :12] = 3
    seg[12:, 12:] = 4
    seg_path = write_segment_raster(tmp_path / "seg4.tif", seg, profile=profile)

    pooled = pool_features_by_segments(feat_path, seg_path)
    # Only classes 1 and 2 appear in training labels; class 3 is absent
    labels = pd.DataFrame(
        {"segment_id": [1, 2, 3], "class_id": [1, 1, 2], "sample_count": [2, 2, 2], "purity": 1.0}
    )
    res = classify_segments(
        pooled, labels, class_schema=schema, n_estimators=20, random_state=0, n_jobs=1
    )
    assert res.probabilities.shape == (4, 3)
    np.testing.assert_allclose(res.probabilities[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(res.probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_assign_segment_labels_purity_tiebreak_and_crs_reprojection(tmp_path: Path) -> None:
    _, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    seg = np.full((24, 24), 10, dtype=np.int32)
    seg[:, 12:] = 20
    seg_path = write_segment_raster(tmp_path / "seg2.tif", seg, profile=profile)

    transform = profile["transform"]
    # Segment 10: 2 votes for class 2, 1 vote for class 1 (purity = 2/3)
    # Segment 20: 1 vote class 3, 1 vote class 1 (tie -> smallest class_id=1, purity=0.5)
    pts = [
        (* (transform * (2.5, 2.5)), 2),
        (* (transform * (4.5, 4.5)), 2),
        (* (transform * (6.5, 6.5)), 1),
        (* (transform * (15.5, 4.5)), 3),
        (* (transform * (18.5, 8.5)), 1),
    ]
    df = pd.DataFrame(pts, columns=["x", "y", "class_id"])

    labels_all = assign_segment_labels(seg_path, df)
    by_seg = labels_all.set_index("segment_id")
    assert int(by_seg.loc[10, "class_id"]) == 2
    assert pytest.approx(float(by_seg.loc[10, "purity"]), abs=1e-6) == 2.0 / 3.0
    assert int(by_seg.loc[20, "class_id"]) == 1
    assert pytest.approx(float(by_seg.loc[20, "purity"]), abs=1e-6) == 0.5

    # Filtering by min_purity=0.6 drops the 50/50 tied segment 20
    labels_pure = assign_segment_labels(seg_path, df, min_purity=0.6)
    assert list(labels_pure["segment_id"]) == [10]

    # Reproject points from EPSG:4326 back into EPSG:32648 raster
    from rasterio.warp import transform as warp_transform

    lons, lats = warp_transform(
        "EPSG:32648", "EPSG:4326", df["x"].tolist(), df["y"].tolist()
    )
    df_wgs84 = pd.DataFrame({"x": lons, "y": lats, "class_id": df["class_id"]})
    labels_reproj = assign_segment_labels(seg_path, df_wgs84, samples_crs="EPSG:4326")
    pd.testing.assert_frame_equal(labels_all, labels_reproj)


def test_gee_snic_drive_export_and_overflow_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.obia.gee_snic.require_ee", lambda *_, **__: fake_ee)

    segmenter = GEESnicSegmenter(
        SNICParams(size=10, hash_modulus=200_000_000, id_stride=300_000_000)
    )
    # Tile index 7 -> 7 * 300M + 200M = 2.3B >= 2^31 - 1 -> emits UserWarning
    with pytest.warns(UserWarning, match="int32 limit"):
        segmenter.tile_offset(7)

    _, tasks = segmenter.export_tiles(
        _FakeEEImage("feat"),
        bounds=(124.0, -9.0, 124.5, -8.5),
        export_prefix="snic_drive_test",
        tile_size_deg=0.5,
        destination="drive",
        drive_folder="MyDriveFolder",
        dry_run=False,
    )
    assert len(tasks) == 1
    assert tasks[0].kwargs["folder"] == "MyDriveFolder"
    assert tasks[0].kwargs["formatOptions"] == {"cloudOptimized": True, "noData": 0}
    assert ("unmask", 0) in tasks[0].kwargs["image"].ops


def test_pool_features_handles_wrapped_negative_and_clamped_segments(tmp_path: Path) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    seg = np.full((24, 24), -500, dtype=np.int32)  # Valid wrapped negative ID
    seg[4:8, 4:8] = GEE_INT32_CLAMP_VALUE  # Clamped island repaired to distinct negative ID
    seg[16:20, 16:20] = GEE_INT32_CLAMP_VALUE  # Second disjoint clamped island
    seg[0:2, 0:2] = 0
    seg_path = write_segment_raster(tmp_path / "seg_wrapped.tif", seg, profile=profile)

    pooled = pool_features_by_segments(feat_path, seg_path, repair_overflow=True)
    # Expect 3 distinct segments: -500 plus two repaired negative IDs
    assert len(pooled) == 3
    assert -500 in set(pooled["segment_id"].tolist())
    assert GEE_INT32_CLAMP_VALUE not in set(pooled["segment_id"].tolist())


def test_run_geobia_reuses_precomputed_segment_raster(tmp_path: Path) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    schema = _three_class_schema()
    samples_df = _make_training_points_for_synthetic_grid(profile)

    seg = np.zeros((24, 24), dtype=np.int32)
    seg[:12, :] = 100
    seg[12:, :12] = 200
    seg[12:, 12:] = 300
    seg[0:2, 0:2] = 0
    precomputed_seg = write_segment_raster(tmp_path / "gee_snic_export.tif", seg, profile=profile)

    outputs = run_geobia_classification(
        feat_path,
        samples_df,
        tmp_path / "geobia_reused",
        class_schema=schema,
        segment_raster_path=precomputed_seg,
        n_estimators=20,
        random_state=0,
        n_jobs=1,
    )
    assert outputs.segment_raster == precomputed_seg
    assert outputs.probabilities.exists()
    stack = read_probability_stack(outputs.probabilities)
    assert stack.manifest.paradigm == "geobia"


def test_local_slic_start_label_above_one_single_pixel_and_valid_mask_forwarding(
    tmp_path: Path,
) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    # start_label > 1 must work without skimage ValueError and shift all valid labels >= 100
    seg_shift = LocalSlicSegmenter(SLICParams(n_segments=12, start_label=100))
    arr = np.ones((2, 8, 8), dtype=np.float32)
    arr[0, :4, :] = 0.0
    arr[1, :, :4] = 0.0
    labels = seg_shift.segment_array(arr)
    assert int(labels.min()) >= 100

    # Single valid pixel edge case must receive start_label (not 0)
    one_px_mask = np.zeros((8, 8), dtype=bool)
    one_px_mask[3, 3] = True
    single_out = seg_shift.segment_array(arr, valid_mask=one_px_mask)
    assert int(single_out[3, 3]) == 100
    assert int(np.count_nonzero(single_out)) == 1

    # segment(..., output_path=..., valid_mask=...) must forward valid_mask to disk
    mask = np.ones((24, 24), dtype=bool)
    mask[12:, :] = False
    out_path = tmp_path / "masked_slic.tif"
    seg_shift.segment(feat_path, output_path=out_path, valid_mask=mask)
    loaded = read_segment_raster(out_path, reference_profile=profile)
    assert np.all(loaded.segments[12:, :] == 0)
    assert np.all(loaded.segments[2:12, 2:] >= 100)

    # output_path with ndarray must raise ValueError
    with pytest.raises(ValueError, match="output_path is only supported"):
        seg_shift.segment(arr, output_path=tmp_path / "bad.tif")


def test_gee_snic_band_normalization_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.obia.gee_snic.require_ee", lambda *_, **__: fake_ee)
    segmenter = GEESnicSegmenter(SNICParams(size=10))
    res = segmenter.segment(
        _FakeEEImage("feat"),
        bands=("red", "nir"),
        band_min=(0.0, 0.0),
        band_max=(0.3, 0.6),
    )
    assert ("select", "clusters") in res.ops

    with pytest.raises(ValueError, match="strictly less than band_max"):
        segmenter.segment(
            _FakeEEImage("feat"),
            bands=("red", "nir"),
            band_min=(0.5, 0.0),
            band_max=(0.3, 0.6),
        )


def test_assign_segment_labels_handles_geometry_col_dataframe_and_vector_gdf(
    tmp_path: Path,
) -> None:
    _, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    seg = np.full((24, 24), 10, dtype=np.int32)
    seg[:, 12:] = 20
    seg_path = write_segment_raster(tmp_path / "seg_vec.tif", seg, profile=profile)
    transform = profile["transform"]

    # Plain DataFrame with a 'geometry' string column must NOT be misrouted to vector path
    x1, y1 = transform * (3.5, 3.5)
    x2, y2 = transform * (16.5, 3.5)
    df_with_geom = pd.DataFrame(
        {
            "x": [x1, x2, np.nan],
            "y": [y1, y2, y2],
            "class_id": [1, 2, 1],
            "geometry": ["POINT (1 2)", "POINT (3 4)", "POINT (5 6)"],
        }
    )
    labels_df = assign_segment_labels(seg_path, df_with_geom)
    assert dict(zip(labels_df["segment_id"], labels_df["class_id"], strict=True)) == {10: 1, 20: 2}

    # Duck-typed GeoDataFrame-like object with GeoJSON polygon geometries & rasterize int32
    x0, y0 = transform * (2, 2)
    x1b, y1b = transform * (6, 6)
    poly_left = {
        "type": "Polygon",
        "coordinates": [[(x0, y0), (x1b, y0), (x1b, y1b), (x0, y1b), (x0, y0)]],
    }

    class _FakeGeom:
        is_empty = False

        def __init__(self, geojson: dict[str, Any]) -> None:
            self.__geo_interface__ = geojson

    class _FakeGDF:
        crs = "EPSG:32648"
        columns = ("geometry", "class_id")

        def __init__(self) -> None:
            self.geometry = [_FakeGeom(poly_left)]
            self._data = {"class_id": [3]}

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def to_crs(self, target_crs: Any) -> _FakeGDF:
            self.crs = target_crs
            return self

    vec_labels = assign_segment_labels(seg_path, _FakeGDF())
    assert dict(zip(vec_labels["segment_id"], vec_labels["class_id"], strict=True)) == {10: 3}


def test_classify_segments_excludes_pixel_count_and_guards_bad_probabilities(
    tmp_path: Path,
) -> None:
    feat_path, profile = _write_synthetic_feature_raster(tmp_path / "features.tif")
    seg = np.zeros((24, 24), dtype=np.int32)
    seg[:12, :12] = 1
    seg[:12, 12:] = 2
    seg[12:, :12] = 3
    seg[12:, 12:] = 4
    seg_path = write_segment_raster(tmp_path / "seg_nc.tif", seg, profile=profile)

    pooled = pool_features_by_segments(feat_path, seg_path, include_morphology=False)
    schema = ClassSchema(
        classes=(
            ClassDefinition(id=10, name="Forest"),
            ClassDefinition(id=20, name="Cropland"),
            ClassDefinition(id=50, name="Water"),
        )
    )
    labels = pd.DataFrame(
        {"segment_id": [1, 2, 3, 4], "class_id": [10, 10, 20, 50], "sample_count": 1, "purity": 1.0}
    )
    res = classify_segments(
        pooled, labels, class_schema=schema, n_estimators=20, random_state=0, n_jobs=1
    )
    assert "pixel_count" not in res.feature_names
    assert set(res.predicted_class_ids.tolist()).issubset({10, 20, 50})

    class _ZeroEstimator:
        classes_ = np.array([10, 20, 50])

        def fit(self, x: np.ndarray, y: np.ndarray) -> _ZeroEstimator:
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((len(x), 3), dtype=np.float64)

    with pytest.raises(ValueError, match="zero total probability mass"):
        classify_segments(pooled, labels, class_schema=schema, estimator=_ZeroEstimator())

    # Passing GEESnicSegmenter directly to run_geobia_classification gives clear guidance
    with pytest.raises(TypeError, match="export_tiles"):
        run_geobia_classification(
            feat_path,
            _make_training_points_for_synthetic_grid(profile),
            tmp_path / "bad_snic",
            class_schema=schema,
            segmenter=GEESnicSegmenter(),
        )



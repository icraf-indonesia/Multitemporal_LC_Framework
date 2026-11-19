"""Tests for Class Probability Stack, Segment Label Raster, CSV, and GEE export I/O."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from lc_framework.config import ClassDefinition, ClassSchema
from lc_framework.io import (
    GEE_INT32_CLAMP_VALUE,
    generate_tile_grid,
    manifest_path_for,
    plan_tiled_gee_exports,
    read_probability_stack,
    read_sample_features_csv,
    read_segment_raster,
    repair_gee_int32_overflow,
    submit_gee_export_tasks,
    write_probability_stack,
    write_sample_features_csv,
)


def _schema() -> ClassSchema:
    return ClassSchema(
        classes=[
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
            ClassDefinition(id=3, name="Water", color="#0288d1"),
        ]
    )


def _sample_profile(height: int = 8, width: int = 10) -> dict:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "crs": "EPSG:32748",
        "transform": from_origin(500000.0, 9000000.0, 10.0, 10.0),
    }


def _sample_probabilities(height: int = 8, width: int = 10) -> np.ndarray:
    probs = np.zeros((3, height, width), dtype=np.float32)
    probs[0] = 0.60
    probs[1] = 0.25
    probs[2] = 0.15
    # Mark top-left 2x2 pixels as nodata
    probs[:, :2, :2] = np.nan
    return probs


def test_uint16_probability_stack_roundtrip_and_manifest_metadata(tmp_path: Path) -> None:
    schema = _schema()
    probs = _sample_probabilities()
    out_tif = tmp_path / "probs_2025.tif"

    written = write_probability_stack(
        out_tif,
        probs,
        profile=_sample_profile(),
        class_schema=schema,
        encoding="uint16",
        epoch="2025",
        paradigm="pixel",
        source="gee_rf",
    )

    mpath = manifest_path_for(written)
    assert written.exists()
    assert mpath.exists()

    with rasterio.open(written) as src:
        assert src.dtypes == ("uint16", "uint16", "uint16")
        assert src.nodata == 65535
        assert src.descriptions == ("1:Forest", "2:Cropland", "3:Water")

    loaded = read_probability_stack(written)
    assert loaded.manifest.class_schema == schema
    assert loaded.manifest.scale_factor == 10000.0
    assert loaded.manifest.epoch == "2025"
    assert loaded.manifest.paradigm == "pixel"
    assert not loaded.valid_mask[0, 0]
    assert loaded.valid_mask[3, 4]
    assert np.isnan(loaded.probabilities[:, 0, 0]).all()
    np.testing.assert_allclose(
        loaded.probabilities[:, 3, 4],
        np.array([0.60, 0.25, 0.15], dtype=np.float32),
        atol=1e-4,
    )


def test_float32_probability_stack_roundtrip(tmp_path: Path) -> None:
    schema = _schema()
    probs = _sample_probabilities()
    out_tif = tmp_path / "probs_f32.tif"

    write_probability_stack(
        out_tif,
        probs,
        profile=_sample_profile(),
        class_schema=schema,
        encoding="float32",
    )

    loaded = read_probability_stack(out_tif)
    assert loaded.manifest.dtype == "float32"
    assert loaded.manifest.scale_factor == 1.0
    assert loaded.manifest.nodata is None
    np.testing.assert_allclose(
        loaded.probabilities[:, 5, 5],
        np.array([0.60, 0.25, 0.15], dtype=np.float32),
        atol=1e-6,
    )


def test_write_probability_stack_rejects_unnormalized_or_out_of_range_values(
    tmp_path: Path,
) -> None:
    schema = _schema()
    bad_sum = np.full((3, 4, 4), 0.5, dtype=np.float32)  # sums to 1.5
    with pytest.raises(ValueError, match="sum to 1"):
        write_probability_stack(
            tmp_path / "bad_sum.tif",
            bad_sum,
            profile=_sample_profile(4, 4),
            class_schema=schema,
        )

    bad_neg = np.array(
        [[[-0.1, 0.5]], [[0.6, 0.3]], [[0.5, 0.2]]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        write_probability_stack(
            tmp_path / "bad_neg.tif",
            bad_neg,
            profile=_sample_profile(1, 2),
            class_schema=schema,
        )


def test_standalone_read_probability_stack_without_manifest(tmp_path: Path) -> None:
    """External uint16 GeoTIFF without sidecar manifest still loads cleanly (ADR-0005)."""
    ext_tif = tmp_path / "external_gee_export.tif"
    prof = _sample_profile(4, 4)
    prof.update(count=3, dtype="uint16", nodata=0)
    data = np.zeros((3, 4, 4), dtype=np.uint16)
    data[0, 1:, 1:] = 7000
    data[1, 1:, 1:] = 2000
    data[2, 1:, 1:] = 1000
    with rasterio.open(ext_tif, "w", **prof) as dst:
        dst.write(data)

    loaded = read_probability_stack(ext_tif, class_schema=_schema())
    assert loaded.manifest.paradigm == "external"
    assert not loaded.valid_mask[0, 0]  # all-zero background treated as nodata
    assert loaded.valid_mask[2, 2]
    np.testing.assert_allclose(loaded.probabilities[:, 2, 2], [0.7, 0.2, 0.1], atol=1e-4)

    # Also works with only n_classes (no manifest, no explicit ClassSchema)
    loaded_by_count = read_probability_stack(ext_tif, n_classes=3)
    assert len(loaded_by_count.manifest.class_schema) == 3
    with pytest.raises(ValueError, match="n_classes"):
        read_probability_stack(ext_tif, n_classes=4)

    # Rejects unnormalized external raster on read unless validate=False
    bad_ext = tmp_path / "bad_external.tif"
    bad_data = np.full((3, 4, 4), 8000, dtype=np.uint16)
    with rasterio.open(bad_ext, "w", **prof) as dst:
        dst.write(bad_data)
    with pytest.raises(ValueError, match="sum to 1"):
        read_probability_stack(bad_ext, n_classes=3)


def test_repair_gee_int32_overflow_localizes_small_clusters_and_drops_megaclusters() -> None:
    seg = np.zeros((10, 10), dtype=np.int32)
    seg[0, 0] = -1  # nodata sentinel preserved
    seg[0, 1] = -2_000_000_001  # existing wrapped negative ID near id_base must not collide
    seg[0, 2] = 42  # normal positive ID preserved

    # Two separate small clamped clusters (size 2 each <= max_clamp_cluster_px=6)
    seg[2:4, 2] = GEE_INT32_CLAMP_VALUE
    seg[2:4, 5] = GEE_INT32_CLAMP_VALUE

    # One oversized fused mega-cluster (size 12 > max_clamp_cluster_px=6)
    seg[6:9, 4:8] = GEE_INT32_CLAMP_VALUE

    repaired = repair_gee_int32_overflow(seg, max_clamp_cluster_px=6)

    assert repaired[0, 0] == -1
    assert repaired[0, 1] == -2_000_000_001
    assert repaired[0, 2] == 42
    # Two small clamped clusters receive distinct negative IDs (< -1) with zero collision
    assert repaired[2, 2] < -1
    assert repaired[2, 5] < -1
    assert repaired[2, 2] != repaired[2, 5]
    assert repaired[2, 2] != repaired[0, 1]
    assert repaired[2, 5] != repaired[0, 1]
    assert repaired[2, 2] == repaired[3, 2]
    # Oversized fused mega-cluster falls back to 0 (pure pixel fallback)
    assert np.all(repaired[6:9, 4:8] == 0)


def test_read_segment_raster_validates_alignment_and_reprojects_when_requested(
    tmp_path: Path,
) -> None:
    ref = _sample_profile(8, 10)
    seg_path = tmp_path / "segments.tif"
    seg_prof = _sample_profile(8, 10)
    seg_prof.update(count=1, dtype="int32")
    arr = np.arange(1, 81, dtype=np.int32).reshape(8, 10)
    arr[0, 0] = GEE_INT32_CLAMP_VALUE
    with rasterio.open(seg_path, "w", **seg_prof) as dst:
        dst.write(arr, 1)

    loaded = read_segment_raster(seg_path, reference_profile=ref)
    assert loaded.segments.shape == (8, 10)
    assert loaded.segments[0, 0] < -1  # overflow repaired

    # Misaligned grid raises unless align_to_reference=True
    mismatched_ref = _sample_profile(4, 5)
    with pytest.raises(ValueError, match="grid"):
        read_segment_raster(seg_path, reference_profile=mismatched_ref)

    aligned = read_segment_raster(
        seg_path,
        reference_profile=mismatched_ref,
        align_to_reference=True,
    )
    assert aligned.segments.shape == (4, 5)


def test_sample_features_csv_roundtrip_and_feature_selection_columns(tmp_path: Path) -> None:
    schema = _schema()
    df = pd.DataFrame(
        {
            "sample_id": ["p1", "p2", "p3"],
            "class_id": [1, 2, 3],
            "ndvi_dry": [0.81, 0.45, -0.10],
            "swir1_dry": [0.12, 0.25, 0.03],
        }
    )
    csv_path = tmp_path / "samples.csv"
    write_sample_features_csv(df, csv_path, class_col="class_id", class_schema=schema)

    x_df, y_series, feature_names = read_sample_features_csv(
        csv_path,
        class_col="class_id",
        ignore_cols=("sample_id", "class_name"),
    )
    assert feature_names == ["ndvi_dry", "swir1_dry"]
    assert list(x_df.columns) == ["ndvi_dry", "swir1_dry"]
    assert list(y_series) == [1, 2, 3]
    raw_df = pd.read_csv(csv_path)
    assert list(raw_df["class_name"]) == ["Forest", "Cropland", "Water"]


def test_tiled_gee_export_planning_and_dry_run_skip_existing() -> None:
    tiles = generate_tile_grid((125.0, -9.0, 126.0, -8.0), tile_size_deg=0.5)
    assert len(tiles) == 4

    existing = {"projects/demo/assets/tile_r0_c0"}
    plan = plan_tiled_gee_exports(
        tiles,
        export_prefix="projects/demo/assets/tile",
        existing_assets=existing,
        skip_existing=True,
    )
    assert len(plan) == 4
    skipped = [p for p in plan if p.skip]
    to_run = [p for p in plan if not p.skip]
    assert len(skipped) == 1
    assert skipped[0].asset_id == "projects/demo/assets/tile_r0_c0"
    assert len(to_run) == 3

    # Dry run does not invoke export builder
    builder = MagicMock()
    dry_tasks = submit_gee_export_tasks(plan, task_factory=builder, dry_run=True)
    assert dry_tasks == []
    builder.assert_not_called()

    # Live submission starts only non-skipped tasks
    fake_task = MagicMock()
    builder.return_value = fake_task
    started = submit_gee_export_tasks(plan, task_factory=builder, dry_run=False)
    assert len(started) == 3
    assert builder.call_count == 3
    assert fake_task.start.call_count == 3

"""Unit tests for standalone Classification Assembly (Ticket #04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lc_framework.assembly import (
    STABILITY_NODATA,
    MaskRule,
    apply_mask_rules,
    apply_mmu,
    apply_mmu_array,
    bayesian_smooth,
    bayesian_smooth_raster,
    blend_pixel_and_object_probabilities,
    compute_confusion_index,
    derive_hard_labels,
    derive_hard_labels_raster,
    heal_array_seams,
    heal_probability_array_spikes,
    heal_probability_raster_spikes,
    heal_raster_seams,
    logits_to_probs,
    probs_to_logits,
    run_hybrid_assembly,
)
from lc_framework.config import ClassDefinition, ClassSchema, ProjectConfig
from lc_framework.io import GEE_INT32_CLAMP_VALUE, read_probability_stack, write_probability_stack


def _schema() -> ClassSchema:
    return ClassSchema(
        classes=[
            ClassDefinition(id=10, name="Forest", color="#1b5e20"),
            ClassDefinition(id=20, name="Cropland", color="#fbc02d"),
            ClassDefinition(id=30, name="Water", color="#0288d1"),
        ]
    )


def _profile(height: int = 8, width: int = 8, pixel_size: float = 10.0) -> dict:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "crs": "EPSG:32748",
        "transform": from_origin(500000.0, 9000000.0, pixel_size, pixel_size),
    }


def test_blend_pixel_and_object_probabilities_invariants_and_noise_suppression() -> None:
    schema = _schema()
    probs = np.zeros((3, 6, 6), dtype=np.float32)
    # Segment 1 (left half): mostly Forest (0.85), with one noisy Cropland pixel at (2, 1)
    probs[0, :, :3] = 0.85
    probs[1, :, :3] = 0.10
    probs[2, :, :3] = 0.05
    probs[:, 2, 1] = [0.42, 0.50, 0.08]  # isolated noisy pixel inside Forest segment

    # Segment 2 (right half): Water
    probs[0, :, 3:] = 0.05
    probs[1, :, 3:] = 0.10
    probs[2, :, 3:] = 0.85

    # Mark (0, 0) as nodata
    probs[:, 0, 0] = np.nan

    segments = np.ones((6, 6), dtype=np.int32)
    segments[:, 3:] = 2
    # Unsegmented pixel at (5, 5) falls back to pure pixel probabilities
    segments[5, 5] = 0

    result = blend_pixel_and_object_probabilities(
        probs,
        segments,
        class_schema=schema,
        smoothness_weights={"Forest": 4.0, "Cropland": 1.0, "Water": 4.0},
        tau=0.65,
        adaptive_sensitivity=0.6,
        heal_segment_seams=False,
    )

    # Valid pixels sum strictly to 1.0 and lie in [0, 1]
    assert not result.valid_mask[0, 0]
    assert np.isnan(result.probabilities[:, 0, 0]).all()
    valid_sums = np.sum(result.probabilities[:, result.valid_mask], axis=0)
    np.testing.assert_allclose(valid_sums, 1.0, atol=1e-5)
    assert np.all(result.probabilities[:, result.valid_mask] >= 0.0)
    assert np.all(result.probabilities[:, result.valid_mask] <= 1.0)

    # Noisy pixel inside Forest segment is pulled toward Forest (ID 10)
    assert result.hard_labels[2, 1] == 10
    assert result.hard_labels[0, 0] == 0

    # Unsegmented pixel (5, 5) preserves pure pixel probability exactly
    np.testing.assert_allclose(result.probabilities[:, 5, 5], probs[:, 5, 5], atol=1e-5)

    # Confusion Index lies in [0, 1] on valid pixels (NaN on nodata); stability uses 255 for nodata
    assert np.isnan(result.confusion_index[0, 0])
    assert np.all(result.confusion_index[result.valid_mask] >= 0.0)
    assert np.all(result.confusion_index[result.valid_mask] <= 1.0)
    assert result.stability[3, 4] == 1
    assert result.stability[0, 0] == STABILITY_NODATA


def test_compute_confusion_index_matches_second_over_winner_ratio() -> None:
    probs = np.array(
        [
            [[0.8, 0.5, np.nan]],
            [[0.2, 0.4, np.nan]],
            [[0.0, 0.1, np.nan]],
        ],
        dtype=np.float32,
    )
    ci, stable = compute_confusion_index(probs, tau=0.65)
    np.testing.assert_allclose(ci[0, 0], 0.2 / 0.8, atol=1e-5)  # 0.25 -> stable
    np.testing.assert_allclose(ci[0, 1], 0.4 / 0.5, atol=1e-5)  # 0.80 -> unstable
    assert np.isnan(ci[0, 2])
    assert stable[0, 0] == 1
    assert stable[0, 1] == 0
    assert stable[0, 2] == STABILITY_NODATA


def test_bayesian_smooth_preserves_probability_simplex_and_smooths_salt_pepper() -> None:
    probs = np.zeros((3, 9, 9), dtype=np.float32)
    probs[0] = 0.85
    probs[1] = 0.10
    probs[2] = 0.05
    # Single noisy center pixel
    probs[:, 4, 4] = [0.35, 0.55, 0.10]
    probs[:, 0, 0] = np.nan

    logits = probs_to_logits(probs)
    recovered = logits_to_probs(logits, normalize=True)
    np.testing.assert_allclose(recovered[:, 2, 2], probs[:, 2, 2], atol=1e-3)

    smoothed = bayesian_smooth(
        probs,
        smoothness={"Forest": 3.0, "Cropland": 1.0, "Water": 1.0},
        window_size=5,
        class_bands=("Forest", "Cropland", "Water"),
    )
    assert np.isnan(smoothed[:, 0, 0]).all()
    valid = np.all(np.isfinite(smoothed), axis=0)
    np.testing.assert_allclose(np.sum(smoothed[:, valid], axis=0), 1.0, atol=1e-5)
    # Center pixel's Forest probability increases after smoothing toward neighborhood
    assert smoothed[0, 4, 4] > probs[0, 4, 4]


def test_seam_healing_fills_thin_gaps_without_touching_broad_background() -> None:
    labels = np.ones((10, 14), dtype=np.int32) * 10
    labels[:, 4] = 0  # 1-px vertical seam -> should be healed
    labels[:, 9:] = 0  # 5-px broad background (> max_seam_width=2) -> far edge stays 0

    healed = heal_array_seams(labels, nodata=0, max_seam_width=2)
    assert np.all(healed[:, 4] == 10)
    assert np.all(healed[:, 12:] == 0)

    # Probability 1-px computation spike healing
    p = np.zeros((3, 5, 5), dtype=np.float32)
    p[0] = 0.8
    p[1] = 0.1
    p[2] = 0.1
    p[:, :, 2] = np.array([0.1, 0.8, 0.1], dtype=np.float32)[:, None]  # 1-px vertical seam
    healed_p = heal_probability_array_spikes(p, spike_threshold=0.05, major_spike_threshold=0.2)
    np.testing.assert_allclose(healed_p[:, 2, 2], [0.8, 0.1, 0.1], atol=1e-5)


def test_derive_hard_labels_and_mmu_filtering_for_arbitrary_class_ids(tmp_path: Path) -> None:
    schema = _schema()
    probs = np.zeros((3, 8, 8), dtype=np.float32)
    probs[0] = 0.8
    probs[1] = 0.1
    probs[2] = 0.1
    probs[:, 3:5, 3:5] = np.array([0.1, 0.8, 0.1], dtype=np.float32)[:, None, None]  # 4-px patch
    probs[:, 0, 0] = np.nan

    labels = derive_hard_labels(probs, class_schema=schema, nodata=0)
    assert labels[0, 0] == 0
    assert labels[1, 1] == 10
    assert labels[3, 3] == 20

    # MMU min size 6 removes the 4-px Cropland (20) patch and fills with surrounding Forest (10)
    filtered = apply_mmu_array(labels, min_patch_sizes={20: 6, 10: 0}, nodata=0)
    assert filtered[0, 0] == 0
    assert np.all(filtered[3:5, 3:5] == 10)

    # Also test file-based apply_mmu round-trip
    in_tif = tmp_path / "hard.tif"
    out_tif = tmp_path / "hard_mmu.tif"
    prof = _profile(8, 8)
    prof.update(count=1, dtype="uint16", nodata=0)
    with rasterio.open(in_tif, "w", **prof) as dst:
        dst.write(labels.astype(np.uint16), 1)
    apply_mmu(in_tif, out_tif, min_patch_sizes={20: 6})
    with rasterio.open(out_tif) as src:
        out_arr = src.read(1)
    assert np.all(out_arr[3:5, 3:5] == 10)


def test_apply_mask_rules_handles_different_resolution_mask_via_warped_vrt(tmp_path: Path) -> None:
    # 10m LULC raster (8x8)
    lulc_path = tmp_path / "lulc.tif"
    lulc = np.full((8, 8), 20, dtype=np.uint16)  # Cropland
    prof_10m = _profile(8, 8, pixel_size=10.0)
    prof_10m.update(count=1, dtype="uint16", nodata=0)
    with rasterio.open(lulc_path, "w", **prof_10m) as dst:
        dst.write(lulc, 1)

    # 20m Water mask (4x4 covering same spatial extent; left half = 1)
    mask_path = tmp_path / "water_mask_20m.tif"
    mask_arr = np.zeros((4, 4), dtype=np.uint8)
    mask_arr[:, :2] = 1
    prof_20m = _profile(4, 4, pixel_size=20.0)
    prof_20m.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(mask_path, "w", **prof_20m) as dst:
        dst.write(mask_arr, 1)

    out_path = tmp_path / "lulc_masked.tif"
    rule = MaskRule(
        from_classes=(20,),
        to_class=30,
        require_mask_path=mask_path,
        require_value=1,
        name="Force water in water mask",
    )
    apply_mask_rules(lulc_path, out_path, [rule])

    with rasterio.open(out_path) as src:
        refined = src.read(1)
    assert np.all(refined[:, :4] == 30)
    assert np.all(refined[:, 4:] == 20)


def test_run_hybrid_assembly_end_to_end_standalone_and_with_project_config(tmp_path: Path) -> None:
    schema = _schema()
    probs = np.zeros((3, 8, 8), dtype=np.float32)
    probs[0, :, :4] = 0.80
    probs[1, :, :4] = 0.15
    probs[2, :, :4] = 0.05
    probs[0, :, 4:] = 0.10
    probs[1, :, 4:] = 0.10
    probs[2, :, 4:] = 0.80
    probs[:, 2, 2] = [0.40, 0.55, 0.05]  # noisy pixel inside Forest segment
    probs[:, 0, 0] = np.nan

    prob_path = tmp_path / "pixel_probs.tif"
    write_probability_stack(
        prob_path,
        probs,
        profile=_profile(8, 8),
        class_schema=schema,
        encoding="uint16",
        epoch="2025",
    )

    seg_path = tmp_path / "segments.tif"
    seg = np.ones((8, 8), dtype=np.int32)
    seg[:, 4:] = 2
    seg[7, 7] = GEE_INT32_CLAMP_VALUE  # triggers overflow repair
    seg_prof = _profile(8, 8)
    seg_prof.update(count=1, dtype="int32")
    with rasterio.open(seg_path, "w", **seg_prof) as dst:
        dst.write(seg, 1)

    cfg = ProjectConfig(
        project_name="test_hybrid",
        crs="EPSG:32748",
        scale_m=10,
        class_schema=schema,
        smoothness_weights={"Forest": 4.0, "Cropland": 1.0, "Water": 4.0},
    )

    outputs = run_hybrid_assembly(
        prob_path,
        seg_path,
        tmp_path / "out",
        config=cfg,
        mmu_table={20: 4},
    )

    assert outputs.probabilities.exists()
    assert outputs.lulc.exists()
    assert outputs.confusion_index.exists()
    assert outputs.stability.exists()

    # Refined probability stack conforms to Class Probability Stack contract
    refined_stack = read_probability_stack(outputs.probabilities)
    assert refined_stack.manifest.paradigm == "hybrid"
    assert refined_stack.manifest.class_schema == schema
    assert refined_stack.manifest.epoch == "2025"

    with rasterio.open(outputs.lulc) as src:
        lulc_arr = src.read(1)
        assert lulc_arr[0, 0] == 0
        assert lulc_arr[2, 2] == 10  # smoothed into Forest
        assert lulc_arr[4, 6] == 30  # Water preserved
        assert src.tags().get("lc_framework_class_schema") is not None

    # Verify CI uses NaN nodata (never colliding with 0.0) and stability uses 255 nodata
    with rasterio.open(outputs.confusion_index) as src_ci:
        assert np.isnan(src_ci.nodata)
        assert np.isnan(src_ci.read(1)[0, 0])
    with rasterio.open(outputs.stability) as src_st:
        assert src_st.nodata == STABILITY_NODATA
        assert src_st.read(1)[0, 0] == STABILITY_NODATA

    # Standalone run WITHOUT ProjectConfig and WITHOUT manifest on an external uint16 raster
    ext_prob = tmp_path / "external_no_manifest.tif"
    ext_prof = _profile(8, 8)
    ext_prof.update(count=3, dtype="uint16", nodata=65535)
    ext_u16 = np.full((3, 8, 8), 65535, dtype=np.uint16)
    valid = np.all(np.isfinite(probs), axis=0)
    ext_u16[:, valid] = np.rint(probs[:, valid] * 10000.0).astype(np.uint16)
    with rasterio.open(ext_prob, "w", **ext_prof) as dst:
        dst.write(ext_u16)

    standalone_out = run_hybrid_assembly(ext_prob, seg_path, tmp_path / "standalone_out")
    assert standalone_out.probabilities.exists()
    assert standalone_out.lulc.exists()


def test_standalone_raster_wrappers_preserve_class_schema_metadata(tmp_path: Path) -> None:
    schema = _schema()
    probs = np.zeros((3, 7, 7), dtype=np.float32)
    probs[0] = 0.8
    probs[1] = 0.1
    probs[2] = 0.1
    probs[:, :, 3] = np.array([0.1, 0.8, 0.1], dtype=np.float32)[:, None]

    prob_path = tmp_path / "probs.tif"
    write_probability_stack(prob_path, probs, profile=_profile(7, 7), class_schema=schema)

    healed_path = heal_probability_raster_spikes(
        prob_path,
        tmp_path / "probs_healed.tif",
        spike_threshold=0.05,
        major_spike_threshold=0.2,
    )
    assert read_probability_stack(healed_path).manifest.class_schema == schema

    smooth_path = bayesian_smooth_raster(
        healed_path,
        tmp_path / "probs_smooth.tif",
        smoothness=2.0,
        window_size=5,
    )
    assert read_probability_stack(smooth_path).manifest.class_schema == schema

    hard_path = derive_hard_labels_raster(smooth_path, tmp_path / "hard.tif")
    healed_hard = heal_raster_seams(hard_path, tmp_path / "hard_healed.tif")
    with rasterio.open(healed_hard) as src:
        assert src.tags().get("lc_framework_class_schema") is not None

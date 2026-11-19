"""Offline unit tests for GEE feature engineering and sample CSV export (Ticket #07)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lc_framework.config import ClassDefinition, ClassSchema
from lc_framework.features import (
    SUPPORTED_SPECTRAL_INDICES,
    FeatureStackSpec,
    apply_speckle_filter,
    build_feature_stack,
    compute_phenological_deltas,
    compute_seasonal_indices,
    export_sample_features_to_drive,
    extract_sample_features,
    get_glcm_texture,
    get_palsar_backscatter,
    get_scansar_features,
    get_seasonal_s1,
    get_topography,
    preprocess_s1_collection,
)


def _schema() -> ClassSchema:
    return ClassSchema(
        classes=[
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
            ClassDefinition(id=3, name="Water", color="#0288d1"),
        ]
    )


def _make_fake_ee() -> MagicMock:
    fake_ee = MagicMock(name="ee")

    class FakeImageCollection:
        def __init__(self, source=None) -> None:
            self.source = source
            self.calls: list[tuple[str, tuple, dict]] = []

        def filterBounds(self, aoi):
            self.calls.append(("filterBounds", (aoi,), {}))
            return self

        def filterDate(self, start, end):
            self.calls.append(("filterDate", (start, end), {}))
            return self

        def filter(self, flt):
            self.calls.append(("filter", (flt,), {}))
            return self

        def select(self, bands):
            self.calls.append(("select", (bands,), {}))
            return self

        def map(self, fn):
            probe = MagicMock(name="sar_probe_img")
            probe.select.return_value.reduceRegion.return_value.get.return_value = MagicMock()
            self.last_mapped = fn(probe)
            self.calls.append(("map", (fn,), {}))
            return self

        def median(self):
            return MagicMock(name="median_img")

        def first(self):
            return MagicMock(name="first_img")

        def count(self):
            return MagicMock(name="count_img")

    fake_ee.ImageCollection = FakeImageCollection
    fake_ee.Image.cat.side_effect = lambda imgs: MagicMock(name="cat_img", inputs=imgs)
    return fake_ee


def test_compute_seasonal_indices_and_phenological_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework._gee.require_ee", lambda *_, **__: fake_ee)

    synth_wet = MagicMock(name="synth_wet")
    synth_dry = MagicMock(name="synth_dry")
    indices = ("NDVI", "EVI", "MNDWI", "NDBI", "BSI")

    wet_idx = compute_seasonal_indices(synth_wet, indices=indices, suffix="wet")
    dry_idx = compute_seasonal_indices(synth_dry, indices=indices, suffix="dry")
    assert wet_idx is not None
    assert dry_idx is not None

    # Every requested index invokes synth.expression(...) and renames with _wet / _dry
    assert synth_wet.expression.call_count == len(indices)
    renamed_wet = [c.args[0] for c in synth_wet.expression.return_value.rename.call_args_list]
    assert renamed_wet == [f"{idx}_wet" for idx in indices]

    deltas = compute_phenological_deltas(wet_idx, dry_idx, indices=indices)
    assert deltas is not None
    wet_selects = [c.args[0] for c in wet_idx.select.call_args_list]
    dry_selects = [c.args[0] for c in dry_idx.select.call_args_list]
    assert wet_selects == [f"{idx}_wet" for idx in indices]
    assert dry_selects == [f"{idx}_dry" for idx in indices]

    assert "NDVI" in SUPPORTED_SPECTRAL_INDICES
    with pytest.raises(ValueError, match="Unsupported spectral index"):
        compute_seasonal_indices(synth_wet, indices=("NOT_AN_INDEX",), suffix="wet")


def test_sar_s1_scansar_palsar_and_speckle_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ee = _make_fake_ee()
    created_collections: list[str] = []
    orig_ic = fake_ee.ImageCollection

    def _capture_ic(source=None):
        created_collections.append(str(source))
        return orig_ic(source)

    fake_ee.ImageCollection = _capture_ic
    monkeypatch.setattr("lc_framework._gee.require_ee", lambda *_, **__: fake_ee)
    aoi = MagicMock(name="aoi")

    for method in ("BOXCAR", "LEE", "GAMMA_MAP"):
        filtered = apply_speckle_filter(
            MagicMock(name="sar_img"),
            method=method,
            kernel_size=5,
            enl=5.0,
            bands=("VV", "VH"),
        )
        assert filtered is not None

    with pytest.raises(ValueError, match="Unsupported speckle filter"):
        apply_speckle_filter(MagicMock(name="sar_img"), method="UNKNOWN")  # type: ignore[arg-type]

    s1_col = preprocess_s1_collection(
        "2025-01-01",
        "2025-12-31",
        aoi,
        orbit="DESCENDING",
        apply_terrain_flattening=True,
    )
    assert s1_col is not None
    assert "COPERNICUS/S1_GRD_FLOAT" in created_collections

    s1_seasonal = get_seasonal_s1(2025, aoi, include_ratio=True)
    assert s1_seasonal is not None
    # Verify contiguous wrap-around wet season starts in year - 1 (2024-12-01)
    fake_ee.Date.fromYMD.assert_any_call(2024, 12, 1)
    # Verify expected seasonal S1 band renames (including ratio)
    s1_cat_inputs = fake_ee.Image.cat.call_args.args[0]
    assert len(s1_cat_inputs) == 6  # VH_dry, VH_wet, VV_dry, VV_wet, 2 ratios

    scansar = get_scansar_features(2025, aoi)
    assert scansar is not None
    assert "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR" in created_collections

    # PALSAR-1 (<= 2012) vs PALSAR-2 (>= 2013) mission gap handling
    get_palsar_backscatter(2010, aoi)
    get_palsar_backscatter(2025, aoi)
    assert "JAXA/ALOS/PALSAR/YEARLY/SAR" in created_collections
    assert "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH" in created_collections
    palsar_cat_inputs = fake_ee.Image.cat.call_args.args[0]
    assert len(palsar_cat_inputs) == 3  # palsar_HH, palsar_HV, palsar_HH_HV_diff


def test_topography_glcm_and_feature_stack_toggles(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ee = _make_fake_ee()
    queried_collections: list[str] = []
    orig_ic = fake_ee.ImageCollection

    def _capture_ic(source=None):
        queried_collections.append(str(source))
        return orig_ic(source)

    fake_ee.ImageCollection = _capture_ic
    monkeypatch.setattr("lc_framework._gee.require_ee", lambda *_, **__: fake_ee)
    aoi = MagicMock(name="aoi")

    topo = get_topography(aoi, compute_tpi=True, compute_twi=True)
    assert topo is not None
    fake_ee.Image.assert_any_call("NASA/NASADEM_HGT/001")
    fake_ee.Image.assert_any_call("MERIT/Hydro/v1_0_1")
    fake_ee.Terrain.products.assert_called_once()
    # Elevation is NOT clipped before Terrain.products (prevents AOI edge artifacts)
    fake_ee.Image.return_value.select.return_value.clip.assert_not_called()

    nir_src = MagicMock(name="dry_img")
    glcm = get_glcm_texture(nir_src, band="nir", size=1, scale_factor=1000.0)
    assert glcm is not None
    nir_src.select.assert_called_once_with(["nir"], ["nir"])
    glcm_chain = nir_src.select.return_value.multiply.return_value.toInt32.return_value
    glcm_chain.glcmTexture.assert_called_once_with(size=1)
    glcm_chain.glcmTexture.return_value.select.assert_called_once_with(
        ["nir_contrast", "nir_ent", "nir_asm", "nir_idm"]
    )

    # Feature stack with SAR disabled -> never queries Sentinel-1 / PALSAR collections
    spec = FeatureStackSpec(
        indices=("NDVI", "MNDWI"),
        include_deltas=True,
        include_s1=False,
        include_scansar=False,
        include_palsar=False,
        include_topography=True,
        include_glcm=True,
    )
    wet_img = MagicMock(name="wet_synth")
    dry_img = MagicMock(name="dry_synth")
    harm_img = MagicMock(name="harm_img")
    stack = build_feature_stack(
        aoi=aoi,
        year=2025,
        wet_image=wet_img,
        dry_image=dry_img,
        harmonics_image=harm_img,
        spec=spec,
    )
    assert stack is not None
    assert queried_collections == []
    # Layers: wet_opt, dry_opt, wet_idx, dry_idx, deltas, harmonics, topo, glcm = 8
    assert len(fake_ee.Image.cat.call_args.args[0]) == 8


def test_extract_sample_features_and_export_csv_to_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ee = _make_fake_ee()
    started: list[dict] = []

    def _fake_to_drive(**kwargs):
        task = SimpleNamespace(kwargs=kwargs)
        task.start = lambda: started.append(kwargs)
        return task

    fake_ee.batch.Export.table.toDrive.side_effect = _fake_to_drive
    monkeypatch.setattr("lc_framework._gee.require_ee", lambda *_, **__: fake_ee)

    schema = _schema()
    feature_img = MagicMock(name="feature_stack")
    samples_fc = MagicMock(name="samples_fc")

    sampled = extract_sample_features(
        feature_img,
        samples_fc,
        class_schema=schema,
        class_property="class_id",
        scale_m=10.0,
        tile_scale=4,
    )
    feature_img.sampleRegions.assert_called_once_with(
        collection=samples_fc,
        properties=["class_id"],
        scale=10.0,
        tileScale=4,
        geometries=False,
    )
    # Attaches ClassSchema id -> name lookup dictionary
    fake_ee.Dictionary.assert_called_once_with({"1": "Forest", "2": "Cropland", "3": "Water"})
    assert sampled is feature_img.sampleRegions.return_value.map.return_value

    # Dry run does not submit task
    dry = export_sample_features_to_drive(
        sampled,
        description="samples_2025",
        folder="lc_framework_exports",
        dry_run=True,
    )
    assert dry is None
    assert started == []

    # Live export writes CSV to Drive
    live = export_sample_features_to_drive(
        sampled,
        description="samples_2025",
        folder="lc_framework_exports",
        file_name_prefix="samples_2025_features",
        dry_run=False,
    )
    assert live is not None
    assert len(started) == 1
    assert started[0]["fileFormat"] == "CSV"
    assert started[0]["folder"] == "lc_framework_exports"
    assert started[0]["fileNamePrefix"] == "samples_2025_features"

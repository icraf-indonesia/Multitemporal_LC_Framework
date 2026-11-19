"""Offline unit tests for GEE optical inputs, composites, and CCDC Temporal Segments."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lc_framework.temporal import (
    OPTICAL_BANDS,
    CCDCModel,
    CCDCParams,
    CCDCProfile,
    ReflectanceBounds,
    export_ccdc_tiles,
    fill_isolated,
    get_harmonics,
    get_landsat_collection,
    get_multi_year_stacks,
    get_s2_collection,
    get_synthetic_stack,
    load_ccdc_mosaic,
    nearest_segment,
    prepare_ccdc_for_export,
    run_ccdc,
    seasonal_composite,
    stable_in_period,
    to_monthly_median,
)


def _make_fake_ee() -> MagicMock:
    """Create a chainable fake ``ee`` module for offline GEE graph verification."""
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

        def linkCollection(self, other, bands):
            self.calls.append(("linkCollection", (other, bands), {}))
            return self

        def map(self, fn):
            self.calls.append(("map", (fn,), {}))
            # Invoke mapper once on a probe image to verify band mapping & scaling
            probe_img = MagicMock(name="probe_img")
            self.last_mapped = fn(probe_img)
            self.last_probe = probe_img
            return self

        def merge(self, other):
            self.calls.append(("merge", (other,), {}))
            return self

        def mosaic(self):
            m = MagicMock(name="mosaic_img")
            self.calls.append(("mosaic", (), {}))
            return m

        def median(self):
            return MagicMock(name="median_img")

        def mean(self):
            return MagicMock(name="mean_img")

        def reduce(self, reducer):
            m = MagicMock(name="reduced_img")
            m.reducer = reducer
            return m

        def size(self):
            return MagicMock(name="col_size")

    fake_ee.ImageCollection = FakeImageCollection
    fake_ee.ImageCollection.fromImages = MagicMock(
        side_effect=lambda imgs: FakeImageCollection(imgs)
    )
    return fake_ee


def test_landsat_collection_harmonizes_tm_and_oli_bands_and_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    created: list = []
    orig_ic = fake_ee.ImageCollection

    def _capture_ic(source=None):
        ic = orig_ic(source)
        created.append(ic)
        return ic

    fake_ee.ImageCollection = _capture_ic
    monkeypatch.setattr("lc_framework.temporal.ccdc.specs._require_ee", lambda: fake_ee)

    aoi = MagicMock(name="aoi")
    get_landsat_collection(
        aoi,
        "2015-01-01",
        "2025-12-31",
        reflectance_scale=10000.0,
        sensors=("L5", "L8"),
    )

    assert [ic.source for ic in created] == [
        "LANDSAT/LT05/C02/T1_L2",
        "LANDSAT/LC08/C02/T1_L2",
    ]
    # L5 maps TM bands -> OPTICAL_BANDS; L8 maps OLI bands -> OPTICAL_BANDS
    created[0].last_probe.select.assert_any_call(
        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        list(OPTICAL_BANDS),
    )
    created[1].last_probe.select.assert_any_call(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        list(OPTICAL_BANDS),
    )
    # Verify QA_PIXEL bitmasking covers fill(0), dilated(1), cloud(3), shadow(4), snow(5)
    qa_mock = created[0].last_probe.select.return_value
    qa_bits = {call.args[0] for call in qa_mock.bitwiseAnd.call_args_list}
    assert qa_bits == {1 << 0, 1 << 1, 1 << 3, 1 << 4, 1 << 5}
    # Verify Collection 2 scaling: multiply(0.0000275).subtract(0.2).multiply(10000.0)
    sr_chain = created[0].last_probe.select.return_value
    sr_chain.multiply.assert_any_call(0.0000275)
    sr_chain.multiply.return_value.subtract.assert_any_call(0.2)
    sr_chain.multiply.return_value.subtract.return_value.multiply.assert_any_call(10000.0)

    with pytest.raises(ValueError, match="Unsupported Landsat sensor"):
        get_landsat_collection(aoi, "2020-01-01", "2021-01-01", sensors=("L99",))


def test_s2_collection_links_cloud_score_plus_and_handles_reflectance_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.temporal.ccdc.specs._require_ee", lambda: fake_ee)
    aoi = MagicMock(name="aoi")

    # CCDC scale = 10000 -> no redundant multiply
    col_ccdc = get_s2_collection(
        aoi,
        "2020-01-01",
        "2025-12-31",
        clear_threshold=0.60,
        qa_band="cs_cdf",
        reflectance_scale=10000.0,
    )
    col_ccdc.last_probe.select.assert_any_call(
        ["B2", "B3", "B4", "B8", "B11", "B12"],
        list(OPTICAL_BANDS),
    )
    col_ccdc.last_probe.select.return_value.multiply.assert_not_called()

    # Direct composite scale = 1.0 -> multiplies by 1e-4
    col_direct = get_s2_collection(
        aoi,
        "2024-01-01",
        "2025-01-01",
        reflectance_scale=1.0,
    )
    col_direct.last_probe.select.return_value.multiply.assert_called_once_with(1.0 / 10000.0)

    with pytest.raises(ValueError, match="qa_band"):
        get_s2_collection(aoi, "2024-01-01", "2025-01-01", qa_band="invalid")


def test_monthly_median_and_seasonal_composite_handle_wraparound_and_december(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.temporal.ccdc.specs._require_ee", lambda: fake_ee)

    col = fake_ee.ImageCollection("S2")
    monthly = to_monthly_median(col, "2024-01-01", "2024-12-31")
    assert any(c[0] == "filter" for c in monthly.calls)
    # Must use .ceil() so full-year ranges include December
    fake_ee.Date.return_value.difference.return_value.ceil.assert_called_once()

    # End month == 12 must use advance(1, "month"), never fromYMD(year, 13, 1)
    seasonal_composite(col, 2025, 10, 12, reducer="median")
    for call in fake_ee.Date.fromYMD.call_args_list:
        _, month, _ = call.args
        assert 1 <= month <= 12

    # Wrap-around wet season (Dec -> Apr) spans year-1 to year and adds calendarRange filter
    col_wrap = fake_ee.ImageCollection("S2")
    seasonal_composite(col_wrap, 2025, 12, 4, reducer=25)
    assert any(c[0] == "filter" for c in col_wrap.calls)
    fake_ee.Reducer.percentile.assert_called_once_with([25])


def test_ccdc_specs_validation_and_presets() -> None:
    l_params = CCDCParams.landsat()
    assert l_params.tmask_bands == ("green", "swir1")
    assert len(l_params.breakpoint_bands) == 5

    s2_params = CCDCParams.sentinel2(min_observations=8)
    assert s2_params.tmask_bands is None
    assert s2_params.min_observations == 8

    prof = CCDCProfile.google_global()
    assert prof.coef_scale == 1.0
    assert prof.coef_band_name("nir") == "NIR_coefs"
    assert prof.harmonic_band_names("nir") == ("nir_amp", "nir_phase")

    with pytest.raises(ValueError, match="strictly less"):
        ReflectanceBounds(min=1.0, max=0.5)


def test_ccdc_fit_reconstruct_harmonics_stability_and_model_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    monkeypatch.setattr("lc_framework.temporal.ccdc.specs._require_ee", lambda: fake_ee)

    col = MagicMock(name="optical_col")
    ccdc_img = run_ccdc(col, params=CCDCParams.sentinel2())
    assert ccdc_img is fake_ee.Algorithms.TemporalSegmentation.Ccdc.return_value
    call_kwargs = fake_ee.Algorithms.TemporalSegmentation.Ccdc.call_args.kwargs
    assert call_kwargs["lambda"] == 20.0
    assert call_kwargs["dateFormat"] == 1
    assert "tmaskBands" not in call_kwargs

    trimmed = prepare_ccdc_for_export(ccdc_img)
    assert trimmed is ccdc_img.select.return_value

    loaded = load_ccdc_mosaic(["projects/demo/t1", "projects/demo/t2"], aoi=MagicMock(name="aoi"))
    assert loaded is not None

    seg = nearest_segment(ccdc_img, 2025.65)
    ccdc_img.select.assert_any_call("tStart")
    ccdc_img.select.assert_any_call("tEnd")
    assert seg.gap is not None

    img_to_fill = MagicMock(name="img")
    fill_isolated(img_to_fill, min_neighbors=5)
    img_to_fill.focalMedian.assert_called_once_with(1, "square", "pixels")
    valid_nb = img_to_fill.select.return_value.mask.return_value.reduceNeighborhood.return_value
    valid_nb.gte.assert_called_once_with(5)

    prof = CCDCProfile.standard()
    get_synthetic_stack(ccdc_img, 2025.65, profile=prof)
    stack_cat = fake_ee.Image.cat.return_value
    stack_cat.reduce.return_value.gt.assert_any_call(prof.bounds.min)
    stack_cat.reduce.return_value.lt.assert_any_call(prof.bounds.max)

    get_harmonics(ccdc_img, 2025.50, bands=("nir", "red"))
    ccdc_img.select.assert_any_call("nir_coefs")
    ccdc_img.select.assert_any_call("red_coefs")
    coef_proj = ccdc_img.select.return_value.arraySlice.return_value.arrayProject.return_value
    coef_proj.arrayGet.assert_any_call([2])
    coef_proj.arrayGet.assert_any_call([3])
    coef_proj.arrayGet.return_value.divide.return_value.hypot.return_value.rename.assert_any_call(
        "nir_amp"
    )
    coef_proj.arrayGet.return_value.divide.return_value.atan2.return_value.rename.assert_any_call(
        "nir_phase"
    )

    series = get_multi_year_stacks(ccdc_img, [2020, 2025], season_fraction=0.65)
    assert set(series.keys()) == {2020, 2025}

    stab = stable_in_period(ccdc_img, 2020.0, 2025.99, max_drift=0.12)
    ccdc_img.select.assert_any_call("tBreak")
    ccdc_img.select.assert_any_call("changeProb")
    ccdc_img.select.return_value.eq.assert_any_call(1)
    assert stab is not None

    model = CCDCModel(image=ccdc_img, profile=prof)
    assert model.synthetic(2025.65) is not None
    assert model.harmonics(2025.50) is not None
    assert model.stable_in_period(2020.0, 2025.99) is not None

    with pytest.raises(ValueError, match="end_year"):
        stable_in_period(ccdc_img, 2025.0, 2020.0)


def test_export_ccdc_tiles_uses_sample_pyramiding_and_skip_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ee = _make_fake_ee()
    started_tasks: list[dict] = []

    def _fake_to_asset(**kwargs):
        task = SimpleNamespace(kwargs=kwargs, started=False)

        def _start():
            task.started = True
            started_tasks.append(kwargs)

        task.start = _start
        return task

    fake_ee.batch.Export.image.toAsset.side_effect = _fake_to_asset
    monkeypatch.setattr("lc_framework.temporal.ccdc.specs._require_ee", lambda: fake_ee)

    col = MagicMock(name="monthly_col")
    plans, dry_tasks = export_ccdc_tiles(
        col,
        bounds=(125.0, -9.0, 126.0, -8.0),
        export_prefix="projects/demo/assets/s2_ccdc",
        tile_size_deg=0.5,
        scale_m=10.0,
        crs="EPSG:32751",
        existing_assets={"projects/demo/assets/s2_ccdc_r0_c0"},
        skip_existing=True,
        dry_run=True,
    )
    assert len(plans) == 4
    assert dry_tasks == []
    assert len(started_tasks) == 0

    plans2, live_tasks = export_ccdc_tiles(
        col,
        bounds=(125.0, -9.0, 126.0, -8.0),
        export_prefix="projects/demo/assets/s2_ccdc",
        tile_size_deg=0.5,
        scale_m=10.0,
        crs="EPSG:32751",
        existing_assets={"projects/demo/assets/s2_ccdc_r0_c0"},
        skip_existing=True,
        dry_run=False,
    )
    assert len(plans2) == 4
    assert len(live_tasks) == 3
    assert len(started_tasks) == 3
    for kw in started_tasks:
        assert kw["pyramidingPolicy"] == {".default": "sample"}
        assert kw["crs"] == "EPSG:32751"
        assert math.isclose(kw["scale"], 10.0)

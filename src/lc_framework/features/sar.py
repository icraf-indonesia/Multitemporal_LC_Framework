"""Sentinel-1 ARD (terrain flattening + speckle filtering) and ALOS PALSAR/ScanSAR (GEE)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from lc_framework import _gee

if TYPE_CHECKING:
    import ee

SpeckleMethod = Literal["GAMMA_MAP", "LEE", "BOXCAR"]
OrbitPass = Literal["BOTH", "ASCENDING", "DESCENDING"]
TerrainModel = Literal["VOLUME", "DIRECT"]
SpeckleMode = Literal["POST_COMPOSITE", "MONO", "NONE"]

_DEG2RAD = math.pi / 180.0
_PALSAR_LIN_CF = 10.0 ** (-8.3)


def lin_to_db(image: ee.Image, bands: Sequence[str] | None = None) -> ee.Image:
    """Convert linear-power backscatter bands to decibels: ``10 * log10(max(x, 1e-10))``."""
    ee_mod = _gee.require_ee("SAR dB conversion")
    band_names = (
        ee_mod.List(list(bands)) if bands is not None else image.bandNames().remove("angle")
    )
    db = image.select(band_names).max(1e-10).log10().multiply(10.0).rename(band_names)
    return image.addBands(db, None, True)


def _mask_s1_border_noise(image: Any, ee_mod: Any) -> Any:
    """Mask Sentinel-1 IW edge incidence angles (30.64-45.24 deg) and non-positive pixels."""
    ang = image.select("angle")
    pol_bands = image.bandNames().remove("angle")
    ang_mask = ang.gt(30.63993).And(ang.lt(45.23993))
    pos_mask = image.select(pol_bands).gt(0).reduce(ee_mod.Reducer.min())
    return image.updateMask(ang_mask.And(pos_mask)).copyProperties(
        image, ["system:time_start", "orbitProperties_pass"]
    )


def apply_speckle_filter(
    image: ee.Image,
    *,
    method: SpeckleMethod = "GAMMA_MAP",
    kernel_size: int = 5,
    enl: float | ee.Image = 5.0,
    bands: Sequence[str] | None = None,
) -> ee.Image:
    """Apply a linear-power spatial speckle filter (``GAMMA_MAP``, ``LEE``, or ``BOXCAR``)."""
    method_upper = method.upper()
    if method_upper not in ("GAMMA_MAP", "LEE", "BOXCAR"):
        raise ValueError(
            f"Unsupported speckle filter {method!r}; expected 'GAMMA_MAP', 'LEE', or 'BOXCAR'."
        )
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size!r}.")

    ee_mod = _gee.require_ee("SAR speckle filtering")
    radius = max(1, kernel_size // 2)
    band_list = list(bands) if bands is not None else None
    ee_bands = (
        ee_mod.List(band_list) if band_list is not None else image.bandNames().remove("angle")
    )

    if method_upper == "BOXCAR":
        kernel = ee_mod.Kernel.square(radius=radius, units="pixels", normalize=True)
        filtered = image.select(ee_bands).convolve(kernel).rename(ee_bands)
        return image.addBands(filtered, None, True)

    enl_img = enl if not isinstance(enl, int | float) else ee_mod.Image.constant(float(enl))
    one = ee_mod.Image.constant(1.0)
    img = image.select(ee_bands)
    mean_bands = (
        [f"{b}_mean" for b in band_list]
        if band_list is not None
        else ee_bands.map(lambda b: ee_mod.String(b).cat("_mean"))
    )

    if method_upper == "LEE":
        eta_sq = one.divide(enl_img)
        reducers = ee_mod.Reducer.mean().combine(
            reducer2=ee_mod.Reducer.variance(), sharedInputs=True
        )
        stats = img.reduceNeighborhood(
            reducer=reducers,
            kernel=ee_mod.Kernel.square(radius, "pixels"),
            optimization="window",
        )
        var_bands = (
            [f"{b}_variance" for b in band_list]
            if band_list is not None
            else ee_bands.map(lambda b: ee_mod.String(b).cat("_variance"))
        )
        z_bar = stats.select(mean_bands).rename(ee_bands)
        var_z = stats.select(var_bands).rename(ee_bands)
        var_x = var_z.subtract(z_bar.pow(2).multiply(eta_sq)).divide(one.add(eta_sq))
        weight = var_x.divide(var_z.max(1e-12)).clamp(0, 1)
        output = one.subtract(weight).multiply(z_bar.abs()).add(weight.multiply(img))
        return image.addBands(output.rename(ee_bands), None, True)

    # Lopes et al. (1990) Gamma MAP filter
    reducers = ee_mod.Reducer.mean().combine(reducer2=ee_mod.Reducer.stdDev(), sharedInputs=True)
    stats = img.reduceNeighborhood(
        reducer=reducers,
        kernel=ee_mod.Kernel.square(radius, "pixels"),
        optimization="window",
    )
    std_bands = (
        [f"{b}_stdDev" for b in band_list]
        if band_list is not None
        else ee_bands.map(lambda b: ee_mod.String(b).cat("_stdDev"))
    )
    z = stats.select(mean_bands).rename(ee_bands)
    sig_z = stats.select(std_bands).rename(ee_bands)

    ci = sig_z.divide(z.max(1e-12))
    cu = one.divide(enl_img.sqrt())
    cmax = cu.multiply(math.sqrt(2.0))

    cu_sq = cu.pow(2)
    ci_sq = ci.pow(2)
    alpha = one.add(cu_sq).divide(ci_sq.subtract(cu_sq).max(1e-6))

    term = z.multiply(alpha.subtract(enl_img).subtract(one))
    disc = term.pow(2).add(alpha.multiply(4.0).multiply(enl_img).multiply(img).multiply(z))
    r_hat = term.add(disc.max(0).sqrt()).divide(alpha.multiply(2.0))
    output = r_hat.where(ci.lte(cu), z).where(ci.gte(cmax), img).rename(ee_bands)
    return image.addBands(output, None, True)


def _apply_s1_terrain_flattening(
    collection: Any,
    *,
    model: TerrainModel,
    dem: Any | None,
    ee_mod: Any,
) -> Any:
    """Radiometric terrain flattening (Vollrath et al. 2020 VOLUME / DIRECT)."""
    dem_img = dem if dem is not None else ee_mod.Image("NASA/NASADEM_HGT/001").select("elevation")
    terrain = ee_mod.Terrain.products(dem_img)
    slope_rad = terrain.select("slope").multiply(_DEG2RAD)
    aspect_rad = terrain.select("aspect").multiply(_DEG2RAD)
    tan_alpha_cos = slope_rad.tan().multiply(aspect_rad.cos())
    tan_alpha_sin = slope_rad.tan().multiply(aspect_rad.sin())
    ninety_rad = ee_mod.Image.constant(math.pi / 2.0)

    def _correct_scene(image: Any) -> Any:
        angle_img = image.select("angle")
        theta_i_rad = angle_img.multiply(_DEG2RAD)
        pol_bands = ["VV", "VH"]
        gamma0 = image.select(pol_bands).divide(theta_i_rad.cos())

        az = ee_mod.Terrain.aspect(angle_img).reduceRegion(
            reducer=ee_mod.Reducer.mean(),
            geometry=image.geometry(),
            scale=5000,
            maxPixels=1e8,
        )
        phi_i_rad = ee_mod.Image.constant(ee_mod.Number(az.get("aspect")).multiply(_DEG2RAD))
        cos_phi = phi_i_rad.cos()
        sin_phi = phi_i_rad.sin()
        alpha_r_rad = tan_alpha_cos.multiply(cos_phi).add(tan_alpha_sin.multiply(sin_phi)).atan()

        if model == "DIRECT":
            alpha_az_rad = (
                tan_alpha_cos.multiply(sin_phi).subtract(tan_alpha_sin.multiply(cos_phi)).atan()
            )
            scf = (
                ninety_rad.subtract(theta_i_rad)
                .cos()
                .divide(
                    alpha_az_rad.cos().multiply(
                        ninety_rad.subtract(theta_i_rad).add(alpha_r_rad).cos()
                    )
                )
            )
        else:
            scf = (
                ninety_rad.subtract(theta_i_rad)
                .add(alpha_r_rad)
                .tan()
                .divide(ninety_rad.subtract(theta_i_rad).tan())
            )

        layover = alpha_r_rad.lt(theta_i_rad)
        shadow = alpha_r_rad.gt(theta_i_rad.subtract(ninety_rad))
        ls_mask = layover.And(shadow).And(scf.gt(0))
        return (
            gamma0.multiply(scf)
            .updateMask(ls_mask)
            .rename(pol_bands)
            .addBands(angle_img)
            .copyProperties(image, ["system:time_start", "orbitProperties_pass"])
        )

    return collection.map(_correct_scene)


def preprocess_s1_collection(
    start_date: str | Any,
    end_date: str | Any,
    aoi: ee.Geometry,
    *,
    orbit: OrbitPass = "BOTH",
    apply_border_noise: bool = True,
    apply_speckle_filter_per_scene: bool = False,
    speckle_filter: SpeckleMethod = "GAMMA_MAP",
    kernel_size: int = 5,
    enl: float = 5.0,
    apply_terrain_flattening: bool = True,
    terrain_model: TerrainModel = "VOLUME",
    dem: ee.Image | None = None,
    output_format: Literal["LINEAR", "DB"] = "LINEAR",
) -> ee.ImageCollection:
    """Build an ARD-preprocessed Sentinel-1 IW dual-pol (VV+VH) collection."""
    orbit_upper = orbit.upper()
    if orbit_upper not in ("BOTH", "ASCENDING", "DESCENDING"):
        raise ValueError(f"orbit must be 'BOTH', 'ASCENDING', or 'DESCENDING', got {orbit!r}.")

    ee_mod = _gee.require_ee("Sentinel-1 ARD preprocessing")
    s1 = (
        ee_mod.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee_mod.Filter.eq("instrumentMode", "IW"))
        .filter(ee_mod.Filter.eq("resolution_meters", 10))
        .filter(ee_mod.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee_mod.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])
    )
    if orbit_upper != "BOTH":
        s1 = s1.filter(ee_mod.Filter.eq("orbitProperties_pass", orbit_upper))
    if apply_border_noise:
        s1 = s1.map(lambda img: _mask_s1_border_noise(img, ee_mod))
    if apply_speckle_filter_per_scene:
        s1 = s1.map(
            lambda img: apply_speckle_filter(
                img,
                method=speckle_filter,
                kernel_size=kernel_size,
                enl=enl,
                bands=("VV", "VH"),
            ).copyProperties(img, ["system:time_start", "orbitProperties_pass"])
        )
    if apply_terrain_flattening:
        s1 = _apply_s1_terrain_flattening(s1, model=terrain_model, dem=dem, ee_mod=ee_mod)
    if output_format.upper() == "DB":
        s1 = s1.map(
            lambda img: lin_to_db(img, ("VV", "VH")).copyProperties(
                img, ["system:time_start", "orbitProperties_pass"]
            )
        )
    return s1


def get_seasonal_s1(
    year: int,
    aoi: ee.Geometry,
    *,
    wet_months: tuple[int, int] = (12, 4),
    dry_months: tuple[int, int] = (5, 11),
    orbit: OrbitPass = "BOTH",
    apply_border_noise: bool = True,
    apply_terrain_flattening: bool = True,
    terrain_model: TerrainModel = "VOLUME",
    dem: ee.Image | None = None,
    apply_speckle: bool = True,
    speckle_mode: SpeckleMode = "POST_COMPOSITE",
    speckle_filter: SpeckleMethod = "GAMMA_MAP",
    kernel_size: int = 5,
    enl: float | None = None,
    fill_masked_gaps: bool = True,
    include_ratio: bool = False,
) -> ee.Image:
    """Generate seasonal Sentinel-1 ARD gamma0 composites (``VH_dry, VH_wet, VV_dry, VV_wet``)."""
    ee_mod = _gee.require_ee("seasonal Sentinel-1 features")
    # If wet season wraps across New Year (e.g. Dec..Apr), start in year - 1 so it is contiguous
    if wet_months[0] > wet_months[1]:
        start_date = ee_mod.Date.fromYMD(year - 1, wet_months[0], 1)
    else:
        start_date = ee_mod.Date.fromYMD(year, 1, 1)
    end_date = ee_mod.Date.fromYMD(year + 1, 1, 1)
    mode = speckle_mode.upper()

    s1_ard = preprocess_s1_collection(
        start_date,
        end_date,
        aoi,
        orbit=orbit,
        apply_border_noise=apply_border_noise,
        apply_speckle_filter_per_scene=apply_speckle and mode == "MONO",
        speckle_filter=speckle_filter,
        kernel_size=kernel_size,
        enl=enl if enl is not None else 5.0,
        apply_terrain_flattening=apply_terrain_flattening,
        terrain_model=terrain_model,
        dem=dem,
        output_format="LINEAR",
    )

    wet_col = s1_ard.filter(ee_mod.Filter.calendarRange(wet_months[0], wet_months[1], "month"))
    dry_col = s1_ard.filter(ee_mod.Filter.calendarRange(dry_months[0], dry_months[1], "month"))
    wet_lin = wet_col.select(["VV", "VH"]).median()
    dry_lin = dry_col.select(["VV", "VH"]).median()

    if fill_masked_gaps and apply_terrain_flattening:
        fallback = (
            preprocess_s1_collection(
                start_date,
                end_date,
                aoi,
                orbit=orbit,
                apply_border_noise=apply_border_noise,
                apply_speckle_filter_per_scene=False,
                apply_terrain_flattening=False,
                output_format="LINEAR",
            )
            .map(
                lambda img: img.select(["VV", "VH"]).divide(
                    img.select("angle").multiply(_DEG2RAD).cos()
                )
            )
            .median()
        )
        wet_lin = wet_lin.unmask(dry_lin).unmask(fallback)
        dry_lin = dry_lin.unmask(wet_lin).unmask(fallback)

    def _finalize(comp_lin: Any, season_col: Any) -> Any:
        if apply_speckle and mode == "POST_COMPOSITE":
            enl_param: Any = (
                enl
                if enl is not None
                else season_col.select("VV").count().unmask(0).multiply(3.18).max(5)
            )
            comp_lin = apply_speckle_filter(
                comp_lin,
                method=speckle_filter,
                kernel_size=kernel_size,
                enl=enl_param,
                bands=("VV", "VH"),
            )
        return lin_to_db(comp_lin, ("VV", "VH"))

    wet_db = _finalize(wet_lin, wet_col)
    dry_db = _finalize(dry_lin, dry_col)

    vh_dry = dry_db.select("VH").rename("VH_dry")
    vh_wet = wet_db.select("VH").rename("VH_wet")
    vv_dry = dry_db.select("VV").rename("VV_dry")
    vv_wet = wet_db.select("VV").rename("VV_wet")
    out_bands = [vh_dry, vh_wet, vv_dry, vv_wet]
    if include_ratio:
        out_bands.append(vh_dry.subtract(vv_dry).rename("VH_VV_ratio_dry"))
        out_bands.append(vh_wet.subtract(vv_wet).rename("VH_VV_ratio_wet"))

    return ee_mod.Image.cat(out_bands).clip(aoi)


def get_scansar_features(
    year: int,
    aoi: ee.Geometry,
    *,
    apply_qa_mask: bool = True,
    mask_ocean: bool = False,
    fill_masked_gaps: bool = True,
    apply_speckle: bool = True,
    speckle_filter: SpeckleMethod = "GAMMA_MAP",
    kernel_size: int = 3,
    enl: float | None = None,
) -> ee.Image:
    """Annual ALOS-2 PALSAR-2 ScanSAR L2.2 gamma0 features (``ScanSAR_HH_dB, ScanSAR_HV_dB``)."""
    ee_mod = _gee.require_ee("ALOS-2 ScanSAR features")
    start_date = ee_mod.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, "year")

    raw_col = (
        ee_mod.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee_mod.Filter.listContains("Polarizations", "HH"))
        .filter(ee_mod.Filter.listContains("Polarizations", "HV"))
    )

    def _to_linear(img: Any, use_qa: bool) -> Any:
        pol = img.select(["HH", "HV"]).toFloat()
        pos_mask = pol.select("HH").gt(0).And(pol.select("HV").gt(0))
        lin = pol.pow(2).multiply(_PALSAR_LIN_CF).updateMask(pos_mask)
        if use_qa:
            msk_val = img.select("MSK").bitwiseAnd(7)
            qa_valid = msk_val.eq(1) if mask_ocean else msk_val.eq(1).Or(msk_val.eq(4))
            lin = lin.updateMask(qa_valid)
        return lin.copyProperties(img, ["system:time_start"])

    lin_col = raw_col.map(lambda img: _to_linear(img, apply_qa_mask))
    comp_lin = lin_col.median()

    if fill_masked_gaps and apply_qa_mask:
        fallback_lin = raw_col.map(lambda img: _to_linear(img, False)).median()
        comp_lin = comp_lin.unmask(fallback_lin)

    if apply_speckle:
        enl_param: Any = (
            enl
            if enl is not None
            else lin_col.select("HH").count().unmask(0).multiply(2.55).max(4)
        )
        comp_lin = apply_speckle_filter(
            comp_lin,
            method=speckle_filter,
            kernel_size=kernel_size,
            enl=enl_param,
            bands=("HH", "HV"),
        )

    return (
        lin_to_db(comp_lin, ("HH", "HV"))
        .select(["HH", "HV"], ["ScanSAR_HH_dB", "ScanSAR_HV_dB"])
        .clip(aoi)
    )


def get_palsar_backscatter(
    year: int,
    aoi: ee.Geometry | None = None,
    *,
    apply_qa_mask: bool = True,
    mask_ocean: bool = False,
    fill_masked_gaps: bool = True,
    apply_speckle: bool = True,
    speckle_filter: SpeckleMethod = "LEE",
    kernel_size: int = 3,
    enl: float = 4.0,
) -> ee.Image:
    """Annual ALOS PALSAR / PALSAR-2 L-band mosaic (``palsar_HH, palsar_HV, palsar_HH_HV_diff``)."""
    ee_mod = _gee.require_ee("ALOS PALSAR features")
    if year <= 2012:
        sar_col = "JAXA/ALOS/PALSAR/YEARLY/SAR"
        clamped_year = min(max(year, 2007), 2010)
    else:
        sar_col = "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH"
        clamped_year = min(max(year, 2015), 2024)

    def _calibrate(img: Any, use_qa: bool) -> Any:
        pol = img.select(["HH", "HV"]).toFloat()
        pos_mask = pol.select("HH").gt(0).And(pol.select("HV").gt(0))
        lin = pol.pow(2).multiply(_PALSAR_LIN_CF).updateMask(pos_mask)
        if use_qa:
            qa = img.select("qa")
            qa_valid = qa.eq(255) if mask_ocean else qa.eq(255).Or(qa.eq(50))
            lin = lin.updateMask(qa_valid)
        return lin

    target_col = ee_mod.ImageCollection(sar_col).filterDate(
        f"{clamped_year}-01-01", f"{clamped_year + 1}-01-01"
    )
    window_col = ee_mod.ImageCollection(sar_col).filterDate(
        f"{clamped_year - 1}-01-01", f"{clamped_year + 2}-01-01"
    )
    target_lin = target_col.map(lambda img: _calibrate(img, apply_qa_mask)).median()
    comp_lin = target_lin.unmask(
        window_col.map(lambda img: _calibrate(img, apply_qa_mask)).median()
    )
    if fill_masked_gaps and apply_qa_mask:
        comp_lin = comp_lin.unmask(window_col.map(lambda img: _calibrate(img, False)).median())

    if apply_speckle:
        comp_lin = apply_speckle_filter(
            comp_lin,
            method=speckle_filter,
            kernel_size=kernel_size,
            enl=enl,
            bands=("HH", "HV"),
        )

    db_img = lin_to_db(comp_lin.resample("bilinear"), ("HH", "HV"))
    hh_db = db_img.select("HH").rename("palsar_HH")
    hv_db = db_img.select("HV").rename("palsar_HV")
    diff_db = hh_db.subtract(hv_db).rename("palsar_HH_HV_diff")
    result = ee_mod.Image.cat([hh_db, hv_db, diff_db])
    return result.clip(aoi) if aoi is not None else result

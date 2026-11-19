"""Unit tests for standalone Evaluation & Multi-Epoch Trajectory Analysis (Ticket #05)."""

from __future__ import annotations

import builtins
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lc_framework.config import ClassDefinition, ClassSchema
from lc_framework.evaluation import (
    analyze_multi_epoch_rasters,
    build_sankey_data,
    compute_log_loss,
    compute_thematic_accuracy,
    compute_transition_summary,
    count_raster_pixels,
    evaluate_probability_stack,
    export_sankey_html,
    suggest_active_learning_samples,
)
from lc_framework.io import write_probability_stack


def _schema() -> ClassSchema:
    return ClassSchema(
        classes=[
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
            ClassDefinition(id=3, name="Water", color="#0288d1"),
        ]
    )


def _profile(height: int = 10, width: int = 10, pixel_size: float = 100.0) -> dict:
    # 100m x 100m pixels => 10,000 m^2 = 1.0 ha per pixel for clean hand-worked math
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "crs": "EPSG:32748",
        "transform": from_origin(500000.0, 9000000.0, pixel_size, pixel_size),
    }


def test_olofsson_area_adjusted_accuracy_matches_worked_example(tmp_path: Path) -> None:
    """Verify Olofsson et al. (2014) Eq. 1-11 against an independent hand-worked 2-class table.

    Map strata:
      Class 1 (Forest):   N_1 = 800 px (W_1 = 0.8), sample n_1. = 20 -> 18 correct, 2 ref=Cropland
      Class 2 (Cropland): N_2 = 200 px (W_2 = 0.2), sample n_2. = 20 -> 4 ref=Forest, 16 correct

    Hand-worked Olofsson values:
      p_11 = 0.8 * (18/20) = 0.72,  p_12 = 0.8 * (2/20) = 0.08
      p_21 = 0.2 * (4/20)  = 0.04,  p_22 = 0.2 * (16/20) = 0.16
      OA_area = p_11 + p_22 = 0.88
      p_.1 = 0.76 (760 ha at 1 ha/px), p_.2 = 0.24 (240 ha)
      UA_1 = 0.90, UA_2 = 0.80
      PA_area_1 = 0.72 / 0.76 = 0.9473684...
      PA_area_2 = 0.16 / 0.24 = 0.6666667...
      Var(OA_area) = 0.8^2 * 0.9 * 0.1 / 19 + 0.2^2 * 0.8 * 0.2 / 19
                   = (0.0576 + 0.0064) / 19 = 0.064 / 19
      SE(OA_area)  = sqrt(0.064 / 19) = 0.058038...
      Var(p_.1)    = 0.8^2 * 0.9 * 0.1 / 19 + 0.2^2 * 0.2 * 0.8 / 19 = 0.064 / 19
      SE(Area_1_ha)= 1000 * sqrt(0.064 / 19) = 58.038... ha
    """
    schema = ClassSchema(
        classes=[
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
        ]
    )
    y_pred = [1] * 20 + [2] * 20
    y_true = [1] * 18 + [2] * 2 + [1] * 4 + [2] * 16

    res = compute_thematic_accuracy(
        y_true=y_true,
        y_pred=y_pred,
        class_schema=schema,
        mapped_pixel_counts={1: 800, 2: 200},
        pixel_area_m2=10_000.0,  # 1.0 ha per pixel
        confidence_level=0.95,
    )

    expected_se = math.sqrt(0.064 / 19.0)
    assert res.overall_metrics["overall_accuracy_sample"] == 34 / 40
    assert math.isclose(res.overall_metrics["overall_accuracy_area_weighted"], 0.88, rel_tol=1e-6)
    assert math.isclose(res.overall_metrics["overall_accuracy_area_se"], expected_se, rel_tol=1e-6)
    assert math.isclose(
        res.overall_metrics["overall_accuracy_area_ci"], 1.9599639845 * expected_se, rel_tol=1e-5
    )

    cm = res.class_metrics.set_index("class_name")
    assert list(cm.index) == ["Forest", "Cropland"]
    assert math.isclose(cm.loc["Forest", "user_accuracy"], 0.90, rel_tol=1e-6)
    assert math.isclose(cm.loc["Cropland", "user_accuracy"], 0.80, rel_tol=1e-6)
    assert math.isclose(cm.loc["Forest", "producer_accuracy_area"], 0.72 / 0.76, rel_tol=1e-6)
    assert math.isclose(cm.loc["Cropland", "producer_accuracy_area"], 0.16 / 0.24, rel_tol=1e-6)
    assert math.isclose(cm.loc["Forest", "adjusted_area_ha"], 760.0, rel_tol=1e-6)
    assert math.isclose(cm.loc["Cropland", "adjusted_area_ha"], 240.0, rel_tol=1e-6)
    assert math.isclose(cm.loc["Forest", "adjusted_area_se_ha"], 1000.0 * expected_se, rel_tol=1e-5)

    saved = res.save_reports(tmp_path / "reports", prefix="test_eval")
    assert saved["overall_metrics_json"].exists()
    assert saved["class_metrics_csv"].exists()
    assert saved["confusion_matrix_counts_csv"].exists()
    assert saved["confusion_matrix_area_prop_csv"].exists()


def test_compute_log_loss_and_evaluate_probability_stack(tmp_path: Path) -> None:
    schema = _schema()
    # Known-good hand-worked log-loss: -0.5 * (ln(0.8) + ln(0.5))
    y_true = [1, 3]
    y_prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32)
    ll = compute_log_loss(y_true, y_prob, class_schema=schema)
    expected_ll = -0.5 * (math.log(0.8) + math.log(0.5))
    assert math.isclose(ll, expected_ll, rel_tol=1e-6)

    # End-to-end evaluation directly from a Class Probability Stack GeoTIFF
    probs = np.zeros((3, 6, 6), dtype=np.float32)
    probs[0, :, :3] = 0.85
    probs[1, :, :3] = 0.10
    probs[2, :, :3] = 0.05
    probs[0, :, 3:] = 0.05
    probs[1, :, 3:] = 0.15
    probs[2, :, 3:] = 0.80

    prob_path = tmp_path / "probs.tif"
    prof = _profile(6, 6, pixel_size=100.0)
    write_probability_stack(prob_path, probs, profile=prof, class_schema=schema)

    # Sample centers inside left half (Forest) and right half (Water)
    coords = [
        (500050.0, 8999950.0),
        (500150.0, 8999850.0),
        (500350.0, 8999950.0),
        (500450.0, 8999850.0),
    ]
    eval_res = evaluate_probability_stack(prob_path, coords, y_true=[1, 1, 3, 3])
    assert eval_res.overall_metrics["overall_accuracy_area_weighted"] == 1.0
    assert math.isclose(
        eval_res.overall_metrics["log_loss"],
        -0.5 * (math.log(0.85) + math.log(0.80)),
        rel_tol=1e-3,
    )


def test_active_learning_suggestions_rank_uncertainty_and_enforce_min_distance(
    tmp_path: Path,
) -> None:
    schema = _schema()
    probs = np.zeros((3, 10, 10), dtype=np.float32)
    # Default confident Forest everywhere (CI = 0.03/0.95, margin = 0.08, entropy = 0.21 < 0.5)
    probs[0] = 0.95
    probs[1] = 0.03
    probs[2] = 0.02
    # Four high-uncertainty pixels:
    # (1, 1) and (1, 2) are 100m apart (closer than min_distance_m=250) -> only highest kept
    probs[:, 1, 1] = [0.46, 0.45, 0.09]  # CI = 0.45/0.46 = 0.978
    probs[:, 1, 2] = [0.47, 0.44, 0.09]  # CI = 0.44/0.47 = 0.936 (too close to (1,1))
    # (8, 8) is far away (> 250m) -> kept
    probs[:, 8, 8] = [0.48, 0.43, 0.09]  # CI = 0.43/0.48 = 0.896
    # (8, 1) is far away, but we'll place an existing sample right on top of it -> excluded
    probs[:, 8, 1] = [0.45, 0.45, 0.10]  # CI = 1.0

    prob_path = tmp_path / "probs_al.tif"
    write_probability_stack(prob_path, probs, profile=_profile(10, 10, 100.0), class_schema=schema)

    existing_xy = [(500150.0, 8999150.0)]  # center of pixel (row=8, col=1)
    out_csv = tmp_path / "suggested.csv"

    for method in ("confusion_index", "margin", "entropy"):
        df = suggest_active_learning_samples(
            prob_path,
            n_samples=10,
            min_distance_m=250.0,
            min_uncertainty=0.5,
            existing_coords_xy=existing_xy,
            method=method,
            output_csv_path=out_csv if method == "confusion_index" else None,
        )
        assert len(df) == 2
        coords_rc = list(zip(df["row"].tolist(), df["col"].tolist(), strict=True))
        assert coords_rc == [(1, 1), (8, 8)]
        assert df["uncertainty"].iloc[0] >= df["uncertainty"].iloc[1]

    assert out_csv.exists()


def test_pontius_millones_transition_summary_and_multi_epoch_sankey_without_puyravaud(
    tmp_path: Path,
) -> None:
    """Verify Pontius & Millones (2011) Quantity/Allocation/Exchange/Shift & zero Puyravaud.

    10x10 raster at 1 ha/px (100 ha total):
      t1: 60 ha Forest (rows 0..5), 40 ha Cropland (rows 6..9)
      t2: 10 ha Forest -> Cropland (row 5), 5 ha Cropland -> Forest (row 6, cols 0..4)
      So transition matrix:
        Forest -> Forest:   50,  Forest -> Cropland:   10  (Loss=10, Gain=5, Net=-5)
        Cropland -> Forest:  5,  Cropland -> Cropland: 35  (Loss=5, Gain=10, Net=+5)
      Pontius & Millones:
        Forest:   Q = |5 - 10| = 5, A = 2*min(10,5) = 10, Exchange = 2*min(10,5) = 10, Shift = 0
        Cropland: Q = |10 - 5| = 5, A = 2*min(5,10) = 10, Exchange = 10, Shift = 0
        Overall:  Q = 5 ha (5%), A = 10 ha (10%), Total Disagreement = 15 ha (15%)
    """
    schema = ClassSchema(
        classes=[
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
        ]
    )
    t1 = np.ones((10, 10), dtype=np.uint8)
    t1[6:, :] = 2

    t2 = t1.copy()
    t2[5, :] = 2
    t2[6, :5] = 1

    t3 = t2.copy()
    t3[4, :] = 2  # another 10 ha Forest -> Cropland

    summary = compute_transition_summary(
        t1,
        t2,
        pixel_area_m2=10_000.0,
        t1_label="2015",
        t2_label="2020",
        class_schema=schema,
    )

    assert summary.transition_matrix_ha.loc["Forest", "Forest"] == 50.0
    assert summary.transition_matrix_ha.loc["Forest", "Cropland"] == 10.0
    assert summary.transition_matrix_ha.loc["Cropland", "Forest"] == 5.0
    assert summary.transition_matrix_ha.loc["Cropland", "Cropland"] == 35.0

    # Strictly NO Puyravaud / annual rate column
    assert all("puyravaud" not in c.lower() for c in summary.class_summary_ha.columns)
    assert all("annual_rate" not in c.lower() for c in summary.class_summary_ha.columns)

    assert summary.overall_disagreement["overall_quantity_disagreement_ha"] == 5.0
    assert summary.overall_disagreement["overall_allocation_disagreement_ha"] == 10.0
    assert summary.overall_disagreement["overall_exchange_ha"] == 10.0
    assert summary.overall_disagreement["overall_shift_ha"] == 0.0
    assert summary.overall_disagreement["overall_total_disagreement_ha"] == 15.0

    # Multi-epoch raster analysis + Sankey HTML export across 3 arbitrary epochs
    prof = _profile(10, 10, pixel_size=100.0)
    prof.update(count=1, dtype="uint8", nodata=0)
    raster_paths: dict[str, Path] = {}
    for label, arr in (("2015", t1), ("2020", t2), ("2025", t3)):
        p = tmp_path / f"lulc_{label}.tif"
        with rasterio.open(p, "w", **prof) as dst:
            dst.write(arr, 1)
        raster_paths[label] = p

    counts, pix_m2 = count_raster_pixels(raster_paths["2015"])
    assert counts == {1: 60, 2: 40}
    assert pix_m2 == 10_000.0

    traj = analyze_multi_epoch_rasters(raster_paths, class_schema=schema)
    assert traj.epochs == ["2015", "2020", "2025"]
    assert len(traj.interval_summaries) == 2
    assert traj.net_summary.t1_label == "2015"
    assert traj.net_summary.t2_label == "2025"

    sankey = build_sankey_data(traj, min_flow_ha=1.0)
    assert len(sankey["nodes"]) == 6  # 2 classes x 3 epochs
    assert len(sankey["links"]) > 0

    html_path = export_sankey_html(traj, tmp_path / "transitions.html")
    assert html_path.exists()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()


def test_unsampled_stratum_warning_and_new_class_svg_sankey_without_plotly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Olofsson warns loudly when a mapped stratum has pixels but zero validation samples
    with pytest.warns(UserWarning, match="zero validation samples"):
        res = compute_thematic_accuracy(
            y_true=[1, 1, 2, 2],
            y_pred=[1, 1, 2, 2],
            mapped_pixel_counts={1: 50, 2: 30, 3: 20},
            pixel_area_m2=10_000.0,
        )
    assert math.isclose(res.overall_metrics["unsampled_mapped_weight"], 0.20, rel_tol=1e-6)

    # 2. Class 3 appears for the first time in epoch 3 (without ClassSchema) + plotly blocked -> SVG
    prof = _profile(6, 6, pixel_size=100.0)
    prof.update(count=1, dtype="uint8", nodata=0)
    e1 = np.ones((6, 6), dtype=np.uint8)
    e2 = e1.copy()
    e2[3:, :] = 2
    e3 = e2.copy()
    e3[5, :] = 3  # brand-new class appearing only in epoch 3

    paths = {}
    for lbl, arr in (("2015", e1), ("2020", e2), ("2025", e3)):
        p = tmp_path / f"e_{lbl}.tif"
        with rasterio.open(p, "w", **prof) as dst:
            dst.write(arr, 1)
        paths[lbl] = p

    traj = analyze_multi_epoch_rasters(paths)
    sankey = build_sankey_data(traj, min_flow_ha=0.0)
    assert len(sankey["nodes"]) == 9  # 3 classes across 3 epochs, no KeyError

    real_import = builtins.__import__

    def _block_plotly(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plotly" or name.startswith("plotly."):
            raise ImportError("plotly blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_plotly)
    for mod in list(sys.modules):
        if mod == "plotly" or mod.startswith("plotly."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    svg_html = export_sankey_html(traj, tmp_path / "svg_sankey.html")
    content = svg_html.read_text(encoding="utf-8")
    assert "<svg " in content
    assert "<path " in content

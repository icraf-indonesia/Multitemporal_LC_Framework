"""Import-isolation tests enforcing ADR-0005 standalone modularity."""

from __future__ import annotations

import subprocess
import sys


def _run_isolated_import(statement: str, blocked: tuple[str, ...]) -> None:
    """Execute an import statement in a fresh interpreter with optional deps blocked."""
    code = f"""
import builtins, sys
BLOCKED = {set(blocked)!r}
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split('.')[0]
    if root in BLOCKED:
        raise ImportError(f"Blocked optional dependency: {{name}}")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
for mod in list(sys.modules):
    if mod.split('.')[0] in BLOCKED:
        del sys.modules[mod]
{statement}
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Isolated import failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    )


def test_core_package_and_config_import_without_optional_dependencies() -> None:
    optional_deps = ("ee", "luma_ge", "sklearn", "lightgbm", "xgboost", "skimage", "geopandas")
    _run_isolated_import(
        "import lc_framework; "
        "from lc_framework.config import ProjectConfig, ClassSchema, ClassDefinition; "
        "from lc_framework.io import read_probability_stack, write_probability_stack, "
        "read_segment_raster, repair_gee_int32_overflow; "
        "import lc_framework.luma_adapter",
        blocked=optional_deps,
    )


def test_assembly_imports_and_runs_without_gee_luma_or_ml() -> None:
    optional_deps = ("ee", "luma_ge", "sklearn", "lightgbm", "xgboost", "skimage", "geopandas")
    _run_isolated_import(
        "import numpy as np; "
        "from lc_framework.assembly import ("
        "blend_pixel_and_object_probabilities, bayesian_smooth, apply_mmu_array, "
        "derive_hard_labels, heal_array_seams, run_hybrid_assembly"
        "); "
        "p = np.full((2, 5, 5), 0.2, dtype=np.float32); p[0] = 0.8; "
        "s = np.ones((5, 5), dtype=np.int32); "
        "res = blend_pixel_and_object_probabilities(p, s, heal_segment_seams=False); "
        "sm = bayesian_smooth(res.probabilities, window_size=3); "
        "lbl = derive_hard_labels(sm); "
        "mmu = apply_mmu_array(lbl, min_patch_sizes={1: 2}); "
        "assert mmu.shape == (5, 5)",
        blocked=optional_deps,
    )


def test_evaluation_imports_and_runs_without_gee_luma_ml_or_plotly() -> None:
    optional_deps = (
        "ee",
        "luma_ge",
        "sklearn",
        "lightgbm",
        "xgboost",
        "skimage",
        "geopandas",
        "plotly",
    )
    _run_isolated_import(
        "import numpy as np; "
        "from lc_framework.evaluation import ("
        "compute_thematic_accuracy, compute_log_loss, compute_transition_summary, "
        "build_sankey_data, export_sankey_html, suggest_active_learning_samples"
        "); "
        "res = compute_thematic_accuracy([1, 1, 2, 2], [1, 2, 2, 2], "
        "mapped_pixel_counts={1: 100, 2: 100}, pixel_area_m2=100.0); "
        "assert res.overall_metrics['overall_accuracy_area_weighted'] > 0; "
        "ll = compute_log_loss([1, 2], np.array([[0.8, 0.2], [0.1, 0.9]]), class_ids=[1, 2]); "
        "assert ll > 0; "
        "tr = compute_transition_summary("
        "np.array([[1, 1], [2, 2]]), np.array([[1, 2], [2, 2]]), "
        "pixel_area_m2=10000.0, t1_label='2020', t2_label='2025'); "
        "assert tr.overall_disagreement['overall_total_disagreement_ha'] == 1.0",
        blocked=optional_deps,
    )


def test_rfecv_runs_without_gee_or_luma_and_artifact_loads_without_sklearn() -> None:
    # SelectedFeaturesArtifact loads with zero optional deps (even without sklearn or ee)
    _run_isolated_import(
        "from lc_framework.features import SelectedFeaturesArtifact, load_selected_features; "
        "art = SelectedFeaturesArtifact("
        "selected_features=('ndvi_dry', 'mndwi_dry'), "
        "all_candidate_features=('ndvi_dry', 'mndwi_dry', 'noise'), "
        "scoring='balanced_accuracy', best_score=0.92, n_samples=60); "
        "assert art.selected_features == ('ndvi_dry', 'mndwi_dry')",
        blocked=("ee", "luma_ge", "sklearn", "lightgbm", "xgboost", "skimage", "geopandas"),
    )
    # Full RFECV runs standalone with GEE and luma_ge completely blocked
    _run_isolated_import(
        "import numpy as np, pandas as pd; "
        "from lc_framework.features import run_rfecv_selection; "
        "rng = np.random.default_rng(0); "
        "X = pd.DataFrame({'f1': np.r_[rng.normal(2,0.2,12), rng.normal(-2,0.2,12)], "
        "'f2': rng.normal(0,1,24)}); "
        "y = pd.Series([1]*12 + [2]*12); "
        "res = run_rfecv_selection(X, y, cv=2, n_estimators=10, n_jobs=1); "
        "assert 'f1' in res.selected_features",
        blocked=("ee", "luma_ge", "lightgbm", "xgboost", "skimage", "geopandas"),
    )


def test_temporal_and_features_import_lazily_without_ee() -> None:
    _run_isolated_import(
        "import sys; import lc_framework.features as features; "
        "assert 'lc_framework.temporal' not in sys.modules; "
        "spec = features.FeatureStackSpec(); assert 'NDVI' in spec.indices; "
        "import lc_framework.temporal as temporal; "
        "p = temporal.CCDCParams.sentinel2(); prof = temporal.CCDCProfile.standard(); "
        "assert prof.coef_band_name('nir') == 'nir_coefs'; "
        "assert temporal.OPTICAL_BANDS == ('blue', 'green', 'red', 'nir', 'swir1', 'swir2')",
        blocked=("ee", "luma_ge", "sklearn", "lightgbm", "xgboost", "skimage", "geopandas"),
    )


def test_obia_imports_lazily_without_optional_dependencies() -> None:
    _run_isolated_import(
        "import sys, numpy as np; import lc_framework.obia as obia; "
        "assert 'lc_framework.temporal' not in sys.modules; "
        "assert 'lc_framework.features' not in sys.modules; "
        "assert 'lc_framework.assembly' not in sys.modules; "
        "assert 'lc_framework.evaluation' not in sys.modules; "
        "assert 'lc_framework.luma_adapter' not in sys.modules; "
        "snic = obia.GEESnicSegmenter(obia.SNICParams(size=10)); "
        "slic = obia.LocalSlicSegmenter(obia.SLICParams(n_segments=20)); "
        "assert isinstance(snic, obia.Segmenter); "
        "assert isinstance(slic, obia.Segmenter); "
        "m = obia.valid_segment_mask(np.array([[0, -1, 5, -9]], dtype=np.int32)); "
        "assert m.tolist() == [[False, False, True, True]]",
        blocked=("ee", "luma_ge", "sklearn", "lightgbm", "xgboost", "skimage", "geopandas"),
    )


def test_classification_imports_lazily_without_optional_dependencies() -> None:
    _run_isolated_import(
        "import sys; import lc_framework.classification as cls; "
        "assert 'lc_framework.temporal' not in sys.modules; "
        "assert 'lc_framework.features' not in sys.modules; "
        "assert 'lc_framework.obia' not in sys.modules; "
        "assert 'lc_framework.assembly' not in sys.modules; "
        "assert 'lc_framework.evaluation' not in sys.modules; "
        "assert 'lc_framework.luma_adapter' not in sys.modules; "
        "p = cls.GEERFParams(number_of_trees=100); "
        "assert p.to_ee_dict()['numberOfTrees'] == 100; "
        "grid = cls.default_param_space('random_forest'); "
        "assert 'n_estimators' in grid",
        blocked=("ee", "luma_ge", "sklearn", "lightgbm", "xgboost", "skimage", "geopandas"),
    )

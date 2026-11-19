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


def test_temporal_ccdc_specs_import_lazily_without_ee() -> None:
    _run_isolated_import(
        "from lc_framework.temporal.ccdc.specs import CCDCParams, CCDCProfile, ReflectanceBounds; "
        "p = CCDCParams.sentinel2(); prof = CCDCProfile.standard(); "
        "assert prof.coef_band_name('nir') == 'nir_coefs'",
        blocked=("ee", "luma_ge", "sklearn"),
    )

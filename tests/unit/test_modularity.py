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


def test_temporal_ccdc_specs_import_lazily_without_ee() -> None:
    _run_isolated_import(
        "from lc_framework.temporal.ccdc.specs import CCDCParams, CCDCProfile, ReflectanceBounds; "
        "p = CCDCParams.sentinel2(); prof = CCDCProfile.standard(); "
        "assert prof.coef_band_name('nir') == 'nir_coefs'",
        blocked=("ee", "luma_ge", "sklearn"),
    )

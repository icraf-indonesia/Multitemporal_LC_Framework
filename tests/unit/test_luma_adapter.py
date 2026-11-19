"""Tests for the external luma-stack adapter (ADR-0001, ADR-0004)."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lc_framework.config import ClassDefinition, ClassSchema, EpochConfig, ProjectConfig
from lc_framework.luma_adapter import (
    build_luma_multitemporal_analysis,
    normalize_luma_probability_image,
    schema_from_luma_manager,
    schema_to_luma_classes,
    schema_to_luma_manager,
)


def _schema() -> ClassSchema:
    return ClassSchema(
        classes=[
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
            ClassDefinition(id=3, name="Water", color="#0288d1"),
        ]
    )


def _project_config() -> ProjectConfig:
    return ProjectConfig(
        project_name="demo_mt",
        crs="EPSG:32748",
        scale_m=30,
        class_schema=_schema(),
        epochs=(
            EpochConfig(label="2015", start_date="2015-01-01", end_date="2015-12-31", sensor="L8"),
            EpochConfig(label="2025", start_date="2025-01-01", end_date="2025-12-31", sensor="S2"),
        ),
    )


def test_missing_luma_ge_raises_actionable_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _block_luma(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "luma_ge" or name.startswith("luma_ge."):
            raise ImportError("No module named 'luma_ge'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_luma)
    for key in list(sys.modules):
        if key == "luma_ge" or key.startswith("luma_ge."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    with pytest.raises(ImportError, match="luma-stack"):
        schema_to_luma_manager(_schema())


def test_bidirectional_class_schema_and_luma_scheme_manager_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLulcSchemeManager:
        def __init__(self) -> None:
            self.classes: list[dict] = []

        def add_class(self, class_id: int, class_name: str, color_code: str) -> tuple[bool, str]:
            self.classes.append(
                {"ID": int(class_id), "Class Name": class_name, "Color Code": color_code}
            )
            return True, "ok"

    fake_mod = SimpleNamespace(LULC_Scheme_Manager=FakeLulcSchemeManager)
    monkeypatch.setitem(sys.modules, "luma_ge", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "luma_ge.classification_scheme", fake_mod)

    schema = _schema()
    luma_dicts = schema_to_luma_classes(schema)
    assert luma_dicts == [
        {"ID": 1, "Class Name": "Forest", "Color Code": "#1b5e20"},
        {"ID": 2, "Class Name": "Cropland", "Color Code": "#fbc02d"},
        {"ID": 3, "Class Name": "Water", "Color Code": "#0288d1"},
    ]

    manager = schema_to_luma_manager(schema)
    assert len(manager.classes) == 3

    roundtripped = schema_from_luma_manager(manager)
    assert roundtripped == schema


def test_normalize_luma_probability_image_renames_and_scales_for_uint16_and_float() -> None:
    schema = ClassSchema(
        classes=[
            ClassDefinition(id=1, name="Primary Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Built-up", color="#e53935"),
            ClassDefinition(id=3, name="Water", color="#0288d1"),
        ]
    )

    # Automatic sanitized prob_<name> mode -> float32 [0, 1] with clamping
    fake_img = MagicMock(name="luma_soft_img")
    selected = fake_img.select.return_value
    clamped = selected.divide.return_value.clamp.return_value
    renamed_f32 = clamped.toFloat.return_value.rename.return_value

    out_f32 = normalize_luma_probability_image(
        fake_img,
        schema,
        band_naming="name",
        source_scale=100.0,
        as_uint16=False,
    )
    assert out_f32 is renamed_f32
    fake_img.select.assert_called_once_with(
        ["prob_Primary_Forest", "prob_Built_up", "prob_Water"]
    )
    selected.divide.assert_called_once_with(100.0)
    selected.divide.return_value.clamp.assert_called_once_with(0.0, 1.0)
    clamped.toFloat.return_value.rename.assert_called_once_with(["prob_1", "prob_2", "prob_3"])

    # uint16 EECU-optimized scaling: divide(100).clamp(0,1).multiply(10000).round().toUint16()
    fake_img2 = MagicMock(name="luma_soft_img_u16")
    sel2 = fake_img2.select.return_value
    clamped2 = sel2.divide.return_value.clamp.return_value
    u16_renamed = clamped2.multiply.return_value.round.return_value.toUint16.return_value.rename
    out_u16 = normalize_luma_probability_image(
        fake_img2,
        schema,
        source_scale=100.0,
        as_uint16=True,
    )
    assert out_u16 is u16_renamed.return_value
    fake_img2.select.assert_called_once_with(["prob_1", "prob_2", "prob_3"])
    sel2.divide.return_value.clamp.assert_called_once_with(0.0, 1.0)
    clamped2.multiply.assert_called_once_with(10000.0)
    u16_renamed.assert_called_once_with(["prob_1", "prob_2", "prob_3"])


def test_build_luma_multitemporal_analysis_delegates_epochs_from_project_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    acquire_calls: list[dict] = []

    class FakePeriodConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.label = kwargs["label"]

    class FakeMultitemporalAnalysis:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def acquire_imagery(self, **kwargs) -> dict:
            acquire_calls.append(kwargs)
            return {"ok": True}

    fake_mt_mod = SimpleNamespace(
        PeriodConfig=FakePeriodConfig,
        MultitemporalAnalysis=FakeMultitemporalAnalysis,
    )
    monkeypatch.setitem(sys.modules, "luma_ge", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "luma_ge.multitemporal", fake_mt_mod)

    cfg = _project_config()
    fake_aoi = MagicMock(name="ee_aoi")
    mt = build_luma_multitemporal_analysis(
        cfg,
        aoi=fake_aoi,
        reference_period="2025",
        acquire=True,
        acquire_kwargs={"verbose": False},
    )

    assert isinstance(mt, FakeMultitemporalAnalysis)
    assert captured["aoi"] is fake_aoi
    assert captured["reference_period"] == "2025"
    assert captured["scale"] == 30
    assert [p.label for p in captured["periods"]] == ["2015", "2025"]
    assert acquire_calls == [{"verbose": False}]

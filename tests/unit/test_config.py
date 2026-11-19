"""Tests for ProjectConfig and ClassSchema loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lc_framework.config import ClassDefinition, ClassSchema, ProjectConfig


def _sample_config_dict() -> dict:
    return {
        "project_name": "test_basin_2025",
        "crs": "EPSG:32748",
        "scale_m": 10,
        "aoi_asset": "projects/my-gee/assets/aoi_basin",
        "export_prefix": "projects/my-gee/assets/lc_2025",
        "seasons": {"wet": 0.16, "dry": 0.65, "harmonic_mid": 0.50},
        "epochs": [
            {
                "label": "2020",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "sensor": "S2",
            },
            {
                "label": "2025",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "sensor": "S2",
            },
        ],
        "class_schema": [
            {"id": 1, "name": "Forest", "color": "#1b5e20"},
            {"id": 2, "name": "Cropland", "color": "#fbc02d"},
            {"id": 3, "name": "Water", "color": "#0288d1"},
        ],
    }


def test_class_schema_orders_by_id_and_builds_canonical_band_names() -> None:
    schema = ClassSchema(
        classes=[
            ClassDefinition(id=3, name="Water", color="#0288d1"),
            ClassDefinition(id=1, name="Forest", color="#1b5e20"),
            ClassDefinition(id=2, name="Cropland", color="#fbc02d"),
        ]
    )

    assert schema.class_ids == (1, 2, 3)
    assert schema.class_names == ("Forest", "Cropland", "Water")
    assert schema.gee_prob_band_names == ("prob_1", "prob_2", "prob_3")
    assert schema.geotiff_band_descriptions == ("1:Forest", "2:Cropland", "3:Water")
    assert schema.name_for(2) == "Cropland"
    assert schema.id_for("Water") == 3
    assert schema.palette == ("#1b5e20", "#fbc02d", "#0288d1")


def test_class_schema_rejects_duplicate_ids_duplicate_names_and_bad_hex_color() -> None:
    with pytest.raises(ValueError, match="Duplicate class id"):
        ClassSchema(
            classes=[
                ClassDefinition(id=1, name="Forest", color="#1b5e20"),
                ClassDefinition(id=1, name="Water", color="#0288d1"),
            ]
        )

    with pytest.raises(ValueError, match="Duplicate class name"):
        ClassSchema(
            classes=[
                ClassDefinition(id=1, name="Forest", color="#1b5e20"),
                ClassDefinition(id=2, name="forest", color="#0288d1"),
            ]
        )

    with pytest.raises(ValueError, match="color"):
        ClassDefinition(id=1, name="Forest", color="not-a-hex")


def test_project_config_loads_from_yaml_and_json_roundtrip(tmp_path: Path) -> None:
    data = _sample_config_dict()
    yaml_path = tmp_path / "project.yaml"
    json_path = tmp_path / "project.json"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    json_path.write_text(json.dumps(data), encoding="utf-8")

    cfg_yaml = ProjectConfig.from_yaml(yaml_path)
    cfg_json = ProjectConfig.from_json(json_path)

    assert cfg_yaml == cfg_json
    assert cfg_yaml.project_name == "test_basin_2025"
    assert cfg_yaml.crs == "EPSG:32748"
    assert cfg_yaml.scale_m == 10
    assert cfg_yaml.class_schema.class_ids == (1, 2, 3)
    assert [e.label for e in cfg_yaml.epochs] == ["2020", "2025"]

    out_yaml = tmp_path / "saved.yaml"
    cfg_yaml.to_yaml(out_yaml)
    assert ProjectConfig.from_yaml(out_yaml) == cfg_yaml


def test_project_config_rejects_invalid_seasonal_fractions_and_nonpositive_scale() -> None:
    bad_season = _sample_config_dict()
    bad_season["seasons"] = {"wet": 1.25}
    with pytest.raises(ValueError, match="season"):
        ProjectConfig.model_validate(bad_season)

    bad_scale = _sample_config_dict()
    bad_scale["scale_m"] = 0
    with pytest.raises(ValueError):
        ProjectConfig.model_validate(bad_scale)

    bad_epoch = _sample_config_dict()
    bad_epoch["epochs"] = [
        {"label": "bad", "start_date": "2025-12-31", "end_date": "2025-01-01", "sensor": "S2"}
    ]
    with pytest.raises(ValueError, match="end_date"):
        ProjectConfig.model_validate(bad_epoch)


def test_example_project_yaml_is_valid_and_study_area_agnostic() -> None:
    example_path = Path(__file__).resolve().parents[2] / "configs" / "example_project.yaml"
    assert example_path.exists()
    cfg = ProjectConfig.from_yaml(example_path)
    assert len(cfg.class_schema.classes) >= 3
    assert "timor" not in cfg.project_name.lower()

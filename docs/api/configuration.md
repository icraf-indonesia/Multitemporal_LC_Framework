# Project Configuration & Class Schema

`LC_Framework` keeps all study-area specifics (AOI, projection, spatial resolution, mapping epochs, seasonal offsets, and land cover classes) outside library code in a validated YAML or JSON **`ProjectConfig`** file.

## Loading a Configuration

```python
from lc_framework.config import ProjectConfig

cfg = ProjectConfig.from_yaml("configs/example_project.yaml")

print(cfg.class_schema.class_ids)                 # (1, 2, 3, 4, 5)
print(cfg.class_schema.gee_prob_band_names)       # ('prob_1', 'prob_2', ...)
print(cfg.class_schema.geotiff_band_descriptions) # ('1:Forest', '2:Agroforestry', ...)
```

## Key Fields

| Field | Required | Description |
| :--- | :---: | :--- |
| `project_name` | Yes | Short identifier for the study area / run. |
| `crs` | Yes | Target coordinate reference system (e.g. `EPSG:32748`). |
| `scale_m` | Yes | Pixel resolution in meters (`> 0`). |
| `class_schema` | Yes | Ordered list of `{id, name, color}` definitions (`#RRGGBB`). Duplicate IDs or names are rejected. |
| `aoi_asset` | No | GEE FeatureCollection asset ID for the study area. |
| `export_prefix` | No | GEE asset or Drive folder prefix for exported artifacts. |
| `seasons` | No | Fractional-year offsets in `[0.0, 1.0)` (default `wet: 0.16`, `dry: 0.65`, `harmonic_mid: 0.50`). |
| `epochs` | No | List of `{label, start_date, end_date, sensor, cloud_cover, cs_threshold}` periods. |
| `smoothness_weights` | No | Optional per-class spatial smoothness weights (keys must match `class_schema` names). |

## Standalone Usage (Without `ProjectConfig`)

Following [ADR-0005](../adr/0005-standalone-modular-subpackages.md), downstream modules such as `lc_framework.io`, `lc_framework.assembly`, and `lc_framework.evaluation` also work with a standalone `ClassSchema` or plain file paths — you never have to create a full `ProjectConfig` just to run Classification Assembly on an existing GeoTIFF.

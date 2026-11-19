# Integrating `luma-stack` (`luma_ge`) with `LC_Framework`

Per [ADR-0001](../adr/0001-luma-stack-external-dependency.md) and [ADR-0004](../adr/0004-ccdc-owned-luma-multitemporal-delegated.md), `luma-stack` stays an **external** Python package integrated through `lc_framework.luma_adapter`. You do **not** need `luma-stack` installed to use standalone stages like `lc_framework.assembly` or `lc_framework.evaluation`.

## Installing `luma-stack` in Google Colab

```bash
pip install "lc-framework[gee]"
pip install git+https://github.com/<your-org>/luma-stack.git
```

## 1. Sharing a `ClassSchema` with `LULC_Scheme_Manager`

```python
from lc_framework.config import ProjectConfig
from lc_framework.luma_adapter import schema_from_luma_manager, schema_to_luma_manager

cfg = ProjectConfig.from_yaml("configs/example_project.yaml")

# LC_Framework -> luma-stack
luma_scheme = schema_to_luma_manager(cfg.class_schema)

# luma-stack -> LC_Framework
recovered_schema = schema_from_luma_manager(luma_scheme)
```

## 2. Normalizing `luma-stack` Soft Classification Outputs

`luma-stack`'s `Generate_LULC.soft_classification()` defaults to integer percentages (`probability_scale=100`) and may name bands `prob_<name>`. Normalize it into an EECU-efficient `uint16` `[0, 10000]` `Class Probability Stack` before exporting to Drive/Assets for `lc_framework.assembly`:

```python
from lc_framework.luma_adapter import normalize_luma_probability_image

prob_stack_ee = normalize_luma_probability_image(
    luma_soft_img,
    cfg.class_schema,
    prob_band_map=model_info["prob_band_map"],
    source_scale=100.0,
    as_uint16=True,
)
```

## 3. Delegating Direct Multitemporal Analysis

When you want direct multi-period composites instead of CCDC `Temporal Segment` reconstruction, delegate to `luma-stack`:

```python
from lc_framework.luma_adapter import build_luma_multitemporal_analysis

mt = build_luma_multitemporal_analysis(
    cfg,
    aoi=aoi_geom,
    reference_period="2025",
)
```

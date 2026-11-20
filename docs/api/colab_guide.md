# Colab Troubleshooting & EECU Optimization Guide

This guide documents how to run `LC_Framework` cleanly in Google Colab with minimal package conflicts and low Google Earth Engine (GEE) EECU/memory footprint.

---

## 1. Modular Colab Installation Slices

Install **only the extra needed by your notebook** (`ADR-0005`). Avoid installing GEE, boosting frameworks, or geospatial vector stacks inside notebooks that only perform local assembly or accuracy evaluation.

| Workflow | Colab `pip` Command |
| :--- | :--- |
| **Config, I/O, Hybrid Assembly & Evaluation** | `pip install -q "git+https://github.com/<your-org>/LC_Framework.git"` |
| **GEE CCDC, Features, AlphaEarth, GEE RF, GEE SNIC** | `pip install -q "lc-framework[gee] @ git+https://github.com/<your-org>/LC_Framework.git"` |
| **RFECV & Local ML (`RandomForest`, `ExtraTrees`, `LightGBM`, `XGBoost`)** | `pip install -q "lc-framework[ml] @ git+https://github.com/<your-org>/LC_Framework.git"` |
| **Local SLIC & Pure `Geo-Object Classification`** | `pip install -q "lc-framework[obia] @ git+https://github.com/<your-org>/LC_Framework.git"` |
| **Interactive Plotly Sankey** | `pip install -q "lc-framework[viz] @ git+https://github.com/<your-org>/LC_Framework.git"` |
| **External `luma-stack` Integration** | `pip install -q "lc-framework[gee] @ git+https://github.com/<your-org>/LC_Framework.git" "git+https://github.com/<your-org>/luma-stack.git"` |

> [!TIP]
> If Colab raises a version conflict after installing `[ml]` or `[obia]` on top of pre-installed Colab wheels, restart the Colab runtime **once** right after `%pip install` so compiled `numpy`/`scipy`/`rasterio` C-extensions reload cleanly.

---

## 2. GEE EECU & Worker Memory Optimization

### A. Export Probabilities as `uint16` (`[0, 10000]`, `nodata=65535`)
Floating-point `float32` national multi-class probability mosaics consume twice the EECU computation/egress and Drive storage of quantized `uint16`.
- **`gee_rf.export_probability_tiles`**: defaults to `encoding="uint16"`, scaling `[0.0, 1.0]` probabilities by `10000`, unmasking background pixels to `65535`, and setting `formatOptions={"cloudOptimized": True, "noData": 65535}`. When `manifest_dir` is provided, writes validated `<tile>.manifest.json` files to disk (and always returns the validated `ProbabilityManifest` objects in `result.manifests`).
- **Precision preserved**: `uint16` `[0..10000]` retains `0.0001` (0.01%) probability granularity — far more accurate than `uint8` `0..100` (1% steps) while staying compact.

### B. Use Tiled Exports with Idempotent `skip_existing` Resumability
Never force a monolithic multi-gigapixel AOI through a single Drive export when GEE workers hit memory or 2 GB file caps:
- `export_ccdc_tiles`, `export_probability_tiles`, and `GEESnicSegmenter.export_tiles` automatically subdivide `bounds` via `generate_tile_grid(bounds, tile_size_deg)` and skip completed assets when `existing_assets` is supplied (`skip_existing=True`).
- Always test a tile plan with **`dry_run=True`** first before starting background tasks.

### C. Match Pyramiding Policy to Data Semantics
Wrong pyramiding silently corrupts downstream zoom levels and reprojected reads:

| Artifact | Pyramiding Policy | Why |
| :--- | :--- | :--- |
| **`Class Probability Stack`** | `{".default": "mean"}` | Continuous `[0, 10000]` probabilities average smoothly across pyramid levels |
| **`Segment Label Raster`** (SNIC) | `{".default": "mode"}` or `"sample"` | Nearest/mode keeps integer segment IDs discrete without creating fake blended IDs |
| **CCDC `Temporal Segment` Arrays** | `{".default": "sample"}` | Required by GEE for variable-length array-valued bands (`tStart`, `tEnd`, `*_coefs`) |

### D. Prevent SNIC `int32` Clamping Overflow Across Tiles
Exporting GEE SNIC tiles without ID offsetting causes cross-tile ID collisions; offsetting raw 64-bit SNIC IDs directly can clamp at `2,147,483,647` (`int32` max) and fuse adjacent objects:
- `GEESnicSegmenter` hashes raw cluster IDs modulo `hash_modulus=200_000_000` and adds deterministic `tile_index * 300_000_000`.
- On local read, `read_segment_raster(..., repair_overflow=True)` and `GEESnicSegmenter.load_exported_segment_raster` automatically isolate any clamped islands via connected-component labeling and reset oversized mega-clusters to `0`.

### E. Offload Feature Selection & Heavy ML to Colab CSV Workflows
Avoid running iterative RFECV or hyperparameter grid searches inside GEE's `.getInfo()` budget:
1. Sample pixels once with `extract_sample_features(...)` + `export_sample_features_to_drive(...)` (`lc_framework.features`).
2. Run `run_rfecv_from_csv(...)` (`lc_framework.features.rfecv`) locally in Colab to emit `rfecv_selected_features.json`.
3. Feed that lightweight JSON directly into either `train_gee_rf(...)` (only selected bands enter GEE RF) or `tune_pixel_classifier(...)` (`lc_framework.classification.local_ml`).

### F. AlphaEarth `Sample Augmentation` Scaling Safeguards
- Over AOIs larger than a few hundred $\text{km}^2$, call `export_augmented_samples_to_asset` / `export_augmented_samples_to_drive` directly rather than `.getInfo()` interactively.
- If concurrent GEE aggregations hit quota errors on many classes, set `SampleAugmentationParams(materialize_per_class=True, wait_seconds=2.0)` to stagger similarity computation per class with `bestEffort=True`.

---

## 3. Common Colab Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `ValueError: Segment Label Raster grid does not match reference_profile` | GEE SNIC raster (e.g. 10 m) differs in grid/extent from probability or feature COG (e.g. 20 m) | Pass `align_to_reference=True` to `read_segment_raster`, `GEESnicSegmenter.load_exported_segment_raster`, `assign_segment_labels`, or `project_segment_probabilities` (`run_hybrid_assembly`, `pool_features_by_segments`, and `run_geobia_classification` already enable nearest-neighbor `WarpedVRT` alignment by default) |
| `ImportError: ... Install it with: pip install 'lc-framework[...]'` | Optional stage dependency not included in lean core | Install the indicated extra (`[gee]`, `[ml]`, `[obia]`, or `[viz]`) |
| Colab RAM exhaustion during local raster inference, masking, or pixel counting | Full-scene float32 arrays or large windows exceed VM RAM | Lower `block_size=256` in `predict_raster_probabilities` (default `512`) or `block_size=1024` in `apply_mask_rules`, `count_raster_pixels`, and `suggest_active_learning_samples` (default `2048`); for `pool_features_by_segments` / `LocalSlicSegmenter`, tile huge scenes spatially first |
| `XGBoost` complains `Expected [0 1 2], got [1 2 5]` on custom estimator | Non-contiguous `ClassSchema` IDs passed directly | Use `train_pixel_classifier` / `tune_pixel_classifier`, which automatically handle `LabelEncoder` translation (even inside `Pipeline`) |
| `evaluate_probability_stack` reports `Class_1..Class_N` names or `Reference class ID ... not found` | External GeoTIFF lacks a sidecar `*.manifest.json`, so `read_probability_stack` falls back to sequential `1..N` IDs (`Class_1..Class_N`) | Pass explicit `class_schema=cfg.class_schema` to `read_probability_stack` or `evaluate_probability_stack` |
| Validation accuracy suspiciously inflated after `Sample Augmentation` | Embedding-discovered points leaked into validation split | Filter out `sample_source == "alphaearth_similarity"` before computing Olofsson accuracy & log-loss |

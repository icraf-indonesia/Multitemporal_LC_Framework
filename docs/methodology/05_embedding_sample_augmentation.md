# AlphaEarth Embedding Sample Augmentation Methodology

## 1. Purpose in `LC_Framework`

Training sample scarcity — especially for rare land-cover strata or newly mapped epochs — frequently limits probabilistic classifiers. `lc_framework.samples` implements **`Sample Augmentation`** over Google's Annual Satellite Embedding dataset (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), discovering high-similarity candidate training locations across an area of interest while preserving canonical **`Class Schema`** governance and explicit sample provenance (`sample_source="alphaearth_similarity"`).

---

## 2. Prototype Cosine Similarity (Per-Sample vs. Mean Centroid)

Each pixel in `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` encodes multi-sensor seasonal/structural appearance as a 64-dimensional unit-norm vector $\mathbf{e}(x) \in \mathbb{R}^{64}$ ($\|\mathbf{e}(x)\|_2 = 1$). Because all vectors are unit-normalized, the dot product gives cosine similarity directly:

$$s(\mathbf{e}_i, \mathbf{e}(x)) = \langle \mathbf{e}_i,\, \mathbf{e}(x) \rangle \in [-1, 1]$$

Following Google's reference AlphaEarth similarity-search formulation (`compute_class_cosine_similarity`), for class $c$ with $N_c$ reference points $\{\mathbf{e}_{c,1}, \dots, \mathbf{e}_{c,N_c}\}$, `LC_Framework` computes a separate similarity image for each prototype point and averages across points:

$$\bar{S}_c(x) = \frac{1}{N_c} \sum_{i=1}^{N_c} \langle \mathbf{e}_{c,i},\, \mathbf{e}(x) \rangle$$

While $\bar{S}_c(x)$ equals the dot product with the unnormalized class mean vector mathematically inside Euclidean space, computing per-sample images in Earth Engine keeps individual prototype handling clean and lets operators inspect or slice homogeneous sub-clusters without forcing a premature normalized centroid projection.

---

## 3. Per-Class Thresholding & Centroid Deduplication

Similarity distributions vary sharply across land-cover classes (e.g., dense closed-canopy forest typically exhibits higher inter-pixel cosine similarity than heterogeneous cropland mosaics or coastal fringes). Therefore:

1. **Per-Class Thresholds (`resolve_class_thresholds`)**: Supply class-specific cutoffs keyed by `ClassSchema` ID or canonical class name (e.g. `{"Forest": 0.92, "Cropland": 0.87}`), rather than assuming one universal threshold across all classes.
2. **Connected-Component Vectorization & Centroid Deduplication (`extract_class_candidate_centroids`)**: Pixels satisfying $\bar{S}_c(x) > \tau_c$ are masked and converted to polygons via `reduceToVectors(..., eightConnected=...)`, then collapsed to polygon centroids (`centroid(maxError=1)`). This naturally deduplicates contiguous uniform patches into a single candidate location instead of flooding the training dataset with highly spatially autocorrelated neighboring pixels.
3. **Provenance Tagging**: Each candidate centroid is emitted as a clean GeoJSON-compatible point feature carrying `{class_property, class_name, confidence, sample_source="alphaearth_similarity"}` (default `class_property="class_id"`, with `confidence` sampled over the matched patch geometry) so downstream validation and active learning can always separate human-interpreted reference samples (`"reference"`) from embedding-augmented points.

> [!WARNING]
> **Prevent Validation Leakage:** Augmented points discovered by embedding similarity must **only** enrich training sets after visual QA — never mix `"alphaearth_similarity"` candidates into probability-based reference accuracy assessment ([Sections 1–2 of Methodology 06](06_accuracy_and_change_evaluation.md)).

---

## 4. Diagnostic Softmax Overlay (Uncalibrated QA Aid Only)

For quick visual triage of class competition (where two classes achieve nearly equal cosine similarity before committing to thresholds), `build_diagnostic_similarity_overlay` optionally computes a temperature-scaled softmax across selected `sim_<id>` bands:

$$q_c(x) = \frac{\exp\!\bigl(T \cdot (\bar{S}_c(x) - \max_k \bar{S}_k(x))\bigr)}{\sum_{j} \exp\!\bigl(T \cdot (\bar{S}_j(x) - \max_k \bar{S}_k(x))\bigr)}$$

> [!IMPORTANT]
> **Not a Calibrated `Class Probability Stack`:**
> `DiagnosticSoftmaxOverlay` explicitly sets `calibrated=False`, emits band names prefixed with **`qa_softmax_<id>`** (never canonical `prob_<id>`), and writes GEE image metadata `is_calibrated_probability=False` and `overlay_role="diagnostic_qa_only"`. Softmax over cosine similarities depends arbitrarily on temperature $T$ and ignores prior class proportions; treat it solely as a visual QA overlay.

---

## 5. Scaling & Export Recommendations for Large AOIs

- Interactive `.getInfo()` or `reduceToVectors` timeouts can occur over AOIs exceeding a few hundred $\text{km}^2$. Export candidate collections via `export_augmented_samples_to_asset(...)` or `export_augmented_samples_to_drive(...)` instead of calling `.size().getInfo()` interactively.
- When hitting Earth Engine concurrent aggregation limits across many classes, set `SampleAugmentationParams(materialize_per_class=True, wait_seconds=2.0)` to stagger similarity band materialization one class at a time.

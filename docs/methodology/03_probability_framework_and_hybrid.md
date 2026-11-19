# Probability Framework & Hybrid Classification Assembly Methodology

## 1. Why Probability-First Land Cover Mapping?

Traditional remote sensing workflows commit early to **hard class labels** (`argmax`), discarding information about runner-up classes, mixed pixels, and transition zones. `LC_Framework` standardizes every classification paradigm (Pixel-Based, Geo-Object, and Hybrid) on a **`Class Probability Stack`**:

$$\mathbf{p}(x) = \bigl(p_1(x),\, p_2(x),\, \dots,\, p_C(x)\bigr), \quad p_c(x) \in [0, 1], \quad \sum_{c=1}^{C} p_c(x) = 1$$

aligned to an ordered **`Class Schema`**. Hard maps, uncertainty layers, mask refinements, and Minimum Mapping Unit (MMU) cleanups are strictly downstream **Classification Assembly** operations.

### EECU-Optimized Storage Encoding
To halve Google Earth Engine EECU compute/export overhead and Drive storage on national-scale mosaics, `Class Probability Stack` artifacts default to **`uint16` Cloud-Optimized GeoTIFFs (COGs)** scaled by $10{,}000$ (`0..10000`, nodata `65535`), paired with a sidecar `*.manifest.json` storing the `Class Schema`, CRS, epoch, and scale factor.

---

## 2. Object-Refined Pixel Probabilities (Adaptive Hybrid Blending)

Pixel classifiers capture fine spatial detail and linear features (roads, narrow rivers, small clearings) but suffer from salt-and-pepper noise. Geo-objects (e.g., SNIC or SLIC superpixels) produce cartographically clean polygons but risk over-smoothing heterogeneous boundaries.

**Object-Refined Pixel Probabilities** (following Malinverni & Sanità, 2026) combines both through per-class **adaptive convex blending**:

1. **Per-Object Mean Probability**: For every valid pixel $x$ inside geo-object $s(x)$:
   $$\bar{p}_c(x) = \frac{1}{|s(x)|} \sum_{y \in s(x)} p_c(y)$$
   Unsegmented pixels (segment ID `0` or `-1`) fall back cleanly to pure pixel probabilities ($w_{\text{eff}, c} = 0$).

2. **Class Base Object Weight**: Each class's smoothness parameter $\sigma_c$ maps to a bounded base object weight:
   $$w_{\text{base}, c} = \operatorname{clip}\!\left(\frac{\sigma_c}{5.0},\, 0.2,\, 0.9\right)$$
   Homogeneous classes (e.g., intact forest, waterbodies) receive higher smoothness weights; heterogeneous mosaics (e.g., agroforestry, built-up) receive lower weights.

3. **Adaptive Disagreement Damping**: When pixel and object disagree strongly ($\delta_c(x) = |\bar{p}_c(x) - p_c(x)|$), the object weight is automatically damped using `adaptive_sensitivity` $\alpha \in [0, 1]$ (default $0.6$):
   $$w_{\text{eff}, c}(x) = w_{\text{base}, c}\,\bigl(1 - \alpha\,\delta_c(x)\bigr)$$

4. **Convex Blend & Simplex Renormalization**:
   $$\tilde{p}_c(x) = w_{\text{eff}, c}(x)\,\bar{p}_c(x) + \bigl(1 - w_{\text{eff}, c}(x)\bigr)\,p_c(x), \qquad \hat{p}_c(x) = \frac{\tilde{p}_c(x)}{\sum_{k=1}^{C} \tilde{p}_k(x)}$$

   Renormalization guarantees that the hybrid output remains a valid `Class Probability Stack` suitable for downstream log-loss evaluation, multi-epoch smoothing, or active learning.

---

## 3. Confusion Index & Binary Stability Map

From the winning probability $W(x) = \max_c \hat{p}_c(x)$ and runner-up probability $S(x)$, the **Confusion Index** measures local class ambiguity:

$$CI(x) = \operatorname{clip}\!\left(\frac{S(x)}{W(x)},\, 0,\, 1\right)$$

- $CI \approx 0$: high confidence (the top class strongly dominates the runner-up).
- $CI \approx 1$: maximal confusion (two classes are nearly tied).

A pixel is flagged **stable** when $CI(x) \le \tau$ (default $\tau = 0.65$), producing a binary stability layer useful for QA review and active learning sample prioritization.

---

## 4. Empirical-Bayes Spatial Smoothing (Logit Space)

When geo-objects are not used (or as an optional spatial smoother on pure pixel stacks), `lc_framework.assembly.bayesian_smooth` applies empirical-Bayes shrinkage in logit space:

1. Transform probabilities to clipped logits $\ell_c = \operatorname{logit}(\operatorname{clip}(p_c, \varepsilon, 1 - \varepsilon))$ with $\varepsilon = 10^{-3}$.
2. Within an odd square window (default $7 \times 7$), compute local mean $\mu_c$ and local variance $v_c$ over neighborhood pixels above the local $(1 - f)$ percentile (`neigh_fraction` $f$, default $0.5$), protecting sharp class boundaries.
3. Shrink each pixel logit toward its neighborhood mean weighted by class smoothness $\sigma_c^2$:
   $$\ell_c^{\text{post}} = \ell_c \frac{v_c}{\sigma_c^2 + v_c} + \mu_c \frac{\sigma_c^2}{\sigma_c^2 + v_c}$$
4. Apply the logistic sigmoid $\operatorname{expit}(\ell_c^{\text{post}})$ and renormalize across classes so probabilities sum to $1.0$.

---

## 5. Tile Seam Healing, Mask Rules & MMU Filtering

- **GEE `int32` Segment Overflow Repair**: SNIC tile exports with large ID strides can saturate at $2^{31}-1$ (`2147483647`). Connected components of clamped pixels are split into collision-free negative IDs; oversized fused mega-clusters (`> max_clamp_cluster_px`, default `400` px) fall back to `0` (pure pixel probabilities) so ocean and land never fuse.
- **Seam Healing**: Euclidean distance transform fills thin 1–4 px internal tile gaps without altering broad exterior nodata, and 1-px GEE tile computation spikes are replaced by linear neighbor averages.
- **Multi-Resolution `MaskRule` Engine**: Auxiliary raster masks (e.g., settlement footprints, river corridors) of any CRS or pixel resolution are aligned on the fly via nearest-neighbor `WarpedVRT`.
- **Minimum Mapping Unit (MMU)**: 8-connected patches smaller than a per-class pixel threshold are iteratively absorbed via 3×3 focal mode filtering while leaving protected classes (`min_size = 0`) untouched.

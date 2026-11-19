# CCDC Temporal Segmentation & Synthetic Reflectance Methodology

## 1. Scientific Rationale

In cloud-prone tropical and wet–dry seasonal environments, direct optical composites often blend images from different phenological states, retain residual cloud shadows, or contain persistent gaps during the wet season. **Continuous Change Detection and Classification (CCDC)** (Zhu & Woodcock, 2014) fits piecewise harmonic regression models per pixel between detected spectral breakpoints, creating continuous **`Temporal Segment`** records that support three core mapping functions:

1. **Cloud-Free Seasonal Synthetic Reflectance**: Reconstructs harmonized reflectance at consistent fractional-year targets (for example, peak wet `Year + 0.16` and peak dry `Year + 0.65`, configured per project in `ProjectConfig.seasons`).
2. **Phenological Descriptors**: Extracts first-order annual harmonic amplitude and phase to distinguish evergreen canopy, deciduous plantations, and seasonal agriculture.
3. **Multi-Epoch Training Sample Stability Filtering**: Rejects pixels with confirmed abrupt breaks or strong gradual spectral drift so baseline samples can be transferred safely to historical epochs.

*(Note: Per [ADR-0004](../adr/0004-ccdc-owned-luma-multitemporal-delegated.md), `lc_framework.temporal` natively owns CCDC temporal segmentation, while direct multi-period GEE orchestration remains available via `lc_framework.luma_adapter`.)*

---

## 2. Harmonized Six-Band Optical Schema

Sensor-specific ingestion (`inputs_landsat.py` and `inputs_s2.py`) maps every optical sensor into a single six-band lowercase schema before downstream temporal modeling or feature engineering:

| Canonical Band | Landsat 5/7 (TM/ETM+) | Landsat 8/9 (OLI/OLI-2) | Sentinel-2 L2A (MSI) |
| :--- | :--- | :--- | :--- |
| `blue` | `SR_B1` | `SR_B2` | `B2` |
| `green` | `SR_B2` | `SR_B3` | `B3` |
| `red` | `SR_B3` | `SR_B4` | `B4` |
| `nir` | `SR_B4` | `SR_B5` | `B8` |
| `swir1` | `SR_B5` | `SR_B6` | `B11` |
| `swir2` | `SR_B7` | `SR_B7` | `B12` |

### Cloud Masking & Radiometric Scaling
- **Landsat Collection 2 L2**: Masks `QA_PIXEL` Fill/SLC-off (bit 0), Dilated Cloud (bit 1), Cloud (bit 3), Cloud Shadow (bit 4), and Snow (bit 5), then applies $\rho_{\text{scaled}} = (0.0000275 \cdot \text{DN} - 0.2) \cdot S_{\text{ref}}$.
- **Sentinel-2 SR Harmonized**: Links `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` and retains pixels where `cs_cdf >= clear_threshold` (default `0.60`), scaling native `[0, 10000]` values by $S_{\text{ref}} / 10000$.

> [!IMPORTANT]
> **Why CCDC Uses `reflectance_scale = 10000.0`:** Earth Engine's `ee.Algorithms.TemporalSegmentation.Ccdc` calibrates its LASSO penalty (`lambda = 20`) for reflectance in `[0, 10000]`. Passing `[0, 1]` reflectance with `lambda = 20` over-penalizes harmonic terms toward zero. Use `10000.0` for CCDC (and divide by `coef_scale = 10000.0` during reconstruction) and `1.0` for direct seasonal composites.

---

## 3. Fractional-Year Conventions (`dateFormat = 1`)

All `lc_framework.temporal` CCDC operations use **fractional years** (`dateFormat = 1`), where every date is encoded as:

$$t = \text{Year} + \frac{\text{DayOfYear} - 1}{\text{DaysInYear}} \in [\text{Year},\, \text{Year} + 1)$$

- **Segment timing arrays** (`tStart`, `tEnd`, `tBreak`) store one fractional-year value per fitted `Temporal Segment` along the 1D array axis.
- **Seasonal synthesis targets** are configured in `ProjectConfig.seasons` as offsets in `[0.0, 1.0)` added to each calendar year (for example: `wet: 0.16` $\approx$ late February / early March, `harmonic_mid: 0.50` $\approx$ July 1, `dry: 0.65` $\approx$ late August).
- **Tolerance windows** (`tol_years`, default `1.0`) and **stability periods** (`start_year`, `end_year`) are expressed in the same fractional-year units, eliminating calendar-day ambiguity across leap years.

---

## 4. Monthly Pre-Compositing & Harmonic Model

Aggregating dense Sentinel-2 or multi-sensor Landsat observations with `to_monthly_median` (timestamped at the 15th of each month, dropping empty months where `n == 0`) suppresses residual cloud fringes, balances wet/dry sampling density, and lowers GEE EECU cost.

Within each `Temporal Segment` $[t_{\text{start}}, t_{\text{end}}]$, CCDC fits an 8-parameter harmonic model per band $i$:

$$\hat{\rho}_i(t) = a_{0,i} + c_{1,i} t + \sum_{k=1}^{3} \bigl[a_{k,i}\cos(k\omega t) + b_{k,i}\sin(k\omega t)\bigr], \qquad \omega = 2\pi$$

stored in `*_coefs` as $[a_{0,i}, c_{1,i}, a_{1,i}, b_{1,i}, a_{2,i}, b_{2,i}, a_{3,i}, b_{3,i}]$.

### Tiled Export Requirement
When exporting CCDC array images (`export_ccdc_tiles`), retain only `['tStart', 'tEnd', 'tBreak', 'changeProb', '.*_coefs']` and always specify **`pyramidingPolicy={'.default': 'sample'}`** so Earth Engine preserves array dimensions across pyramid levels.

---

## 5. Reconstruction Safeguards

1. **Nearest-Segment Lookup (`nearest_segment`)**: Computes per-pixel distance $d(t, s) = \max(0,\, t_{\text{start}, s} - t,\, t - t_{\text{end}, s})$ and selects $s^* = \arg\min_s d(t, s)$, masking pixels whose gap exceeds `tol_years` (default `1.0` year).
2. **Linear Trend Clamping ($t_c$)**: Evaluates the linear trend at $t_c = \operatorname{clamp}(t, t_{\text{start}, s^*}, t_{\text{end}, s^*})$ while evaluating seasonal harmonics at exact fractional date $t$, preventing steep slopes from overshooting outside fitted bounds.
3. **All-Band Plausibility Masking & Conditional Hole Repair**: Pixels are marked valid (`synth_valid = 1`) only when `gap_years <= tol_years` and every reconstructed band lies strictly in `(-0.05, 1.0)`. Isolated 1–2 pixel dropouts are repaired via conditional $3 \times 3$ focal median (`fill_isolated`) only where at least `min_neighbors = 5` of 8 neighbors are valid.
4. **First-Harmonic Polar Metrics (`get_harmonics`)**:
   $$\text{Amplitude}_i = \sqrt{a_{1,i}^2 + b_{1,i}^2}, \qquad \text{Phase}_i = \operatorname{atan2}(b_{1,i},\, a_{1,i}) \in [-\pi, \pi]$$
5. **Multi-Epoch Stability Mask (`stable_in_period`)**: Marks a pixel stable (`1`) over $[t_a, t_b]$ iff (i) zero confirmed breakpoints (`changeProb == 1`) fall in $[t_a, t_b]$, (ii) both epoch midpoints fall within `tol_years` on the exact same `Temporal Segment` index, and (iii) cumulative linear drift $|c_{1,i} / \text{coef\_scale}| \cdot (t_b - t_a) \le \text{max\_drift}$ across all monitored `drift_bands`.

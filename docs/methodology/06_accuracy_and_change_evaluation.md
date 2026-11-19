# Accuracy Assessment, Active Learning & Multi-Epoch Change Evaluation

## 1. Area-Adjusted Thematic Accuracy (Olofsson et al., 2014)

Raw sample confusion matrices overestimate or distort map accuracy whenever classes are sampled disproportionately (stratified random sampling). `lc_framework.evaluation.compute_thematic_accuracy` follows **Olofsson et al. (2014)** and **QGIS AcATaMa** conventions:

- **Rows $i$**: Map (Predicted) strata with mapped pixel counts $N_i$ and stratum weights $W_i = N_i / \sum_k N_k$.
- **Columns $j$**: Reference (Ground-Truth) classes.
- **Estimated Area Proportions Matrix** (Eq. 1):
  $$\hat{p}_{ij} = W_i \frac{n_{ij}}{n_{i\cdot}}$$
- **Area-Weighted Overall Accuracy & Variance** (Eq. 4–5):
  $$\hat{O} = \sum_{i=1}^{q} \hat{p}_{ii}, \qquad \widehat{V}(\hat{O}) = \sum_{i=1}^{q} W_i^2 \frac{\hat{U}_i(1 - \hat{U}_i)}{n_{i\cdot} - 1}$$
- **User's Accuracy** ($\hat{U}_i = \hat{p}_{ii} / \hat{p}_{i\cdot}$) and **Area-Weighted Producer's Accuracy** ($\hat{P}_j = \hat{p}_{jj} / \hat{p}_{\cdot j}$, Eq. 3 & 7).
- **Error-Adjusted Class Area & Standard Error** (Eq. 8–11):
  $$\hat{A}_j = A_{\text{total}} \cdot \hat{p}_{\cdot j}, \qquad S(\hat{p}_{\cdot j}) = \sqrt{\sum_{i=1}^{q} W_i^2 \frac{\frac{n_{ij}}{n_{i\cdot}}\!\left(1 - \frac{n_{ij}}{n_{i\cdot}}\right)}{n_{i\cdot} - 1}}$$
  with two-sided confidence interval $\pm\, z_{1-\alpha/2} \cdot A_{\text{total}} \cdot S(\hat{p}_{\cdot j})$ (default 95%, $z \approx 1.96$).

---

## 2. Probability Calibration Evaluation (Log-Loss)

Because `LC_Framework` operates natively on **`Class Probability Stack`** outputs, hard accuracy alone cannot distinguish a well-calibrated classifier from an overconfident one. Multi-class **Log-Loss** evaluates the probability assigned to the true reference class $y_k$:

$$\mathcal{L}_{\text{log}} = -\frac{1}{N} \sum_{k=1}^{N} \log\!\Bigl(\operatorname{clip}\bigl(p_{y_k}(x_k),\, \varepsilon,\, 1 - \varepsilon\bigr)\Bigr)$$

Lower values indicate better-calibrated probabilities, complementing Olofsson area-adjusted accuracy when comparing Pixel-Based Classification, Geo-Object Classification, and Object-Refined Pixel Probabilities.

---

## 3. Uncertainty-Driven Active Learning

`lc_framework.evaluation.suggest_active_learning_samples` prioritizes locations where additional human reference labels yield the highest model improvement, supporting three normalized uncertainty criteria in $[0, 1]$:

| Criterion | Formula | Interpretation |
| :--- | :--- | :--- |
| **`confusion_index`** | $CI = S / W$ | Ratio of runner-up to winning class probability |
| **`margin`** | $1 - (W - S)$ | Complement of the top-two probability gap |
| **`entropy`** | $-\frac{1}{\log C}\sum_{c=1}^{C} p_c \log p_c$ | Full multi-class Shannon entropy normalized by $\log C$ |

Candidate pixels are streamed in bounded memory, filtered by minimum uncertainty and optional target classes, and spatially thinned using a `cKDTree` so suggested points maintain at least `min_distance_m` spacing from both existing training samples and each other.

---

## 4. Multi-Epoch Transitions & Pontius–Millones Disagreement

For any sequence of two or more configured epochs ($t_1, t_2, \dots, t_T$), `lc_framework.evaluation.analyze_multi_epoch_rasters` computes pairwise consecutive transition matrices (in hectares) plus the net ($t_1 \to t_T$) matrix, and decomposes change following **Pontius & Millones (2011)** and **Pontius & Santacruz (2014)**:

- **Persistence**: diagonal area $A_{ii}$ unchanged from $t_1$ to $t_2$.
- **Loss** ($L_i = A_{i\cdot} - A_{ii}$) and **Gain** ($G_i = A_{\cdot i} - A_{ii}$).
- **Quantity Disagreement**: $Q_i = |G_i - L_i|$ (net change in total class area).
- **Allocation Disagreement (Swap)**: $A_i = 2 \min(L_i, G_i)$ (spatial relocation of class $i$ without net area change).
- **Exchange** ($E_i = 2 \sum_{j \ne i} \min(A_{ij}, A_{ji})$) and **Shift** ($S_i = A_i - E_i$).
- **Overall Landscape Totals**: half the sum across categories ($Q = \tfrac{1}{2}\sum_i Q_i$, $A = \tfrac{1}{2}\sum_i A_i$, total disagreement $D = Q + A$).

Transitions across all epochs can be exported directly to an interactive HTML **Sankey** diagram via `export_sankey_html`.

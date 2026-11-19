"""Thematic accuracy assessment (Olofsson et al., 2014), log-loss, and active learning.

Implemented strictly on NumPy, SciPy, Pandas, and Rasterio so Evaluation runs
completely standalone without GEE, scikit-learn, or GeoPandas (ADR-0005).
"""

from __future__ import annotations

import heapq
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.spatial import cKDTree
from scipy.stats import norm

from lc_framework.config import ClassSchema
from lc_framework.io import read_probability_stack

UncertaintyMethod = Literal["confusion_index", "margin", "entropy"]


@dataclass(frozen=True)
class ThematicAccuracyResult:
    """Container for sample-based and Olofsson (2014) area-adjusted accuracy reports."""

    overall_metrics: dict[str, float]
    class_metrics: pd.DataFrame
    confusion_matrix_counts: pd.DataFrame
    confusion_matrix_area_prop: pd.DataFrame | None = None

    def save_reports(
        self,
        output_dir: str | Path,
        *,
        prefix: str = "accuracy",
    ) -> dict[str, Path]:
        """Write summary metrics JSON, per-class CSV, and confusion matrices to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        metrics_path = out_dir / f"{prefix}_overall_metrics.json"
        metrics_path.write_text(json.dumps(self.overall_metrics, indent=2), encoding="utf-8")
        paths["overall_metrics_json"] = metrics_path

        class_csv = out_dir / f"{prefix}_class_accuracy_area.csv"
        self.class_metrics.to_csv(class_csv, index=False, encoding="utf-8")
        paths["class_metrics_csv"] = class_csv

        cm_csv = out_dir / f"{prefix}_confusion_matrix_counts.csv"
        self.confusion_matrix_counts.to_csv(cm_csv, encoding="utf-8")
        paths["confusion_matrix_counts_csv"] = cm_csv

        if self.confusion_matrix_area_prop is not None:
            cm_prop_csv = out_dir / f"{prefix}_confusion_matrix_area_proportions.csv"
            self.confusion_matrix_area_prop.to_csv(cm_prop_csv, encoding="utf-8")
            paths["confusion_matrix_area_prop_csv"] = cm_prop_csv

        return paths


def compute_pixel_area_m2(crs: CRS | None, transform: Any, height: int) -> float:
    """Compute single-pixel area in square meters for projected or geographic CRS."""
    pixel_width = abs(float(transform.a))
    pixel_height = abs(float(transform.e))

    if crs is not None and crs.is_geographic:
        wgs84_a = 6_378_137.0
        wgs84_f = 1.0 / 298.257_223_563
        wgs84_b = wgs84_a * (1.0 - wgs84_f)
        centre_lat_deg = float(transform.f) + (height / 2.0) * float(transform.e)
        centre_lat_rad = math.radians(centre_lat_deg)
        sin_lat = math.sin(centre_lat_rad)
        cos_lat = math.cos(centre_lat_rad)
        e2 = 1.0 - (wgs84_b / wgs84_a) ** 2
        denom = math.sqrt(1.0 - e2 * sin_lat**2)
        meridional = wgs84_a * (1.0 - e2) / (denom**3)
        prime_vertical = wgs84_a / denom
        m_per_deg_lat = meridional * math.pi / 180.0
        m_per_deg_lon = prime_vertical * cos_lat * math.pi / 180.0
        return (pixel_width * m_per_deg_lon) * (pixel_height * m_per_deg_lat)

    return pixel_width * pixel_height


def count_raster_pixels(
    raster_path: str | Path,
    *,
    band: int = 1,
    nodata_values: Sequence[int | float] = (0, -1, 255),
    block_size: int = 2048,
) -> tuple[dict[int, int], float]:
    """Count mapped pixels per class in a classification GeoTIFF via windowed streaming."""
    totals: dict[int, int] = {}
    nodata_set = set(nodata_values)

    with rasterio.open(raster_path) as src:
        if src.nodata is not None and np.isfinite(src.nodata):
            nodata_set.add(int(src.nodata))
        pixel_area_m2 = compute_pixel_area_m2(src.crs, src.transform, src.height)

        for r_off in range(0, src.height, block_size):
            win_h = min(block_size, src.height - r_off)
            for c_off in range(0, src.width, block_size):
                win_w = min(block_size, src.width - c_off)
                win = Window(c_off, r_off, win_w, win_h)
                arr = src.read(band, window=win, masked=True)
                valid = ~np.ma.getmaskarray(arr)
                data = np.asarray(arr.data)
                valid &= np.isfinite(data)
                for nd in nodata_set:
                    valid &= data != nd
                if not np.any(valid):
                    continue
                uniq, cnts = np.unique(data[valid].astype(np.int64), return_counts=True)
                for u, c in zip(uniq.tolist(), cnts.tolist(), strict=True):
                    totals[int(u)] = totals.get(int(u), 0) + int(c)

    return dict(sorted(totals.items())), pixel_area_m2


def _resolve_name_map(
    class_schema: ClassSchema | None,
    class_names: Mapping[int, str] | None,
) -> dict[int, str]:
    if class_schema is not None:
        return {c.id: c.name for c in class_schema.classes}
    if class_names is not None:
        return {int(k): str(v) for k, v in class_names.items()}
    return {}


def _build_confusion_matrix(
    y_pred: NDArray[np.int64],
    y_true: NDArray[np.int64],
    class_list: Sequence[int],
) -> NDArray[np.float64]:
    """Build (q, q) confusion matrix with Rows = Map (Predicted i), Cols = Reference (j)."""
    q = len(class_list)
    id_to_idx = {cid: idx for idx, cid in enumerate(class_list)}
    n_ij = np.zeros((q, q), dtype=np.float64)
    for pred_id, true_id in zip(y_pred.tolist(), y_true.tolist(), strict=True):
        if pred_id in id_to_idx and true_id in id_to_idx:
            n_ij[id_to_idx[pred_id], id_to_idx[true_id]] += 1.0
    return n_ij


def compute_thematic_accuracy(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    *,
    class_schema: ClassSchema | None = None,
    class_names: Mapping[int, str] | None = None,
    labels: Sequence[int] | None = None,
    mapped_pixel_counts: Mapping[int, int] | None = None,
    pixel_area_m2: float | None = None,
    confidence_level: float = 0.95,
) -> ThematicAccuracyResult:
    """Compute sample-based and Olofsson et al. (2014) area-adjusted thematic accuracy.

    Confusion matrix rows ``i`` represent Map (Predicted) classes and columns ``j``
    represent Reference (Ground-Truth) classes.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level!r}.")

    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have equal length ({len(y_true_arr)} vs {len(y_pred_arr)})."
        )
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute thematic accuracy on empty sample arrays.")

    name_map = _resolve_name_map(class_schema, class_names)
    mapped_clean: dict[int, int] | None = None
    if mapped_pixel_counts is not None:
        mapped_clean = {int(k): int(v) for k, v in mapped_pixel_counts.items() if int(k) > 0}

    if labels is not None:
        class_list = sorted({int(c) for c in labels})
    elif class_schema is not None:
        class_list = list(class_schema.class_ids)
    else:
        observed = set(np.unique(y_true_arr).tolist()) | set(np.unique(y_pred_arr).tolist())
        if mapped_clean is not None:
            observed |= set(mapped_clean.keys())
        observed |= set(name_map.keys())
        class_list = sorted(c for c in observed if c > 0)

    q = len(class_list)
    z_score = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

    n_ij = _build_confusion_matrix(y_pred_arr, y_true_arr, class_list)
    n_total = float(n_ij.sum())
    if n_total <= 0:
        raise ValueError("No sample pairs matched the evaluated class labels.")

    n_i_dot = n_ij.sum(axis=1)
    n_dot_j = n_ij.sum(axis=0)
    diag_ii = np.diag(n_ij)

    label_names = [name_map.get(c, f"Class_{c}") for c in class_list]
    row_col_labels = [f"{c}: {name}" for c, name in zip(class_list, label_names, strict=True)]

    cm_counts_df = pd.DataFrame(n_ij.astype(np.int64), index=row_col_labels, columns=row_col_labels)
    cm_counts_df.index.name = "Map (i) \\ Reference (j)"
    cm_counts_df["Total_Map (n_i.)"] = n_i_dot.astype(np.int64)
    col_totals = pd.Series(
        [*n_dot_j.astype(np.int64).tolist(), int(n_total)],
        index=cm_counts_df.columns,
        name="Total_Ref (n_.j)",
    )
    cm_counts_df = pd.concat([cm_counts_df, col_totals.to_frame().T])

    with np.errstate(divide="ignore", invalid="ignore"):
        ua_sample = np.where(n_i_dot > 0, diag_ii / n_i_dot, np.nan)
        ua_sample_se = np.where(
            n_i_dot > 1,
            np.sqrt(np.maximum(ua_sample * (1.0 - ua_sample) / (n_i_dot - 1.0), 0.0)),
            np.nan,
        )
        pa_sample = np.where(n_dot_j > 0, diag_ii / n_dot_j, np.nan)
        pa_sample_se = np.where(
            n_dot_j > 1,
            np.sqrt(np.maximum(pa_sample * (1.0 - pa_sample) / (n_dot_j - 1.0), 0.0)),
            np.nan,
        )
        f1_per_class = np.where(
            (ua_sample + pa_sample) > 0,
            2.0 * ua_sample * pa_sample / (ua_sample + pa_sample),
            0.0,
        )
        f1_per_class = np.where(
            np.isnan(ua_sample) & np.isnan(pa_sample), np.nan, np.nan_to_num(f1_per_class, nan=0.0)
        )

    oa_sample = float(diag_ii.sum() / n_total)
    oa_sample_se = (
        math.sqrt(oa_sample * (1.0 - oa_sample) / (n_total - 1.0)) if n_total > 1 else 0.0
    )
    p_expected = float(np.sum(n_i_dot * n_dot_j) / (n_total**2))
    kappa = (oa_sample - p_expected) / (1.0 - p_expected) if p_expected < 1.0 else 1.0

    active_ref = n_dot_j > 0
    balanced_acc = float(np.nanmean(pa_sample[active_ref])) if np.any(active_ref) else float("nan")
    f1_macro = float(np.nanmean(f1_per_class[active_ref])) if np.any(active_ref) else float("nan")
    if np.any(active_ref):
        ref_w = n_dot_j[active_ref] / n_dot_j[active_ref].sum()
        f1_weighted = float(np.nansum(f1_per_class[active_ref] * ref_w))
    else:
        f1_weighted = float("nan")

    overall_metrics: dict[str, float] = {
        "overall_accuracy_sample": oa_sample,
        "overall_accuracy_sample_se": oa_sample_se,
        "overall_accuracy_sample_ci": z_score * oa_sample_se,
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "kappa": float(kappa),
        "confidence_level": float(confidence_level),
        "z_score": z_score,
        "total_samples": float(n_total),
        "num_classes_evaluated": float(np.sum(active_ref | (n_i_dot > 0))),
    }

    class_rows: list[dict[str, Any]] = []
    for idx, cid in enumerate(class_list):
        class_rows.append(
            {
                "class_id": int(cid),
                "class_name": label_names[idx],
                "ref_samples": int(n_dot_j[idx]),
                "map_samples": int(n_i_dot[idx]),
                "correct_samples": int(diag_ii[idx]),
                "user_accuracy": float(ua_sample[idx]) if np.isfinite(ua_sample[idx]) else np.nan,
                "user_accuracy_se": (
                    float(ua_sample_se[idx]) if np.isfinite(ua_sample_se[idx]) else np.nan
                ),
                "user_accuracy_ci": (
                    float(z_score * ua_sample_se[idx]) if np.isfinite(ua_sample_se[idx]) else np.nan
                ),
                "commission_error": (
                    float(1.0 - ua_sample[idx]) if np.isfinite(ua_sample[idx]) else np.nan
                ),
                "producer_accuracy_sample": (
                    float(pa_sample[idx]) if np.isfinite(pa_sample[idx]) else np.nan
                ),
                "producer_accuracy_sample_se": (
                    float(pa_sample_se[idx]) if np.isfinite(pa_sample_se[idx]) else np.nan
                ),
                "omission_error_sample": (
                    float(1.0 - pa_sample[idx]) if np.isfinite(pa_sample[idx]) else np.nan
                ),
                "f1_score": float(f1_per_class[idx]) if np.isfinite(f1_per_class[idx]) else np.nan,
            }
        )

    cm_area_prop_df: pd.DataFrame | None = None
    if mapped_clean is not None and sum(mapped_clean.values()) > 0:
        n_mapped = np.array([float(mapped_clean.get(c, 0)) for c in class_list], dtype=np.float64)
        n_mapped_total = float(np.sum(n_mapped))
        w_i = n_mapped / n_mapped_total

        unsampled_mask = (n_mapped > 0) & (n_i_dot == 0)
        unsampled_weight = float(w_i[unsampled_mask].sum())
        if unsampled_weight > 0.0:
            unsampled_ids = [class_list[i] for i in np.nonzero(unsampled_mask)[0].tolist()]
            warnings.warn(
                f"Mapped strata {unsampled_ids} have mapped pixels (weight={unsampled_weight:.4f}) "
                "but zero validation samples; Olofsson area proportions sum to < 1.0.",
                UserWarning,
                stacklevel=2,
            )
        overall_metrics["unsampled_mapped_weight"] = unsampled_weight

        with np.errstate(divide="ignore", invalid="ignore"):
            q_ij = np.where(n_i_dot[:, None] > 0, n_ij / n_i_dot[:, None], 0.0)
            denom_i = np.where(n_i_dot[:, None] > 1, n_i_dot[:, None] - 1.0, np.nan)
            raw_var_ij = (w_i[:, None] ** 2) * q_ij * (1.0 - q_ij) / denom_i
            cell_var_ij = np.nan_to_num(raw_var_ij, nan=0.0)

        p_ij = w_i[:, None] * q_ij
        p_i_dot = p_ij.sum(axis=1)
        p_dot_j = p_ij.sum(axis=0)
        p_diag = np.diag(p_ij)

        # Olofsson et al. (2014) Eq. 4 & 5: Overall area-weighted accuracy & SE
        oa_area = float(np.sum(p_diag))
        oa_area_se = math.sqrt(max(float(np.sum(np.diag(cell_var_ij))), 0.0))
        overall_metrics["overall_accuracy_area_weighted"] = oa_area
        overall_metrics["overall_accuracy_area_se"] = oa_area_se
        overall_metrics["overall_accuracy_area_ci"] = z_score * oa_area_se
        overall_metrics["total_mapped_pixels"] = n_mapped_total

        # Olofsson Eq. 3 & 7: Area-weighted Producer's Accuracy & SE
        with np.errstate(divide="ignore", invalid="ignore"):
            pa_area = np.where(p_dot_j > 0, p_diag / p_dot_j, np.nan)
        pa_area_se = np.full(q, np.nan, dtype=np.float64)
        for j in range(q):
            if p_dot_j[j] <= 0 or not np.isfinite(pa_area[j]):
                continue
            delta_j = np.zeros(q, dtype=np.float64)
            delta_j[j] = 1.0
            var_pj = float(np.sum(cell_var_ij[:, j] * ((delta_j - pa_area[j]) ** 2))) / (
                p_dot_j[j] ** 2
            )
            pa_area_se[j] = math.sqrt(max(var_pj, 0.0))

        # Olofsson Eq. 8-11: Error-adjusted area proportions & SE
        p_dot_j_se = np.sqrt(np.maximum(np.sum(cell_var_ij, axis=0), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_area = np.where(p_dot_j > 0, p_dot_j_se / p_dot_j, np.nan)

        pix_m2 = float(pixel_area_m2) if pixel_area_m2 is not None else 1.0
        total_area_ha = n_mapped_total * (pix_m2 / 10_000.0)
        total_area_km2 = n_mapped_total * (pix_m2 / 1_000_000.0)
        if pixel_area_m2 is not None:
            overall_metrics["pixel_area_m2"] = pix_m2
            overall_metrics["total_mapped_area_ha"] = total_area_ha
            overall_metrics["total_mapped_area_km2"] = total_area_km2

        for idx in range(q):
            row = class_rows[idx]
            row["mapped_pixels"] = int(n_mapped[idx])
            row["map_weight_Wi"] = float(w_i[idx])
            row["producer_accuracy_area"] = (
                float(pa_area[idx]) if np.isfinite(pa_area[idx]) else np.nan
            )
            row["producer_accuracy_area_se"] = (
                float(pa_area_se[idx]) if np.isfinite(pa_area_se[idx]) else np.nan
            )
            row["producer_accuracy_area_ci"] = (
                float(z_score * pa_area_se[idx]) if np.isfinite(pa_area_se[idx]) else np.nan
            )
            row["omission_error_area"] = (
                float(1.0 - pa_area[idx]) if np.isfinite(pa_area[idx]) else np.nan
            )
            row["adjusted_area_prop"] = float(p_dot_j[idx])
            row["adjusted_area_prop_se"] = float(p_dot_j_se[idx])
            row["adjusted_area_cv"] = float(cv_area[idx]) if np.isfinite(cv_area[idx]) else np.nan
            if pixel_area_m2 is not None:
                mapped_ha = float(n_mapped[idx] * (pix_m2 / 10_000.0))
                adj_ha = float(total_area_ha * p_dot_j[idx])
                se_ha = float(total_area_ha * p_dot_j_se[idx])
                ci_ha = float(z_score * se_ha)
                row["mapped_area_ha"] = mapped_ha
                row["adjusted_area_ha"] = adj_ha
                row["adjusted_area_se_ha"] = se_ha
                row["adjusted_area_ci_ha"] = ci_ha
                row["adjusted_area_lower_ha"] = max(0.0, adj_ha - ci_ha)
                row["adjusted_area_upper_ha"] = adj_ha + ci_ha

                adj_km2 = float(total_area_km2 * p_dot_j[idx])
                se_km2 = float(total_area_km2 * p_dot_j_se[idx])
                ci_km2 = float(z_score * se_km2)
                row["adjusted_area_km2"] = adj_km2
                row["adjusted_area_se_km2"] = se_km2
                row["adjusted_area_ci_km2"] = ci_km2

        cm_area_prop_df = pd.DataFrame(p_ij, index=row_col_labels, columns=row_col_labels)
        cm_area_prop_df.index.name = "Map (i) \\ Reference (j)"
        cm_area_prop_df["W_i (Map_Prop)"] = w_i
        cm_area_prop_df["Total_Map_Prop (p_i.)"] = p_i_dot
        prop_totals = pd.Series(
            [*p_dot_j.tolist(), float(w_i.sum()), float(p_i_dot.sum())],
            index=cm_area_prop_df.columns,
            name="Adjusted_Area_Prop (p_.j)",
        )
        cm_area_prop_df = pd.concat([cm_area_prop_df, prop_totals.to_frame().T])

    return ThematicAccuracyResult(
        overall_metrics=overall_metrics,
        class_metrics=pd.DataFrame(class_rows),
        confusion_matrix_counts=cm_counts_df,
        confusion_matrix_area_prop=cm_area_prop_df,
    )


def compute_log_loss(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_prob: np.ndarray,
    *,
    class_schema: ClassSchema | None = None,
    class_ids: Sequence[int] | None = None,
    eps: float = 1e-6,
) -> float:
    """Compute multi-class negative log-likelihood on predicted class probabilities."""
    if not (0.0 < eps < 0.5):
        raise ValueError(f"eps must lie in (0, 0.5), got {eps!r}.")

    y_arr = np.asarray(y_true, dtype=np.int64).ravel()
    p_arr = np.asarray(y_prob, dtype=np.float64)
    if p_arr.ndim != 2 or p_arr.shape[0] != y_arr.size:
        raise ValueError(
            f"y_prob must have shape (N, C) matching y_true length ({y_arr.size}), "
            f"got {p_arr.shape}."
        )
    if y_arr.size == 0:
        raise ValueError("Cannot compute log_loss on empty arrays.")

    if class_schema is not None:
        ordered_ids = list(class_schema.class_ids)
    elif class_ids is not None:
        ordered_ids = [int(c) for c in class_ids]
    else:
        ordered_ids = list(range(1, p_arr.shape[1] + 1))

    if len(ordered_ids) != p_arr.shape[1]:
        raise ValueError(
            f"Class count ({len(ordered_ids)}) does not match y_prob columns ({p_arr.shape[1]})."
        )

    id_to_col = {cid: idx for idx, cid in enumerate(ordered_ids)}
    losses: list[float] = []
    for row_idx, cid in enumerate(y_arr.tolist()):
        if cid not in id_to_col:
            raise ValueError(f"Reference class ID {cid} not found in class_ids {ordered_ids}.")
        p_true = float(np.clip(p_arr[row_idx, id_to_col[cid]], eps, 1.0 - eps))
        losses.append(-math.log(p_true))

    return float(np.mean(losses))


def evaluate_probability_stack(
    prob_path: str | Path,
    sample_coords_xy: Sequence[tuple[float, float]],
    y_true: Sequence[int] | np.ndarray,
    *,
    class_schema: ClassSchema | None = None,
    confidence_level: float = 0.95,
    eps: float = 1e-6,
) -> ThematicAccuracyResult:
    """Evaluate a Class Probability Stack GeoTIFF for Olofsson accuracy and log-loss."""
    stack = read_probability_stack(prob_path, class_schema=class_schema)
    schema = stack.manifest.class_schema
    class_ids = np.asarray(schema.class_ids, dtype=np.int64)

    transform = stack.profile["transform"]
    height = int(stack.profile["height"])
    width = int(stack.profile["width"])
    crs_raw = stack.profile.get("crs")
    crs = CRS.from_user_input(crs_raw) if crs_raw is not None else None
    pixel_area_m2 = compute_pixel_area_m2(crs, transform, height)

    # Derive mapped hard pixel counts across valid pixels
    safe = np.where(np.isfinite(stack.probabilities), stack.probabilities, -np.inf)
    winners = np.argmax(safe, axis=0)
    mapped_labels = class_ids[winners[stack.valid_mask]]
    uniq, cnts = np.unique(mapped_labels, return_counts=True)
    mapped_counts = {int(u): int(c) for u, c in zip(uniq.tolist(), cnts.tolist(), strict=True)}

    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    if len(sample_coords_xy) != y_true_arr.size:
        raise ValueError("sample_coords_xy and y_true must have identical lengths.")

    sampled_probs: list[np.ndarray] = []
    sampled_pred: list[int] = []
    kept_true: list[int] = []
    for (x, y), ref_cls in zip(sample_coords_xy, y_true_arr.tolist(), strict=True):
        r, c = rasterio.transform.rowcol(transform, x, y)
        if not (0 <= r < height and 0 <= c < width) or not stack.valid_mask[r, c]:
            continue
        p_vec = stack.probabilities[:, r, c]
        sampled_probs.append(p_vec)
        sampled_pred.append(int(class_ids[int(np.argmax(p_vec))]))
        kept_true.append(ref_cls)

    if not kept_true:
        raise ValueError(
            "No sample coordinates fell on valid pixels of the Class Probability Stack."
        )

    result = compute_thematic_accuracy(
        y_true=kept_true,
        y_pred=sampled_pred,
        class_schema=schema,
        mapped_pixel_counts=mapped_counts,
        pixel_area_m2=pixel_area_m2,
        confidence_level=confidence_level,
    )
    ll = compute_log_loss(
        kept_true,
        np.vstack(sampled_probs),
        class_schema=schema,
        eps=eps,
    )
    merged_metrics = dict(result.overall_metrics)
    merged_metrics["log_loss"] = ll
    return ThematicAccuracyResult(
        overall_metrics=merged_metrics,
        class_metrics=result.class_metrics,
        confusion_matrix_counts=result.confusion_matrix_counts,
        confusion_matrix_area_prop=result.confusion_matrix_area_prop,
    )


def _uncertainty_from_probs(
    probs: NDArray[np.float32],
    valid: NDArray[np.bool_],
    method: UncertaintyMethod,
) -> NDArray[np.float32]:
    num_classes, height, width = probs.shape
    unc = np.full((height, width), np.nan, dtype=np.float32)
    if not valid.any():
        return unc

    p_valid = probs[:, valid].T
    row_sums = p_valid.sum(axis=1, keepdims=True)
    positive = row_sums[:, 0] > 0.0
    if not positive.any():
        return unc

    p_norm = np.zeros_like(p_valid, dtype=np.float32)
    p_norm[positive] = np.clip(p_valid[positive] / row_sums[positive], 0.0, 1.0)

    if method == "confusion_index":
        sorted_p = np.sort(p_norm, axis=1)
        w = sorted_p[:, -1]
        s = sorted_p[:, -2] if num_classes >= 2 else np.zeros_like(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.clip(np.where(w > 1e-6, s / w, 0.0), 0.0, 1.0)
    elif method == "margin":
        sorted_p = np.sort(p_norm, axis=1)
        w = sorted_p[:, -1]
        s = sorted_p[:, -2] if num_classes >= 2 else np.zeros_like(w)
        scores = np.clip(1.0 - (w - s), 0.0, 1.0)
    elif method == "entropy":
        denom = math.log(max(2, num_classes))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(p_norm > 1e-12, np.log(np.maximum(p_norm, 1e-12)), 0.0)
        scores = np.clip(-np.sum(p_norm * log_p, axis=1) / denom, 0.0, 1.0)
    else:
        raise ValueError("method must be 'confusion_index', 'margin', or 'entropy'.")

    scores[~positive] = np.nan
    unc[valid] = scores.astype(np.float32, copy=False)
    return unc


def _to_metric_xy(
    coords_xy: np.ndarray,
    src: rasterio.DatasetReader,
) -> np.ndarray:
    """Convert CRS coordinates to isotropic meters (scaling lon by cos(lat) if geographic)."""
    arr = np.asarray(coords_xy, dtype=np.float64)
    if src.crs is not None and src.crs.is_geographic:
        center_lat = src.bounds.bottom + 0.5 * (src.bounds.top - src.bounds.bottom)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * max(0.1, math.cos(math.radians(center_lat)))
        scaled = arr.copy()
        scaled[..., 0] *= m_per_deg_lon
        scaled[..., 1] *= m_per_deg_lat
        return scaled
    return arr


def suggest_active_learning_samples(
    uncertainty_or_prob_path: str | Path,
    *,
    n_samples: int = 50,
    min_distance_m: float = 200.0,
    min_uncertainty: float = 0.5,
    existing_coords_xy: Sequence[tuple[float, float]] | None = None,
    lulc_path: str | Path | None = None,
    target_classes: Sequence[int] | None = None,
    method: UncertaintyMethod = "confusion_index",
    output_csv_path: str | Path | None = None,
    block_size: int = 2048,
) -> pd.DataFrame:
    """Suggest spatially separated high-uncertainty points for active learning.

    Accepts either a single-band uncertainty GeoTIFF (such as ``Confusion Index``)
    or a multi-band ``Class Probability Stack`` GeoTIFF, enforcing spatial separation
    via ``scipy.spatial.cKDTree`` with zero GeoPandas dependency.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if min_distance_m < 0:
        raise ValueError("min_distance_m must be non-negative.")

    target_set = {int(c) for c in target_classes} if target_classes is not None else None
    pool_limit = max(n_samples * 200, 20_000)

    with rasterio.open(uncertainty_or_prob_path) as src:
        height, width = src.height, src.width
        crs = src.crs
        transform = src.transform

        existing_tree: cKDTree | None = None
        if existing_coords_xy:
            coords_arr = np.asarray(existing_coords_xy, dtype=np.float64)
            if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
                raise ValueError("existing_coords_xy must be a sequence of (x, y) pairs.")
            existing_tree = cKDTree(_to_metric_xy(coords_arr, src))

        lulc_src: rasterio.DatasetReader | None = None
        lulc_vrt: WarpedVRT | None = None
        heap: list[tuple[float, int, int, int | None]] = []
        try:
            if lulc_path is not None:
                lulc_src = rasterio.open(lulc_path)
                lulc_vrt = WarpedVRT(
                    lulc_src,
                    crs=crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=Resampling.nearest,
                )

            for r_off in range(0, height, block_size):
                win_h = min(block_size, height - r_off)
                for c_off in range(0, width, block_size):
                    win_w = min(block_size, width - c_off)
                    win = Window(c_off, r_off, win_w, win_h)
                    blk = src.read(window=win, masked=True)
                    raw = np.ma.getdata(blk).astype(np.float32, copy=False)
                    mask = np.ma.getmaskarray(blk)
                    valid = (~mask.any(axis=0)) & np.isfinite(raw).all(axis=0)

                    if src.count == 1:
                        unc = np.where(valid, raw[0], np.nan)
                    else:
                        unc = _uncertainty_from_probs(raw, valid, method=method)

                    cand_mask = valid & np.isfinite(unc) & (unc >= min_uncertainty)

                    lulc_blk: NDArray[np.integer[Any]] | None = None
                    if lulc_vrt is not None:
                        l_ma = lulc_vrt.read(1, window=win, masked=True)
                        lulc_blk = np.ma.getdata(l_ma)
                        cand_mask &= (~np.ma.getmaskarray(l_ma)) & (lulc_blk > 0)
                        if target_set is not None:
                            cand_mask &= np.isin(lulc_blk, list(target_set))

                    if not cand_mask.any():
                        continue

                    rr, cc = np.where(cand_mask)
                    vals = unc[rr, cc]
                    order = np.argsort(-vals)
                    for idx in order[:pool_limit]:
                        score = float(vals[idx])
                        r_abs = r_off + int(rr[idx])
                        c_abs = c_off + int(cc[idx])
                        pred_cls = (
                            int(lulc_blk[rr[idx], cc[idx]]) if lulc_blk is not None else None
                        )
                        item = (score, r_abs, c_abs, pred_cls)
                        if len(heap) < pool_limit:
                            heapq.heappush(heap, item)
                        elif score > heap[0][0]:
                            heapq.heapreplace(heap, item)
        finally:
            if lulc_vrt is not None:
                lulc_vrt.close()
            if lulc_src is not None:
                lulc_src.close()

        candidates_sorted = sorted(heap, key=lambda t: (-t[0], t[1], t[2]))
        rows: list[dict[str, Any]] = []
        accepted_metric_xy: list[np.ndarray] = []

        for score, r_abs, c_abs, pred_cls in candidates_sorted:
            x, y = rasterio.transform.xy(transform, r_abs, c_abs, offset="center")
            pt_metric = _to_metric_xy(np.array([float(x), float(y)], dtype=np.float64), src)
            if existing_tree is not None and min_distance_m > 0:
                dist_ex, _ = existing_tree.query(pt_metric)
                if float(dist_ex) < min_distance_m:
                    continue

            if min_distance_m > 0 and accepted_metric_xy:
                accepted_tree = cKDTree(np.vstack(accepted_metric_xy))
                dist_acc, _ = accepted_tree.query(pt_metric)
                if float(dist_acc) < min_distance_m:
                    continue

            accepted_metric_xy.append(pt_metric)
            rows.append(
                {
                    "rank": len(accepted_metric_xy),
                    "x": float(x),
                    "y": float(y),
                    "row": r_abs,
                    "col": c_abs,
                    "uncertainty": score,
                    "predicted_class": pred_cls,
                }
            )
            if len(accepted_metric_xy) >= n_samples:
                break

    df = pd.DataFrame(
        rows,
        columns=["rank", "x", "y", "row", "col", "uncertainty", "predicted_class"],
    )
    if output_csv_path is not None:
        out = Path(output_csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df

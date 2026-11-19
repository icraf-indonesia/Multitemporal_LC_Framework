"""Multi-epoch land cover transition matrices, Pontius & Millones disagreement, and Sankey outputs.

Study-area-agnostic implementation supporting arbitrary configured epochs,
runnable strictly on the lean core install (ADR-0005). Uses Plotly when
installed (``[viz]``) or falls back to a self-contained HTML Sankey table/SVG.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from lc_framework.config import ClassSchema
from lc_framework.evaluation.accuracy import compute_pixel_area_m2


@dataclass(frozen=True)
class IntervalTransitionSummary:
    """Transition matrix and Pontius & Millones / Santacruz disagreement for ``t1 -> t2``."""

    t1_label: str
    t2_label: str
    transition_matrix_ha: pd.DataFrame
    class_summary_ha: pd.DataFrame
    overall_disagreement: dict[str, float]


@dataclass(frozen=True)
class MultiEpochTrajectoryResult:
    """Complete multi-epoch transition and disagreement summary across ``>= 2`` epochs."""

    epochs: list[str]
    interval_summaries: list[IntervalTransitionSummary]
    net_summary: IntervalTransitionSummary
    class_colors: dict[str, str]


def _resolve_class_maps(
    arr_t1: np.ndarray,
    arr_t2: np.ndarray,
    class_schema: ClassSchema | None,
    class_names: Mapping[int, str] | None,
    nodata_values: Sequence[int | float],
) -> tuple[list[int], list[str], dict[str, str]]:
    if class_schema is not None:
        ids = list(class_schema.class_ids)
        names = list(class_schema.class_names)
        colors = {c.name: c.color for c in class_schema.classes}
        return ids, names, colors

    nd_set = {int(v) for v in nodata_values if np.isfinite(v)}
    if class_names is not None:
        ids = sorted(int(k) for k in class_names if int(k) not in nd_set)
        names = [str(class_names[cid]) for cid in ids]
        colors = {name: "#607d8b" for name in names}
        return ids, names, colors

    observed = set(np.unique(arr_t1).tolist()) | set(np.unique(arr_t2).tolist())
    ids = sorted(int(v) for v in observed if int(v) not in nd_set and int(v) > 0)
    names = [f"Class_{cid}" for cid in ids]
    colors = {name: "#607d8b" for name in names}
    return ids, names, colors


def compute_transition_summary(
    arr_t1: np.ndarray,
    arr_t2: np.ndarray,
    *,
    pixel_area_m2: float,
    t1_label: str,
    t2_label: str,
    class_schema: ClassSchema | None = None,
    class_names: Mapping[int, str] | None = None,
    nodata_values: Sequence[int | float] = (0, -1, 255),
) -> IntervalTransitionSummary:
    """Compute cross-tabulated transition matrix and Pontius & Millones disagreement.

    Decomposes total change into Quantity Disagreement and Allocation Disagreement
    (Pontius & Millones, 2011), and further decomposes Allocation into Exchange and
    Shift (Pontius & Santacruz, 2014).
    """
    if pixel_area_m2 <= 0:
        raise ValueError(f"pixel_area_m2 must be > 0, got {pixel_area_m2!r}.")

    a1 = np.asarray(arr_t1, dtype=np.int64)
    a2 = np.asarray(arr_t2, dtype=np.int64)
    if a1.shape != a2.shape:
        raise ValueError(f"Epoch arrays must share identical shape ({a1.shape} vs {a2.shape}).")

    class_ids, labels, _ = _resolve_class_maps(a1, a2, class_schema, class_names, nodata_values)
    if not class_ids:
        raise ValueError("No valid classes found for transition analysis.")

    q = len(class_ids)
    id_to_idx = {cid: idx for idx, cid in enumerate(class_ids)}

    valid = np.isin(a1, class_ids) & np.isin(a2, class_ids)
    sub_counts: np.ndarray = np.zeros((q, q), dtype=np.int64)
    if np.any(valid):
        idx1 = np.vectorize(id_to_idx.__getitem__)(a1[valid])
        idx2 = np.vectorize(id_to_idx.__getitem__)(a2[valid])
        flat = np.bincount(idx1 * q + idx2, minlength=q * q)
        sub_counts = np.asarray(flat, dtype=np.int64).reshape((q, q))

    pixel_area_ha = float(pixel_area_m2) / 10_000.0
    sub_areas = np.round(sub_counts.astype(np.float64) * pixel_area_ha, 4)
    total_landscape_ha = max(float(sub_areas.sum()), 1e-9)

    matrix_df = pd.DataFrame(sub_areas, index=labels, columns=labels)
    matrix_df.index.name = f"From ({t1_label})"
    matrix_df.columns.name = f"To ({t2_label})"

    row_totals = sub_areas.sum(axis=1)
    col_totals = sub_areas.sum(axis=0)
    diagonal = np.diag(sub_areas)

    losses = row_totals - diagonal
    gains = col_totals - diagonal
    net_change = col_totals - row_totals

    # Pontius & Millones (2011) Quantity and Allocation Disagreement
    quantity_disagreement = np.abs(gains - losses)
    allocation_disagreement = 2.0 * np.minimum(losses, gains)
    total_change = losses + gains

    # Pontius & Santacruz (2014) Exchange and Shift decomposition
    pairwise_min = np.minimum(sub_areas, sub_areas.T)
    np.fill_diagonal(pairwise_min, 0.0)
    exchange = 2.0 * pairwise_min.sum(axis=1)
    shift = np.maximum(0.0, allocation_disagreement - exchange)

    summary_df = pd.DataFrame(
        {
            f"area_{t1_label}_ha": np.round(row_totals, 4),
            f"area_{t2_label}_ha": np.round(col_totals, 4),
            "persistence_ha": np.round(diagonal, 4),
            "loss_ha": np.round(losses, 4),
            "gain_ha": np.round(gains, 4),
            "net_change_ha": np.round(net_change, 4),
            "quantity_disagreement_ha": np.round(quantity_disagreement, 4),
            "allocation_disagreement_ha": np.round(allocation_disagreement, 4),
            "exchange_ha": np.round(exchange, 4),
            "shift_ha": np.round(shift, 4),
            "total_change_ha": np.round(total_change, 4),
            "quantity_disagreement_pct": np.round(
                (quantity_disagreement / total_landscape_ha) * 100.0, 3
            ),
            "allocation_disagreement_pct": np.round(
                (allocation_disagreement / total_landscape_ha) * 100.0, 3
            ),
        },
        index=labels,
    )
    summary_df.index.name = "class_name"

    overall_q = float(np.round(0.5 * quantity_disagreement.sum(), 4))
    overall_a = float(np.round(0.5 * allocation_disagreement.sum(), 4))
    overall_e = float(np.round(0.5 * exchange.sum(), 4))
    overall_s = float(np.round(0.5 * shift.sum(), 4))
    overall_d = float(np.round(overall_q + overall_a, 4))

    overall_disagreement = {
        "total_landscape_ha": float(np.round(total_landscape_ha, 4)),
        "overall_persistence_ha": float(np.round(diagonal.sum(), 4)),
        "overall_quantity_disagreement_ha": overall_q,
        "overall_allocation_disagreement_ha": overall_a,
        "overall_exchange_ha": overall_e,
        "overall_shift_ha": overall_s,
        "overall_total_disagreement_ha": overall_d,
        "overall_quantity_disagreement_pct": float(
            np.round((overall_q / total_landscape_ha) * 100.0, 3)
        ),
        "overall_allocation_disagreement_pct": float(
            np.round((overall_a / total_landscape_ha) * 100.0, 3)
        ),
        "overall_total_disagreement_pct": float(
            np.round((overall_d / total_landscape_ha) * 100.0, 3)
        ),
    }

    return IntervalTransitionSummary(
        t1_label=t1_label,
        t2_label=t2_label,
        transition_matrix_ha=matrix_df,
        class_summary_ha=summary_df,
        overall_disagreement=overall_disagreement,
    )


def analyze_multi_epoch_rasters(
    epoch_rasters: Mapping[str, str | Path],
    *,
    class_schema: ClassSchema | None = None,
    class_names: Mapping[int, str] | None = None,
    nodata_values: Sequence[int | float] = (0, -1, 255),
    align_to_first: bool = True,
) -> MultiEpochTrajectoryResult:
    """Analyze consecutive intervals and net transitions across ``>= 2`` epoch LULC rasters."""
    epochs = list(epoch_rasters.keys())
    if len(epochs) < 2:
        raise ValueError(f"Multi-epoch analysis requires at least 2 epochs, got {len(epochs)}.")

    arrays: dict[str, np.ndarray] = {}
    first_path = Path(epoch_rasters[epochs[0]])
    with rasterio.open(first_path) as ref_src:
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_width = ref_src.width
        ref_height = ref_src.height
        pixel_area_m2 = compute_pixel_area_m2(ref_crs, ref_transform, ref_height)
        arrays[epochs[0]] = ref_src.read(1)

    for label in epochs[1:]:
        path = Path(epoch_rasters[label])
        with rasterio.open(path) as src:
            same_grid = (
                src.width == ref_width
                and src.height == ref_height
                and src.crs == ref_crs
                and src.transform == ref_transform
            )
            if same_grid:
                arrays[label] = src.read(1)
            elif align_to_first:
                with WarpedVRT(
                    src,
                    crs=ref_crs,
                    transform=ref_transform,
                    width=ref_width,
                    height=ref_height,
                    resampling=Resampling.nearest,
                ) as vrt:
                    arrays[label] = vrt.read(1)
            else:
                raise ValueError(
                    f"Epoch raster {label!r} grid does not match {epochs[0]!r}. "
                    "Pass align_to_first=True to align via nearest-neighbor WarpedVRT."
                )

    interval_summaries: list[IntervalTransitionSummary] = []
    for idx in range(len(epochs) - 1):
        t1, t2 = epochs[idx], epochs[idx + 1]
        interval_summaries.append(
            compute_transition_summary(
                arrays[t1],
                arrays[t2],
                pixel_area_m2=pixel_area_m2,
                t1_label=t1,
                t2_label=t2,
                class_schema=class_schema,
                class_names=class_names,
                nodata_values=nodata_values,
            )
        )

    net_summary = compute_transition_summary(
        arrays[epochs[0]],
        arrays[epochs[-1]],
        pixel_area_m2=pixel_area_m2,
        t1_label=epochs[0],
        t2_label=epochs[-1],
        class_schema=class_schema,
        class_names=class_names,
        nodata_values=nodata_values,
    )

    all_colors: dict[str, str] = {}
    for idx in range(len(epochs) - 1):
        _, _, c_map = _resolve_class_maps(
            arrays[epochs[idx]],
            arrays[epochs[idx + 1]],
            class_schema,
            class_names,
            nodata_values,
        )
        all_colors.update(c_map)

    return MultiEpochTrajectoryResult(
        epochs=epochs,
        interval_summaries=interval_summaries,
        net_summary=net_summary,
        class_colors=all_colors,
    )


def build_sankey_data(
    result: MultiEpochTrajectoryResult,
    *,
    min_flow_ha: float = 0.0,
) -> dict[str, Any]:
    """Build Sankey node/link structures across all epochs in ``result``."""
    if not result.interval_summaries:
        raise ValueError("result contains no interval summaries.")

    # Collect union of class names across all intervals so newly appearing classes never KeyError
    class_names: list[str] = []
    seen_classes: set[str] = set()
    for interval in result.interval_summaries:
        for cname in list(interval.transition_matrix_ha.index) + list(
            interval.transition_matrix_ha.columns
        ):
            name_str = str(cname)
            if name_str not in seen_classes:
                seen_classes.add(name_str)
                class_names.append(name_str)

    nodes: list[str] = []
    node_colors: list[str] = []
    node_index: dict[tuple[str, str], int] = {}

    for epoch in result.epochs:
        for cname in class_names:
            node_index[(epoch, cname)] = len(nodes)
            nodes.append(f"{cname} ({epoch})")
            node_colors.append(result.class_colors.get(cname, "#607d8b"))

    links: list[dict[str, Any]] = []
    for interval in result.interval_summaries:
        mat = interval.transition_matrix_ha
        for src_cls in mat.index:
            for dst_cls in mat.columns:
                val = float(mat.loc[src_cls, dst_cls])
                if val > min_flow_ha:
                    links.append(
                        {
                            "source": node_index[(interval.t1_label, str(src_cls))],
                            "target": node_index[(interval.t2_label, str(dst_cls))],
                            "value": round(val, 4),
                            "from_epoch": interval.t1_label,
                            "to_epoch": interval.t2_label,
                            "from_class": str(src_cls),
                            "to_class": str(dst_cls),
                        }
                    )

    return {"nodes": nodes, "node_colors": node_colors, "links": links}


def _render_standalone_svg_sankey(
    result: MultiEpochTrajectoryResult,
    data: dict[str, Any],
) -> str:
    """Render a lightweight self-contained interactive SVG Sankey diagram (zero dependencies)."""
    n_epochs = len(result.epochs)
    width = max(760, 240 * n_epochs)
    height = 420
    margin_x, margin_y = 60.0, 40.0
    bar_w = 22.0
    plot_h = height - 2.0 * margin_y
    col_step = (width - 2.0 * margin_x - bar_w) / max(1, n_epochs - 1)

    # Compute node totals per (epoch, class)
    node_totals: dict[int, float] = {i: 0.0 for i in range(len(data["nodes"]))}
    out_totals: dict[int, float] = {i: 0.0 for i in range(len(data["nodes"]))}
    in_totals: dict[int, float] = {i: 0.0 for i in range(len(data["nodes"]))}
    for lk in data["links"]:
        out_totals[lk["source"]] += float(lk["value"])
        in_totals[lk["target"]] += float(lk["value"])
    for idx in node_totals:
        node_totals[idx] = max(out_totals[idx], in_totals[idx])

    n_classes = len(data["nodes"]) // n_epochs
    max_col_total = max(
        (
            sum(node_totals[e * n_classes + c] for c in range(n_classes))
            for e in range(n_epochs)
        ),
        default=1.0,
    )
    max_col_total = max(max_col_total, 1e-6)
    pad_px = 6.0
    avail_h = max(40.0, plot_h - pad_px * max(0, n_classes - 1))

    node_box: dict[int, tuple[float, float, float]] = {}  # idx -> (x, y, h)
    svg_parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px; '
        f'background:#ffffff; border:1px solid #e5e7eb; border-radius:6px;">'
    ]
    for e_idx, epoch in enumerate(result.epochs):
        x = margin_x + e_idx * col_step
        svg_parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="22" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="#111827">{html.escape(epoch)}</text>'
        )
        y_cursor = margin_y
        for c_idx in range(n_classes):
            nid = e_idx * n_classes + c_idx
            val = node_totals[nid]
            h = max(2.0, (val / max_col_total) * avail_h) if val > 0 else 0.0
            node_box[nid] = (x, y_cursor, h)
            if h > 0:
                color = html.escape(data["node_colors"][nid])
                label = html.escape(data["nodes"][nid])
                svg_parts.append(
                    f'<rect x="{x:.1f}" y="{y_cursor:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                    f'fill="{color}" rx="2"><title>{label}: {val:.2f} ha</title></rect>'
                )
                y_cursor += h + pad_px

    out_offset: dict[int, float] = {i: 0.0 for i in range(len(data["nodes"]))}
    in_offset: dict[int, float] = {i: 0.0 for i in range(len(data["nodes"]))}
    for lk in data["links"]:
        s_id, t_id, val = int(lk["source"]), int(lk["target"]), float(lk["value"])
        sx, sy, sh = node_box[s_id]
        tx, ty, th = node_box[t_id]
        if sh <= 0 or th <= 0 or node_totals[s_id] <= 0 or node_totals[t_id] <= 0:
            continue
        lh_s = (val / node_totals[s_id]) * sh
        lh_t = (val / node_totals[t_id]) * th
        y0 = sy + out_offset[s_id] + 0.5 * lh_s
        y1 = ty + in_offset[t_id] + 0.5 * lh_t
        out_offset[s_id] += lh_s
        in_offset[t_id] += lh_t
        x0 = sx + bar_w
        x1 = tx
        cx = 0.5 * (x0 + x1)
        stroke_w = max(1.0, 0.5 * (lh_s + lh_t))
        color = html.escape(data["node_colors"][s_id])
        tip = html.escape(
            f"{lk['from_class']} ({lk['from_epoch']}) -> "
            f"{lk['to_class']} ({lk['to_epoch']}): {val:.2f} ha"
        )
        path_d = (
            f"M {x0:.1f} {y0:.1f} "
            f"C {cx:.1f} {y0:.1f}, {cx:.1f} {y1:.1f}, {x1:.1f} {y1:.1f}"
        )
        svg_parts.append(
            f'<path d="{path_d}" '
            f'fill="none" stroke="{color}" stroke-opacity="0.45" stroke-width="{stroke_w:.2f}">'
            f"<title>{tip}</title></path>"
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def export_sankey_html(
    result: MultiEpochTrajectoryResult,
    output_path: str | Path,
    *,
    title: str = "Multi-Epoch Land Cover Transitions",
    min_flow_ha: float = 0.0,
) -> Path:
    """Export multi-epoch transitions as an interactive HTML Sankey diagram.

    Uses ``plotly`` when available (``[viz]`` extra), and otherwise writes a
    self-contained interactive HTML/SVG Sankey diagram with zero external dependencies.
    """
    data = build_sankey_data(result, min_flow_ha=min_flow_ha)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Sankey(
                    node={
                        "label": data["nodes"],
                        "color": data["node_colors"],
                        "pad": 15,
                        "thickness": 20,
                    },
                    link={
                        "source": [lk["source"] for lk in data["links"]],
                        "target": [lk["target"] for lk in data["links"]],
                        "value": [lk["value"] for lk in data["links"]],
                    },
                )
            ]
        )
        fig.update_layout(title_text=title, font_size=12)
        fig.write_html(str(out), include_plotlyjs="cdn")
        return out
    except ImportError:
        pass

    svg_sankey = _render_standalone_svg_sankey(result, data)
    rows_html = "\n".join(
        f"<tr><td>{html.escape(lk['from_epoch'])} &rarr; {html.escape(lk['to_epoch'])}</td>"
        f"<td>{html.escape(lk['from_class'])}</td>"
        f"<td>{html.escape(lk['to_class'])}</td>"
        f"<td style='text-align:right'>{lk['value']:.2f}</td></tr>"
        for lk in data["links"]
    )
    payload_json = json.dumps(data)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
    path:hover {{ stroke-opacity: 0.85; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 820px; margin-top: 1.5rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.45rem 0.7rem; }}
    th {{ background: #f3f4f6; text-align: left; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Epochs: {html.escape(" -> ".join(result.epochs))}.
     Hover ribbons/bars for transition areas (ha).</p>
  {svg_sankey}
  <script id="sankey-data" type="application/json">{payload_json}</script>
  <table>
    <thead>
      <tr>
        <th>Interval</th><th>From Class</th><th>To Class</th>
        <th style="text-align:right">Area (ha)</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out

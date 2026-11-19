"""Stage 7: Standalone Accuracy Assessment, Active Learning & Multi-Epoch Trajectories.

Runnable with only the lean core install — zero Google Earth Engine, luma-stack,
scikit-learn, GeoPandas, or Plotly required (ADR-0005).
"""

from __future__ import annotations

from lc_framework.evaluation.accuracy import (
    ThematicAccuracyResult,
    UncertaintyMethod,
    compute_log_loss,
    compute_pixel_area_m2,
    compute_thematic_accuracy,
    count_raster_pixels,
    evaluate_probability_stack,
    suggest_active_learning_samples,
)
from lc_framework.evaluation.trajectory import (
    IntervalTransitionSummary,
    MultiEpochTrajectoryResult,
    analyze_multi_epoch_rasters,
    build_sankey_data,
    compute_transition_summary,
    export_sankey_html,
)

__all__ = [
    "IntervalTransitionSummary",
    "MultiEpochTrajectoryResult",
    "ThematicAccuracyResult",
    "UncertaintyMethod",
    "analyze_multi_epoch_rasters",
    "build_sankey_data",
    "compute_log_loss",
    "compute_pixel_area_m2",
    "compute_thematic_accuracy",
    "compute_transition_summary",
    "count_raster_pixels",
    "evaluate_probability_stack",
    "export_sankey_html",
    "suggest_active_learning_samples",
]

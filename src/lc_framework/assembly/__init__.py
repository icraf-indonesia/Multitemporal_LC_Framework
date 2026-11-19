"""Stage 6: Standalone Classification Assembly (Object-Refined Pixel Probabilities).

Runnable with only the lean core install — zero Google Earth Engine, luma-stack,
or ML library dependencies required (ADR-0005).
"""

from __future__ import annotations

from lc_framework.assembly.hybrid import (
    DEFAULT_ADAPTIVE_SENSITIVITY,
    DEFAULT_STABILITY_TAU,
    STABILITY_NODATA,
    HybridAssemblyOutputs,
    HybridBlendResult,
    blend_pixel_and_object_probabilities,
    compute_confusion_index,
    run_hybrid_assembly,
)
from lc_framework.assembly.postprocess import (
    MaskRule,
    apply_mask_rules,
    apply_mmu,
    apply_mmu_array,
    derive_hard_labels,
    derive_hard_labels_raster,
)
from lc_framework.assembly.smoothing import (
    bayesian_smooth,
    bayesian_smooth_raster,
    heal_array_seams,
    heal_probability_array_spikes,
    heal_probability_raster_spikes,
    heal_raster_seams,
    local_mean_variance,
    logits_to_probs,
    probs_to_logits,
)

__all__ = [
    "DEFAULT_ADAPTIVE_SENSITIVITY",
    "DEFAULT_STABILITY_TAU",
    "STABILITY_NODATA",
    "HybridAssemblyOutputs",
    "HybridBlendResult",
    "MaskRule",
    "apply_mask_rules",
    "apply_mmu",
    "apply_mmu_array",
    "bayesian_smooth",
    "bayesian_smooth_raster",
    "blend_pixel_and_object_probabilities",
    "compute_confusion_index",
    "derive_hard_labels",
    "derive_hard_labels_raster",
    "heal_array_seams",
    "heal_probability_array_spikes",
    "heal_probability_raster_spikes",
    "heal_raster_seams",
    "local_mean_variance",
    "logits_to_probs",
    "probs_to_logits",
    "run_hybrid_assembly",
]

"""Sample Augmentation workflows (GEE AlphaEarth embedding similarity search)."""

from __future__ import annotations

from lc_framework.samples.augmentation import (
    ALPHAEARTH_COLLECTION,
    DEFAULT_SIMILARITY_THRESHOLD,
    UNCALIBRATED_QA_DISCLAIMER,
    DiagnosticSoftmaxOverlay,
    SampleAugmentationParams,
    SampleAugmentationResult,
    build_class_similarity_stack,
    build_diagnostic_similarity_overlay,
    compute_class_cosine_similarity,
    export_augmented_samples_to_asset,
    export_augmented_samples_to_drive,
    extract_all_candidate_centroids,
    extract_class_candidate_centroids,
    get_alphaearth_mosaic,
    merge_augmented_samples,
    resolve_class_ids,
    resolve_class_thresholds,
    run_sample_augmentation,
)

__all__ = [
    "ALPHAEARTH_COLLECTION",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DiagnosticSoftmaxOverlay",
    "SampleAugmentationParams",
    "SampleAugmentationResult",
    "UNCALIBRATED_QA_DISCLAIMER",
    "build_class_similarity_stack",
    "build_diagnostic_similarity_overlay",
    "compute_class_cosine_similarity",
    "export_augmented_samples_to_asset",
    "export_augmented_samples_to_drive",
    "extract_all_candidate_centroids",
    "extract_class_candidate_centroids",
    "get_alphaearth_mosaic",
    "merge_augmented_samples",
    "resolve_class_ids",
    "resolve_class_thresholds",
    "run_sample_augmentation",
]

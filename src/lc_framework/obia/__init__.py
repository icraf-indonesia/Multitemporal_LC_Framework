"""Pluggable segmentation (GEE SNIC & local SLIC) and pure Geo-Object Classification."""

from __future__ import annotations

from lc_framework.obia.base import (
    SEGMENT_NODATA,
    Segmenter,
    valid_segment_mask,
    write_segment_raster,
)
from lc_framework.obia.gee_snic import (
    DEFAULT_SNIC_HASH_MODULUS,
    DEFAULT_SNIC_ID_STRIDE,
    GEESnicSegmenter,
    SNICParams,
)
from lc_framework.obia.geobia import (
    GeobiaClassificationResult,
    GeobiaPipelineOutputs,
    assign_segment_labels,
    classify_segments,
    pool_features_by_segments,
    project_segment_probabilities,
    run_geobia_classification,
)
from lc_framework.obia.local_slic import LocalSlicSegmenter, SLICParams

__all__ = [
    "DEFAULT_SNIC_HASH_MODULUS",
    "DEFAULT_SNIC_ID_STRIDE",
    "GEESnicSegmenter",
    "GeobiaClassificationResult",
    "GeobiaPipelineOutputs",
    "LocalSlicSegmenter",
    "SEGMENT_NODATA",
    "SLICParams",
    "SNICParams",
    "Segmenter",
    "assign_segment_labels",
    "classify_segments",
    "pool_features_by_segments",
    "project_segment_probabilities",
    "run_geobia_classification",
    "valid_segment_mask",
    "write_segment_raster",
]

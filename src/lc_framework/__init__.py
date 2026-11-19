"""LC_Framework: Unified probability-based LULC classification framework.

Top-level imports intentionally stay lean and never eagerly import optional
dependencies (earthengine-api, luma_ge, scikit-learn, lightgbm, xgboost,
scikit-image). See ADR-0005.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

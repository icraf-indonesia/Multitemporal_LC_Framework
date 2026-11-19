"""Monthly median pre-compositing and direct seasonal composites (GEE)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lc_framework.temporal.ccdc import specs as ccdc_specs

if TYPE_CHECKING:
    import ee

CompositeReducer = Literal["median", "mean"] | int | float


def to_monthly_median(
    collection: ee.ImageCollection,
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    """Aggregate an optical collection into monthly medians timestamped on the 15th.

    Drops empty months (``n == 0``) to suppress residual cloud fringes and reduce
    GEE EECU compute cost ahead of CCDC fitting.
    """
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date!r}) must be >= start_date ({start_date!r}).")

    ee_mod = ccdc_specs._require_ee()
    start = ee_mod.Date(start_date)
    end = ee_mod.Date(end_date)
    # Use ceil() so full-year ranges like 2024-01-01..2024-12-31 include December
    n_months = end.difference(start, "month").ceil()

    def _month_composite(month_offset: Any) -> Any:
        month_start = start.advance(month_offset, "month")
        monthly_imgs = collection.filterDate(month_start, month_start.advance(1, "month"))
        return (
            monthly_imgs.median()
            .set("system:time_start", month_start.advance(15, "day").millis())
            .set("n", monthly_imgs.size())
        )

    composites = ee_mod.List.sequence(0, n_months.subtract(1)).map(_month_composite)
    return ee_mod.ImageCollection.fromImages(composites).filter(ee_mod.Filter.gt("n", 0))


def seasonal_composite(
    collection: ee.ImageCollection,
    year: int,
    start_month: int,
    end_month: int,
    *,
    reducer: CompositeReducer = "median",
) -> ee.Image:
    """Generate a direct seasonal composite for ``year`` and months ``[start_month, end_month]``.

    Supports wrap-around seasons (e.g. ``start_month=12, end_month=4`` spans
    ``year - 1`` through ``year``) and handles ``end_month=12`` safely via date advance.
    """
    if not (1 <= start_month <= 12) or not (1 <= end_month <= 12):
        raise ValueError(
            f"start_month ({start_month}) and end_month ({end_month}) must lie in 1..12."
        )

    ee_mod = ccdc_specs._require_ee()
    if start_month > end_month:
        start_dt = ee_mod.Date.fromYMD(year - 1, start_month, 1)
        end_dt = ee_mod.Date.fromYMD(year, end_month, 1).advance(1, "month")
        filtered = collection.filterDate(start_dt, end_dt).filter(
            ee_mod.Filter.calendarRange(start_month, end_month, "month")
        )
    else:
        start_dt = ee_mod.Date.fromYMD(year, start_month, 1)
        end_dt = ee_mod.Date.fromYMD(year, end_month, 1).advance(1, "month")
        filtered = collection.filterDate(start_dt, end_dt)

    if reducer == "median":
        return filtered.median()
    if reducer == "mean":
        return filtered.mean()
    if isinstance(reducer, int | float) and not isinstance(reducer, bool):
        if not (0 <= float(reducer) <= 100):
            raise ValueError(f"Percentile reducer must lie in [0, 100], got {reducer!r}.")
        return filtered.reduce(ee_mod.Reducer.percentile([reducer]))
    raise ValueError(
        f"reducer must be 'median', 'mean', or a percentile in [0, 100]; got {reducer!r}."
    )

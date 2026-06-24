"""Feature construction — all built past-only to avoid future leakage (FR-009).

Responsibility: turn the gap-free series and supporting tables into model features:
deterministic trend + Fourier seasonality, calendar/payday/earthquake flags, the
effective locale-scoped holiday calendar, forward-filled oil, promotions, and
lag/rolling features. Every feature must be knowable at prediction time.

Implemented in tasks T023 (deterministic), T026 (holidays), T027 (calendar),
T028 (oil + promo), T029 (lags/rolling).
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# Fourier harmonics, chosen from the EDA evidence (periodogram + by-weekday/by-month):
# ~2-3 weekly harmonics capture the weekend jump, ~3-5 annual harmonics the yearly wave.
# The sharp December spike is left to holiday features rather than piling on harmonics.
# See analyze/concepts/seasonality-fourier.md and analyze/eda/03-seasonality.md.
WEEKLY_FOURIER_ORDER: int = 3
ANNUAL_FOURIER_ORDER: int = 4

# statsmodels CalendarFourier period aliases (pandas 3.0): weekly and year-end.
_WEEKLY_FREQ: str = "W"
_ANNUAL_FREQ: str = "YE"


def make_deterministic_features(
    index: pd.DatetimeIndex,
    *,
    trend_order: int = 1,
    weekly_order: int = WEEKLY_FOURIER_ORDER,
    annual_order: int = ANNUAL_FOURIER_ORDER,
    drop: bool = True,
) -> pd.DataFrame:
    """Build date-only deterministic features: a trend plus weekly + annual Fourier terms.

    Returns a table whose columns depend **only on the calendar date** — a constant, a
    linear ``trend`` (1, 2, 3, … over the index), and sine/cosine pairs approximating the
    weekly (period 7) and annual (period ~365) cycles via statsmodels ``CalendarFourier``
    and ``DeterministicProcess``. Because nothing is read from past *sales*, every row is
    knowable in advance and these features extend straight into the future 16-day horizon
    with no leakage (research R3; data-model.md; Constitution IV).

    **Pass one continuous index covering both the fit window and the forecast horizon**,
    then slice the result. The ``trend`` is a running counter that starts at 1 on the first
    date of ``index``; building train and horizon features in *separate* calls would restart
    the counter and break the trend's meaning. The index must also be gap-free daily so the
    Fourier phases stay aligned with real weekdays (a dropped day slides the weekly/annual
    waves out of phase — see analyze/data-traps/01-missing-calendar-days.md); this is
    asserted below, mirroring :func:`validation.assert_gapfree`.

    Args:
        index: A daily ``DatetimeIndex`` (or datetime-like values) for the dates to build
            features for — typically the union of the training dates and the forecast
            horizon. Order and duplicates are normalized away internally.
        trend_order: Polynomial trend order (1 = linear). Defaults to 1.
        weekly_order: Number of weekly Fourier harmonics. Defaults to 3 (EDA-backed).
        annual_order: Number of annual Fourier harmonics. Defaults to 4 (EDA-backed).
        drop: Drop perfectly collinear terms (passed to ``DeterministicProcess``). Defaults
            to True.

    Returns:
        A DataFrame indexed by the sorted, de-duplicated daily dates, with one column per
        deterministic term (``const``, ``trend``, and the weekly/annual ``sin``/``cos``
        pairs). The caller joins these to each series by date.

    Raises:
        AssertionError: If ``index`` is empty or, after sorting/de-duplication, the dates do
            not form a contiguous daily run (a gap would put the Fourier terms out of phase).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index)).unique().sort_values()

    assert len(idx) > 0, "make_deterministic_features received an empty index."
    expected_days = (idx[-1] - idx[0]).days + 1
    assert len(idx) == expected_days, (
        "Index is not gap-free daily — Fourier terms would fall out of phase. "
        f"Spanning {idx[0].date()} → {idx[-1].date()} expects {expected_days} days, "
        f"got {len(idx)}. Reindex gap-free before building deterministic features."
    )

    # A fixed daily freq lets DeterministicProcess/CalendarFourier compute positions reliably.
    idx = pd.DatetimeIndex(idx, freq="D")

    weekly = CalendarFourier(freq=_WEEKLY_FREQ, order=weekly_order)
    annual = CalendarFourier(freq=_ANNUAL_FREQ, order=annual_order)
    dp = DeterministicProcess(
        index=idx,
        constant=True,
        order=trend_order,
        additional_terms=[weekly, annual],
        drop=drop,
    )
    return dp.in_sample()

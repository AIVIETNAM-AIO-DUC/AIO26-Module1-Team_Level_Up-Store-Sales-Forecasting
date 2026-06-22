"""Model wrappers: baseline, deterministic, and hybrid.

Responsibility: provide a consistent fit/predict interface for each modeling stage —
seasonal-naive baseline, LinearRegression on deterministic features, and the hybrid
(linear fit + XGBoost on residuals). All operate in log space and clip predictions to
non-negative; randomness is seeded for reproducibility (Constitution III/IV).

Implemented in tasks T020 (seasonal_naive_predict), T024 (DeterministicModel),
T031 (HybridModel), T032 (sparse-series fallback).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .validation import HORIZON_DAYS, SERIES_KEY

# A weekly season: the dominant cycle in daily retail sales (research R2 / EDA Section 4).
SEASON_LENGTH: int = 7


def seasonal_naive_predict(
    train: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    *,
    season_length: int = SEASON_LENGTH,
    date_col: str = "date",
    target_col: str = "sales",
    key: Sequence[str] = SERIES_KEY,
) -> pd.DataFrame:
    """Seasonal-naive forecast: repeat each series' last observed week across the horizon.

    For every ``key`` series, each forecast day is predicted by the observed value from the
    **same weekday of the prior week** (FR-007). Because the horizon (16 days) is longer
    than the weekly season (7 days), this is equivalent to repeating the last *complete*
    training week: day-of-week ``w`` in the horizon always takes the most recent training
    observation whose weekday is ``w``.

    This is the bar every later model must beat (Stage 1 baseline; SC-002).

    **Leak-safe by construction**: predictions are read only from ``train`` (the data
    *before* the holdout/horizon). It never shifts a combined train+holdout series, so a
    holdout day can never read another holdout day's actual (Constitution IV).

    Args:
        train: A gap-free reindexed sales frame ending strictly before the forecast window
            (e.g. the ``train`` half of :func:`validation.train_holdout_split`, or the full
            history when forecasting the real test horizon). Must contain ``key``,
            ``date_col``, and ``target_col``.
        horizon: Number of future days to predict. Defaults to 16.
        season_length: Length of the seasonal cycle in days. Defaults to 7 (weekly).
        date_col: Name of the daily date column.
        target_col: Name of the observed-sales column to carry forward.
        key: Columns identifying a series. Defaults to (store_nbr, family).

    Returns:
        A tidy frame ``[*key, date, sales_pred]`` with one row per series × forecast day
        (e.g. 1,782 × 16 = 28,512 rows). ``date`` runs from the day after ``train`` ends
        for ``horizon`` days. Sparse series that were zero-padded by the gap-free reindex
        naturally predict ~0 (the intended near-zero fallback; research R8).

    Raises:
        AssertionError: If the series do not all share the same last training date (the
            gap-free contract is violated — pass reindexed data, not raw ``load_train()``).
    """
    key = list(key)

    # The vectorized weekday mapping below assumes every series ends on the same calendar
    # day, which holds iff the input was gap-free reindexed. Fail loudly otherwise.
    last_dates = train.groupby(key, observed=True)[date_col].max()
    assert last_dates.nunique() == 1, (
        "Series do not share a common last date — pass gap-free reindexed data "
        f"(found {last_dates.nunique()} distinct end dates: "
        f"{sorted(last_dates.unique())[:5]}...)."
    )
    last_date = last_dates.iloc[0]

    # The last complete week: one observation per weekday for every series.
    window_start = last_date - pd.Timedelta(days=season_length - 1)
    last_week = train.loc[train[date_col] >= window_start, key + [date_col, target_col]].copy()
    last_week["weekday"] = last_week[date_col].dt.weekday
    last_week = last_week.drop(columns=date_col)

    # The forecast calendar, one weekday label per future day.
    future = pd.DataFrame(
        {date_col: pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")}
    )
    future["weekday"] = future[date_col].dt.weekday

    # Each series × each future day, joined to that series' value for the matching weekday.
    series = train[key].drop_duplicates()
    preds = (
        series.merge(future, how="cross")
        .merge(last_week, on=key + ["weekday"], how="left")
        .drop(columns="weekday")
        .rename(columns={target_col: "sales_pred"})
        .sort_values(key + [date_col])
        .reset_index(drop=True)
    )
    return preds[key + [date_col, "sales_pred"]]

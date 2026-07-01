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

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression

from .features import (
    ANNUAL_FOURIER_ORDER,
    WEEKLY_FOURIER_ORDER,
    make_deterministic_features,
)
from .validation import HORIZON_DAYS, SERIES_KEY, clip_nonneg

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
    *before* the validation set/horizon). It never shifts a combined train+validation series,
    so a validation day can never read another validation day's actual (Constitution IV).

    Args:
        train: A gap-free reindexed sales frame ending strictly before the forecast window
            (e.g. the ``train`` half of :func:`validation.train_validation_split`, or the full
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


class DeterministicModel:
    """Trend + Fourier-seasonality model: one ``LinearRegression`` per series, in log space.

    Stage 2 of the teaching progression. The features
    (:func:`features.make_deterministic_features`) depend only on the calendar date, so they
    are identical across every series on a given day. A single global regression on date-only
    features would therefore predict the same value for all 1,782 series — useless. We fit a
    **separate** ``LinearRegression`` per ``(store_nbr, family)`` series on the shared date
    basis, so each series gets its own level, trend slope, and weekly/annual amplitude (Kaggle
    course "Trend"/"Seasonality" lessons; research R3). (For *full-history* series this is
    numerically identical to one multi-output fit — OLS per column — so nothing is lost; the
    per-series loop only lets us handle ragged histories, below.)

    **Two data realities make a naive fit explode — both handled here:**

    1. *Leading-zero blocks.* Many series carry ``sales = 0`` from 2013-01-01 until the store
       opened. A single linear trend through ``[dead zeros → active sales]`` is a step the
       smooth basis cannot represent, and least-squares overshoots wildly (predicting tens of
       thousands for a series whose max is a few thousand — even *in-sample*). Fix: each series
       is fit only on its **active history**, from its first non-zero day to the end. Interior
       zeros (closed Sundays/holidays) are kept — they are real signal.
    2. *Short histories.* A store that opened < 1 year before the cutoff cannot identify an
       **annual** cycle (the yearly sin/cos columns barely vary over a few months, so their
       coefficients blow up and extrapolate insanely). Fix: drop the annual Fourier terms for
       any series whose active span is shorter than ``min_days_for_annual``; it keeps trend +
       weekly seasonality, which a few months *can* support.

    A residual linear-trend overshoot remains for a handful of plateaued long-trending series
    (a linear trend keeps rising) — this is the honest behaviour of a deterministic model, not
    a defect, and it is bounded (no clip hack). Series with **no** sales at all predict 0 (the
    natural near-zero fallback; research R8). On the 16-day validation set this model lands roughly at
    parity with the seasonal-naive baseline: deterministic-only is a fair but modest stage;
    the large gains arrive with lags / promotions / holidays / the hybrid (Stages 3–4).

    Modeled in ``log1p`` space, inverted with ``expm1``, predictions clipped to ``>= 0``
    (FR-004, Constitution IV). Leak-safe: features are a pure function of the date, and
    :meth:`predict` builds the horizon features on the contiguous ``train ∪ horizon`` index
    (so the trend counter continues), then keeps only the future rows — no past sales leak in.
    Granularity here is implicitly per-series; the global-vs-per-group *decision* is a separate
    Stage-4 concern (R9 / T033).

    Typical use::

        model = DeterministicModel().fit(train)
        preds = model.predict()          # tidy [*key, date, sales_pred] over the 16-day horizon
    """

    def __init__(
        self,
        *,
        trend_order: int = 1,
        weekly_order: int = WEEKLY_FOURIER_ORDER,
        annual_order: int = ANNUAL_FOURIER_ORDER,
        min_days_for_annual: int = 365,
        key: Sequence[str] = SERIES_KEY,
        date_col: str = "date",
        target_col: str = "sales",
    ) -> None:
        """Configure the deterministic basis and the series/date/target column names.

        Args:
            trend_order: Polynomial trend order passed to the feature builder (1 = linear).
                Defaults to 1; a clip-free validation sweep showed dropping the trend underfits
                genuinely-trending series badly.
            weekly_order: Number of weekly Fourier harmonics. Defaults to the EDA-backed 3.
            annual_order: Number of annual Fourier harmonics. Defaults to the EDA-backed 4.
            min_days_for_annual: A series needs at least this many days of active history for
                its annual Fourier terms to be identifiable; shorter series are fit with trend
                + weekly only. Defaults to 365 (one year).
            key: Columns identifying a series. Defaults to (store_nbr, family).
            date_col: Name of the daily date column.
            target_col: Name of the observed-sales column to model.
        """
        self.trend_order = trend_order
        self.weekly_order = weekly_order
        self.annual_order = annual_order
        self.min_days_for_annual = min_days_for_annual
        self.key = list(key)
        self.date_col = date_col
        self.target_col = target_col

        # Per-series fitted models: col -> (LinearRegression, uses_annual) or None (all-zero).
        self._models: dict[object, tuple[LinearRegression, bool] | None] = {}
        self._series_columns: pd.Index | None = None
        self._train_index: pd.DatetimeIndex | None = None
        self._annual_cols: list[str] = []

    def _features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Build the deterministic feature table for ``index`` with this model's fixed orders.

        Centralizing this guarantees ``fit`` and ``predict`` use identical Fourier orders, so
        the column set always matches (mismatched columns would make sklearn error or
        misalign).
        """
        return make_deterministic_features(
            index,
            trend_order=self.trend_order,
            weekly_order=self.weekly_order,
            annual_order=self.annual_order,
        )

    def fit(self, train: pd.DataFrame) -> DeterministicModel:
        """Fit one linear model per series on its active history (see class docstring).

        Args:
            train: A gap-free reindexed sales frame with ``key``, ``date_col``, and
                ``target_col`` (e.g. the ``train`` half of
                :func:`validation.train_validation_split`, or the full history before forecasting
                the real test horizon).

        Returns:
            ``self`` (fitted), so calls can be chained.

        Raises:
            AssertionError: If the wide pivot has any missing cells — i.e. the series do not
                form a complete date × series grid. Pass gap-free reindexed data (T006) so
                every series spans the same calendar (closed days are zeros, not gaps).
        """
        wide = train.pivot_table(
            index=self.date_col, columns=self.key, values=self.target_col
        )

        n_missing = int(wide.isna().sum().sum())
        assert n_missing == 0, (
            f"Wide sales matrix has {n_missing} missing cell(s) — series do not form a "
            "complete date × series grid. Pass gap-free reindexed data "
            "(reindex_series_gapfree, T006)."
        )

        self._series_columns = wide.columns
        self._train_index = pd.DatetimeIndex(wide.index)

        x_full = self._features(self._train_index)
        self._annual_cols = [c for c in x_full.columns if "YE" in c]
        x_noannual = x_full.drop(columns=self._annual_cols)

        self._models = {}
        for col in wide.columns:
            y = wide[col]
            nonzero = y[y > 0]
            if nonzero.empty:
                self._models[col] = None  # never sold → predict 0 (near-zero fallback, R8)
                continue

            active = self._train_index >= nonzero.index.min()
            uses_annual = int(active.sum()) >= self.min_days_for_annual
            x = x_full if uses_annual else x_noannual

            model = LinearRegression().fit(
                x.loc[active], np.log1p(y[active].to_numpy(dtype=np.float64))
            )
            self._models[col] = (model, uses_annual)
        return self

    def predict(self, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
        """Predict the next ``horizon`` days for every fitted series, in raw sales units.

        Builds the horizon's deterministic features on the contiguous ``train ∪ horizon``
        index (continuing the trend counter), keeps the future rows, predicts each series with
        its own fitted model in log space, inverts with ``expm1``, and clips to ``>= 0``.

        Args:
            horizon: Number of future days to predict. Defaults to 16 (the competition
                horizon / local validation length).

        Returns:
            A tidy frame ``[*key, date, sales_pred]`` with one row per series × forecast day
            (1,782 × 16 = 28,512 rows), matching :func:`seasonal_naive_predict` so both can
            be scored the same way. ``date`` runs from the day after training ends.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._train_index is None or self._series_columns is None:
            raise RuntimeError("DeterministicModel.predict called before fit.")

        last_date = self._train_index.max()
        future = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        # train ∪ horizon is contiguous, so the trend counter continues and the gap-free
        # assert inside the feature builder will not false-trip; keep only the future rows.
        combined = self._train_index.append(future)
        x_future_full = self._features(combined).loc[future]
        x_future_noannual = x_future_full.drop(columns=self._annual_cols)

        preds: dict[object, npt.NDArray] = {}
        for col, fitted in self._models.items():
            if fitted is None:
                preds[col] = np.zeros(horizon, dtype=np.float64)
                continue
            model, uses_annual = fitted
            x = x_future_full if uses_annual else x_future_noannual
            preds[col] = clip_nonneg(np.expm1(model.predict(x)))

        pred_wide = pd.DataFrame(
            preds, index=pd.Index(future, name=self.date_col), columns=self._series_columns
        )
        tidy = (
            pred_wide.stack(self.key, future_stack=True)
            .rename("sales_pred")
            .reset_index()
            .sort_values(self.key + [self.date_col])
            .reset_index(drop=True)
        )
        return tidy[self.key + [self.date_col, "sales_pred"]]


class LinearFeatureModel:
    """Per-series ``LinearRegression`` on a **prebuilt** feature matrix (Stage 3).

    Generalizes :class:`DeterministicModel`'s per-series loop to any set of feature columns —
    the deterministic basis (trend + Fourier) *plus* holiday / calendar / promotion / oil / lag
    features composed by the caller. Keeping feature *building* in the notebook (one tidy matrix
    via the ``features`` builders) and the *fit* here lets us add feature groups "one at a time"
    and re-score, without this class coupling to holidays/stores/oil/transactions.

    Granularity is **per-series** by design: the global-vs-per-group decision is a separate Stage-4
    concern (R9 / T033), and per-series is the per-group arm that decision will benchmark. The same
    two ragged-history defenses as Stage 2 apply, plus one the lag features introduce:

    1. *Leading-zero blocks* — each series is fit only on its **active history** (from its first
       non-zero sales day); interior closed-day zeros are kept.
    2. *Short histories* — annual Fourier columns (those whose name contains ``annual_marker``) are
       dropped for series with under ``min_days_for_annual`` active days, where they're
       unidentifiable.
    3. *Lag warm-up NaNs* — the first rows of each series have undefined lag/rolling features; rows
       with **any** NaN feature are dropped *at fit*. Prediction rows (the 16-day horizon) are
       fully populated and must contain no NaN (asserted in :meth:`predict`).

    Modeled in ``log1p`` space, inverted with ``expm1``, clipped to ``>= 0``. Leak-safety is the
    caller's responsibility (build features once over the full history, split by date, and run
    :func:`validation.assert_no_leak` with ``base_lag``); this class only fits and predicts.

    Typical use::

        model = LinearFeatureModel(feature_cols).fit(train_matrix)
        preds = model.predict(horizon_matrix)   # tidy [*key, date, sales_pred]
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        *,
        min_days_for_annual: int = 365,
        annual_marker: str = "YE",
        key: Sequence[str] = SERIES_KEY,
        date_col: str = "date",
        target_col: str = "sales",
    ) -> None:
        """Configure which columns are features and the series/date/target column names.

        Args:
            feature_cols: The feature column names to fit on (order is preserved and reused at
                predict time so coefficients stay aligned).
            min_days_for_annual: Minimum active days for a series to keep annual Fourier columns.
                Defaults to 365.
            annual_marker: Substring identifying annual Fourier columns to drop for short series.
                Defaults to ``"YE"`` (statsmodels' year-end CalendarFourier names).
            key: Columns identifying a series. Defaults to (store_nbr, family).
            date_col: Name of the daily date column.
            target_col: Name of the observed-sales column to model.
        """
        self.feature_cols = list(feature_cols)
        self.min_days_for_annual = min_days_for_annual
        self.annual_cols = [c for c in self.feature_cols if annual_marker in c]
        self.key = list(key)
        self.date_col = date_col
        self.target_col = target_col

        # series-key -> (LinearRegression, uses_annual) or None (all-zero / no usable rows).
        self._models: dict[object, tuple[LinearRegression, bool] | None] = {}

    def _cols_for(self, uses_annual: bool) -> list[str]:
        """Feature columns for a series, dropping annual Fourier terms when not identifiable."""
        if uses_annual:
            return self.feature_cols
        return [c for c in self.feature_cols if c not in self.annual_cols]

    def fit(self, train: pd.DataFrame) -> LinearFeatureModel:
        """Fit one linear model per series on its active, NaN-free history.

        Args:
            train: Tidy frame ``[*key, date, target, *feature_cols]`` (gap-free per series),
                covering only the dates to train on (e.g. rows before the validation start).

        Returns:
            ``self`` (fitted), so calls can be chained.
        """
        self._models = {}
        for series_key, g in train.groupby(self.key, observed=True):
            g = g.sort_values(self.date_col)
            y = g[self.target_col].to_numpy(dtype=np.float64)

            nonzero = np.flatnonzero(y > 0)
            if nonzero.size == 0:
                self._models[series_key] = None  # never sold → predict 0 (R8)
                continue

            first = int(nonzero[0])
            active_days = len(y) - first
            uses_annual = active_days >= self.min_days_for_annual
            cols = self._cols_for(uses_annual)

            x = g.iloc[first:][cols]
            y_active = y[first:]
            keep = x.notna().all(axis=1).to_numpy()  # drop lag warm-up NaN rows
            x, y_active = x.loc[keep], y_active[keep]

            if len(x) == 0:
                self._models[series_key] = None
                continue

            model = LinearRegression().fit(x, np.log1p(y_active))
            self._models[series_key] = (model, uses_annual)
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Predict for every row of ``future`` using each series' fitted model, in raw sales units.

        Args:
            future: Tidy frame ``[*key, date, *feature_cols]`` for the rows to predict (e.g. the
                16-day validation/horizon). Feature columns must be fully populated (no NaN).

        Returns:
            A tidy frame ``[*key, date, sales_pred]``, one row per input row, sorted by series/date.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            AssertionError: If any feature value is NaN (horizon rows must be fully known).
        """
        if not self._models:
            raise RuntimeError("LinearFeatureModel.predict called before fit.")

        parts: list[pd.DataFrame] = []
        for series_key, g in future.groupby(self.key, observed=True):
            g = g.sort_values(self.date_col)
            fitted = self._models.get(series_key)

            if fitted is None:
                pred = np.zeros(len(g), dtype=np.float64)
            else:
                model, uses_annual = fitted
                x = g[self._cols_for(uses_annual)]
                assert x.notna().all().all(), (
                    f"NaN feature(s) for series {series_key} at prediction time — horizon "
                    "features must be fully populated (check lag warm-up / merges)."
                )
                pred = clip_nonneg(np.expm1(model.predict(x)))

            out = g[self.key + [self.date_col]].copy()
            out["sales_pred"] = pred
            parts.append(out)

        return (
            pd.concat(parts, ignore_index=True)
            .sort_values(self.key + [self.date_col])
            .reset_index(drop=True)
        )

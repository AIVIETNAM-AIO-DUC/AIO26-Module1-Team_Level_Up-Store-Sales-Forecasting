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

# Calendar-feature constants (T027).
# Payday in Ecuador falls on the 15th and the last day of the month, but the EDA shows the
# demand response is a multi-day wave, not a one-day spike (a naive single-day flag is ~1.01x;
# the real lift is a start-of-month surge of ~1.24x). So we also flag the days *following* each
# payday. See analyze/eda/04-calendar-holidays.md.
PAYDAY_WINDOW_DAYS: int = 3

# The 2016-04-16 magnitude-7.8 earthquake drove a ~1-week relief-buying spike. We flag a
# deliberately generous window so the model absorbs it as a known one-off event instead of
# bending trend/seasonality to fit it (research R7; analyze/data-traps/04-earthquake-anomaly.md).
_EARTHQUAKE_START: pd.Timestamp = pd.Timestamp("2016-04-16")
_EARTHQUAKE_END: pd.Timestamp = pd.Timestamp("2016-05-15")

# Lag/rolling constants (T029). Direct forecasting: predict all 16 horizon days at once, so every
# sales-derived lag must reach back at least the horizon length (16 days). The last horizon day is
# 16 days past the last real training day, so any sales-lag < 16 would need a value from inside the
# horizon — i.e. a prediction, not real data. A uniform base_lag keeps every lag/rolling feature
# leak-free with one model and no error compounding (research R11; analyze/concepts/lag-horizon.md).
# Equals validation.HORIZON_DAYS (kept literal here to avoid an import cycle).
BASE_LAG: int = 16
_SERIES_KEY: tuple[str, str] = ("store_nbr", "family")
DEFAULT_SALES_LAGS: tuple[int, ...] = (16, 17, 18)
DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (7, 14, 28)
DEFAULT_TRANSACTIONS_LAGS: tuple[int, ...] = (16,)


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


def effective_holiday_calendar(holidays: pd.DataFrame) -> pd.DataFrame:
    """Resolve ``holidays_events.csv`` into an *effective* dated calendar (research R6).

    The raw table is **not** a yes/no holiday flag (see
    analyze/data-traps/03-holidays.md). Two corrections turn it into one:

    1. **Drop transferred ghosts.** A row with ``transferred = True`` means the holiday
       did *not* happen on its listed date — it was officially moved, and a separate
       ``type = "Transfer"`` row carries the date it was actually observed. Dropping the
       ``transferred`` rows keeps only dates where something real happened.
    2. **Classify the effect.** Most surviving types (Holiday / Transfer / Bridge /
       Additional / Event) are days off → ``effect = "holiday"``. A ``Work Day`` row is
       the *opposite* — a normally-off day the government made into a working one → it is
       tagged ``effect = "work_day"`` so it never leaks into the holiday flag.

    Args:
        holidays: The raw frame from :func:`data.load_holidays` with columns ``date``,
            ``type``, ``locale``, ``locale_name``, ``description``, ``transferred``.

    Returns:
        The de-ghosted calendar — one row per surviving (date, locale-scoped) entry —
        with the original columns plus an ``effect`` column (``"holiday"`` or
        ``"work_day"``). ``type`` is retained so a later feature can still distinguish
        moved vs fixed-date holidays if needed. Locale scoping happens downstream in
        :func:`make_holiday_features`.
    """
    eff = holidays.loc[~holidays["transferred"]].copy()
    eff["effect"] = "holiday"
    eff.loc[eff["type"] == "Work Day", "effect"] = "work_day"
    return eff.reset_index(drop=True)


def make_holiday_features(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Build per-``(store_nbr, date)`` holiday indicator features (research R6; FR-009).

    Starts from :func:`effective_holiday_calendar`, then **scopes each entry by locale**
    so a holiday only touches the stores it actually applies to (see
    analyze/data-traps/03-holidays.md):

    - **National** → every store (cross join);
    - **Regional** → stores whose ``state`` matches ``locale_name``;
    - **Local** → stores whose ``city`` matches ``locale_name`` (typically 1-2 stores).

    Applying a local festival to all 54 stores would pair ``is_holiday = 1`` with normal
    sales elsewhere and teach the model the flag means nothing — locale scoping prevents
    that. Every feature here depends only on the calendar date and known store metadata,
    so all are knowable in advance and safe for the forecast horizon (no leakage).

    The result holds **only the store-days that carry an event**. Callers left-join it
    onto the gap-free series grid by ``(store_nbr, date)`` and fill the absent rows with
    0 (an ordinary day).

    Args:
        holidays: Raw frame from :func:`data.load_holidays`.
        stores: Store metadata from :func:`data.load_stores` (needs ``store_nbr``,
            ``city``, ``state``).

    Returns:
        A tidy frame ``[store_nbr, date, is_holiday, is_work_day, is_national_holiday]``,
        one row per affected store-day, flags in ``{0, 1}`` (``int8``). Multiple events
        on the same store-day collapse via max, so each flag is 1 if *any* matching event
        sets it.
    """
    eff = effective_holiday_calendar(holidays)
    keep = ["date", "locale", "effect"]

    national = eff.loc[eff["locale"] == "National", keep].merge(
        stores[["store_nbr"]], how="cross"
    )
    regional = eff.loc[eff["locale"] == "Regional", [*keep, "locale_name"]].merge(
        stores[["store_nbr", "state"]], left_on="locale_name", right_on="state"
    )
    local = eff.loc[eff["locale"] == "Local", [*keep, "locale_name"]].merge(
        stores[["store_nbr", "city"]], left_on="locale_name", right_on="city"
    )

    expanded = pd.concat(
        [national, regional, local], ignore_index=True
    )[["store_nbr", "date", "locale", "effect"]]

    expanded["is_holiday"] = (expanded["effect"] == "holiday").astype("int8")
    expanded["is_work_day"] = (expanded["effect"] == "work_day").astype("int8")
    expanded["is_national_holiday"] = (
        (expanded["effect"] == "holiday") & (expanded["locale"] == "National")
    ).astype("int8")

    flags = ["is_holiday", "is_work_day", "is_national_holiday"]
    return (
        expanded.groupby(["store_nbr", "date"], as_index=False)[flags]
        .max()
        .sort_values(["store_nbr", "date"])
        .reset_index(drop=True)
    )


def make_calendar_features(
    index: pd.DatetimeIndex, *, payday_window: int = PAYDAY_WINDOW_DAYS
) -> pd.DataFrame:
    """Build date-only calendar features: day-of-week, month, payday window, earthquake flag.

    Every column is a pure function of the calendar date — store-independent and knowable in
    advance — so the features extend straight into the forecast horizon with no leakage (FR-009).
    The caller joins them to each series by date.

    Columns produced:

    - **``dayofweek``** (0=Mon … 6=Sun) and **``month``** (1-12) — raw integer calendar codes.
      The *linear* model already encodes these cycles smoothly via Fourier terms
      (:func:`make_deterministic_features`), so these integers are mainly for the **tree** side
      of the later hybrid (XGBoost on residuals), which can split on them directly.
    - **``is_payday``** — exactly the 15th and the month-end (the literal Ecuadorian payday days).
    - **``is_payday_window``** — a payday day *or* one of the next ``payday_window`` days. The EDA
      shows the demand response is a multi-day wave (a start-of-month surge after the month-end
      payday), so the window captures the real signal a single-day flag misses
      (analyze/eda/04-calendar-holidays.md).
    - **``is_earthquake``** — 1 across the 2016-04-16 → 2016-05-15 relief-spike window (research R7).

    Args:
        index: A daily ``DatetimeIndex`` (or datetime-like values). Order and duplicates are
            normalized away internally. Need not be gap-free — each row is computed independently.
        payday_window: How many days *after* each payday to keep flagged in ``is_payday_window``.
            Defaults to 3 (EDA-backed start-of-month surge).

    Returns:
        A DataFrame indexed by the sorted, de-duplicated dates, with columns ``dayofweek``,
        ``month``, ``is_payday``, ``is_payday_window``, ``is_earthquake`` (all ``int8``).

    Raises:
        AssertionError: If ``index`` is empty.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index)).unique().sort_values()
    assert len(idx) > 0, "make_calendar_features received an empty index."

    out = pd.DataFrame(index=idx)
    out["dayofweek"] = idx.dayofweek.astype("int8")
    out["month"] = idx.month.astype("int8")
    out["is_payday"] = (idx.is_month_end | (idx.day == 15)).astype("int8")

    # Post-payday window: for each date, how many days since the most recent payday (the 15th or a
    # month-end). Generate paydays over a span padded back far enough that every date has one
    # before it, then look up the latest payday <= each date.
    span = pd.date_range(idx.min() - pd.Timedelta(days=40), idx.max(), freq="D")
    paydays = span[span.is_month_end | (span.day == 15)]
    last_payday = paydays[paydays.searchsorted(idx, side="right") - 1]
    days_since = (idx - last_payday).days
    out["is_payday_window"] = ((days_since >= 0) & (days_since <= payday_window)).astype("int8")

    out["is_earthquake"] = (
        (idx >= _EARTHQUAKE_START) & (idx <= _EARTHQUAKE_END)
    ).astype("int8")

    return out


def make_oil_features(
    oil: pd.DataFrame, index: pd.DatetimeIndex, *, col: str = "oil"
) -> pd.DataFrame:
    """Build a gap-free, **forward-filled** daily oil-price feature (research R4; FR-008).

    ``oil.csv`` has two kinds of hole on a daily calendar: weekends/holidays have no row (oil
    trades on business days only) and a few rows carry a blank price (e.g. 2013-01-01). Left over
    as ``NaN`` they crash linear models and spread through any derived feature; filled naively with
    ``interpolate``/``bfill`` they peek at *future* prices and leak. The fix is **forward-fill** —
    carry the last *known* price into each gap (a backward lookup, so past-only and honest) — with
    a single leading **back-fill** to cover the very first blank day, which has no prior price to
    look back to and is never an honest training target anyway (see
    analyze/data-traps/02-oil-gaps.md).

    Note the EDA found oil's apparent sales effect is largely a spurious trend artifact (raw corr
    ≈ −0.62, ≈ 0 once detrended), so this is a *candidate* feature for the validation set to judge, not a
    proven driver.

    Args:
        oil: Raw frame from :func:`data.load_oil` with columns ``date`` and ``dcoilwtico``.
        index: The daily ``DatetimeIndex`` (or datetime-like values) to return prices for —
            typically the union of training dates and the forecast horizon. The continuous fill
            calendar spans this index's min→max so gaps inside it are filled before slicing back.
        col: Output column name. Defaults to ``"oil"``.

    Returns:
        A DataFrame indexed by the sorted, de-duplicated dates of ``index`` with one column
        (``col``) of forward-filled prices and no missing values.

    Raises:
        AssertionError: If ``index`` is empty, or if any price remains missing after the fill.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index)).unique().sort_values()
    assert len(idx) > 0, "make_oil_features received an empty index."

    # Continuous daily calendar across the requested span, so weekend/holiday gaps are filled
    # from the last business-day price before we slice back to the requested dates.
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    price = oil.set_index("date")["dcoilwtico"].sort_index()

    # ffill = backward lookup (past-only, leak-free); the trailing bfill only touches the leading
    # run before the first known price (the one allowed exception).
    filled = price.reindex(full).ffill().bfill()

    out = filled.reindex(idx).rename(col).to_frame()
    out.index.name = None
    assert out[col].notna().all(), "oil price still has gaps after ffill+bfill."
    return out


def make_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the contemporaneous ``onpromotion`` feature — same-day, leak-free (FR-008).

    Unlike ``transactions`` (an *outcome* of the day, shipped with no test file → lag-only),
    ``onpromotion`` is a *decision made in advance*: ``test.csv`` carries it for all 16 horizon
    days with zero nulls. So it may be used on the **same day** as the prediction without any lag
    or leakage (analyze/eda/05-promotions-oil.md; analyze/data-traps/06-transactions-past-only.md).
    Closed days inserted by :func:`data.reindex_series_gapfree` already carry ``onpromotion = 0``.

    Args:
        df: A gap-free sales frame (e.g. from :func:`data.reindex_series_gapfree`) with columns
            ``store_nbr``, ``family``, ``date``, ``onpromotion``.

    Returns:
        A tidy ``[store_nbr, family, date, onpromotion]`` frame, one row per series-day.

    Raises:
        AssertionError: If any ``onpromotion`` value is missing (the contemporaneous guarantee).
    """
    cols = ["store_nbr", "family", "date", "onpromotion"]
    out = df.loc[:, cols].copy()
    assert out["onpromotion"].notna().all(), "onpromotion has NaNs — expected contemporaneous."
    return out.reset_index(drop=True)


def make_lag_features(
    df: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    *,
    base_lag: int = BASE_LAG,
    sales_lags: tuple[int, ...] = DEFAULT_SALES_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
    transactions_lags: tuple[int, ...] = DEFAULT_TRANSACTIONS_LAGS,
    key: tuple[str, str] = _SERIES_KEY,
    date_col: str = "date",
    target_col: str = "sales",
) -> pd.DataFrame:
    """Build leak-safe lag/rolling features via **direct forecasting** (research R11; FR-009).

    All features come from *past* ``sales`` (and optionally *lagged* ``transactions``) and obey one
    rule: **every sales-derived offset is at least ``base_lag`` (16) days**. Because the last
    horizon day is 16 days past the last real training day, any shorter lag would need a value from
    inside the horizon — a prediction, not real data. Anchoring everything at ``shift(base_lag)``
    keeps the whole 16-day horizon computable from real data with one model and no error compounding
    (see analyze/concepts/lag-horizon.md). The complementary ``base_lag`` guard lives in
    :func:`validation.assert_no_leak` (T029a).

    Features (per ``(store_nbr, family)`` series, computed on the gap-free daily index so a lag of
    *N rows* is exactly *N days*):

    - ``sales_lag_{L}`` for each ``L`` in ``sales_lags`` — the sales value ``L`` days ago.
    - ``sales_roll_{W}_lag_{base_lag}_mean`` for each ``W`` in ``roll_windows`` — mean of the ``W``
      days ending at ``shift(base_lag)`` (so the *newest* day in every window is still ``base_lag``
      days back). The ``lag_{base_lag}`` token in the name lets :func:`validation.assert_no_leak`
      verify the anchor offset without inspecting the data (T029a).
    - ``transactions_lag_{L}`` for each ``L`` in ``transactions_lags`` — only if ``transactions`` is
      given. ``transactions`` is **past-only** (no test file), so it may appear *only* as a lag
      (analyze/data-traps/06-transactions-past-only.md). Closed store-days (absent from
      ``transactions.csv``) are treated as 0 before lagging.

    ``onpromotion`` is deliberately **absent** here — it is provided for the horizon, so it is used
    contemporaneously by :func:`make_promotion_features`, not lagged.

    Early rows of each series (before enough history exists) carry ``NaN`` — expected, and handled
    downstream (dropped for the linear fit, or tolerated by the tree booster). The frame must be
    gap-free per series (e.g. from :func:`data.reindex_series_gapfree`) or row-based lags slip.

    Args:
        df: Gap-free sales frame with at least ``key``, ``date_col``, ``target_col``.
        transactions: Optional frame from :func:`data.load_transactions`
            (``store_nbr``, ``date``, ``transactions``). If ``None``, no transaction features.
        base_lag: Minimum legal sales-lag and the rolling anchor. Defaults to 16 (the horizon).
        sales_lags: Sales-lag offsets to emit; each must be ``>= base_lag``.
        roll_windows: Rolling-mean window lengths, all anchored at ``shift(base_lag)``.
        transactions_lags: Transaction-lag offsets; each must be ``>= base_lag``.
        key: Series key. Defaults to ``("store_nbr", "family")``.
        date_col: Date column name. Defaults to ``"date"``.
        target_col: Target column name. Defaults to ``"sales"``.

    Returns:
        A tidy frame ``[*key, date]`` plus one column per requested lag/rolling feature, one row
        per series-day, row-aligned to a date-sorted view of ``df``.

    Raises:
        AssertionError: If any requested sales/transaction offset is ``< base_lag`` (the
            direct-forecasting rule), guarding against a short lag slipping in.
    """
    keys = list(key)
    assert min(sales_lags) >= base_lag, (
        f"sales_lags {sales_lags} must all be >= base_lag ({base_lag}) — a shorter lag cannot be "
        "computed for the far end of the horizon without using predicted sales."
    )
    if transactions is not None:
        assert min(transactions_lags) >= base_lag, (
            f"transactions_lags {transactions_lags} must all be >= base_lag ({base_lag}) — "
            "transactions are past-only and unknown across the horizon."
        )

    df = df.sort_values(keys + [date_col])
    out = df[keys + [date_col]].copy()
    sales_by_series = df.groupby(keys, observed=True)[target_col]

    for lag in sales_lags:
        out[f"sales_lag_{lag}"] = sales_by_series.shift(lag)

    # Rolling means anchored at shift(base_lag); grouped so windows never span two series.
    # The lag_{base_lag} token in the name encodes the anchor offset for the T029a leak guard.
    for window in roll_windows:
        out[f"sales_roll_{window}_lag_{base_lag}_mean"] = sales_by_series.transform(
            lambda s, w=window: s.shift(base_lag).rolling(w).mean()
        )

    if transactions is not None:
        tx = transactions[["store_nbr", date_col, "transactions"]]
        merged = df[keys + [date_col]].merge(tx, on=["store_nbr", date_col], how="left")
        merged["transactions"] = merged["transactions"].fillna(0)
        merged = merged.sort_values(keys + [date_col])
        tx_by_series = merged.groupby(keys, observed=True)["transactions"]
        for lag in transactions_lags:
            out[f"transactions_lag_{lag}"] = tx_by_series.shift(lag).to_numpy()

    return out.reset_index(drop=True)

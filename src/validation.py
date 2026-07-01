"""Validation harness — the test of record for this project (Constitution IV).

Responsibility: the time-respecting 16-day validation split, RMSLE scoring in log space
with non-negative clipping, the no-leak and gap-free assertions, and the iteration
log helper. Every model is judged by these functions on the *same* validation set so
scores are comparable and honest.

Implemented in tasks T007 (assert_gapfree), T008 (train_validation_split),
T009 (rmsle, clip_nonneg), T010 (assert_no_leak), T011 (log_iteration),
T036 (validate_submission).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

SERIES_KEY: list[str] = ["store_nbr", "family"]

# The competition horizon is 16 days; the local validation set mirrors it exactly.
HORIZON_DAYS: int = 16

# Past-only feature sources for the base_lag guard (T029a): a feature derived from these may only
# appear as a lag of >= HORIZON_DAYS days, never contemporaneously. onpromotion and oil are NOT
# listed — they are knowable for the horizon, so they are used same-day and are exempt from the
# guard (research R11; data-model.md).
_LAGGED_SOURCES: tuple[str, ...] = ("sales", "transactions")
# Matches the "lag_<N>" offset token embedded in lag/rolling feature names (e.g. "sales_lag_16",
# "sales_roll_7_lag_16_mean"). N is the minimum offset (days back) the feature reads.
_LAG_OFFSET_RE = re.compile(r"lag_(\d+)")

# iteration_log.md lives at the repo root (src/ -> parents[1]).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
ITERATION_LOG: Path = REPO_ROOT / "iteration_log.md"


def clip_nonneg(pred: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Clip predictions to be non-negative (sales cannot be < 0; RMSLE needs ``>= 0``).

    Applied before scoring and before writing the submission so the same rule holds
    everywhere (Constitution V).
    """
    return np.clip(np.asarray(pred, dtype=np.float64), a_min=0.0, a_max=None)


def rmsle(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Root Mean Squared Logarithmic Error — the official competition metric.

    Computed as RMSE between ``log1p(y_true)`` and ``log1p(y_pred)``. Inputs are in raw
    sales units (not log space); predictions are clipped to ``>= 0`` first because
    ``log1p`` is undefined for negatives. ``log1p`` (= ``log(1 + x)``) maps the many
    zero-sales days to 0 cleanly. Every model is scored with this same function on the
    same validation set so the numbers are comparable (Constitution IV).

    Args:
        y_true: Actual sales (``>= 0``).
        y_pred: Predicted sales (any real values; clipped to ``>= 0`` internally).

    Returns:
        The RMSLE as a float (lower is better; 0.0 is a perfect prediction).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = clip_nonneg(y_pred)
    squared_log_error = (np.log1p(y_pred_arr) - np.log1p(y_true_arr)) ** 2
    return float(np.sqrt(np.mean(squared_log_error)))


def assert_gapfree(
    df: pd.DataFrame,
    *,
    key: Sequence[str] = SERIES_KEY,
    date_col: str = "date",
) -> None:
    """Assert every series has a complete, duplicate-free daily calendar.

    A "series" is one group of ``key`` (default ``(store_nbr, family)``). For each series
    the dates must form a contiguous daily run from its first to its last date — no
    missing days and no duplicates. Lag and seasonality features rely on this regular
    index; call this guard right after reindexing and before building any such feature.

    Args:
        df: A frame with the ``key`` columns and a datetime ``date_col``.
        key: Columns identifying a series. Defaults to (store_nbr, family).
        date_col: Name of the daily date column.

    Raises:
        AssertionError: If any series has missing days or duplicate dates, with a short
            sample of the offending series in the message.
    """
    key = list(key)
    stats = df.groupby(key, observed=True)[date_col].agg(["min", "max", "count", "nunique"])

    expected_days = (stats["max"] - stats["min"]).dt.days + 1
    has_missing = stats["nunique"] != expected_days
    has_duplicates = stats["count"] != stats["nunique"]
    bad = stats[has_missing | has_duplicates]

    if not bad.empty:
        sample = bad.head(5).to_string()
        raise AssertionError(
            f"{len(bad)} series are not gap-free "
            f"(missing days and/or duplicate dates). First offenders:\n{sample}"
        )


def assert_no_leak(
    feature_df: pd.DataFrame,
    prediction_date: str | pd.Timestamp,
    *,
    date_col: str = "date",
    strict: bool = False,
    base_lag: int | None = None,
) -> None:
    """Assert no feature row is dated past the prediction boundary (no future leak).

    Time-series leakage — using information unavailable at prediction time — makes a
    model look great in validation and fail on the real future. This guard enforces the
    boundary explicitly (Constitution IV).

    Two common uses:
      * Training must not see the validation set: ``assert_no_leak(train, val_start,
        strict=True)`` — every training row must be *strictly before* the validation set.
      * A feature set is "as of" a date: ``assert_no_leak(features, as_of)`` — nothing
        may be dated *after* ``as_of``.

    Timestamps are read from ``date_col`` if present, otherwise from a DatetimeIndex.

    **``base_lag`` guard (T029a).** When ``base_lag`` is given, also check the *feature
    columns*: every column derived from a past-only source (``sales``, ``transactions``)
    must encode a ``lag_<N>`` offset with ``N >= base_lag``. This enforces the
    direct-forecasting rule — the minimum sales-lag must be at least the horizon length, or
    the far end of the horizon would need a value from inside the horizon (research R11;
    see analyze/concepts/lag-horizon.md). A sales/transactions column that *looks* like a
    lag/rolling feature but carries no parseable offset fails loudly rather than slipping
    through. ``onpromotion`` and ``oil`` are exempt — they are knowable for the horizon and
    used contemporaneously.

    Args:
        feature_df: Frame whose rows carry a date (column ``date_col`` or a datetime index).
        prediction_date: The cutoff. Rows must be ``<= prediction_date`` (``strict=False``)
            or ``< prediction_date`` (``strict=True``).
        date_col: Name of the date column to check; falls back to the index.
        strict: If True, require dates strictly before ``prediction_date``.
        base_lag: If set, also assert every past-only lag/rolling feature column reads back
            at least this many days. Pass ``HORIZON_DAYS`` (16) for direct forecasting.

    Raises:
        AssertionError: If any row violates the date boundary (with count + worst date), or
            (with ``base_lag``) if a past-only feature uses an offset ``< base_lag`` or
            encodes no offset at all.
        KeyError: If no usable date column/index is found.
    """
    if date_col in feature_df.columns:
        ts = pd.to_datetime(feature_df[date_col])
    elif isinstance(feature_df.index, pd.DatetimeIndex):
        ts = feature_df.index.to_series()
    else:
        raise KeyError(
            f"No '{date_col}' column and the index is not a DatetimeIndex; "
            "cannot check timestamps for leakage."
        )

    cutoff = pd.Timestamp(prediction_date)
    violators = ts >= cutoff if strict else ts > cutoff

    if violators.any():
        n = int(violators.sum())
        worst = ts[violators].max().date()
        boundary = "before" if strict else "on or before"
        raise AssertionError(
            f"Leak detected: {n} row(s) dated past the cutoff "
            f"(features must be {boundary} {cutoff.date()}; worst offender {worst})."
        )

    if base_lag is not None:
        _assert_base_lag(feature_df.columns, base_lag)


def _assert_base_lag(columns: Sequence[str], base_lag: int) -> None:
    """Assert every past-only lag/rolling feature column reads back >= ``base_lag`` days (T029a).

    A column is "past-only" if its name starts with a source in ``_LAGGED_SOURCES`` (``sales`` /
    ``transactions``) and looks like a lag/rolling feature (contains ``"lag"`` or ``"roll"``). Such
    a column must encode at least one ``lag_<N>`` offset, and the smallest one must be ``>=
    base_lag``. Columns from contemporaneous sources (``onpromotion``, ``oil``) never match and are
    exempt by construction.

    Raises:
        AssertionError: If a past-only lag/rolling column encodes no offset, or its minimum
            offset is below ``base_lag`` (the short lag that would break direct forecasting).
    """
    for col in columns:
        name = str(col)
        is_past_only = name.startswith(_LAGGED_SOURCES)
        looks_laggy = ("lag" in name) or ("roll" in name)
        if not (is_past_only and looks_laggy):
            continue

        offsets = [int(m) for m in _LAG_OFFSET_RE.findall(name)]
        if not offsets:
            raise AssertionError(
                f"Feature '{name}' is a past-only ({'/'.join(_LAGGED_SOURCES)}) lag/rolling "
                "feature but encodes no 'lag_<N>' offset, so its leak-safety can't be verified. "
                "Name it with the anchor offset (e.g. 'sales_roll_7_lag_16_mean')."
            )
        if min(offsets) < base_lag:
            raise AssertionError(
                f"base_lag violation: feature '{name}' reads back {min(offsets)} day(s), "
                f"but direct forecasting needs every sales/transactions lag >= {base_lag} "
                "(the horizon length) — a shorter lag would require sales from inside the horizon."
            )


def train_validation_split(
    df: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into a training set and a time-based validation set = the last ``horizon_days``.

    The validation set mirrors the real forecast horizon (16 days) and comes strictly
    *after* the training data, so local RMSLE scores behave like the leaderboard and no
    future day leaks into training (Constitution IV; random K-fold is forbidden for time
    series).

    Used only for *measuring* models. The final model is re-fit on the full ``df``
    (including the validation days) before predicting the real ``test.csv``.

    Args:
        df: A frame with a datetime ``date_col`` (e.g. gap-free reindexed train).
        horizon_days: Length of the validation window. Defaults to 16.
        date_col: Name of the daily date column.

    Returns:
        ``(train, val)``. With data ending 2017-08-15, the validation set is
        2017-07-31 → 2017-08-15 and train is everything before 2017-07-31.

    Raises:
        AssertionError: If either split is empty, the validation set does not strictly
            follow train, or the validation set does not cover exactly ``horizon_days``
            distinct dates.
    """
    last_day = df[date_col].max()
    val_start = last_day - pd.Timedelta(days=horizon_days - 1)

    train = df[df[date_col] < val_start]
    val = df[df[date_col] >= val_start]

    assert not train.empty, "Training split is empty — check the date column / horizon."
    assert not val.empty, "Validation split is empty — check the date column / horizon."
    assert train[date_col].max() < val[date_col].min(), (
        "Validation set must strictly follow training data (time leak): "
        f"train ends {train[date_col].max().date()}, "
        f"validation starts {val[date_col].min().date()}."
    )
    n_val_days = val[date_col].nunique()
    assert n_val_days == horizon_days, (
        f"Validation set covers {n_val_days} distinct dates, expected {horizon_days}."
    )

    return train, val


def log_iteration(
    stage: str,
    rmsle_value: float,
    notes: str = "",
    *,
    log_path: str | Path = ITERATION_LOG,
) -> str:
    """Append one iteration to ``iteration_log.md`` and return the delta string.

    Records ``stage`` (the technique tried), its validation ``rmsle_value``, and the delta
    versus the best score logged so far (negative = improvement, flagged ✅). This makes
    each technique's incremental effect traceable across the project.

    The first real entry replaces the "baseline pending" placeholder row. The running
    best is parsed from the existing table, so this is the single source of progress.

    Args:
        stage: Short label, e.g. "baseline: seasonal-naive" or "deterministic + weekly Fourier".
        rmsle_value: Validation RMSLE for this iteration.
        notes: Optional free-text note for the row.
        log_path: Target log file (defaults to repo-root ``iteration_log.md``).

    Returns:
        The delta cell text that was written (e.g. ``"-0.01234 (better)"`` or ``"-"`` for
        the first entry).
    """
    log_path = Path(log_path)
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    prev_scores: list[float] = []
    kept_lines: list[str] = []
    for line in existing.splitlines():
        if "baseline pending" in line:
            continue  # drop the placeholder once a real entry exists
        kept_lines.append(line)
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                try:
                    prev_scores.append(float(cells[1]))
                except ValueError:
                    pass  # header / separator / non-numeric cell

    if prev_scores:
        delta = rmsle_value - min(prev_scores)
        mark = "(better)" if delta < 0 else ("(same)" if delta == 0 else "(worse)")
        delta_str = f"{delta:+.5f} {mark}"
    else:
        delta_str = "-"  # first entry: nothing to compare against

    row = f"| {stage} | {rmsle_value:.5f} | {delta_str} | {notes} |"
    body = "\n".join(kept_lines).rstrip()
    log_path.write_text(f"{body}\n{row}\n", encoding="utf-8")
    return delta_str

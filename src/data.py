"""Data loading, gap-free reindexing, and submission writing.

Responsibility: read the raw competition CSVs from ``store-sales-data/`` (read-only),
join them as needed, guarantee a gap-free daily index per ``(store_nbr, family)``
series, and write the final submission file. Paths are relative to the repo root so
results are reproducible (Constitution III).

Implemented in tasks T005 (loaders), T006 (gap-free reindex), T035 (write_submission).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Resolve the raw-data directory from THIS file's location, not the current working
# directory. src/data.py -> parents[1] is the repo root -> store-sales-data/. This keeps
# loading reproducible whether run from a notebook, the repo root, or `uv run`.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "store-sales-data"


def load_train() -> pd.DataFrame:
    """Load ``train.csv`` — daily unit sales per (date, store_nbr, family).

    Columns: id, date, store_nbr, family, sales, onpromotion.
    Covers 2013-01-01 .. 2017-08-15 (~3.0M rows, 1,782 series).
    """
    return pd.read_csv(
        DATA_DIR / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"},
    )


def load_test() -> pd.DataFrame:
    """Load ``test.csv`` — the 16-day forecast horizon (2017-08-16 .. 2017-08-31).

    Same columns as train minus ``sales`` (the target we must predict). 28,512 rows.
    """
    return pd.read_csv(
        DATA_DIR / "test.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"},
    )


def load_stores() -> pd.DataFrame:
    """Load ``stores.csv`` — metadata for the 54 stores.

    Columns: store_nbr, city, state, type, cluster. ``city``/``state`` scope local and
    regional holidays; ``cluster`` is a candidate feature.
    """
    return pd.read_csv(
        DATA_DIR / "stores.csv",
        dtype={"store_nbr": "int16", "cluster": "int16"},
    )


def load_transactions() -> pd.DataFrame:
    """Load ``transactions.csv`` — daily transaction counts per store.

    Columns: date, store_nbr, transactions. A demand-intensity signal usable only as a
    *lagged* feature for the horizon (not known contemporaneously at prediction time).
    """
    return pd.read_csv(
        DATA_DIR / "transactions.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "transactions": "int32"},
    )


def load_oil() -> pd.DataFrame:
    """Load ``oil.csv`` — daily WTI oil price (``dcoilwtico``).

    Columns: date, dcoilwtico. Trading days only (weekends/holidays absent) and a few
    blank prices; gaps are forward-filled later (see T028 / research R4).
    """
    return pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])


def load_holidays() -> pd.DataFrame:
    """Load ``holidays_events.csv`` — dated calendar entries.

    Columns: date, type, locale, locale_name, description, transferred. Resolved into an
    *effective* holiday calendar before feature-building (see T026 / research R6).
    """
    return pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])


def load_sample_submission() -> pd.DataFrame:
    """Load ``sample_submission.csv`` — the exact required output shape.

    Columns: id, sales. Used to validate the generated submission's id set and format
    (see T036 / contracts/submission.md).
    """
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


def reindex_series_gapfree(
    df: pd.DataFrame,
    *,
    fill_sales: float = 0.0,
    fill_onpromotion: int = 0,
) -> pd.DataFrame:
    """Reindex every ``(store_nbr, family)`` series onto a complete daily calendar.

    The retailer is closed on a few days (e.g. every Dec 25), so those dates are simply
    absent from ``train.csv`` rather than recorded as zero. Time-series features count
    *rows* as days: a missing row shifts every later lag off by one and warps the
    Fourier seasonality terms. We therefore restore the missing days so the per-series
    daily index is gap-free *before* any lag/seasonality feature is built
    (Constitution → Performance → "Data integrity"; research R5).

    Closed days are filled with ``sales = 0`` (the store sold nothing — true and safe)
    rather than NaN (breaks lag arithmetic) or interpolation (invents sales). A boolean
    ``was_closed`` column flags the synthesized rows so downstream code can see them.

    Args:
        df: A sales frame with at least columns ``store_nbr``, ``family``, ``date``,
            ``sales``, ``onpromotion`` (e.g. the output of :func:`load_train`).
        fill_sales: Value for ``sales`` on inserted closed days. Default 0.0.
        fill_onpromotion: Value for ``onpromotion`` on inserted closed days. Default 0.

    Returns:
        A frame covering every series × every calendar day from the global first to
        last date, sorted by (store_nbr, family, date), with no missing days. Rows that
        did not exist in the input have ``was_closed = True`` and filled sales/promo;
        their ``id`` stays NaN (synthesized rows carry no competition id).
    """
    key = ["store_nbr", "family"]

    # Complete daily calendar spanning the data's full history (all series share it).
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    # Skeleton = every observed series paired with every calendar day (cross join).
    # Using the *observed* pairs (not a blind product) avoids inventing store/family
    # combinations that never existed.
    pairs = df[key].drop_duplicates()
    skeleton = pairs.merge(pd.DataFrame({"date": full_dates}), how="cross")

    out = skeleton.merge(df, on=key + ["date"], how="left", indicator=True)
    out["was_closed"] = out["_merge"].eq("left_only")
    out = out.drop(columns="_merge")

    out["sales"] = out["sales"].fillna(fill_sales)
    out["onpromotion"] = out["onpromotion"].fillna(fill_onpromotion).astype("int32")

    return out.sort_values(key + ["date"]).reset_index(drop=True)

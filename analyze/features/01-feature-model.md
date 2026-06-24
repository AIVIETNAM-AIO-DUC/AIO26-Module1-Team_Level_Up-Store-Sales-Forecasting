# Feature model — adding techniques one at a time (notebook `04_features.ipynb`)

**Scope:** what the feature stage produced — the holdout RMSLE after each feature group is added on
top of the deterministic backbone, which techniques actually helped, and the honest story the
iteration log tells. This is the result page; the *ideas* behind each feature live in the
`concepts/` and `data-traps/` files linked at the bottom.

---

## What the notebook does

Same thin-notebook shape as the earlier stages — all logic in `src/`:

1. **Load + reindex** to a gap-free daily calendar (closed days → `sales = 0`).
2. **Build each feature group** with the `src/features.py` builders — deterministic (trend +
   Fourier), calendar (day-of-week, month, payday window, earthquake flag), oil (forward-filled),
   holidays (effective, locale-scoped), and lags (sales lag/rolling + lagged transactions). All are
   knowable in advance; `onpromotion` is used same-day (it ships in `test.csv`).
3. **Compose one tidy matrix** — merge every group onto the grid by its natural key, then split off
   the same last-16-days holdout.
4. **Add groups one at a time** — fit `LinearFeatureModel` (per-series `LinearRegression` in
   `log1p` space) on a growing column set, re-score RMSLE, and log each result.

`LinearFeatureModel` generalizes the Stage 2 per-series model to a prebuilt feature matrix, keeping
the same ragged-history defenses (active-history fit, annual-Fourier drop for short series) and
adding a lag warm-up NaN-drop. With deterministic features only it reproduces the Stage 2 number
(0.62188) exactly — a useful sanity check that nothing shifted.

## The result

Each row is scored on the same 16-day holdout (2017-07-31 → 2017-08-15):

| Added on top of deterministic | Holdout RMSLE | vs baseline (0.61704) |
|---|---|---|
| deterministic only | 0.62188 | −0.8% (parity) |
| + holidays | 0.62305 | −1.0% (slightly worse) |
| + calendar | 0.60717 | +1.6% |
| + promotions & oil | 0.58226 | +5.6% |
| **+ lags** | **0.50991** | **+17.4%** |

The full feature model reaches **0.50991 — about 17% below the baseline**, comfortably past the
**≥ 10% measurable-margin goal** that the deterministic-only stage left open. That goal is met here.

## Which techniques helped — and which didn't

- **Holidays alone slightly hurt.** A single `is_holiday` flag conflates *demand-boosting* holidays
  with *closure* holidays (many national holidays close stores, dropping sales to 0), so one linear
  coefficient is pulled both ways and learns little. An honest non-improvement, kept in the log — it
  may still earn its place once it interacts with other terms in the Stage-4 tree.
- **Calendar and promotions help steadily.** The payday window and earthquake flag add real
  separable lift, and the promotion signal is strong (promo days sell markedly more).
- **Lags deliver the big jump** (0.582 → 0.510). They recover the *recent level* that smooth trend +
  seasonality averages away — precisely the edge the seasonal-naive baseline held by "copying last
  week." This is the single largest contributor and what carries the model past the baseline.

The takeaway mirrors the staging: a smooth date-only backbone gets you to parity; the recent-level
signal in lags is what wins on a short horizon.

### Ablation — does every group earn its place?

The cumulative table above bundles promotions and oil into one stage. Removing **one group at a
time** from the full set isolates each group's marginal contribution (contribution =
`RMSLE_without − RMSLE_full`; positive = the group helps, negative = it hurts):

| Removed group | RMSLE without it | Contribution | Verdict |
|---|---|---|---|
| lags | 0.58226 | +0.07235 | helps most |
| promotions | 0.54676 | +0.03685 | helps (strong) |
| calendar | 0.51357 | +0.00366 | helps (modest) |
| holidays | 0.50907 | −0.00085 | ~neutral |
| **oil** | **0.49365** | **−0.01626** | **hurts → drop** |

Two lessons: **promotions, not oil, carried the combined stage** — separated, promotions help
strongly while **oil actively hurts** the linear model. That matches the EDA warning that oil's
sales correlation is a spurious trend artifact (raw corr ≈ −0.62, ≈ 0 once detrended). The best
linear set is roughly **everything except oil** (~0.494). We drop oil here and let the Stage-4 tree
decide whether it can extract any non-linear signal. Feature selection is just choosing the column
list passed to `LinearFeatureModel` — nothing else changes.

## Why it stays leak-free

Every feature is built **once over the full history** and split by date — safe because the holdout
span (16 days) equals the minimum sales-lag, `base_lag = 16`. The farthest holdout day's `lag_16`
reaches back exactly to the last training day, never into the holdout (see
[`../concepts/lag-horizon.md`](../concepts/lag-horizon.md)). The notebook asserts this at **every**
stage with the `base_lag` leak guard — if any sales/transactions feature reached back fewer than 16
days, the run would fail loudly rather than report an inflated score.

## Verify

Reproduces the 0.50991 holdout RMSLE of the full feature model from the raw CSVs (run from the repo
root; takes ~30s):

```bash
uv run python -c "
import numpy as np, pandas as pd
from src import data, features, models, validation as V

df = data.reindex_series_gapfree(data.load_train())
KEY = ['store_nbr','family']
dates = pd.DatetimeIndex(np.sort(df['date'].unique()))

det  = features.make_deterministic_features(dates).rename_axis('date').reset_index()
cal  = features.make_calendar_features(dates).rename_axis('date').reset_index()
oil  = features.make_oil_features(data.load_oil(), dates).rename_axis('date').reset_index()
hol  = features.make_holiday_features(data.load_holidays(), data.load_stores())
lags = features.make_lag_features(df, data.load_transactions())

HOL = ['is_holiday','is_work_day','is_national_holiday']
cols = list(det.columns[1:]) + HOL + list(cal.columns[1:]) + ['onpromotion','oil'] \
       + [c for c in lags.columns if c not in KEY+['date']]

m = (df[KEY+['date','sales','onpromotion']]
     .merge(det,on='date').merge(cal,on='date').merge(oil,on='date')
     .merge(hol,on=['store_nbr','date'],how='left'))
m[HOL] = m[HOL].fillna(0).astype('int8')
m = m.merge(lags,on=KEY+['date'],how='left')

start = df['date'].max() - pd.Timedelta(days=V.HORIZON_DAYS-1)
train, hold = m[m['date']<start], m[m['date']>=start]
V.assert_no_leak(train[KEY+['date']+cols], start, strict=True, base_lag=V.HORIZON_DAYS)

preds = models.LinearFeatureModel(cols).fit(train).predict(hold)
sc = hold[KEY+['date','sales']].merge(preds, on=KEY+['date'])
print('holdout RMSLE:', round(V.rmsle(sc['sales'], sc['sales_pred']), 5))  # 0.50991
"
```

Or run the whole notebook end-to-end:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_features.ipynb
```

**Related:** [`../baseline/01-seasonal-naive.md`](../baseline/01-seasonal-naive.md) ·
[`../deterministic/01-trend-fourier.md`](../deterministic/01-trend-fourier.md) ·
[`../concepts/lag-horizon.md`](../concepts/lag-horizon.md) ·
[`../data-traps/03-holidays.md`](../data-traps/03-holidays.md) ·
[`../data-traps/06-transactions-past-only.md`](../data-traps/06-transactions-past-only.md)

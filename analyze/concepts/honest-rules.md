# Five rules of honest time-series forecasting

**Scope:** the five disciplines we carry through **every** modeling stage — so a good validation
score actually means a good real-world score. Each rule is a general principle for time-ordered
data, grounded here in our 2013-01-01 → 2017-08-15 daily sales and the 16-day Aug 16 → Aug 31
horizon. Each rule has a deeper home page; this page is the checklist that ties them together.

---

These five aren't features you add once — they're habits applied at every step, from the first
baseline to the final hybrid model. Skip any one and your scoreboard quietly lies to you.

| # | Rule | What it means | In our data |
|---|------|---------------|-------------|
| 1 | **No leakage** | A *feature* on a training row may only use information that would actually be known on the day that row is predicted. The model must never "see the future." | `oil.csv` skips weekends → fill the gaps with **forward-fill**, not interpolation (interpolation would pull Monday's price back into Friday). `transactions.csv` ends 2017-08-15 and isn't provided for the Aug 16 → Aug 31 horizon, so it's usable only as a lagged feature, never same-day. |
| 2 | **Gap-free before features** | A *lag* is a past value carried onto the current row (`lag_7` = 7 *rows* back). Seasonal features (sin/cos, day-of-week one-hots) likewise assume consecutive rows. Both count by row position, not by date — any missing day silently shifts every lag and breaks every seasonal alignment. | Stores close on Dec 25 → those rows are absent from `train.csv`. Without reindexing, `lag_7` on 2014-01-01 reaches 2013-12-24, not Dec 25. Reindex each (store, family) onto a full daily calendar with `sales = 0`, `was_closed = True` before generating any feature. |
| 3 | **Lags must clear the horizon** | Under *direct forecasting* — one output per horizon day in a single shot, no model output ever fed back as input — every lag must be ≥ H rows old (H = horizon length). A shorter lag wouldn't be knowable on the far edge of H. | H = 16, so `base_lag = 16`. The last day we predict (Aug 31) is 16 days past our last real data (Aug 15); `lag_7` works Aug 16 → 22 but breaks from **Aug 23 onward** (Aug 23's lag_7 source = Aug 16, itself a test day with no real sales). |
| 4 | **Time-respecting validation** | Validation data must be strictly later in time than training data. Random K-fold cuts across time and lets the model train on the future of the days it's about to "predict," producing scores that don't survive the real test. | Real test = Aug 16 → Aug 31, 2017 (16 days right after training ends). Our development **validation set** mirrors that exactly: train on 2013-01-01 → 2017-07-30, validate on 2017-07-31 → 2017-08-15 — same 16-day window, just shifted back 16 days. |
| 5 | **Consistent metric handling** | Compute the metric identically across every model and every prediction — same transform, same clipping, same averaging — or the scoreboard stops being comparable. | Kaggle scores RMSLE on `log1p(sales)`, so every validation score does the same. Sales can't be negative, but a regression model can still emit a small negative — an unclipped negative crashes `log1p`, so clip predictions to ≥ 0 first. |

## Where in code

Each rule is enforced by a guard that *raises* when broken — worth more than a comment reminding
you not to.

| Rule | Helper |
|------|--------|
| 1 No leakage | `src/validation.py :: assert_no_leak()` |
| 2 Gap-free | `src/data.py :: reindex_series_gapfree()` + `src/validation.py :: assert_gapfree()` |
| 3 Lag ≥ horizon | `base_lag = 16` convention + direct-forecasting structure (built alongside the feature pipeline in `src/features.py`) |
| 4 Time-respecting validation split | `src/validation.py :: train_validation_split()` |
| 5 Metric handling | `src/validation.py :: rmsle()` + `clip_nonneg()` |

**Lesson:** the goal isn't a low validation number — it's a validation number you can *trust*.
These five rules are what make the score honest.

**Related:** [rmsle-metric.md](rmsle-metric.md) · [validation.md](validation.md) ·
[baselines.md](baselines.md) · [leakage.md](leakage.md) · [lag-horizon.md](lag-horizon.md)

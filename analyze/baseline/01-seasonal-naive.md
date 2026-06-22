# Baseline — Seasonal-naive result (notebook `02_baseline.ipynb`)

**Scope:** what the baseline stage actually produced — the first holdout score, how it was
measured, and why it is the *floor* every later model must beat. This is the result page; the
*idea* of a seasonal-naive forecast lives in
[`../concepts/baselines.md`](../concepts/baselines.md).

---

## What the notebook does

Five small steps, all leaning on the reusable `src/` code:

1. **Load + reindex** — read `train.csv` and reindex every `(store_nbr, family)` series onto a
   gap-free daily calendar (closed days → `sales = 0`). See
   [`../data-traps/01-missing-calendar-days.md`](../data-traps/01-missing-calendar-days.md).
2. **Holdout split** — carve off the last 16 days (2017-07-31 → 2017-08-15) as the validation
   window; train is everything before. See
   [`../concepts/validation-holdout.md`](../concepts/validation-holdout.md).
3. **Predict** — seasonal-naive: for each series, repeat the last complete training week across
   the 16-day horizon (same weekday → same value).
4. **Score** — RMSLE between holdout actuals and predictions, in log space with non-negative
   clipping. See [`../concepts/rmsle-metric.md`](../concepts/rmsle-metric.md).
5. **Log** — append the score to `iteration_log.md` as the first row of the scoreboard.

The notebook stays thin: the model is one call, scoring is one call. The logic is in
`src/models.py` (`seasonal_naive_predict`) and `src/validation.py` (`train_holdout_split`,
`rmsle`, `log_iteration`).

## The result

| What | Value |
|---|---|
| **Holdout RMSLE** | **0.61704** |
| Holdout window | 2017-07-31 → 2017-08-15 (16 days) |
| Series scored | 1,782 |
| Predictions | 28,512 rows (1,782 × 16), zero missing |

This **0.617 is the bar to beat.** Every later stage (trend + seasonality, then features, then
the hybrid) is judged by how far it pushes this number *down* on the exact same window.

## Why this number is honest (and leak-free)

Three disciplines make the score trustworthy rather than flattering — the same disciplines that
guard every stage (see [`../concepts/honest-rules.md`](../concepts/honest-rules.md)):

- **Time-respecting split.** The holdout is the *last* 16 days, strictly after training — never a
  random sample. A random split would let the model peek at the future and report a fantasy
  score.
- **Leak-free by construction.** The prediction reads only the training half's last week. It
  never shifts a combined train+holdout series, so a holdout day can never copy another holdout
  day's actual value. (The general form of this trap:
  [`../concepts/leakage.md`](../concepts/leakage.md).)
- **Scored in log space.** RMSLE penalizes *relative* error, so a tiny family and GROCERY I are
  weighted comparably — a small store isn't drowned out by a big one.

## What it captures — and what it misses

Seasonal-naive captures exactly **one** thing: the **weekly rhythm**. Each series keeps its own
shape — a Sunday dip here, a weekend peak there — with zero fitting.

What it ignores is precisely the to-do list for the next stages:

| Ignored signal | Picked up in |
|---|---|
| Long-run **trend** (sales roughly doubled 2013→2017) | Stage 2 — deterministic |
| **Holidays / paydays / earthquake** windows | Stage 3 — features |
| **Promotions** (`onpromotion`) and **oil** price | Stage 3 — features |
| Cross-series structure, residual patterns | Stage 4 — hybrid |

That a no-learning rule already reaches 0.617 tells us the weekly cycle is strong and any real
model must clear a meaningful bar — not just beat random guessing.

## A note on sparse series

Series that the gap-fill padded to all-zeros (a dead product family at a store) naturally predict
~0 here — "sells nothing, predict nothing." That is the intended near-zero fallback behavior,
covered in [`../data-traps/05-sparse-series.md`](../data-traps/05-sparse-series.md) and
[`../concepts/baselines.md`](../concepts/baselines.md).

## Verify

Reproduces the 0.61704 holdout RMSLE from the raw CSVs (run from the repo root):

```bash
uv run python -c "
from src.data import load_train, reindex_series_gapfree
from src.validation import train_holdout_split, rmsle, HORIZON_DAYS
from src.models import seasonal_naive_predict

df = reindex_series_gapfree(load_train())
train, holdout = train_holdout_split(df)
preds = seasonal_naive_predict(train, HORIZON_DAYS)

scored = holdout.merge(preds, on=['store_nbr', 'family', 'date'], how='inner')
assert len(scored) == 28512 and scored[['sales','sales_pred']].isna().sum().sum() == 0
print('holdout RMSLE:', round(rmsle(scored['sales'], scored['sales_pred']), 5))  # 0.61704
"
```

Or run the whole notebook end-to-end:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline.ipynb
```

**Related:** [`../concepts/baselines.md`](../concepts/baselines.md) ·
[`../concepts/validation-holdout.md`](../concepts/validation-holdout.md) ·
[`../concepts/rmsle-metric.md`](../concepts/rmsle-metric.md) ·
[`../concepts/leakage.md`](../concepts/leakage.md) ·
[`../concepts/lag-horizon.md`](../concepts/lag-horizon.md)

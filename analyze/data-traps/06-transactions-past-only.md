# Trap 6 — Transactions are past-only (a hard constraint)  ⚠️ DESIGN CONSTRAINT

**Scope:** why `transactions.csv` can only be used as a *lagged* feature. The leakage principle is
[`../concepts/leakage.md`](../concepts/leakage.md); the horizon-coverage evidence is the Section 1
span table (`../eda/01-data-overview.md`).

---

## The setup — why the horizon ends Aug 31, and why that matters per column

You don't choose the prediction window; the Favorita competition does. Train ends 2017-08-15,
test spans 2017-08-16 → 2017-08-31 — **16 days you must forecast**.

```
TRAIN (real sales known)            TEST / horizon (must PREDICT)
... Jul 31 ... Aug 15  │  Aug 16  Aug 17  ...  Aug 30  Aug 31
            ↑ last real day        └──────── 16 days to predict ────────┘
```

For every horizon day the model needs **input features**. Every value in those features must be
*knowable on the day you predict* — otherwise the model is peeking at the future (the leakage
rule in [`../concepts/leakage.md`](../concepts/leakage.md)). So per column, the question is:
**does its file extend through Aug 31?**

| column | covers Aug 16–31? | reason | usage |
|---|---|---|---|
| `onpromotion` | ✅ yes | promotions are planned ahead — Kaggle ships the schedule in `test.csv` | same-day, leak-free |
| `oil` | ✅ yes | prices are historical and knowable for that period | same-day (forward-filled gaps, [Trap 2](02-oil-gaps.md)) |
| `transactions` | ❌ no | it's an *outcome* of the day (counted at close of business) — ships no test file at all | **lag-only** ← this trap |
| `sales` | ❌ no | this is the **target** you're predicting | lag-only (≥ 16 days, [lag-horizon.md](../concepts/lag-horizon.md)) |

So there are really two reasons `transactions` is past-only, working together:

1. **Competition shape** — the 16-day horizon is fixed; you have to fill features for every day in it.
2. **Column nature** — transactions are a same-day *outcome* (you only know them after the day
   ends), so unlike a promo plan, there's no honest way to publish them in advance.

That's why this is a **design constraint**, not a fill choice — no amount of imputation makes a
post-hoc count knowable in advance.

## What you see

`transactions.csv` (how many customer transactions each store had) is rich and tempting — but it
is **not provided for the 16-day test horizon**. You do not know a store's transaction count on a
day you're trying to predict. (Visible in the span table: `transactions` ends 2017-08-15, while
`oil` and the test rows reach 2017-08-31.)

## Why it quietly hurts

Use *same-day* transactions as a feature and your validation score looks fantastic — because in
validation you happen to have that column. On the real future it doesn't exist, so the model
collapses. Same leakage idea as [Trap 2](02-oil-gaps.md), but here it's a **permanent property of
the data**, not a fill choice.

## The fix — same-day fails, lagged works

Two ways you *could* try to use transactions when predicting sales on (say) Aug 20:

- **Same-day:** feature = `transactions[Aug 20]`. To fill that cell at prediction time you'd need
  Aug 20 to exist in `transactions.csv`. It doesn't (file ends Aug 15) → NaN for every horizon day
  → unusable.
- **Lagged 7:** feature = `transactions[Aug 13]`. Aug 13 *is* in the file → cell fills → usable.
  Same for `lag_14`, `lag_28`, rolling means anchored at `shift(7)`, etc.

So transactions enters the model **only as a lagged feature** (e.g. transactions 7+ days ago).

So when this doc and [`leakage.md`](../concepts/leakage.md) say a column must "cover the horizon"
to be used same-day, that's the concrete meaning: its file has to contain a value for every day
you intend to predict. Otherwise that cell of X is empty and the feature can only enter lagged.

Contrast `onpromotion` — same dataset shape, but Kaggle ships the promo schedule inside `test.csv`
for the whole horizon (promotions are decided in advance), so same-day use is leak-free. The
train/test column table in [`../eda/05-promotions-oil.md`](../eda/05-promotions-oil.md) makes the
asymmetry explicit. `transactions` is an *outcome* of the day, not a plan, so it ships no test
file at all → lag-only, forever.

## Verify

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
tr = data.load_transactions()
te = data.load_test()
print('transactions ends:', tr['date'].max())   # 2017-08-15
print('test horizon ends:', te['date'].max())   # 2017-08-31
print('covers horizon?  :', 'yes' if tr['date'].max() >= te['date'].max() else 'NO — past-only')
"
```

## Where

Lag features in `src/features.py`.

**Lesson:** before using any column, ask *"will I have this value at the moment I make the
prediction?"* (the core question in [`../concepts/leakage.md`](../concepts/leakage.md)). If not,
you can only use a lagged version.

**Related:** [`../concepts/leakage.md`](../concepts/leakage.md) ·
[`../concepts/lag-horizon.md`](../concepts/lag-horizon.md) · `../eda/01-data-overview.md`

# Trap 6 — Transactions are past-only (a hard constraint)  ⚠️ DESIGN CONSTRAINT

**Scope:** why `transactions.csv` can only be used as a *lagged* feature. The leakage principle is
[`../concepts/leakage.md`](../concepts/leakage.md); the horizon-coverage evidence is the Section 1
span table (`../eda/01-data-overview.md`).

---

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

## The fix

Use transactions **only as a lagged feature** (e.g. transactions 7+ days ago, which *are* known by
prediction time). Contrast `onpromotion` — same dataset, but provided for the horizon, so it's
used same-day (see `../eda/05-promotions-oil.md`).

## Where

Lag features in `src/features.py`.

**Lesson:** before using any column, ask *"will I have this value at the moment I make the
prediction?"* (the core question in [`../concepts/leakage.md`](../concepts/leakage.md)). If not,
you can only use a lagged version.

**Related:** [`../concepts/leakage.md`](../concepts/leakage.md) ·
[`../concepts/lag-horizon.md`](../concepts/lag-horizon.md) · `../eda/01-data-overview.md`

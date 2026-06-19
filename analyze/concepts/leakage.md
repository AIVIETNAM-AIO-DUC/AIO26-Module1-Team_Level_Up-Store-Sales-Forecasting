# Leakage — the one question behind several traps

**Scope:** the single idea of *leakage* — letting the model use information it would not have at
prediction time — and the "past-only" rule that prevents it. This is the **canonical** treatment;
the specific places leakage shows up each link back here.

---

## The idea

**Leakage** = a feature secretly contains information you would *not* have at the moment you make
the prediction. The model looks brilliant in validation (where that information happens to exist)
and then **collapses on the real future** (where it doesn't).

## The one test to apply to every feature

> **"Will I have this value at the moment I make the prediction?"**

If yes → safe to use as-is. If no → you can only use a **lagged** version (a value from far
enough in the past that it *is* known by prediction time).

## Where it shows up in this project

Each of these has its own page; the leakage *reasoning* is here, the specifics are there:

- **Oil-price gap-filling** — fill forward (carry the last known price), never interpolate.
  Interpolation averages the value *before* and *after* a gap, and "after" is the **future**.
  Details: [`../data-traps/02-oil-gaps.md`](../data-traps/02-oil-gaps.md).
- **Transactions** — *not* provided for the horizon, so usable **only lagged**, never same-day.
  Details: [`../data-traps/06-transactions-past-only.md`](../data-traps/06-transactions-past-only.md).
- **Promotions (`onpromotion`)** — the happy contrast: it *is* known for the horizon (promotions
  are planned ahead), so it's used **same-day, leak-free**. Same dataset as transactions, opposite
  availability. See `../eda/05-promotions-oil.md`.
- **Sales lags** — the horizon is 16 days long with no real sales inside it, so any sales-lag must
  reach back **≥ 16 days**. Full reasoning: [lag-horizon.md](lag-horizon.md).
- **Validation** — even the train/holdout split is a leakage question: the holdout must come
  strictly *after* training (no future days in training). See
  [validation-holdout.md](validation-holdout.md).

## Encode the rule, don't just remember it

When a rule matters, make it an assertion that *fails loudly* rather than a comment you hope to
honour. The leakage rules here are guarded in code by
`src/validation.py :: assert_no_leak()` (with a planned `base_lag ≥ 16` check), and the gap-free
prerequisite by `assert_gapfree()`.

**Related:** [lag-horizon.md](lag-horizon.md) · [`../data-traps/02-oil-gaps.md`](../data-traps/02-oil-gaps.md) ·
[`../data-traps/06-transactions-past-only.md`](../data-traps/06-transactions-past-only.md)

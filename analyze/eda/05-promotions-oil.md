# EDA — Promotions & oil price (notebook Section 6)

**Scope:** what `onpromotion` and the oil price look like over time, including the oil gaps, and
the forward-fill intent. Not yet explored.

**Status:** stub — to be written during the documentation consolidation pass. Verified facts
seeded below.

## Verified findings (seed)

- **`onpromotion` is leak-free, same-day.** Present in both train and test with **0** test
  nulls (promotions are planned ahead) → used directly for the horizon. Contrast `transactions`
  (same dataset, *not* given for the horizon → lag-only). Started **2014-04-01** (all zeros
  before); **20.4%** of open rows have a promotion; mean sales rises steeply across promo
  buckets.
- **…but the promo lift can't be sized in EDA.** It's doubly confounded — by **family**
  (big families promote *and* sell more) and by **time/trend** (promo rows are all 2014-04+, the
  higher-trend period). Leave the magnitude to the model; note it has **no signal for
  2013–early 2014**.
- **Oil needs gap-filling.** 1,218 trading-day rows; **43** blank prices (incl. the first day);
  **486** missing calendar days = **529** holes on a daily calendar → **0** after `ffill`+`bfill`.
  Forward-fill = past-only (leak-free); back-fill touches only the leading 2013-01-01 blank;
  interpolation rejected (peeks at the future).
- **Oil is a candidate, not a proven driver.** Price crashed (2014 avg ≈ 93 → 2016 avg ≈ 43)
  while sales grew, so the raw correlation looks strong (≈ **−0.62**) but is a pure trend
  artifact: detrended (month-over-month % changes) it **collapses to ≈ 0.05 ≈ 0**. A bonus
  "debunk the spurious correlation" cell shows this (raw twin-axis lines vs scatter of changes).
  May still matter per-region/family or lagged — the holdout decides.

**Related:** `../data-traps/02-oil-gaps.md` · `../data-traps/06-transactions-past-only.md` ·
[../concepts/leakage.md](../concepts/leakage.md)

**Related:** `data-traps/02-oil-gaps.md` · [../concepts/leakage.md](../concepts/leakage.md)

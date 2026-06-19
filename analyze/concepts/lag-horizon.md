# Why lags must clear the 16-day horizon

**Scope:** why every sales-lag feature must look back **at least 16 days**, and what *direct
forecasting* with `base_lag = 16` means. This is the subtlest leakage case — the general idea is
[leakage.md](leakage.md).

---

## The setup

```
TRAIN (real sales known)            TEST / horizon (must PREDICT, no real sales)
... Jul 31 ... Aug 15  │  Aug 16  Aug 17  ...  Aug 30  Aug 31
            ↑ last real day        └──────── 16 days to predict ────────┘
```

- **Last day with real sales = Aug 15.**
- To compute `lag_N` for some day, the day *N days earlier* must have **real** sales (on or before
  Aug 15).

## Why `lag_7` breaks partway through the horizon

Does "7 days before" each test day land in the known zone (≤ Aug 15)?

| Predict | 7 days before | Known? |
|---|---|---|
| Aug 16 | Aug 9  | ✅ in train |
| Aug 22 | Aug 15 | ✅ last real day |
| **Aug 23** | **Aug 16** | ❌ itself a test day — no real sales |
| Aug 31 | Aug 24 | ❌ test day |

`lag_7` works for the first 7 test days, then from **Aug 23 onward "7 days ago" points back into
the test period itself**, where no real sales exist.

## Why `lag_16` works for the whole horizon

| Predict | 16 days before | Known? |
|---|---|---|
| Aug 16 | Jul 31 | ✅ in train |
| Aug 31 | **Aug 15** | ✅ last real day |

Even the farthest day reaches back exactly to Aug 15.

### Worked example (real values)

Tail of **store 1, GROCERY I**:

| date | Jul 30 | Jul 31 | … | Aug 14 | Aug 15 (last real) |
|---|---|---|---|---|---|
| sales | 1086 | 2966 | … | 2407 | 2508 |

Building `lag_16` / `lag_17` for the horizon reaches back into that known tail:

| predict day | `lag_16` source → value | `lag_17` source → value |
|---|---|---|
| Aug 16 | Jul 31 → **2966** | Jul 30 → **1086** |
| Aug 31 | Aug 15 → **2508** | Aug 14 → **2407** |

Every value is real. `lag_7` for Aug 31 would need Aug 24 — inside the horizon, no real sales.

## The rule

> The last day you predict is **16 days** past your last real data. A lag that must work for the
> *whole* horizon has to reach back **at least 16 days**. In general, **minimum usable lag =
> horizon length (H).**

## The design decision: direct forecasting with `base_lag = 16`

- **Direct forecasting (chosen).** Predict all 16 days at once using only *real* lags → every
  sales-lag ≥ 16 (`lag_16, lag_17, …`, rolling windows anchored at `shift(16)`). One feature
  matrix, **no error compounding**.
- **Recursive (not chosen).** Predict Aug 16, treat it *as if real*, use as `lag_1` for Aug 17, …
  Allows short lags but each day's error **snowballs**.

*(A middle option — a separate model per horizon day — gets the freshest legal lag per day at the
cost of 16 models. We keep the single-model `base_lag = 16` for simplicity.)*

## Long horizons push you toward seasonality

The rule scales: a 1-year horizon would need `lag_365` as its shortest lag — nearly useless. The
longer the horizon, the less lags help and the more you lean on **date-only features** (trend +
Fourier seasonality, holidays) that are computable for any future date. That's why modeling is
**staged**: a deterministic seasonality/trend backbone first (horizon-robust, no lags), with lags
added *on top* as a short-horizon booster. A no-lag model is perfectly valid.

**Where:** lag/rolling features in `src/features.py`; enforced by the planned `base_lag ≥ 16`
guard in `src/validation.py :: assert_no_leak()`.

**Related:** [leakage.md](leakage.md) · [seasonality-fourier.md](seasonality-fourier.md) ·
[`../data-traps/06-transactions-past-only.md`](../data-traps/06-transactions-past-only.md)

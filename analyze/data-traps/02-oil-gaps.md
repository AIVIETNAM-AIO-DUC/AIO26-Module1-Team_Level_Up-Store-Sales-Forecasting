# Trap 2 — Oil price gaps + blanks (your first taste of leakage)  ✅ FIXED

**Scope:** the two holes in `oil.csv` and the past-only fill that handles them. The *why* of
leakage is [`../concepts/leakage.md`](../concepts/leakage.md); the verified gap counts and the
"oil effect is spurious" finding are in `../eda/05-promotions-oil.md`.

---

## What you see

Oil matters here — Ecuador's economy is oil-dependent, so the price is a plausible spending
driver. But `oil.csv` has two kinds of hole:

1. **Missing rows** — oil trades only on business days, so weekends and holidays have no row.
2. **Blank prices** — some rows exist with an empty `dcoilwtico` (e.g. 2013-01-01, the first day).

On a full daily calendar this is **529 holes** (43 blanks + 486 missing days) — confirmed in
`../eda/05-promotions-oil.md`.

## Why it quietly hurts

Join oil onto the daily calendar naively and the holes go down one of two bad paths.

**Path A — leave the gaps as `NaN` ("NaN features")**
- Many estimators just crash on `NaN` (linear models, plain sklearn pipelines).
- Tree boosters (LightGBM / XGBoost) tolerate `NaN` but only route it down a default branch — they
  get **no real signal from oil** on that day. With ~529 holes in a ~1700-day calendar, roughly a
  third of rows have no oil information.
- `NaN` propagates. Any derived feature — `oil_lag7`, `oil_roll14_mean`, `oil_diff` — that touches
  a gap row inherits the `NaN`, so one missing day silently produces many missing feature values
  downstream.

So `NaN` features are *honest but weak*: the model just has less to learn from.

**Path B — naive fill ("or worse, filled from the future")**

"Naive" fills usually peek forward:
- `interpolate()` averages the value *before* and *after* the gap — and "after" is tomorrow.
- `bfill()` literally copies tomorrow's price into today.
- `fillna(df['oil'].mean())` computed over the full table bakes the test-period mean into training
  rows.

Each of these uses information that didn't exist yet on the date being filled. The model trains on
a feature it could never honestly have at inference time → validation looks great, real
out-of-sample performance won't match. That's the leak (see
[`../concepts/leakage.md`](../concepts/leakage.md)).

| Strategy                 | Looks at         | Honest? | Cost                                              |
|--------------------------|------------------|---------|---------------------------------------------------|
| Leave as `NaN`           | Nothing          | Yes     | Lost signal; `NaN` spreads through derived features |
| `interpolate` / `bfill`  | Past + **future**| **No (leak)** | Inflated CV; silent overfitting                |
| Forward-fill (the fix)   | Past only        | Yes     | None — uses the last value the calendar would actually have known |

So `NaN` is the "worse weak signal" path, future-fill is the "worse leak" path, and forward-fill
(below) is the one that avoids both.

## The fix

Build a continuous daily calendar, left-join oil, then **forward-fill** — carry the *last known*
price forward into each gap. A single leading **back-fill** covers the very first blank day.

### Why forward-fill is leak-free — read the name carefully

The name "forward-fill" is misleading. "Forward" describes the direction the known value is
*carried into the gap*, **not** the direction the algorithm *searches* for a value. The search
actually walks **backward in time**.

Concretely, with a tiny gap:

```
date:   Mon    Tue   Wed   Thu   Fri
oil:   100.0   NaN   NaN   NaN  102.0
```

To fill Thursday's `NaN`, `ffill` walks:

```
Thursday is NaN → need a value.
   step 1: check Wednesday → NaN, keep going.
   step 2: check Tuesday   → NaN, keep going.
   step 3: check Monday    → 100.0. Done. Copy 100.0 into Thursday.
```

The search walked Thu → Wed → Tue → Mon — backward through the calendar. Call that direction
**backward lookup**.

The identity to remember:

```
forward-fill  =  backward lookup   ← reads past values only   → LEAK-FREE
back-fill     =  forward  lookup   ← reads future values      → LEAKS
```

The "forward" in *forward-fill* refers to the value flowing forward in time *into* the gap; the
*lookup* itself is backward. Two arrows pointing opposite ways inside the same operation — that's
where the name trips people up.

Why only backward lookup is leak-free: on Thursday in real life, Monday/Tuesday/Wednesday have
already happened, so you genuinely know those prices — using them to fill Thursday is honest.
Friday hasn't happened yet, so copying Friday into Thursday (which is what `bfill` does) means the
model trains on a value it won't actually have at inference time. That's the leak.

- **Interpolation is rejected for the same reason**: it averages the value *before* and *after* a
  gap, and "after" is the future — using it to fill today's feature lets the model peek ahead.
- The single leading `bfill` on line 60 is the one allowed exception: the very first day of the
  dataset has nothing before it to look back to, and that day is never used as a training target
  in any honest split (you have no prior history to learn from on day 1 anyway), so the leak has
  nothing to leak *into*.

## Verify

The raw holes the fill has to close:

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
print('oil null prices:', int(data.load_oil()['dcoilwtico'].isna().sum()))  # 43
"
```

And that the forward-fill closes every gap past-only (weekend carries Friday, leading day
back-filled, nothing left missing):

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from src import data, features
of = features.make_oil_features(data.load_oil(), pd.date_range('2013-01-01','2017-08-15', freq='D'))
print('gaps after fill :', int(of['oil'].isna().sum()))                         # 0
fri, sat, sun = (float(of.loc[d,'oil']) for d in ['2013-01-04','2013-01-05','2013-01-06'])
print('weekend = Friday:', sat == sun == fri)                                   # True (past-only ffill)
print('leading day fill:', float(of.loc['2013-01-01','oil']) == float(of.loc['2013-01-02','oil']))  # True (bfill)
"
```

## Where

`make_oil_features(oil, index)` in `src/features.py` builds the forward-filled price; the
contemporaneous `onpromotion` feature (same-day, leak-free — `test.csv` ships it for the horizon)
is selected by `make_promotion_features(df)` alongside it. (Note: the EDA showed oil's apparent
sales effect is a **spurious trend artifact** — raw corr ≈ −0.62, ≈ 0 once detrended — so oil is
included only as a *candidate* feature for the holdout to judge. See `../eda/05-promotions-oil.md`.)

**Lesson:** how you fill a gap encodes an assumption about *what you're allowed to know*.
Forward-fill = "only the past." Interpolation = "I peeked at the future."

**Related:** [`../concepts/leakage.md`](../concepts/leakage.md) · `../eda/05-promotions-oil.md`

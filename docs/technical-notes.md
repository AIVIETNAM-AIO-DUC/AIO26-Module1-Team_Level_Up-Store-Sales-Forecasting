# Technical Notes — Store Sales Time Series

**Companion to [`data-analysis.md`](data-analysis.md).** That document is the readable, top-to-bottom
story of the data and our decisions. *This* file holds the deeper technical explanations — the
maths, formulas, and mechanics — so the main lesson stays short. The narrative links here
whenever a topic deserves a full treatment.

**Contents — ordered easiest → most advanced; read top to bottom.**

1. [Forecast baselines: seasonal-naive & near-zero](#forecast-baselines-seasonal-naive--near-zero) — *the simplest way to predict at all*
2. [The metric: RMSLE](#the-metric-rmsle) — *how a prediction gets scored*
3. [Seasonality & Fourier features](#seasonality--fourier-features) — *turning the calendar into wave features*
4. [The 16-day lag horizon (why lags must be ≥ 16)](#the-16-day-lag-horizon-why-lags-must-be--16) — *using past sales without leaking the future*

Each section is self-contained, but later ones occasionally lean on earlier ones (e.g. the lag
horizon refers back to seasonality), so the order above is the smoothest path.

---

## Forecast baselines: seasonal-naive & near-zero

**Level: start here.** These are the simplest possible ways to make a prediction — no training, no
maths — so they're the natural first thing to understand. We use them as **fallback rules** for
series too thin to model (Trap 5 in `data-analysis.md`). Important: they predict the **future** 16
days — they do **not** change any past data. (Don't confuse this with Trap 1, which fills *past*
closed days with `sales = 0`.)

### Why we need a fallback

Our main model learns a pattern from history. A series with almost no sales history has no
pattern to learn, so the model would produce garbage. But the submission **must** contain a value
for all 28,512 rows. So for thin series we skip the model and use a simple, safe rule instead.

### Near-zero

If a series has essentially *no* sales ever, the safest forecast for the next 16 days is just
**≈ 0** — "it sells nothing, so predict nothing." Cheap, and almost always right for a dead
product family at a given store.

### Seasonal-naive

- **Naive** = don't learn anything; just **copy the past forward**.
- **Seasonal** = copy from the same point in the *previous cycle* (here, the weekly cycle).

```
predict next Monday  = last Monday's sales
predict next Tuesday = last Tuesday's sales
…repeat last week's 7-day pattern across the 16-day horizon
```

It captures the weekly rhythm (weekend up, midweek down) with zero fitting, which is often a
surprisingly strong baseline and a sensible floor for any "real" model to beat.

### Sample data

*(Real values from the CSVs.)* Take the **last known week** (the 7 days ending on the last real
day, Tue Aug 15, 2017) of a real series — **store 1, GROCERY I**:

| Last week | Wed 8/9 | Thu 8/10 | Fri 8/11 | Sat 8/12 | Sun 8/13 | Mon 8/14 | Tue 8/15 |
|---|---|---|---|---|---|---|---|
| sales | 2719 | 2591 | 1270 | 1630 | 952 | 2407 | 2508 |

**Seasonal-naive** copies each weekday forward across the 16-day horizon (the 7-day pattern just
repeats):

| Horizon day | Wed 8/16 | Thu 8/17 | Fri 8/18 | Sat 8/19 | Sun 8/20 | … | Wed 8/30 | Thu 8/31 |
|---|---|---|---|---|---|---|---|---|
| predicted | 2719 | 2591 | 1270 | 1630 | 952 | … | 2719 | 2591 |

The series' own shape is preserved — here a Sunday dip to **952** against midweek highs of
~2.5–2.7k — with no model required. (This series happens to dip on Sunday; others peak then —
see the seasonality note.)

**Near-zero** is for a series whose history is basically all zeros — e.g. **store 1, BOOKS**, whose
last week is:

| Last 7 days (store 1, BOOKS) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|---|---|---|---|---|---|---|---|

→ predict **0** for all 16 horizon days. Nothing to copy, so "sells nothing → predict nothing."

### Who decides which rule to use? (not the model)

A common misconception: the trained model does **not** choose between the main prediction,
seasonal-naive, and near-zero. The choice is made *before* modeling by a simple **rule-based gate
(an `if` check)** on each series' history. The thin series are peeled off and never reach the
model — precisely because it would produce garbage on them.

```
For each of the 1,782 series:
   ├─ Has enough history?  ──►  use the MAIN model (the learned one)
   └─ Too thin?            ──►  use a FALLBACK rule
                                   ├─ a little history to copy  ►  seasonal-naive
                                   └─ basically no sales ever   ►  near-zero (~0)
```

So three possible prediction sources, and which one produces a given series' 16 values is decided
by a threshold *we* choose (e.g. "fewer than N non-zero days of history → fallback") — the ML
model has no say in the routing.

### Does the model leave the thin series empty?

No — and that's worth being precise about. A trained regression model is a function: give it
inputs, it always returns *a number*. It physically cannot output a blank. The risk for a
no-history series isn't emptiness; it's that the number is **unreliable garbage** (the model is
extrapolating from nothing). So the question is only *how we arrange the code* to keep that
garbage out of the submission. Two valid wirings:

**Option A — exclude before training (what we use):**

```
1. Code splits series:  healthy  vs  thin
2. Train the model on HEALTHY series only        ← model never sees thin ones
3. Predict healthy series with the model
4. Predict thin series with the fallback rule
5. Stitch both sets together → all 28,512 rows
```

**Option B — predict everything, then overwrite:**

```
1. Train / predict on ALL series with the model
2. Code OVERWRITES the thin-series predictions with the fallback values
```

Either way the model never outputs blanks, and the **code** decides the final value for thin
series — by routing them around the model (A) or replacing the model's bad guess (B). We use
**Option A**: it's cleaner and stops the thin series from misleading the model during training.

### The rule

The model is sophisticated only where it *can* be. For thin series we fall back to seasonal-naive
(if there's a little history to copy) or near-zero (if there's essentially none), so every
required row gets a sensible value instead of a blank or a wild model guess.

**Where**: the fallback path in `src/models.py`.

---

## The metric: RMSLE

**Level: easy–medium.** Before we can say one model is "better" than another, we need to know how
predictions are scored. Our model is graded by **RMSLE** — *Root Mean Squared **Logarithmic**
Error*. Read the name backwards and it's just a recipe, one word per step:

| Word | Step |
|------|------|
| **Error** | how wrong each prediction is: `prediction − actual` |
| **Logarithmic** | but first take the **log** of both numbers |
| **Squared** | square each error (so too-high and too-low both count as positive) |
| **Mean** | average all those squared errors |
| **Root** | take the square root at the end |

As a formula:

```
RMSLE = sqrt( mean( (log1p(prediction) − log1p(actual))² ) )
```

**Why the log? It measures *relative* (ratio) error, not the raw gap.** Compare two forecasts:

- Predict **15** when the truth is **5** → off by 10.
- Predict **3,000** when the truth is **1,000** → off by 2,000.

A plain error metric would call the second one *200× worse*. But both forecasts are "**3× too
high**" — equally bad. The log fixes this, because subtracting logs is the same as dividing:

```
log(a) − log(b) = log(a / b)      ← depends on the RATIO a/b, not the gap
```

So RMSLE asks "**how many times off were you?**" That's the fair way to compare a tiny family
(AUTOMOTIVE, a few sales a day) against a huge one (BEVERAGES, thousands a day). Without it, the
big families would dominate the score and the model would neglect the small ones.

**Why `log1p` instead of plain `log`?** Because `log(0)` is undefined (negative infinity), and
sales are often **0** (closed days, slow products). So we use `log1p(x) = log(1 + x)`:
`log1p(0) = 0` — no crash. Its inverse, `expm1(x) = e^x − 1`, turns a log-space number back into
real sales.

**Two consequences we apply to *every* model**, so scores are comparable:

1. **We model in log space.** Two separate uses of the log, don't mix them up:
   - the *metric* uses logs (above);
   - we also *train* on log-transformed sales — feed the model `log1p(sales)`, let it predict,
     then `expm1` back to real units. Training in the same space it's graded in means "minimize
     ordinary error" lines up directly with "minimize RMSLE." It also tames the huge range of
     sales values (0 to thousands).
2. **We clip predictions to ≥ 0.** A regression model can output a negative number (e.g. −3
   sales), which is both impossible *and* would make `log1p(−3)` undefined and crash the metric.
   So we floor every prediction at 0 before scoring.

### Sample data

Four (actual, prediction) pairs run through the whole pipeline — including a negative prediction
that gets **clipped to 0**. These pairs are illustrative (we have no trained model yet to produce
real predictions), but the *scales* are realistic: an `actual` of 5 is a small-family day (e.g.
AUTOMOTIVE), 1000 a big-family day (e.g. GROCERY I — its real daily sales sit in the thousands,
as the tables below show). The last two columns are the two error styles, side by side:

| actual | pred | pred clipped | gap² (plain RMSE style) | log1p(actual) | log1p(pred) | log-diff | log-diff² (RMSLE style) |
|---|---|---|---|---|---|---|---|
| 5    | 15   | 15 | 100 | 1.7918 | 2.7726 | 0.9808 | 0.962 |
| 1000 | 3000 | 3000 | 4,000,000 | 6.9088 | 8.0067 | 1.0979 | 1.205 |
| 0    | −2   | 0  | 0 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| 200  | 195  | 195 | 25 | 5.3033 | 5.2781 | −0.0252 | 0.001 |

Now aggregate each column (mean → square root):

```
Plain RMSE  = sqrt( (100 + 4,000,000 + 0 + 25) / 4 ) = sqrt(1,000,031) ≈ 1000.0
RMSLE       = sqrt( (0.962 + 1.205 + 0 + 0.001)  / 4 ) = sqrt(0.542)     ≈ 0.736
```

**Read the contrast.** Rows 1 and 2 are *both* "3× too high," equally bad as forecasts:

- **Plain RMSE** is ≈ 1000 — almost entirely dictated by row 2 (its `gap²` of 4,000,000 drowns
  out everything). The small-scale row is invisible.
- **RMSLE** scores row 1 at `0.962` and row 2 at `1.205` — *comparable* numbers. Both "3× off"
  rows are penalized in the same ballpark, so a tiny family and a huge family count fairly.

Also note **row 3**: the negative prediction `−2` would make `log1p(−2)` undefined and crash the
metric — clipping it to `0` keeps it valid (and here matches the actual `0` exactly).

**Where**: `src/validation.py :: rmsle()` and `clip_nonneg()`.

---

## Seasonality & Fourier features

**Level: medium.** Now the first *real* feature family. These terms come from the **statsmodels**
library and are how we model *seasonality*. They also explain why the gap-free reindex (Trap 1 in
`data-analysis.md`) has to run **before** any seasonal feature.

### Seasonality

A **repeating pattern tied to the calendar**. This data has two big ones:

- **Weekly** — across all series, weekends sell more on average (per series-day means: Sat ≈ 433,
  Sun ≈ 463, vs ≈ 285–346 on weekdays). Repeats every **7 days**. Individual series still vary —
  each has its own weekly shape (e.g. store 1's GROCERY I actually *dips* on Sunday).
- **Yearly** — a December surge, summer patterns, etc. Repeats every **365 days**.

"Seasonal" just means "it cycles on a fixed period."

### Regular daily frequency

"One row per day, evenly spaced, **no gaps**." The tools below assume row #1 is day 1, row #2 is
day 2, and so on. That assumption is exactly what missing days break — hence the dependency on
the gap-free reindex.

### Sine / cosine terms — how you represent a cycle

A cycle is a **wave**, and the natural way to write a wave is with `sin` and `cos`. For a weekly
pattern (period = 7):

```
sin(2π × day / 7)   and   cos(2π × day / 7)
```

Each completes exactly **one full wave every 7 days** — up on weekends, down midweek, then
repeats. Why *both* sin and cos? A single sine wave has a fixed peak position, but combining a
sin **and** a cos (with weights the model learns) lets it place the peak *anywhere* and at *any*
height. So with just 2 numbers per cycle, a plain linear regression can fit "this series peaks
on Saturday, that one on Sunday."

### Sample data

Here are the **exact** values of those two columns for one week (`d` = day-of-week index,
0 = Monday; sin/cos rounded to 3 dp), next to the **real** average sales-by-weekday of
**store 1, GROCERY I**:

| `d` | weekday | `sin(2π·d/7)` | `cos(2π·d/7)` | mean sales (store 1, GROCERY I) |
|---|---|---|---|---|
| 0 | Mon | 0.000 | 1.000 | 2383 |
| 1 | Tue | 0.782 | 0.623 | 2409 |
| 2 | Wed | 0.975 | −0.223 | 2770 |
| 3 | Thu | 0.434 | −0.901 | 2229 |
| 4 | Fri | −0.434 | −0.901 | 2414 |
| 5 | Sat | −0.975 | −0.223 | 2323 |
| 6 | Sun | −0.782 | 0.623 | 1031 |

The model just learns two weights (`w_sin`, `w_cos`) plus an intercept so that
`intercept + w_sin·sin + w_cos·cos` approximately traces this shape — here roughly flat Mon–Sat
(~2.3–2.8k) with a sharp **Sunday dip to ~1031**. **Two columns reconstruct the weekly rhythm**,
and the weights adapt per series: a weekend-peaking series simply gets different `w_sin`/`w_cos`.
(One sin/cos pair captures the smooth shape; a sharp single-day dip like Sunday's needs a couple
more Fourier pairs to pin down exactly.) These values repeat every week and are computable for any
future date — which is why they extend cleanly into the horizon.

### `CalendarFourier`

A helper that **generates those sin/cos columns for you** from the dates. "Fourier" = the maths
of building any repeating shape out of sine waves. Instead of making 365 separate "is it Jan 1 /
Jan 2 / …" columns for the yearly cycle, you ask for, say, 4 Fourier pairs and get a handful of
smooth sin/cos columns that approximate the whole annual wave. Compact and smooth.

### `DeterministicProcess`

A helper that **assembles the full feature table of time-based terms**: a constant, a **trend**
(1, 2, 3, … counting upward over time), and the seasonal Fourier terms from `CalendarFourier`.

"**Deterministic**" is the key word: these features depend **only on the date**, not on past
sales. That means you can compute them for *any future day* without knowing the answer — perfect
for forecasting, because you can extend them straight into the 16-day horizon. (Contrast with a
lag feature — covered in the next section — which needs actual past sales.)

### "Out of phase" — why gaps wreck this

**Phase** = where you are in the cycle. The sin/cos value is computed from the row's
**position**, assuming consecutive days. Now suppose a day is missing:

```
Expected (regular):  Fri  Sat  Sun  Mon  …   ← wave's "weekend peak" lands on Sat/Sun ✓
After a gap:         Fri  Sun  Mon  Tue  …   ← Sat's row is gone, everything shifts left
                          ↑ the wave still "thinks" this slot is Saturday
```

The wave keeps assigning its Saturday-peak to a slot that is now actually **Sunday**. The whole
pattern slides over — that's being **out of phase**: the seasonal feature points at the wrong
weekday. The model then "learns" a smeared, wrong rhythm. Nothing errors; the score just quietly
gets worse.

### Sample data (the phase shift in numbers)

Using the `sin` values from the table above, with real store 1 / GROCERY I sales. **Correct**
(no gap — position matches the weekday):

| date | real weekday | position `d` | `sin` used | sales |
|---|---|---|---|---|
| Aug 12 | Sat | 5 | −0.975 | 1630 |
| Aug 13 | Sun | 6 | −0.782 | 952 |

Now **drop the Sat Aug 12 row** (a closed day). Everything after it shifts up one position:

| date | real weekday | position `d` (miscounted) | `sin` used | sales |
|---|---|---|---|---|
| Aug 13 | Sun | **5** (thinks it's Sat) | **−0.975** | 952 |

Sunday's 952 sales are now paired with **Saturday's** sin value (−0.975 instead of −0.782). The
model is being told "Sunday looks like Saturday's wave position" — every weekday past the gap is
mislabelled by one. That's the silent corruption.

**Punchline**: fill the calendar *first* (so position = real date again), *then* build Fourier
seasonality.

**Where**: seasonal/deterministic features in `src/features.py`.

---

## The 16-day lag horizon (why lags must be ≥ 16)

**Level: advanced.** This one ties the previous ideas together and carries the subtlest reasoning
(leakage + the forecast horizon). A **lag** feature ("sales N days ago") is built from *past
sales*. But our prediction window is 16 days long and we have **no real sales inside it**. That
puts a hard floor on how short a lag we can use. This note works out why that floor is **16** and
what it means for feature design.

### The setup

```
TRAIN (real sales known)            TEST / horizon (must PREDICT, no real sales)
... Jul 31 ... Aug 15  │  Aug 16  Aug 17  ...  Aug 30  Aug 31
            ↑ last real day        └──────── 16 days to predict ────────┘
```

- **Last day with real sales = Aug 15.**
- To compute `lag_N` for some day, the day *N days earlier* must have **real** sales (fall on or
  before Aug 15).

### Why `lag_7` breaks partway through the horizon

Check `lag_7` for each test day — does "7 days before" land in the known zone (≤ Aug 15)?

| Predict this day | 7 days before | Known? |
|---|---|---|
| Aug 16 | Aug 9  | ✅ in train |
| …      | …      | ✅ |
| Aug 22 | Aug 15 | ✅ last real day |
| **Aug 23** | **Aug 16** | ❌ Aug 16 is itself a test day — no real sales |
| …      | …      | ❌ |
| Aug 31 | Aug 24 | ❌ test day |

`lag_7` works for the first 7 test days, then from **Aug 23 onward "7 days ago" points back into
the test period itself**, where no real sales exist.

### Why `lag_16` works for the whole horizon

| Predict this day | 16 days before | Known? |
|---|---|---|
| Aug 16 | Jul 31 | ✅ in train |
| Aug 31 | **Aug 15** | ✅ last real day |

Even the farthest day (Aug 31) reaches back exactly to Aug 15. So `lag_16` is computable for all
16 horizon days.

### Sample data

*(Real values from the CSVs.)* The tail of **store 1, GROCERY I**'s real history is:

| date | Jul 30 | Jul 31 | … | Aug 14 | Aug 15 (last real) |
|---|---|---|---|---|---|
| sales | 1086 | 2966 | … | 2407 | 2508 |

Building `lag_16`/`lag_17` for the horizon just reaches back into that known tail:

| predict day | `lag_16` source → value | `lag_17` source → value |
|---|---|---|
| Aug 16 | Jul 31 → **2966** | Jul 30 → **1086** |
| Aug 31 | Aug 15 → **2508** | Aug 14 → **2407** |

Every value comes from real data. Compare what `lag_7` would need for **Aug 31**: Aug 24 — a day
*inside* the horizon with no real sales, so it could only be filled with our own prediction
(that's the recursive route we avoid).

### The rule

> The last day you predict is **16 days** past your last real data. A lag feature that must work
> for the *whole* horizon has to reach back **at least 16 days**. In general, **minimum usable lag
> = horizon length (H).**

### The design decision: direct forecasting with `base_lag = 16`

There are two ways to handle this; we pick the first:

- **Direct forecasting (chosen).** Predict all 16 days in one shot, using only *real* lags →
  every sales-lag is ≥ 16 (`base_lag = 16`: `lag_16, lag_17, lag_18, …`, rolling windows anchored
  at `shift(16)`). Simple, one feature matrix, **no error compounding**.
- **Recursive forecasting (not chosen).** Predict Aug 16, treat that prediction *as if real*, use
  it as `lag_1` for Aug 17, and so on. Allows short lags, but each day's error feeds the next, so
  mistakes **snowball** across 16 days.

*(A middle option — "direct per step" — trains a separate model for each horizon day, so the day-1
model may use `lag_1`, the day-16 model uses `lag_16`. Freshest legal lag per day, at the cost of
16 models. We keep the single-model `base_lag = 16` for simplicity.)*

### Long horizons: why this pushes you toward seasonality

The rule scales literally: a **1-year** horizon would need `lag_365` as its shortest lag. But
that's almost useless — last year's value barely predicts tomorrow, and you'd drop a full year of
training rows to `NaN`. The lesson: **the longer the horizon, the less lags help and the more you
must lean on date-only features** (trend + Fourier seasonality from the previous section, holidays),
which are computable for *any* future date and don't depend on recent sales at all.

That is exactly why the modelling is **staged**: a deterministic seasonality/trend backbone (no
lags, horizon-robust) first, with lags added *on top* as a short-horizon booster. Lags are an
*input to* the model, never a replacement for it — a no-lag model is perfectly valid (it's our
deterministic stage); lags just add recent-momentum signal the calendar can't express.

**Where**: lag/rolling features in `src/features.py`; to be enforced by the `base_lag ≥ 16` guard
in `src/validation.py :: assert_no_leak()`.

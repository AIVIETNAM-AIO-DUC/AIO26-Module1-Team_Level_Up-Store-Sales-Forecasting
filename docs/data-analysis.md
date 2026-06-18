# Store Sales Time Series — A Learner's Walkthrough

**Who this is for**: someone learning ML / time-series forecasting who wants to *understand*
this dataset, not just run code. Read it **top to bottom** — each lesson builds on the one
before it.

**The big idea**: real data is rarely "wrong" in a way that throws an error. It's wrong in
ways that *silently* make your model worse and give you no clue why. This document walks you
from the simplest question ("what are we even predicting?") up to the subtle traps that
separate a good time-series model from a broken one.

Every number below is real — reproduced from the CSVs in `store-sales-data/` with
`uv run python …` (see the last section to run them yourself).

---

## Lesson 1 — What are we actually predicting?

A supermarket chain in Ecuador (Corporación Favorita) wants to predict **daily sales**.

- There are **54 stores**.
- Each store sells **33 product families** (BEVERAGES, AUTOMOTIVE, PRODUCE, …).
- For each store + family, sales are recorded **once per day**.

The task: given years of history (2013-01-01 → 2017-08-15), predict sales for the **next 16
days** (2017-08-16 → 2017-08-31) for every store + family combination.

Why 16 and not some other number? It's simply the window the competition asks for — the gap
between where the training data stops and where the prediction period ends. This fixed window
has a name we'll reuse throughout: the **horizon** (= 16 days). Everything we build is aimed at
predicting one horizon forward.

### The most important word: *series*

A **series** is the day-by-day sales history of **one product family at one store**. It is the
"thing that has a past and a future" — the unit we forecast over time.

- *Store 1, AUTOMOTIVE* → one series
- *Store 1, BEVERAGES* → a different series
- *Store 2, AUTOMOTIVE* → a different series again

So how many series are there? Every store carries every family (the full grid):

```
54 stores × 33 families = 1,782 series
```

`train.csv` is just these 1,782 series **stacked on top of each other** in one long table.
Running `groupby(["store_nbr", "family"])` splits that table back into its individual series —
that's literally how we get the count 1,782.

**Why this matters for everything that follows**: almost every operation happens *per series*,
never mixing across them.

- A **lag** ("sales 7 days ago") must look back within the *same* store + family.
- **Seasonality** is per series — BEVERAGES and AUTOMOTIVE have completely different weekly and
  yearly rhythms.
- The **forecast** covers the 16-day horizon for every series: 16 days × 1,782 series =
  **28,512** predictions — exactly the number of rows we must submit.

Hold on to the number **1,782**. It will keep reappearing as a sanity check.

---

## Lesson 2 — Meet the data (look before you model)

Before touching a model, look at what you have. Here is every file, verified:

| File | Rows | What's in it |
|------|------|-------------|
| `train.csv` | 3,000,888 | id, date, store_nbr, family, sales, onpromotion · 2013-01-01 → 2017-08-15 |
| `test.csv` | 28,512 | same columns minus `sales` · 2017-08-16 → 2017-08-31 (the 16 days to predict) |
| `stores.csv` | 54 | store_nbr, city, state, type, cluster |
| `transactions.csv` | 83,488 | date, store_nbr, transactions |
| `oil.csv` | 1,218 | date, dcoilwtico (oil price) · **43 prices are blank** |
| `holidays_events.csv` | 350 | date, type, locale, locale_name, description, transferred |
| `sample_submission.csv` | 28,512 | id, sales — the exact shape our output must have |

A few facts to internalize now, because the rest of the document leans on them:

- **Every series covers the same span**: 2013-01-01 → 2017-08-15, which is **1,688 calendar
  days**.
- **Horizon** (introduced in Lesson 1) = the 16 days we must predict, 2017-08-16 → 2017-08-31.
- **The scoring metric is RMSLE** (Root Mean Squared *Logarithmic* Error). The short version:
  it grades us on *relative* error ("how many **times** off were you?"), not the raw gap — which
  is why we model in **log space** (`log1p` in, `expm1` out) and **clip predictions to ≥ 0**.
  Full explanation, with a worked example, in
  [`technical-notes.md` → The metric: RMSLE](technical-notes.md#the-metric-rmsle).
- **`onpromotion` is a feature we *will* use.** Unlike transactions (Trap 6), the promotion count
  **is** provided for the 16-day test horizon — promotions are planned in advance — so it's a
  strong *and* leak-free predictor. We feed it to the model directly (same-day, no lag needed).

That's the whole landscape. Now the interesting part: why you can't just load `train.csv` and
fit a model.

---

## Lesson 3 — The silent traps (and how we defuse each one)

This is the heart of the document. The problems are ordered **easiest to subtlest**. Each one
follows the same shape: **what you see → why it quietly hurts → the fix → where the fix lives in
code.**

A theme runs through several of them — **leakage**: accidentally letting the model use
information it would *not* have at prediction time. Watch for it; we introduce it gently here
and formalize it in Lesson 5.

### Trap 1 — Missing calendar days (the counting trap)  ✅ FIXED

**What you see**: the store closes on some days — notably **every Dec 25** — so those dates have
**no row at all** in `train.csv`. They are *absent*, not recorded as a zero.

- Verified: series `store=1, AUTOMOTIVE` has 1,684 rows over a 1,688-day span → **4 missing
  days**, all Christmases: 2013-12-25, 2014-12-25, 2015-12-25, 2016-12-25.
- Across all series: 1,782 × 4 = **7,128 missing rows**.

**Why it quietly hurts**: time-series features count *rows*, assuming each row is the next day.

- A **lag** like `shift(7)` means "go back 7 rows." After a gap, "7 rows back" is no longer "7
  days back" — your "same weekday last week" feature silently points at the wrong day.
- **Seasonality tools** (like `DeterministicProcess` / `CalendarFourier`) assume a *regular*
  daily frequency. Holes push the weekly and yearly sine/cosine terms out of phase — the seasonal
  pattern slides onto the wrong weekday. (What these tools are and why a gap shifts the wave:
  [`technical-notes.md` → Seasonality & Fourier features](technical-notes.md#seasonality--fourier-features).)
- Nothing errors. You just get a worse model and no idea why. This is the canonical "silent
  trap."

**The fix**: rebuild each series onto a *complete* daily calendar and fill the missing days with
**`sales = 0`** (the store sold nothing that day — which is *true*, and keeps the row spacing
regular).

- Why `0` and not `NaN`? `NaN` breaks lag arithmetic and labels a *known* closed day as
  "unknown."
- Why `0` and not interpolation? Interpolation would *invent* sales on a day the store was shut.
- We also tag every inserted row with `was_closed = True`, so the information isn't lost — it can
  even become a feature.

**Where**: `src/data.py :: reindex_series_gapfree()`, proven correct by
`src/validation.py :: assert_gapfree()` (it *checks* that no gaps remain).

#### Read the numbers — they're the proof

```
original rows   : 3,000,888    ← what Kaggle gave us
reindexed rows  : 3,008,016    ← after filling gaps
expected        : 1,782 × 1,688 = 3,008,016
rows inserted   : 7,128        ← 4 per series
```

- **1,782** = series (54 × 33).
- **1,688** = calendar days. From 2013-01-01 to 2017-08-15 there are 1,687 days *between* the
  ends; **+1** to count both ends → 1,688.
- **1,782 × 1,688 = 3,008,016.** A complete grid *must* have exactly (series × days) rows. The
  total matching this product is the proof that the grid is complete — no series short a day,
  none with a duplicate.
- **7,128 inserted = 3,008,016 − 3,000,888** — the closed-store days.
- **4 per series = 7,128 ÷ 1,782.** Why exactly 4? The data ends 2017-08-15, so Dec 25 of 2017
  is past the window, and 2012-12-25 is before the 2013 start. That leaves 2013, 2014, 2015,
  2016 — the four Christmases.

```
3,000,888  (gaps where the store was shut)
+   7,128  (4 Christmases × 1,782 series, restored as sales = 0)
─────────
3,008,016  (a complete, gap-free grid: 1,782 series × 1,688 days)
```

The fact that the inserted count divides *evenly* by 1,782, and the total equals the product, is
itself the evidence the reindex did exactly the right thing.

**Lesson takeaway**: in time series, "no row" and "a zero" are not the same thing — and the
difference can silently break every lag and seasonal feature downstream. Always make the
calendar complete *first*.

### Trap 2 — Oil price gaps + blanks (your first taste of leakage)  ⏳ FIX PLANNED

Oil matters here: Ecuador's economy is oil-dependent, so the oil price is a plausible driver of
spending. But `oil.csv` has two different holes:

1. **Missing rows** — oil only trades on business days, so weekends and holidays have no row at
   all. In the 16-day holdout window alone, 4 days are absent (the two weekends).
2. **Blank prices** — **43 rows** exist but have an empty `dcoilwtico` (for example
   2013-01-01, a holiday).

**Why it quietly hurts**: join oil onto the daily calendar naively and those holes become `NaN`
features — or, worse, get filled from the *future*.

**The fix**: build a continuous daily calendar, left-join oil, then **forward-fill** the price
— carry the *last known* price forward into the gap. A single leading back-fill covers the very
first blank day.

- Why forward-fill and **not** interpolation? Interpolation averages the value *before* and the
  value *after* the gap. The "after" value is the **future** — using it to fill today's feature
  means the model peeks at information it wouldn't have on that day. That is **leakage**: it
  inflates your validation score and then collapses on the real future. Forward-fill only ever
  looks backward, so it's safe.

**Where**: oil features in `src/features.py`.

**Lesson takeaway**: how you fill a gap encodes an assumption about *what you're allowed to
know*. Forward-fill = "only the past." Interpolation = "I peeked at the future." For features,
always choose past-only.

### Trap 3 — Holidays aren't a simple yes/no (domain modeling)  ⏳ FIX PLANNED

**What you see**: `holidays_events.csv` looks like a "was it a holiday?" table, but it isn't.

- A row with `transferred = True` means the holiday did **not** happen on its listed date — it
  was officially moved. A separate row with `type = "Transfer"` carries the date it was *actually*
  observed.
- There are also `Bridge` days (an extra day off) and `Work Day` rows (a normally-off day made
  into a working day).
- `locale` says whether a holiday is National, Regional, or Local — it doesn't apply to every
  store.

**Why it quietly hurts**: trust the raw `date` column and you attach "holiday demand" to the
wrong day. Ignore `locale` and you apply one city's local holiday to all 54 stores — pure noise.

**The fix**: compute an **effective holiday calendar** — drop the `transferred=True` originals,
honor the matching `Transfer`/`Bridge`/`Work Day` rows, and scope each holiday by locale:
National → all stores; Regional → stores in that `state`; Local → stores in that `city`.

**Where**: holiday features in `src/features.py`.

**Lesson takeaway**: a column named like a feature isn't automatically a usable feature.
Understanding the domain (how holidays *actually* work) is part of feature engineering.

### Trap 4 — The 2016 earthquake anomaly  ⏳ FIX PLANNED

**What you see**: a magnitude-7.8 earthquake on **2016-04-16** caused a relief-driven sales
spike that lasted for weeks.

**Why it quietly hurts**: a one-off spike is *not* a repeating pattern. If you leave it
unmarked, your trend and seasonality estimates try to "explain" it and get bent out of shape for
every series.

**The fix**: add a binary (or decaying) **earthquake-window flag** covering roughly 2016-04-16 →
mid-May 2016, so the model can absorb the spike *as a known event* instead of distorting the
baseline. We do **not** delete those rows — that would punch a hole back into the gap-free index
we just built in Trap 1.

**Where**: calendar features in `src/features.py`.

**Lesson takeaway**: don't delete anomalies — *label* them. Give the model a flag so it can say
"this was special" rather than treating the spike as normal behavior.

### Trap 5 — New / sparse series (the coverage trap)  ⏳ FIX PLANNED

**What you see**: some store + family pairs have little or no meaningful history (a product
family that barely sells, or only started recently).

**Why it quietly hurts**: a complex model can't learn a pattern that isn't there — but the
submission **must** contain a prediction for all 28,512 test rows. A missing or `NaN` prediction
is an invalid submission.

**The fix**: for those thin series, fall back to a simple **seasonal-naive / near-zero**
prediction so every row always gets *a* value. (These rules forecast the *future* by copying the
last weekly cycle, or predicting ~0 when there's nothing to copy — they don't alter the past.
Details:
[`technical-notes.md` → Forecast baselines](technical-notes.md#forecast-baselines-seasonal-naive--near-zero).)

**Where**: the fallback path in `src/models.py`.

**Lesson takeaway**: a model only needs to be sophisticated *where it can be*. Always have a
dumb-but-safe fallback so you never ship a blank prediction.

### Trap 6 — Transactions are past-only (a hard constraint)  ⚠️ DESIGN CONSTRAINT

**What you see**: `transactions.csv` (how many customer transactions each store had) is rich and
tempting — but it is **not provided for the 16-day test horizon**. You do not know a store's
transaction count on a day you're trying to predict.

**Why it quietly hurts**: use *same-day* transactions as a feature and your validation score will
look fantastic — because in validation you happen to have that column. On the real future it
doesn't exist, so the model collapses. This is the same leakage idea as Trap 2, but here it's a
permanent property of the data, not a fill choice.

**The fix**: use transactions **only as a lagged feature** (e.g. transactions 7+ days ago, which
*are* known by prediction time).

**Where**: lag features in `src/features.py`.

**Lesson takeaway**: before using any column as a feature, ask "will I have this value *at the
moment I make the prediction*?" If not, you can only use a lagged version of it.

---

## Lesson 4 — How do we know a model is good?

You now have clean features. Next question: when you try two models, how do you decide which is
*actually* better? You need an honest scoreboard.

### Why you can't just check the training score

A model always looks brilliant on the data it trained on — it has effectively seen the answers.
The only honest signal is its error on data it did **not** train on.

You *could* get that by submitting to Kaggle every time, but Kaggle limits submissions per day,
and chasing the public leaderboard leads to **overfitting the leaderboard** — tuning to one
hidden set instead of building something genuinely good. So we keep a private slice of *our own*
data as an instant, unlimited, honest scoreboard. This is called a **holdout**.

### Why the split must respect *time*

In ordinary ML you shuffle rows and validate on a random subset (K-fold). **For time series that
cheats.** If you validate on a day in the *middle* of history, the model gets to train on days
that come *after* it — it literally sees the future relative to what it's predicting. Score looks
great; reality collapses.

So the split is a clean cut in time:

```
train: 2013-01-01 ........... 2017-07-30  │  holdout: 2017-07-31 → 2017-08-15
       (model learns from the past)        │  (model predicts forward, blind)
```

Everything in the holdout comes strictly *after* everything in train. **Random K-fold is
forbidden here.** The split function asserts this ordering so we can't break it by accident.

### Why exactly 16 days

The real competition horizon is 16 days. So we hold out the **last 16 training days** — same
length, same "predict 16 days forward" shape. A shorter window wouldn't faithfully represent the
real task.

### The holdout is NOT `test.csv` — keep them straight

|  | **Holdout** | **`test.csv`** |
|---|---|---|
| Comes from | carved out of `train.csv` | a separate file Kaggle provides |
| Dates | 2017-07-31 → 2017-08-15 | 2017-08-16 → 2017-08-31 |
| Do we know the real sales? | **Yes** — we just hide them | **No** — that's what we predict |
| Who scores it | **we** do, instantly, locally | **Kaggle**, after we submit |
| Purpose | private rehearsal to compare models | the actual graded exam |

The holdout is a *stand-in* for `test.csv`: same shape, but the answers are in our pocket, so we
can check ourselves for free as often as we like.

### Hide for measuring, retrain on everything for the real forecast

The last 16 days are hidden **only while comparing models**. Once we've picked the best approach,
we **re-fit it on ALL training data** (including those last 16 days) before predicting the real
`test.csv` — the most recent days carry the freshest trend and seasonality, so we want them in
the final fit.

```
1. Hide last 16 days → train → score on them   (how we PICK the best model)
2. Best model chosen → re-train on ALL data     (now it uses the last 16 days too)
3. Predict test.csv (2017-08-16 → 08-31) → submit
```

**Where**: `src/validation.py :: train_holdout_split()` does the split and asserts the ordering.

---

## Lesson 5 — The rules that keep us honest

Everything above distills into five rules. These aren't just good intentions — they're
*enforced in code*, so a mistake fails loudly instead of silently corrupting a score.

1. **No leakage.** Every feature must be knowable at prediction time. Lags and rolling windows
   use strictly past days; oil is forward-filled, never interpolated; transactions are lag-only.
   Guard: `src/validation.py :: assert_no_leak()`.
2. **Gap-free before features.** The Trap-1 reindex runs *before* any lag or seasonal feature, so
   "7 rows back" always means "7 days back."
   Guard: `src/validation.py :: assert_gapfree()`.
3. **Lags must clear the horizon (≥ 16 days).** Because the last day we predict (Aug 31) is 16
   days past our last real data (Aug 15), any sales-lag shorter than 16 can't be computed for the
   far end of the horizon without feeding the model its own guesses. We forecast all 16 days at
   once (*direct forecasting*) with `base_lag = 16`. Full reasoning:
   [`technical-notes.md` → The 16-day lag horizon](technical-notes.md#the-16-day-lag-horizon-why-lags-must-be--16).
4. **Time-respecting validation.** One 16-day holdout (2017-07-31 → 2017-08-15) that mirrors the
   real horizon; train strictly on earlier data; random K-fold forbidden.
5. **Consistent metric handling.** RMSLE computed on `log1p`, predictions clipped ≥ 0, applied
   identically to every model so scores are truly comparable.

**Meta-lesson**: when a rule matters, encode it as an assertion. A guard that *raises* the moment
you break a rule is worth more than a comment reminding you not to.

---

## Appendix A — Reproduce the numbers yourself

Everything runs inside the uv-managed environment:

```bash
# Confirm the closed-day gaps and the reindex fix
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
t = data.load_train(); r = data.reindex_series_gapfree(t)
print('inserted closed-days:', int(r['was_closed'].sum()))   # 7128
"

# Confirm the oil blanks
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
print('oil null prices:', int(data.load_oil()['dcoilwtico'].isna().sum()))  # 43
"
```

## Appendix B — Status at a glance

| Trap | Problem | Status | Code |
|----|---------|--------|------|
| 1 | Missing calendar days | ✅ Fixed | `reindex_series_gapfree()` + `assert_gapfree()` |
| 2 | Oil gaps + 43 blanks | ⏳ Planned | oil features in `src/features.py` |
| 3 | Transferred/bridge holidays | ⏳ Planned | holiday features in `src/features.py` |
| 4 | 2016 earthquake spike | ⏳ Planned | calendar flag in `src/features.py` |
| 5 | New/sparse series | ⏳ Planned | model fallback in `src/models.py` |
| 6 | Transactions = past-only | ⚠️ Constraint | lag features in `src/features.py` |

_This is a living document. As each fix lands, update the status table and the matching lesson._

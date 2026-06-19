# The scoring metric: RMSLE

**Scope:** the metric the competition grades us on (Root Mean Squared *Logarithmic* Error), what
it rewards, and the two habits it forces on us (model in log space, clip predictions ≥ 0).

---

## Read the name backwards — it's a recipe

**RMSLE** = *Root Mean Squared **Logarithmic** Error*, one word per step:

| Word | Step |
|------|------|
| **Error** | how wrong each prediction is: `prediction − actual` |
| **Logarithmic** | but first take the **log** of both numbers |
| **Squared** | square each error (too-high and too-low both count positive) |
| **Mean** | average all the squared errors |
| **Root** | square root at the end |

As a formula:

```
RMSLE = sqrt( mean( (log1p(prediction) − log1p(actual))² ) )
```

## Why the log? It measures *relative* error

Compare two forecasts:

- Predict **15** when the truth is **5** → off by 10.
- Predict **3,000** when the truth is **1,000** → off by 2,000.

A plain error metric calls the second *200× worse*. But both are "**3× too high**" — equally bad.
The log fixes this, because subtracting logs is dividing:

```
log(a) − log(b) = log(a / b)      ← depends on the RATIO a/b, not the gap
```

So RMSLE asks "**how many times off were you?**" — the fair way to compare a tiny family
(AUTOMOTIVE, a few sales/day) against a huge one (BEVERAGES, thousands/day). Without it the big
families dominate and the model neglects the small ones.

**Why `log1p` not plain `log`?** `log(0)` is undefined and sales are often **0** (closed days,
slow products). `log1p(x) = log(1 + x)` gives `log1p(0) = 0` — no crash. Its inverse
`expm1(x) = eˣ − 1` turns a log-space number back into real sales.

## Two consequences we apply to *every* model

1. **Model in log space.** Two separate uses of the log — don't mix them: the *metric* uses logs
   (above), and we also *train* on `log1p(sales)`, predict, then `expm1` back. Training in the
   space it's graded in lines "minimize ordinary error" up with "minimize RMSLE," and tames the
   huge range of sales.
2. **Clip predictions to ≥ 0.** A regression model can output a negative number, which is
   impossible *and* makes `log1p(−3)` crash the metric. Floor every prediction at 0 before scoring.

## Worked example (real scales)

Four (actual, prediction) pairs run through the pipeline — including a negative prediction
**clipped to 0**. (Illustrative pairs — no trained model yet — but realistic scales: `actual` 5 ≈
a small-family day, 1000 ≈ a GROCERY I day.)

| actual | pred | pred clipped | gap² (plain RMSE) | log1p(actual) | log1p(pred) | log-diff² (RMSLE) |
|---|---|---|---|---|---|---|
| 5    | 15   | 15 | 100 | 1.7918 | 2.7726 | 0.962 |
| 1000 | 3000 | 3000 | 4,000,000 | 6.9088 | 8.0067 | 1.205 |
| 0    | −2   | 0  | 0 | 0.0000 | 0.0000 | 0.000 |
| 200  | 195  | 195 | 25 | 5.3033 | 5.2781 | 0.001 |

Aggregate each (mean → square root):

```
Plain RMSE  = sqrt( (100 + 4,000,000 + 0 + 25) / 4 ) ≈ 1000.0
RMSLE       = sqrt( (0.962 + 1.205 + 0 + 0.001)  / 4 ) ≈ 0.736
```

**The contrast.** Rows 1 and 2 are *both* "3× too high," equally bad:

- **Plain RMSE ≈ 1000** — almost entirely dictated by row 2 (its gap² of 4,000,000 drowns out the
  rest). The small-scale row is invisible.
- **RMSLE** scores them `0.962` and `1.205` — *comparable*. A tiny family and a huge family count
  fairly.

**Row 3** shows why we clip: the prediction `−2` would make `log1p(−2)` crash; clipped to `0` it's
valid (and here matches the actual `0`).

**Where:** `src/validation.py :: rmsle()` and `clip_nonneg()`.

**Related:** [validation-holdout.md](validation-holdout.md) · [baselines.md](baselines.md)

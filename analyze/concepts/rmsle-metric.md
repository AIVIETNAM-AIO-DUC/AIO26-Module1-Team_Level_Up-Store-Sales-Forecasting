# The scoring metric: RMSLE

**Scope:** the metric the competition grades us on (Root Mean Squared *Logarithmic* Error), what
it rewards, and the two habits it forces on us (train in log space, clip predictions ≥ 0).

---

## Read the name backwards — it's a recipe

**RMSLE** = *Root Mean Squared **Logarithmic** Error*, one word per step:

| Word | Step |
|------|------|
| **Error** | how wrong each prediction is — but measured *after* the log step below |
| **Logarithmic** | first take the **log** of both numbers, then subtract |
| **Squared** | square each error (too-high and too-low both count positive) |
| **Mean** | average all the squared errors |
| **Root** | square root at the end |

As a formula — first the **origin form**, exactly as the competition defines it:

```
                   1    n
RMSLE  =  sqrt(   ───   Σ   ( ln(1 + pᵢ) − ln(1 + aᵢ) )²   )
                   n   i=1

   n  = number of predictions        pᵢ = the i-th prediction
   ln = natural logarithm            aᵢ = the i-th actual value
```

That is just the recipe above written in symbols. For each pair — a prediction `pᵢ`
and its matching actual `aᵢ`:

1. **Logarithmic** — take the log of both numbers: `ln(1 + pᵢ)` and `ln(1 + aᵢ)`.
2. **Error** — subtract the two logs; that difference is the error, measured in log space.
3. **Squared** — square that difference.
4. **Mean** — do this for all `n` pairs and average the squares.
5. **Root** — finally take the square root.

## What `log1p` means, step by step

The code — and the compact formula we use below — writes `log1p(x)` instead of
`ln(1 + x)`. They are the **same function**; `log1p` just spells out "**log** of **1**
**p**lus x":

```
log1p(x) = ln(1 + x)
```

Walk it one step at a time for a single number, say `x = 15`:

```
1. start with the input      x      = 15
2. add one                   1 + x  = 16
3. take the natural log      ln(16) ≈ 2.7726     →  log1p(15) ≈ 2.7726
```

And for `x = 0` (a closed day or an unsold product — very common here):

```
1. x      = 0
2. 1 + x  = 1
3. ln(1)  = 0                                    →  log1p(0) = 0
```

That `+1` is the whole trick: plain `log(0)` is undefined and would crash, but
`log1p(0)` is a clean `0`. (The *why* behind the `+1`, and its inverse `expm1`, get
their own section below.)

With `log1p` defined, the origin formula compresses to the form you'll meet in the
code — same math, shorter to write:

```
RMSLE = sqrt( mean( (log1p(prediction) − log1p(actual))² ) )
```

## Why the log? It measures *ratio*, not raw gap

Taking the log **before** measuring error makes the metric care about the **ratio** of the miss
("how many times off?") instead of the raw gap between two numbers. The identity that makes this
work is short:

```
log(a) − log(b) = log(a / b)      ← subtracting logs is dividing
```

So once you take the log, "distance" stops depending on scale and depends only on the *ratio*
`a / b`. The next section makes this concrete with two real product families from this dataset.

## Unpack — why this matters

Product families live on wildly different scales: `GROCERY I` sells a few **thousand** units/day,
while `BABY CARE` sells only a handful. With plain RMSE (which measures raw gap), the same
*proportional* miss is punished very differently:

| Family | Predicted | Actual | Raw gap | Ratio |
|--------|-----------|--------|---------|-------|
| GROCERY I | 3,000 | 1,000 | **2,000** | 3× |
| BABY CARE |    15 |     5 |     **10** | 3× |

RMSE punishes the GROCERY row **200× more** than the BABY CARE row — even though both forecasts
are "3× too high." A model trained against RMSE learns to pour effort into the big family and
abandon the small one.

**How log fixes it.** Apply the identity from the previous section to both pairs:

- `log(3000) − log(1000) = log(3000 / 1000) = log(3)`
- `log(15)   − log(5)    = log(15 / 5)     = log(3)`

Both pairs collapse to **the same value `log(3)`**. RMSLE scores them *equally*.

**One-line summary:** log converts the number line from "gap scale" to "ratio scale" →
small-family and big-family forecasts get scored fairly.

## `log1p` and `expm1` — why a dedicated function

`log1p(x) = log(1 + x)` is true **by definition** — the `1p` literally means "**1** **p**lus". So
why does the math library ship a separate `log1p` instead of letting us just write `log(1 + x)`?
Two reasons:

**Reason 1 — it dodges `log(0)`.** Sales are often **0** (closed days, slow products), and plain
`log(0)` is undefined (it heads to −∞ and crashes the metric). Adding 1 first moves that danger
point out of the way:

```
log1p(x) = log(1 + x)        →   log1p(0) = log(1) = 0     ← no crash
```

**Reason 2 — it stays accurate when `x` is tiny.** A computer keeps only ~15–16 digits, so the
naive `1 + x` can silently *lose* a very small `x`:

```
x = 0.0000000000000001  (1e-16)

   naive:   1 + x = 1.0   (x is rounded away)  →  log(1.0) = 0      ← wrong
   log1p(x)        ≈ 1e-16                                          ← right
```

`log1p` uses a dedicated algorithm (a series expansion for small `x`) so it never has to form
`1 + x` naively. Its inverse `expm1(x) = eˣ − 1` exists for the same precision reason:

```
expm1(x) = eˣ − 1            →   inverse of log1p, returns real sales
```

`log1p` goes into the log-space; `expm1` comes back out. They're a pair.

## Two rules this forces on every model

**Rule 1 — Train in log space.**
- **What "log space" means:** "space" is just shorthand for *which set of numbers you're working
  with right now*. The pipeline has two:

  | Space      | Values         | Example (sales = 0, 1, 50, 1000)   |
  |------------|----------------|------------------------------------|
  | Raw space  | `y` (sales)    | `0, 1, 50, 1000`                   |
  | Log space  | `log1p(y)`     | `0.000, 0.693, 3.932, 6.908`       |

  `log1p` is the door in, `expm1` is the door out. Inside log space you fit the model, generate
  predictions, and compute the loss; `expm1` carries you back to real sales at the end.
- **Why:** training minimizes whatever loss you give it. If the leaderboard measures *log* error,
  you want the model optimizing *log* error too — otherwise you're optimizing one thing and
  scored on another.
- **How:** fit on `log1p(sales)`, predict in log space, then `expm1` back to real sales for the
  submission.

**Rule 2 — Clip predictions to ≥ 0.**
- **Why:** a regression model can output a negative number, and `log1p(−2)` is undefined — the
  metric crashes on it. Negative sales are also physically impossible.
- **How:** `clip_nonneg(pred)` before scoring or submitting.

## Worked example — clipping in action

Two (actual, prediction) pairs run through the pipeline. The second has a negative prediction
that gets **clipped to 0** before any logs are taken:

| actual | pred | pred clipped | log1p(actual) | log1p(clipped) | log-diff² |
|--------|------|--------------|---------------|----------------|-----------|
|   5    |  15  |     15       |   1.7918      |    2.7726      |   0.962   |
|   0    |  −2  |      0       |   0.0000      |    0.0000      |   0.000   |

```
RMSLE = sqrt( (0.962 + 0.000) / 2 ) ≈ 0.694
```

Row 2 is the point of clipping: without it, `log1p(−2)` would have crashed the calculation.
Clipped to 0 it matches the actual `0` and contributes a clean zero error.

**Where:** `src/validation.py :: rmsle()` and `clip_nonneg()`.

**Related:** [validation-holdout.md](validation-holdout.md) · [baselines.md](baselines.md)

# An honest scoreboard: the time-respecting validation set

**Scope:** how we compare models honestly — by holding back the last 16 training days as a
private, instant stand-in for the real test set — and why the split must respect time.

---

## Why the training score can't be trusted

A model always looks brilliant on the data it trained on — it has effectively seen the answers.
The only honest signal is its error on data it did **not** train on.

You *could* get that by submitting to Kaggle each time, but submissions are limited and chasing
the public leaderboard leads to **overfitting the leaderboard**. So we keep a private slice of our
own data as an instant, unlimited, honest scoreboard — a **validation set**.

## Why the split must respect *time*

In ordinary ML you shuffle rows and validate on a random subset (K-fold). **For time series that
cheats:** validating on a day in the *middle* of history lets the model train on days that come
*after* it — it sees the future relative to what it predicts. Score looks great, reality
collapses. (This is a leakage problem — see [leakage.md](leakage.md).)

So the split is a clean cut in time:

```
train: 2013-01-01 ........... 2017-07-30  │  validation: 2017-07-31 → 2017-08-15
       (model learns from the past)        │  (predicts forward, blind)
```

Everything in the validation set comes strictly *after* training. **Random K-fold is forbidden
here**, and the split function asserts the ordering so we can't break it by accident.

## Why exactly 16 days

The real horizon is 16 days, so we hold back the **last 16 training days** — same length, same
"predict 16 days forward" shape. A shorter window wouldn't faithfully represent the task.

## The validation set is NOT `test.csv`

|  | **Validation set** | **`test.csv` (test set)** |
|---|---|---|
| Comes from | carved out of `train.csv` | a separate file Kaggle provides |
| Dates | 2017-07-31 → 2017-08-15 | 2017-08-16 → 2017-08-31 |
| Real sales known? | **Yes** — we just hide them | **No** — that's what we predict |
| Who scores it | **we** do, instantly, locally | **Kaggle**, after we submit |
| Purpose | private rehearsal to compare models | the actual graded exam |

## Hide to measure, then re-fit on everything

The last 16 days are hidden **only while comparing models**. Once the best approach is chosen, we
**re-fit it on ALL training data** (including those last 16 days) before predicting the real
`test.csv` — the most recent days carry the freshest trend/seasonality.

```
1. Hide last 16 days → train → score on them   (how we PICK the best model)
2. Best model chosen → re-train on ALL data     (now uses the last 16 days too)
3. Predict test.csv (2017-08-16 → 08-31) → submit
```

**Where:** `src/validation.py :: train_validation_split()` does the split and asserts the ordering.
Scoring uses [rmsle-metric.md](rmsle-metric.md).

**Related:** [rmsle-metric.md](rmsle-metric.md) · [baselines.md](baselines.md) · [leakage.md](leakage.md)

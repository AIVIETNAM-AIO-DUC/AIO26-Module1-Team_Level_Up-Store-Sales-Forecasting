# Forecast baselines: seasonal-naive and near-zero

**Scope:** the two simple, dumb-but-safe forecasts — *seasonal-naive* (copy the last weekly
cycle) and *near-zero* (for series with nothing to copy) — and how we route thin series to them.

These predict the **future** 16 days; they do **not** change any past data. (Don't confuse with
the Trap-1 gap-fill, which fills *past* closed days — see
[`../data-traps/01-missing-calendar-days.md`](../data-traps/01-missing-calendar-days.md).)

---

## Why we need a fallback

The main model learns a pattern from history. A series with almost no history has no pattern to
learn, so the model would produce garbage — but the submission **must** contain a value for all
28,512 rows. So for thin series we skip the model and use a simple, safe rule.

## Near-zero

If a series has essentially *no* sales ever, the safest forecast is **≈ 0** — "it sells nothing,
predict nothing." Almost always right for a dead product family at a given store.

## Seasonal-naive

- **Naive** = don't learn anything; just **copy the past forward**.
- **Seasonal** = copy from the same point in the previous cycle (here, the weekly cycle).

```
predict next Monday  = last Monday's sales
predict next Tuesday = last Tuesday's sales
…repeat last week's 7-day pattern across the 16-day horizon
```

It captures the weekly rhythm with zero fitting — a surprisingly strong baseline and the floor
any "real" model must beat.

### Worked example (real values)

The **last known week** (ending Tue Aug 15, 2017) of **store 1, GROCERY I**:

| Last week | Wed 8/9 | Thu 8/10 | Fri 8/11 | Sat 8/12 | Sun 8/13 | Mon 8/14 | Tue 8/15 |
|---|---|---|---|---|---|---|---|
| sales | 2719 | 2591 | 1270 | 1630 | 952 | 2407 | 2508 |

Seasonal-naive repeats that 7-day pattern across the horizon:

| Horizon | Wed 8/16 | Thu 8/17 | Fri 8/18 | Sat 8/19 | Sun 8/20 | … | Wed 8/30 | Thu 8/31 |
|---|---|---|---|---|---|---|---|---|
| predicted | 2719 | 2591 | 1270 | 1630 | 952 | … | 2719 | 2591 |

The series' own shape is preserved — here a Sunday dip to **952** against midweek ~2.5–2.7k — with
no model. (This series dips on Sunday; others peak then — see
[seasonality-fourier.md](seasonality-fourier.md).)

**Near-zero** is for an all-zeros history — e.g. **store 1, BOOKS**, whose last week is `0 0 0 0 0
0 0` → predict **0** for all 16 days.

## Who decides which rule? (not the model)

A common misconception: the trained model does **not** choose between the main prediction,
seasonal-naive, and near-zero. The choice is a **rule-based gate** (an `if` check) on each series'
history, made *before* modeling. Thin series are peeled off and never reach the model.

```
For each of the 1,782 series:
   ├─ Enough history?  ──►  MAIN model (the learned one)
   └─ Too thin?        ──►  FALLBACK rule
                              ├─ a little history to copy  ►  seasonal-naive
                              └─ basically no sales ever   ►  near-zero (~0)
```

A regression model can't output a blank — give it inputs and it returns *a number*; the risk for
a no-history series is that the number is unreliable garbage. We use **exclude-before-training**
(train on healthy series only, predict thin ones with the fallback, stitch together) — cleaner,
and the thin series don't mislead the model during training. (The threshold, e.g. "fewer than N
non-zero days → fallback," is one *we* choose; the connection to thin series is
[`../data-traps/05-sparse-series.md`](../data-traps/05-sparse-series.md).)

**Where:** the fallback path in `src/models.py`. Scored like everything else with
[rmsle-metric.md](rmsle-metric.md) on the [holdout](validation-holdout.md).

**Related:** [`../data-traps/05-sparse-series.md`](../data-traps/05-sparse-series.md) ·
[seasonality-fourier.md](seasonality-fourier.md)

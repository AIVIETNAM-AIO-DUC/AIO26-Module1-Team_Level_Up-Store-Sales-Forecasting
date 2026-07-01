# What we predict: the *series* and the *horizon*

**Scope:** the unit we forecast (one store × family = a *series*) and the fixed prediction window
(the 16-day *horizon*). The dataset tour (which files exist, their shapes) is in
`../eda/01-data-overview.md`.

---

## The task

A supermarket chain in Ecuador (Corporación Favorita) wants to predict **daily unit sales**.

- **54 stores**, each selling **33 product families** (BEVERAGES, AUTOMOTIVE, PRODUCE, …).
- For each store + family, sales are recorded **once per day**.

Given history from 2013-01-01 → 2017-08-15, predict sales for the **next 16 days**
(2017-08-16 → 2017-08-31), for every store + family.

## The most important word: *series*

A **series** is the day-by-day sales history of **one product family at one store** — the "thing
that has a past and a future," the unit we forecast over time.

- *Store 1, AUTOMOTIVE* → one series
- *Store 1, BEVERAGES* → a different series
- *Store 2, AUTOMOTIVE* → a different series again

How many? Every store carries every family (the full grid):

```
54 stores × 33 families = 1,782 series
```

`train.csv` is just these 1,782 series **stacked** in one long table; `groupby(["store_nbr",
"family"])` splits it back into them — that's literally how we get 1,782.

## Why this matters for everything else

Almost every operation happens **per series**, never mixing across them:

- A **lag** ("sales 7 days ago") must look back within the *same* store + family.
- **Seasonality** is per series — families have completely different weekly/yearly rhythms.
- The **forecast** covers the 16-day horizon for every series.

## The horizon

The **horizon** is the fixed prediction window: **16 days**, 2017-08-16 → 2017-08-31. It's set by
the competition (the gap between where training stops and where prediction ends), not chosen by
us. Everything we build aims at predicting one horizon forward — and the horizon length has a
sharp consequence for lag features (see [lag-horizon.md](lag-horizon.md)).

The submission therefore has:

```
16 days × 1,782 series = 28,512 predictions   ← exactly the rows we must submit
```

Hold on to **1,782** and **28,512** — they reappear constantly as sanity checks.

**Related:** `../eda/01-data-overview.md` · [lag-horizon.md](lag-horizon.md) ·
[validation.md](validation.md)

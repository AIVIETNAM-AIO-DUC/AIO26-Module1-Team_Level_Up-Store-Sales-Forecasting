# Trap 2 — Oil price gaps + blanks (your first taste of leakage)  ⏳ FIX PLANNED

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

Join oil onto the daily calendar naively and those holes become `NaN` features — or worse, get
filled from the **future**, which leaks (see [`../concepts/leakage.md`](../concepts/leakage.md)).

## The fix

Build a continuous daily calendar, left-join oil, then **forward-fill** — carry the *last known*
price forward into each gap. A single leading **back-fill** covers the very first blank day.

- Forward-fill only ever looks **backward**, so it's leak-free.
- **Interpolation is rejected**: it averages the value *before* and *after* a gap, and "after" is
  the future — using it to fill today's feature lets the model peek ahead.

## Where

Oil features in `src/features.py`. (Note: the EDA showed oil's apparent sales effect is a
**spurious trend artifact** — raw corr ≈ −0.62, ≈ 0 once detrended — so oil is included only as a
*candidate* feature for the holdout to judge. See `../eda/05-promotions-oil.md`.)

**Lesson:** how you fill a gap encodes an assumption about *what you're allowed to know*.
Forward-fill = "only the past." Interpolation = "I peeked at the future."

**Related:** [`../concepts/leakage.md`](../concepts/leakage.md) · `../eda/05-promotions-oil.md`

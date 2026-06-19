# Seasonality and Fourier terms

**Scope:** what seasonality is, and how a repeating cycle is turned into model features with
**Fourier terms** (sine/cosine pairs) and *harmonics*. *Discovering* which cycles exist is a
separate step — see [periodogram.md](periodogram.md). This file is also the canonical home for the
"out of phase" mechanics that [`../data-traps/01-missing-calendar-days.md`](../data-traps/01-missing-calendar-days.md)
depends on.

---

## Seasonality

A **repeating pattern tied to the calendar.** This data has two big ones:

- **Weekly** — weekends sell more on average; repeats every **7 days**. Each series still has its
  own weekly shape (store 1's GROCERY I actually *dips* on Sunday).
- **Annual** — a December surge, summer patterns; repeats every **365 days**.

The empirical evidence (weekend ≈1.39× weekdays, December peak, the 7-day periodogram spike) is in
`../eda/03-seasonality.md`.

## Sine / cosine terms — how you represent a cycle

A cycle is a **wave**, written with `sin` and `cos`. For a weekly pattern (period 7):

```
sin(2π × day / 7)   and   cos(2π × day / 7)
```

Each completes one full wave every 7 days. **Why both?** A single sine has a fixed peak position;
combining a sin *and* a cos (with weights the model learns) lets the peak land *anywhere*, at *any*
height. So 2 numbers per cycle let a plain linear regression fit "this series peaks Saturday, that
one Sunday."

### Worked example (real values)

The two columns for one week (`d` = day-of-week, 0 = Monday; sin/cos to 3 dp), beside the **real**
average sales-by-weekday of **store 1, GROCERY I**:

| `d` | weekday | `sin(2π·d/7)` | `cos(2π·d/7)` | mean sales (store 1, GROCERY I) |
|---|---|---|---|---|
| 0 | Mon | 0.000 | 1.000 | 2383 |
| 1 | Tue | 0.782 | 0.623 | 2409 |
| 2 | Wed | 0.975 | −0.223 | 2770 |
| 3 | Thu | 0.434 | −0.901 | 2229 |
| 4 | Fri | −0.434 | −0.901 | 2414 |
| 5 | Sat | −0.975 | −0.223 | 2323 |
| 6 | Sun | −0.782 | 0.623 | 1031 |

The model learns weights `w_sin`, `w_cos` (+ intercept) so `intercept + w_sin·sin + w_cos·cos`
traces this shape — roughly flat Mon–Sat with a sharp **Sunday dip to ~1031**. Two columns
reconstruct the weekly rhythm, and the weights adapt per series.

## Harmonics — how many waves per cycle

One sin/cos pair captures a smooth shape. A *sharp* feature (the weekend jump, a sudden dip) needs
extra pairs — **harmonics** — at 2×, 3× the base frequency. The periodogram's 3.5-day spike *is*
the weekly cycle's 2nd harmonic. From the EDA evidence we use roughly **2–3 weekly** harmonics and
**3–5 annual** harmonics — and leave the *sharp* December spike to holiday features rather than
piling on annual harmonics (see `../eda/03-seasonality.md`).

## `CalendarFourier` and `DeterministicProcess`

From **statsmodels**:

- **`CalendarFourier`** generates the sin/cos columns from dates. Instead of 365 "is it Jan 1 /
  Jan 2 / …" columns, you ask for a few Fourier pairs and get smooth columns approximating the
  annual wave.
- **`DeterministicProcess`** assembles the full time-feature table: a constant, a **trend** (1, 2,
  3, … over time), and those Fourier terms.

"**Deterministic**" is the key word — these depend **only on the date**, not past sales, so they're
computable for *any* future day and extend straight into the 16-day horizon. (Contrast a lag
feature, which needs actual past sales — see [lag-horizon.md](lag-horizon.md).)

## "Out of phase" — why gaps wreck this

**Phase** = where you are in the cycle. The sin/cos value is computed from the row's **position**,
assuming consecutive days. Drop a day and everything after it shifts:

```
Expected (regular):  Fri  Sat  Sun  Mon  …   ← weekend peak lands on Sat/Sun ✓
After a gap:         Fri  Sun  Mon  Tue  …   ← Sat's row gone, everything shifts left
                          ↑ the wave still "thinks" this slot is Saturday
```

In numbers, using the `sin` column above with real store 1 / GROCERY I sales. **Correct** (no gap):

| date | weekday | position `d` | `sin` used | sales |
|---|---|---|---|---|
| Aug 12 | Sat | 5 | −0.975 | 1630 |
| Aug 13 | Sun | 6 | −0.782 | 952 |

Now **drop Sat Aug 12** (a closed day) — everything after shifts up one position:

| date | weekday | position `d` (miscounted) | `sin` used | sales |
|---|---|---|---|---|
| Aug 13 | Sun | **5** (thinks it's Sat) | **−0.975** | 952 |

Sunday's 952 is now paired with **Saturday's** sin value. Every weekday past the gap is mislabelled
by one — a smeared, wrong rhythm, and nothing errors.

**Punchline:** fill the calendar *first* (so position = real date), *then* build Fourier
seasonality. That's why [`../data-traps/01-missing-calendar-days.md`](../data-traps/01-missing-calendar-days.md)
runs before any seasonal feature.

**Where:** seasonal/deterministic features in `src/features.py`.

**Related:** [periodogram.md](periodogram.md) · `../eda/03-seasonality.md` ·
[`../data-traps/01-missing-calendar-days.md`](../data-traps/01-missing-calendar-days.md)

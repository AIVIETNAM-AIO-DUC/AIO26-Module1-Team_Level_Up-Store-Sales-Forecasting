# EDA — Weekly & annual seasonality (notebook Section 4)

**Scope:** the repeating calendar patterns — the weekly cycle and the annual shape — and how many
Fourier harmonics each implies. The *mechanics* of Fourier terms and the periodogram are in
[../concepts/seasonality-fourier.md](../concepts/seasonality-fourier.md) and
[../concepts/periodogram.md](../concepts/periodogram.md).

**Status:** stub — to be written during the documentation consolidation pass. Verified facts
seeded below.

## Verified findings (seed)

- **Weekly:** weekend runs **~39%** above weekday average (weekend/weekday ratio ≈ **1.39**);
  **Sunday** is the peak (≈ 821,794), **Thursday** the trough (≈ 503,173).
- **Annual:** clear **December** peak (≈ 782,483) and **February** trough (≈ 571,895), with a
  July bump. Used *mean*-per-month (not sum) because training ends mid-Aug 2017, so Sep–Dec have
  one fewer year of data.
- **Periodogram (2–60 day band):** strongest short cycles are **7.0** and **3.5** days (and a
  minor ~15-day). The annual cycle is not resolvable here → the month profile is its evidence.

## Implication (seed)

- ~**2–3 weekly** Fourier harmonics (the sharp weekend jump needs the 3.5-day 2nd harmonic).
- ~**3–5 annual** harmonics for the smooth yearly shape.
- Leave the sharp December spike to explicit **holiday features**, not more harmonics.
- The month profile is not detrended (mixes growth with seasonality), yet December still tops it
  *despite missing its highest-sales year* — which strengthens the holiday-surge conclusion.

**Related:** [../concepts/periodogram.md](../concepts/periodogram.md) ·
[../concepts/seasonality-fourier.md](../concepts/seasonality-fourier.md) · `eda/04-calendar-holidays.md`

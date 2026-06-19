# The periodogram — discovering cycle lengths from the data

**Scope:** how we let the *data itself* reveal which repeating cycles it contains (using the
FFT), and how to read the resulting plot. This file is about **discovery**; turning a discovered
cycle into a model feature lives in [seasonality-fourier.md](seasonality-fourier.md).

---

## The idea: a prism for time series

Our daily-sales line is one wiggly curve over ~4.6 years. Hidden inside it are repeating
**cycles** of different lengths — a 7-day weekly rhythm, maybe a yearly one. A **periodogram**
splits that curve into pure waves and measures **how much of the variation sits at each cycle
length**. A tall spike at 7 days means "a strong 7-day repeat lives in this data."

Think of a **prism**: white light goes in, and out comes a spectrum showing how much of each
colour is present. The periodogram does the same to a time series, and the tool that does the
splitting is the **FFT** (Fast Fourier Transform) — just think of it as a "cycle-finder."

The point of doing this: instead of *assuming* the data has a weekly cycle, we let the data
**confirm** it.

## Frequency vs period (the one confusing part)

The FFT speaks in **frequency** = "cycles per day." We think in **period** = "days per cycle."
They are reciprocals:

```
period = 1 / frequency
```

- A 7-day cycle → frequency 1/7 ≈ 0.14 cycles/day.
- A slow, year-long cycle → a tiny frequency.

So **long, slow cycles sit at low frequency; short, fast cycles at high frequency.** Keep that
straight and the rest follows.

## How the "power" of a cycle is computed

For each candidate cycle length, the FFT essentially **fits a pair of waves** — one `cos` and one
`sin` — to the data and reads off two weights, `a` and `b`. (Two waves, not one, so the cycle's
peak can land anywhere — same reason explained in
[seasonality-fourier.md](seasonality-fourier.md).) The **power** is then:

```
power = a² + b²        (= |fft|² in code)
```

Intuitively: if the data really rises and falls on that cycle, the matching wave lines up and `a`
and `b` come out large → tall spike. If there's no such rhythm, the wave matches in some places
and cancels in others → power ≈ 0. Squaring drops the phase (we only care *how strong*, not
*where* the peak is) and gives an "energy."

## Two presentation choices that matter

When we actually plot it (notebook Section 4), two deliberate choices keep the picture honest:

- **Subtract the mean first.** The average sales level would otherwise show up as a giant spike
  at "frequency 0" (an infinitely long cycle) and dominate everything. Subtracting the mean
  removes it. *Note:* this removes the flat level, **not** the slow multi-year growth.
- **Look only at the 2–60 day band, on a log y-axis.** The slow growth lives at very long periods
  and produces a huge low-frequency spike that would dwarf the weekly one. Restricting to cycles
  of **2–60 days** zooms into the seasonal range (2 days is the shortest a daily series can
  resolve). A **log** y-axis then lifts the smaller spikes so their heights are comparable to the
  dominant one.

## Reading our figure

In this data the periodogram shows:

- the **tallest** spike at **7 days** — the weekly cycle, the dominant short-term pattern;
- a clear **second** spike at **3.5 days** — the *2nd harmonic* of the weekly cycle, present
  because the weekend jump is sharp rather than a smooth sine (so one wave can't capture it);
- a minor bump near **15 days**.

What it **cannot** show here is the annual (365-day) cycle: 365 is outside the 2–60 band, and
~4.6 years is too few repeats to resolve a clean yearly line anyway. That's why the **annual**
signal is read from the month-of-year profile (see `../eda/03-seasonality.md`), not the
periodogram.

## What it feeds

The periodogram is *evidence*, not a feature. Seeing a real, dominant 7-day spike (plus its
3.5-day harmonic) is what justifies using a weekly Fourier term with a few harmonics. How that
becomes a model feature — and how many harmonics to use — is
[seasonality-fourier.md](seasonality-fourier.md).

**Related:** [seasonality-fourier.md](seasonality-fourier.md) · `../eda/03-seasonality.md`

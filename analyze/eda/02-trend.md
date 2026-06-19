# EDA — Trend over time (notebook Section 3)

**Scope:** the slow drift of sales across the ~4.6 years — overall and per family — and what it
implies for modeling the trend.

**Status:** stub — to be written during the documentation consolidation pass. Verified facts
seeded below.

## Verified findings (seed)

- Overall daily total grows strongly: mean daily total roughly **doubles** from ~386k (2013) to
  ~856k (2017), but growth **decelerates** (big early jumps, smaller later ones).
- Families live on wildly different scales: the top ~6 families are ~83% of all sales (GROCERY I
  dominates); many families are tiny.
- Shape is shared, size is not: most families show the same upward drift + seasonal wiggle.

## Implication (seed)

- A trend term is essential; allow mild curvature (a straight line over/under-shoots the ends).
- Model per series/family, in `log1p` space, so small families aren't swamped.

**Related:** [../concepts/seasonality-fourier.md](../concepts/seasonality-fourier.md) · `eda/03-seasonality.md`

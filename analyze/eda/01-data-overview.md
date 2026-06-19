# EDA — Data overview & integrity (notebook Section 1–2)

**Scope:** the first look at the raw files — shapes, date spans, the headline counts — and the
gap-free reindex that the rest of the analysis depends on.

**Status:** stub — to be written during the documentation consolidation pass. Verified facts
seeded below so they aren't lost.

## Verified findings (seed)

- 7 CSVs load cleanly. `train` = 3,000,888 rows; `test` = 28,512; spans 2013-01-01 → 2017-08-15
  (train) and 2017-08-16 → 2017-08-31 (test).
- Headline counts asserted: **54** stores, **33** families, **54 × 33 = 1,782** series.
- After gap-free reindex: 3,000,888 → **3,008,016** rows (= 1,782 × 1,688 days); **7,128**
  closed-day rows inserted with `sales = 0`, `was_closed = True`; `assert_gapfree` passes.

**Related:** `data-traps/01-missing-calendar-days.md` · [../concepts/series-and-horizon.md](../concepts/series-and-horizon.md)

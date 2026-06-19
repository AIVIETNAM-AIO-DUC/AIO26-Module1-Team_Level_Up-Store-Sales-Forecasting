# EDA — Calendar & holiday effects (notebook Section 5)

**Scope:** what the holiday calendar and payday dates (15th & month-end) look like in the data,
and which become model features. Not yet explored.

**Status:** stub — to be written during the documentation consolidation pass. Verified facts
seeded below.

## Verified findings (seed)

- **Holiday table is not yes/no.** 350 rows; **12** `transferred = True` (holiday did *not*
  occur on that date); types include `Holiday`/`Additional`/`Bridge`/`Transfer`/`Event` and
  **5** `Work Day` rows (a normally-off day made into a working day). Locales: National /
  Regional / Local.
- **National-holiday effect:** 82 national day-off dates in the train span; open national
  holidays average ≈ **1.20×** a normal day (756k vs 632k), the 4 Christmases = **0** (closed).
  *Caveat:* not weekday-controlled, and the weekly swing (≈1.39) is larger — so 1.20 is
  suggestive, not exact.
- **Payday is a multi-day wave, not a one-day spike.** Naive "is it the 15th/month-end?" flag
  ≈ **1.01×** (negligible), but the day-of-month profile shows a real cycle: start-of-month
  surge (day 2 ≈ 743k vs mid-month trough ≈ 599k, ≈ **1.24×**) + a smaller post-15th bump.

## Implication (seed)

- Build an **effective holiday calendar** (reconcile transferred/bridge/work-day) and scope by
  locale (National → all; Regional → state; Local → city).
- Encode **days-since-payday / a short post-payday window**, not a single-day flag.

**Related:** `../data-traps/03-holidays.md` · `eda/03-seasonality.md`

**Related:** `data-traps/03-holidays.md` · `eda/03-seasonality.md`

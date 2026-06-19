# EDA — Anomalies (notebook Section 7)

**Scope:** one-off events that distort the baseline — chiefly the 2016-04-16 earthquake window
and major closures. Not yet explored.

**Status:** stub — to be written during the documentation consolidation pass. Verified facts
seeded below.

## Verified findings (seed)

- **2016 earthquake (2016-04-16, M7.8):** vs an 8-week pre-quake baseline (764,910), the **first
  week** spiked to ≈ **1.41×** (peak **2016-04-18** = 1,345,921 ≈ **1.76×**) on relief buying;
  **week +1 was already back to ~1.00×**. The acute spike is ≈ **1 week**, not weeks.
- **The "week +2" bump is NOT an earthquake echo** — it's the start-of-month payday surge + May 1
  Labour Day. Proof: the same Apr 30–May 6 window is *higher* in 2017 (959k, no quake) than 2016
  (901k). (Same trap as the spurious oil correlation — don't credit a routine pattern to a special
  event.)
- **Closures:** only the 4 Dec 25 are true *gaps* (sales = 0, restored in Section 2); the 4 Jan 1
  are *recorded* near-zero days (stores barely open), already covered by the calendar features.
- Bonus: the holidays file itself marks the aftermath with an `Event` row `Terremoto Manabi+15`.

## Implication (seed)

- Add an **earthquake-window flag** (a deliberately generous ≈ 2016-04-16 → mid-May envelope) so
  the model absorbs the shock as a known event. **Label anomalies, don't delete** them (deleting
  re-opens the gap-free calendar).

**Related:** `../data-traps/04-earthquake-anomaly.md` · `eda/04-calendar-holidays.md`

**Related:** `data-traps/04-earthquake-anomaly.md`

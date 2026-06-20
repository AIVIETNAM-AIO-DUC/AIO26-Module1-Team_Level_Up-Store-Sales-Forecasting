# Trap 1 — Missing calendar days (the counting trap)  ✅ FIXED

**Scope:** the closed-store days that are *absent* (not zero) in `train.csv`, why that silently
breaks lag/seasonal features, and the gap-free reindex that fixes it. The *mechanics* of how a
gap throws seasonality out of phase live in
[../concepts/seasonality-fourier.md](../concepts/seasonality-fourier.md); leakage is
[../concepts/leakage.md](../concepts/leakage.md).

---

## What you see

The store closes on a few days — notably **every Dec 25** — so those dates have **no row at all**
in `train.csv`. They are *absent*, not recorded as a zero.

- The series `store 1, AUTOMOTIVE` has 1,684 rows over a 1,688-day span → **4 missing days**, all
  Christmases (2013–2016).
- Across all series: 1,782 × 4 = **7,128 missing rows**.

## Why it quietly hurts

Time-series features locate "the past" by **row position** in the DataFrame, not by calendar
date. `df["sales"].shift(7)` simply means "take the value of the row 7 positions back" — it
never looks at the `date` column. Row-counting and day-counting only agree when no day is
missing.

### A concrete example

Series `store=1, AUTOMOTIVE` around Christmas 2016 (store closes Dec 25):

| index | date                       | sales |
|------:|----------------------------|------:|
| 100   | Dec 20                     | 5     |
| 101   | Dec 21                     | 6     |
| 102   | Dec 22                     | 4     |
| 103   | Dec 23                     | 7     |
| 104   | Dec 24                     | 8     |
| —     | *Dec 25 (no row — closed)* | —     |
| 105   | Dec 26                     | 5     |
| 106   | Dec 27                     | 6     |
| 107   | Dec 28                     | 9     |

At row `index=107` (Dec 28), `shift(7)` grabs `index=100` → **Dec 20**. But "7 days before
Dec 28" should be **Dec 21**. Off by one day. The lag that was supposed to be "last Wednesday"
is now "last Tuesday."

The same shift breaks **seasonality (Fourier) terms**, which assume a regular daily frequency:
a hole slides the weekly and yearly waves out of phase, so the pattern lands on the wrong
weekday. (The exact phase shift, in numbers, is worked through in
[../concepts/seasonality-fourier.md](../concepts/seasonality-fourier.md).)

Nothing errors — you just get a worse model and no idea why. This is the canonical *silent trap*.

## The fix

Rebuild each series onto a **complete daily calendar** and fill the missing days with
**`sales = 0`** (the store sold nothing — which is *true*, and keeps the row spacing regular).

- Why `0` and not `NaN`? `NaN` breaks lag arithmetic and labels a *known* closed day as "unknown."
- Why `0` and not interpolation? Interpolation would *invent* sales on a day the store was shut.
- Every inserted row is tagged `was_closed = True`, so the information isn't lost and can even
  become a feature.

A closed day (sales = 0) and a zero-demand day are different things; the `was_closed` flag is how
we keep them straight (see `../eda/06-anomalies.md` for the Dec 25 vs Jan 1 contrast).

## The numbers are the proof

```
original rows   : 3,000,888    ← what Kaggle gave us
reindexed rows  : 3,008,016    ← after filling gaps
expected        : 1,782 × 1,688 = 3,008,016
rows inserted   : 7,128        ← 4 per series
```

- **1,688** calendar days = 2013-01-01 → 2017-08-15 inclusive.
- A complete grid *must* have exactly `series × days` rows. The total matching `1,782 × 1,688`
  is the proof the grid is complete — no series short a day, none duplicated.
- **4 per series** because only the 2013–2016 Christmases fall inside the window (2017's is past
  the train end, 2012's is before the start).

## Verify

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
t = data.load_train(); r = data.reindex_series_gapfree(t)
print('inserted closed-days:', int(r['was_closed'].sum()))   # 7128
"
```

## Where

`src/data.py :: reindex_series_gapfree()`, proven correct by
`src/validation.py :: assert_gapfree()` (it *checks* that no gaps remain). Empirically confirmed
in the notebook — see `../eda/01-data-overview.md`.

**Lesson:** in time series, "no row" and "a zero" are not the same thing, and the difference can
silently break every lag and seasonal feature downstream. Make the calendar complete **first**.

**Related:** [../concepts/seasonality-fourier.md](../concepts/seasonality-fourier.md) ·
`../eda/01-data-overview.md`

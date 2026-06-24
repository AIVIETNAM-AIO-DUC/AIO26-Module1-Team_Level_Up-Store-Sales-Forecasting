# Trap 4 — The 2016 earthquake anomaly  ✅ FIXED

**Scope:** the one-off relief-driven spike after the 2016-04-16 earthquake, why an unmarked
anomaly distorts trend/seasonality, and the "label, don't delete" flag. The measured size and
duration are in `../eda/06-anomalies.md`.

---

## What you see

A magnitude-7.8 earthquake on **2016-04-16** caused a relief-buying spike. The EDA measured it: the
**first week** ran ≈ **1.41×** the pre-quake baseline (peak ≈ **1.76×** on 2016-04-18), then
**week +1 was already back to normal** — an acute spike of about a week, not weeks.
(`../eda/06-anomalies.md` also shows the apparent "week +2 echo" is really the payday + May-1
Labour-Day calendar effect, not the quake.)

## Why it quietly hurts

A one-off spike is **not** a repeating pattern. Leave it unmarked and the trend and seasonality
terms try to "explain" it, bending out of shape for *every* year.

## The fix

Add a binary (or decaying) **earthquake-window flag** covering a deliberately generous envelope
(≈ 2016-04-16 → mid-May 2016) so the model can absorb the spike *as a known event*. We do **not**
delete those rows — that would punch a hole back into the gap-free index built in
[`01-missing-calendar-days.md`](01-missing-calendar-days.md).

## Verify

The aftermath Event rows the holidays file itself carries:

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
h = data.load_holidays()
mask = h['description'].fillna('').str.contains('Terremoto')
print(h[mask][['date','type','locale','description']].head())  # the aftermath Event rows
"
```

And that the flag covers exactly the intended window:

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from src import features
cf = features.make_calendar_features(pd.date_range('2016-01-01','2016-12-31', freq='D'))
print('earthquake days flagged:', int(cf['is_earthquake'].sum()))      # 30 (Apr 16 - May 15)
print('2016-04-15:', int(cf.loc['2016-04-15','is_earthquake']))         # 0 (day before)
print('2016-04-16:', int(cf.loc['2016-04-16','is_earthquake']))         # 1 (quake day)
print('2016-05-15:', int(cf.loc['2016-05-15','is_earthquake']))         # 1 (window end)
print('2016-05-16:', int(cf.loc['2016-05-16','is_earthquake']))         # 0 (after)
"
```

## Where

`make_calendar_features(index)` in `src/features.py` emits the date-only `is_earthquake` flag
(1 across 2016-04-16 → 2016-05-15). The rows are kept, not deleted, so the gap-free index from
[`01-missing-calendar-days.md`](01-missing-calendar-days.md) stays intact. (Bonus: the holidays
file itself marks the aftermath with an `Event` row, `Terremoto Manabi+15`.)

**Lesson:** don't delete anomalies — *label* them. Give the model a flag so it can say "this was
special" rather than treating the spike as normal behaviour.

**Related:** `../eda/06-anomalies.md` · [`01-missing-calendar-days.md`](01-missing-calendar-days.md)

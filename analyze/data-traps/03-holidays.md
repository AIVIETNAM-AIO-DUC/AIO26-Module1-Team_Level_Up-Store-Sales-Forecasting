# Trap 3 — Holidays aren't a simple yes/no (domain modeling)  ⏳ FIX PLANNED

**Scope:** why `holidays_events.csv` can't be read as a plain "was it a holiday?" table, and how
to build an *effective* holiday calendar. The verified composition (350 rows, 12 transferred, the
~1.20× national-holiday lift) is in `../eda/04-calendar-holidays.md`.

---

## What you see

The table looks like a holiday flag but isn't:

- A row with `transferred = True` means the holiday did **not** happen on its listed date — it was
  officially moved. A separate `type = "Transfer"` row carries the date it was *actually* observed.
- There are `Bridge` days (an extra day off) and `Work Day` rows (a normally-off day turned into a
  working one — the *opposite* of a holiday).
- `locale` is National / Regional / Local — a holiday doesn't apply to every store.

(Counts confirmed in `../eda/04-calendar-holidays.md`: 12 `transferred`, 5 `Work Day`.)

## Why it quietly hurts

Trust the raw `date` and you attach "holiday demand" to the wrong day. Ignore `locale` and you
apply one city's local holiday to all 54 stores — pure noise.

## The fix

Compute an **effective holiday calendar**:

- drop the `transferred = True` originals, honour the matching `Transfer` / `Bridge` / `Work Day`
  rows;
- scope each holiday by locale — **National** → all stores; **Regional** → stores in that `state`;
  **Local** → stores in that `city`.

### Worked example — Independencia de Guayaquil, 2012

The raw table carries **two rows for one real observance**:

```
date        type      transferred   meaning
2012-10-09  Holiday   True          official date — but NOT observed here
2012-10-12  Transfer  False         the day the holiday actually happened
```

Read naively, you'd attach holiday demand to **Oct 9** (because the type literally says
`Holiday`) and treat **Oct 12** as a normal Friday — both wrong.

The fix is two filters:

1. `eff = h[~h['transferred']]` — drops the `2012-10-09` ghost (`transferred=True`).
2. Tag everything that survives as `effect='holiday'` (Holiday / Transfer / Bridge /
   Additional / Event), except `Work Day` rows — those mean the store is *open* on a
   normally-off day → `effect='work_day'`.

After both filters, the Guayaquil pair collapses to one correct entry:

```
date        type      effect
2012-10-12  Transfer  holiday
```

Oct 9 is gone (correctly — no spike there); Oct 12 is flagged as a holiday (correctly).
The `type` column survives, so a later feature can still distinguish *moved* vs.
*fixed-date* holidays if the EDA suggests they behave differently.

The same two filters resolve all 12 transferred originals in the file (verified in
`../eda/04-calendar-holidays.md`).

### Worked example — Local scope, `Fundacion de Manta`, 2014

`locale` answers *which stores does this holiday apply to?* and has three values:

| locale     | matched against              | applies to                       |
|------------|------------------------------|----------------------------------|
| National   | (everything)                 | all 54 stores                    |
| Regional   | `locale_name` ↔ store `state`| every store in that state        |
| Local      | `locale_name` ↔ store `city` | only stores in that city (1–2)   |

A **Local** holiday is a city-only festival:

```
date        type     locale  locale_name  description
2014-03-02  Holiday  Local   Manta        Fundacion de Manta
```

After locale scoping, only the Manta store (store_nbr 52 in our data) sees the row.
Store 1 (Quito) gets nothing for that date — because demand in Quito doesn't move when
Manta throws a party.

**The trap if you ignore locale:** you mark Mar 2 as a holiday across all 54 stores. The
model then sees `holiday=1` paired with normal Quito sales, learns the flag means nothing,
and the *real* signal in Manta gets averaged into noise.

The fix is a locale-by-locale merge against `stores.csv`:

```python
nat = eff[eff['locale']=='National'].merge(stores[['store_nbr']], how='cross')
reg = eff[eff['locale']=='Regional'].merge(
    stores[['store_nbr','state']], left_on='locale_name', right_on='state')
loc = eff[eff['locale']=='Local'].merge(
    stores[['store_nbr','city']],  left_on='locale_name', right_on='city')
```

One Local row typically expands into 1–2 store-day rows, not 54.

### Worked example — `Work Day`, the anti-holiday

A `Work Day` row is the **opposite** of a holiday. It marks a date that would normally be
off (a Sunday-after-a-bridge, a long-weekend recovery) but the government declared
everyone has to work to make up for an extended break:

```
date        type      description
2013-01-05  Work Day  Recupero puente Navidad             ← "recovering the Christmas bridge"
2013-01-12  Work Day  Recupero puente primer dia del ano  ← "recovering New Year's bridge"
2014-12-20  Work Day  Recupero Puente Navidad
```

Spanish `recupero` = "make-up". The pattern: government grants a `Bridge` day (e.g. Dec 26
off so people get a long break), then later issues a `Work Day` (e.g. Jan 5) on a normally-
closed day to compensate.

**The trap if you trust `type` blindly:** you'd see "type = Work Day" sitting in a holiday
table and assume it's *a* kind of holiday. It isn't — it's the inverse. Flagging it as one
makes the model see `holiday=1` on a day with normal-or-elevated sales (people shopping on
an unexpectedly open Sunday) and the holiday coefficient gets dragged toward zero.

That's why the classifier maps these to `effect='work_day'`, not `effect='holiday'`:

```python
def classify(t):
    if t == 'Work Day':
        return 'work_day'   # store OPEN on a normally-off day → NOT a holiday
    return 'holiday'        # Holiday / Transfer / Bridge / Additional / Event
```

Downstream you have a choice: drop these rows (treat them as ordinary days) or give them
their own flag (`is_work_day_recovery`) if the unusual-Sunday-shopping effect is large
enough to model. The point is they must **not** end up in your holiday flag.

### Three mechanisms in one table

|                       | Transfer                                  | Local                                   | Work Day                                 |
|-----------------------|-------------------------------------------|-----------------------------------------|------------------------------------------|
| What can mislead you  | Original date carries `transferred=True`  | One city's festival on the table        | An "anti-holiday" with `type` ≠ Holiday  |
| Stores affected       | All / region / city (per its own locale)  | Just that city (1–2 stores)             | All 54 stores                            |
| Effect on demand      | Spike on the *moved* date                 | Spike in that city only                 | Open on a normally-off day               |
| Effect tag in the fix | `effect='holiday'`                        | `effect='holiday'`                      | `effect='work_day'`                      |
| If you get it wrong   | Spike attached to the wrong day           | 52 stores' holiday flag becomes noise   | Holiday flag points the wrong way        |

All three bugs share the same root cause — trusting `date`, `type`, and "one row = all
stores" at face value. The fix is one shape: re-derive *what actually happened, per store,
per day* before any feature touches the data.

## Verify

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
h = data.load_holidays()
print('total rows     :', len(h))                              # 350
print('transferred    :', int(h['transferred'].sum()))         # 12
print('Work Day rows  :', int((h['type'] == 'Work Day').sum())) # 5
print(h['locale'].value_counts().to_dict())                    # National/Regional/Local breakdown
"
```

## Where

Holiday features in `src/features.py`. The payday signal (a related calendar effect) is best
encoded as a **post-payday window**, not a single-day flag — see `../eda/04-calendar-holidays.md`.

**Lesson:** a column named like a feature isn't automatically a usable feature. Understanding the
domain (how holidays *actually* work) is part of feature engineering.

**Related:** `../eda/04-calendar-holidays.md`

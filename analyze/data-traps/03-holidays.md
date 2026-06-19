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

## Where

Holiday features in `src/features.py`. The payday signal (a related calendar effect) is best
encoded as a **post-payday window**, not a single-day flag — see `../eda/04-calendar-holidays.md`.

**Lesson:** a column named like a feature isn't automatically a usable feature. Understanding the
domain (how holidays *actually* work) is part of feature engineering.

**Related:** `../eda/04-calendar-holidays.md`

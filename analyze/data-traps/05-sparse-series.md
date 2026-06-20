# Trap 5 — New / sparse series (the coverage trap)  ⏳ FIX PLANNED

**Scope:** store × family pairs with little or no meaningful history, and why every one still
needs *a* prediction. The fallback rules themselves (seasonal-naive / near-zero) and the routing
gate live in [`../concepts/baselines.md`](../concepts/baselines.md).

---

## What you see

Some store + family pairs have little or no meaningful history — a family that barely sells, or
one that only started recently.

## Why it quietly hurts

A complex model can't learn a pattern that isn't there — but the submission **must** contain a
prediction for all **28,512** rows. A missing or `NaN` prediction is an invalid submission, and a
model extrapolating from no history produces unreliable garbage.

## The fix

For thin series, fall back to a simple **seasonal-naive / near-zero** prediction so every row
always gets a sensible value. A rule-based gate (chosen by *us*, e.g. "fewer than N non-zero days
→ fallback") routes thin series around the model *before* training, so they neither get garbage
predictions nor mislead the model. The mechanics — what each rule predicts, and the
exclude-before-training wiring — are in [`../concepts/baselines.md`](../concepts/baselines.md).

## Verify

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
t = data.load_train()
totals = t.groupby(['store_nbr','family'])['sales'].sum()
nonzero_days = (t['sales'] > 0).groupby([t['store_nbr'], t['family']]).sum()
print('total (store,family) pairs :', int(len(totals)))                  # 1782
print('pairs with zero total sales:', int((totals == 0).sum()))          # the all-zero extreme
print('pairs with < 30 non-zero days:', int((nonzero_days < 30).sum()))  # the thin tail
"
```

## Where

The fallback path in `src/models.py`.

**Lesson:** a model only needs to be sophisticated *where it can be*. Always have a
dumb-but-safe fallback so you never ship a blank prediction.

**Related:** [`../concepts/baselines.md`](../concepts/baselines.md)

# Store Sales — Learner's Docs (index)

A guided, beginner-friendly walkthrough of this time-series project. **Each file has one small
scope** — read or revisit one idea at a time. Every number in these docs is reproduced from the
real CSVs with `uv run`.

## How the docs are organized

- **`concepts/`** — one self-contained idea per file (what a *series* is, the metric, leakage,
  seasonality, the periodogram, …). Read these to *understand* a topic.
- **`data-traps/`** — the silent ways real data quietly breaks a model, and how we defuse each
  one. Format per file: *what you see → why it hurts → the fix → where in code*.
- **`eda/`** — what the exploratory notebook (`notebooks/01_eda.ipynb`) actually showed, section
  by section, with the verified numbers and the modeling implication.

## Suggested reading order

1. `concepts/series-and-horizon.md` — what we predict and the unit we predict over
2. `concepts/rmsle-metric.md` — how we're scored, and why we work in log space
3. `data-traps/` (01 → 06) — the data problems, easiest to subtlest
4. `concepts/leakage.md` — the one idea behind several traps
5. `concepts/validation-holdout.md` — how we get an honest scoreboard
6. `eda/` (01 → 06) — the evidence behind every modeling choice
7. `concepts/seasonality-fourier.md`, `concepts/periodogram.md` — the seasonality machinery
8. `concepts/baselines.md`, `concepts/lag-horizon.md` — forecasting mechanics

## Appendix A — Reproduce the numbers yourself

Every figure in these docs is reproducible inside the uv-managed environment:

```bash
# Confirm the closed-day gaps and the reindex fix (Trap 1)
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
t = data.load_train(); r = data.reindex_series_gapfree(t)
print('inserted closed-days:', int(r['was_closed'].sum()))   # 7128
"

# Confirm the oil blanks (Trap 2)
uv run python -c "
import sys; sys.path.insert(0,'.')
from src import data
print('oil null prices:', int(data.load_oil()['dcoilwtico'].isna().sum()))  # 43
"
```

The exploratory notebook `notebooks/01_eda.ipynb` runs the full analysis end-to-end (execute it
with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`).

## Appendix B — Data-trap status at a glance

| Trap | Problem | Status | Home |
|----|---------|--------|------|
| 1 | Missing calendar days | ✅ Fixed | [data-traps/01](data-traps/01-missing-calendar-days.md) |
| 2 | Oil gaps + 43 blanks | ⏳ Planned | [data-traps/02](data-traps/02-oil-gaps.md) |
| 3 | Transferred/bridge holidays | ⏳ Planned | [data-traps/03](data-traps/03-holidays.md) |
| 4 | 2016 earthquake spike | ⏳ Planned | [data-traps/04](data-traps/04-earthquake-anomaly.md) |
| 5 | New/sparse series | ⏳ Planned | [data-traps/05](data-traps/05-sparse-series.md) |
| 6 | Transactions = past-only | ⚠️ Constraint | [data-traps/06](data-traps/06-transactions-past-only.md) |

_This is a living set of docs — as each fix lands, update the status here and the matching page._

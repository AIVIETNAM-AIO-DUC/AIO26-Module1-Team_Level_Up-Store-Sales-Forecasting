# The five rules that keep us honest

**Scope:** the five disciplines we carry through **every** modeling stage — so a good validation
score actually means a good real-world score. Each rule has a deeper home page; this page is just
the checklist that ties them together.

---

These five aren't features you add once — they're habits applied at every step, from the first
baseline to the final hybrid model. Skip any one and your scoreboard quietly lies to you.

| # | Rule | Why it matters | Home |
|---|------|----------------|------|
| 1 | **Score with RMSLE in log space** | We're graded on *relative* error. Training and scoring in `log1p` keeps tiny families (a few sales/day) comparable to huge ones (thousands/day), so the big stores don't dominate and the small ones aren't neglected. | [rmsle-metric.md](rmsle-metric.md) |
| 2 | **Clip predictions ≥ 0** | Sales can't be negative, but a regression model in log space can still emit a negative after `expm1`. An unclipped negative is impossible *and* crashes the metric. Floor every prediction at 0 before scoring. | [rmsle-metric.md](rmsle-metric.md) |
| 3 | **Validate on a single time-respecting 16-day holdout** (no random folds) | Random k-fold leaks the future into the past. The holdout must mimic the real task exactly: train on the past, predict the next 16 days. | [validation-holdout.md](validation-holdout.md) |
| 4 | **Keep a seasonal-naive / near-zero fallback** | Every one of the **28,512** submission rows must get a value. Thin or brand-new series can't be modeled — route them to a safe, dumb prediction instead of shipping garbage or a blank. | [baselines.md](baselines.md) |
| 5 | **Log each iteration's RMSLE** | So you can *see* what actually helped. Without a written scoreboard you're guessing whether a feature improved anything — keep whatever lowers RMSLE on the *same* holdout. | [validation-holdout.md](validation-holdout.md) |

## Where in code

Four of the five are already built and guarded as reusable helpers; the fallback is the one still
to come.

| Rule | Helper |
|------|--------|
| 1 RMSLE | `src/validation.py :: rmsle()` |
| 2 Clip | `src/validation.py :: clip_nonneg()` |
| 3 Holdout | `src/validation.py :: train_holdout_split()` |
| 4 Fallback | the fallback path in `src/models.py` (built alongside the baselines) |
| 5 Iteration log | `src/validation.py :: log_iteration()` |

**Lesson:** the goal isn't a low validation number — it's a validation number you can *trust*.
These five rules are what make the score honest.

**Related:** [rmsle-metric.md](rmsle-metric.md) · [validation-holdout.md](validation-holdout.md) ·
[baselines.md](baselines.md) · [leakage.md](leakage.md)

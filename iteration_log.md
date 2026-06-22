# Iteration Log — Store Sales Time Series Forecasting

Running record of every modeling iteration. Each row logs the technique added, its
holdout RMSLE (scored on the shared 16-day window 2017-07-31 → 2017-08-15, in log
space with non-negative clipping), and the change versus the previous best.
Populated by `src.validation.log_iteration`.

| Stage / technique added | Holdout RMSLE | Δ vs previous best | Notes |
|-------------------------|---------------|--------------------|-------|
| baseline: seasonal-naive (weekly) | 0.61704 | - | same-weekday-prior-week; repeats last training week over 16-day horizon |

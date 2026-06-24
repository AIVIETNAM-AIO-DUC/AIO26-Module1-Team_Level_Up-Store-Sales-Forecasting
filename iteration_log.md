# Iteration Log — Store Sales Time Series Forecasting

Running record of every modeling iteration. Each row logs the technique added, its
holdout RMSLE (scored on the shared 16-day window 2017-07-31 → 2017-08-15, in log
space with non-negative clipping), and the change versus the previous best.
Populated by `src.validation.log_iteration`.

| Stage / technique added | Holdout RMSLE | Δ vs previous best | Notes |
|-------------------------|---------------|--------------------|-------|
| baseline: seasonal-naive (weekly) | 0.61704 | - | same-weekday-prior-week; repeats last training week over 16-day horizon |
| deterministic: trend + weekly/annual Fourier (LinearRegression) | 0.62188 | +0.00484 (worse) | per-series fit on active history; annual dropped for <1yr series; log1p space |
| feature model: + holidays | 0.62305 | +0.00601 (worse) | 19 features; per-series LinearRegression, log1p; no-leak base_lag=16 |
| feature model: + calendar | 0.60717 | -0.00987 (better) | 24 features; per-series LinearRegression, log1p; no-leak base_lag=16 |
| feature model: + promotions & oil | 0.58226 | -0.02491 (better) | 26 features; per-series LinearRegression, log1p; no-leak base_lag=16 |
| feature model: + lags | 0.50991 | -0.07235 (better) | 33 features; per-series LinearRegression, log1p; no-leak base_lag=16 |

"""Feature construction — all built past-only to avoid future leakage (FR-009).

Responsibility: turn the gap-free series and supporting tables into model features:
deterministic trend + Fourier seasonality, calendar/payday/earthquake flags, the
effective locale-scoped holiday calendar, forward-filled oil, promotions, and
lag/rolling features. Every feature must be knowable at prediction time.

Implemented in tasks T023 (deterministic), T026 (holidays), T027 (calendar),
T028 (oil + promo), T029 (lags/rolling).
"""

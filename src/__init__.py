"""Reusable logic for the Store Sales time series forecasting project.

Notebooks (``notebooks/01_eda`` … ``05_hybrid``) handle teaching and exploration;
stabilized, reusable functions live here so the same scoring and feature logic is
shared across every stage (Constitution III: code quality & reproducibility).

Modules:
    data        Load/join raw CSVs; gap-free per-series daily reindex; write submission.
    features    Deterministic, calendar/holiday, oil, promotion, and lag features (past-only).
    validation  16-day holdout split; RMSLE in log space; no-leak / gap-free assertions; iteration log.
    models      Seasonal-naive baseline, deterministic, and hybrid model wrappers.
"""

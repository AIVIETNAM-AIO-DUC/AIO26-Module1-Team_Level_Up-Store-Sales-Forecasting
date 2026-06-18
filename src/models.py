"""Model wrappers: baseline, deterministic, and hybrid.

Responsibility: provide a consistent fit/predict interface for each modeling stage —
seasonal-naive baseline, LinearRegression on deterministic features, and the hybrid
(linear fit + XGBoost on residuals). All operate in log space and clip predictions to
non-negative; randomness is seeded for reproducibility (Constitution III/IV).

Implemented in tasks T020 (seasonal_naive_predict), T024 (DeterministicModel),
T031 (HybridModel), T032 (sparse-series fallback).
"""

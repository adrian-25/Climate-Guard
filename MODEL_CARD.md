# Model Card — ClimateGuard v1.0.0

## Intended use

Research and education: a next-day heatwave-risk estimate for the five validated cities. It supports awareness, not official alerting, clinical decisions, emergency dispatch, or unvalidated locations.

## Data & method

Daily Open-Meteo/ERA5 reanalysis from 1990-01-01 to 2025-08-31. The locked Random Forest uses 110 current-weather, lag, rolling, trend, anomaly, calendar, and city features. Training ends 2022-12-31; the held-out period is 2023-01-01 to 2025-08-30. Labels are IMD-inspired project labels, not IMD-certified observations.

## Held-out results

F1 0.6947, precision 0.5789, recall 0.8684, Brier 0.0083 on 4,865 days / 38 positives. Bootstrap 95% intervals: F1 0.5819–0.7984, precision 0.4513–0.7143, recall 0.7532–0.9677. Persistence F1 is 0.6316; an IMD-style qualifying-day proxy reaches 0.7470. See `evaluation/results/latest.json`.

## Limitations

Small positive sample, zero held-out positives for Ahmedabad and Mumbai, ERA5-to-forecast distribution shift, proxy outcomes for live tracking, and uncertain calibration at high probability. The default model must not be switched without an evaluated, versioned comparison and owner approval.

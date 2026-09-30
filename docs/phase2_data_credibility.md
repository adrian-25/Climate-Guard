# Phase 2 data credibility report

## Current labelled data

The current labelled dataset contains 65,135 daily records: five cities, 13,027
records per city, from 1990-01-01 through 2025-08-31.

| City | Labelled heatwave days | Held-out test positives |
|---|---:|---:|
| New Delhi | 213 | 18 |
| Lucknow | 141 | 16 |
| Nagpur | 119 | 4 |
| Ahmedabad | 32 | 0 |
| Mumbai | 0 | 0 |

The test set has 4,865 rows and only 38 positive next-day labels. Consequently,
aggregated evaluation is more informative than city-specific scores; Ahmedabad and
Mumbai cannot support a meaningful held-out heatwave evaluation.

## Expansion feasibility

Expansion is technically feasible with the same Open-Meteo Historical Weather API.
Its ERA5 archive is global from 1940 at 0.25° resolution, supports the daily
variables used by ClimateGuard, and permits multiple coordinate queries. For a
multi-decade climate study, use a consistent ERA5/ERA5-Land choice rather than a
mixed best-match source. See the official documentation:
https://open-meteo.com/en/docs/historical-weather-api

Expansion is **not** yet implemented. It requires a new city manifest, regenerated
labels and features, chronological split, retraining, and full comparison against
`v1.0.0`. Any resulting dataset and model must be versioned; `v1.0.0` remains the
active default until explicit approval.

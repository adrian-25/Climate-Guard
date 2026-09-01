# ClimateGuard Final Model — Prediction Contract

**Version:** 1.0.0  
**Phase:** 14 — Final Model Selection  
**Status:** Production Candidate  
**Date:** 2026-09-02  
**Author:** Adrian (Part 1 — Dataset & ML)

---

## 1. Model Artifact

| Item | Path | Size |
|---|---|---|
| Model file | `models/final/climateguard_final_model.joblib` | 1.86 MB |
| Feature list | `models/final/feature_list.json` | 10 KB |
| Metadata | `models/final/metadata.json` | 6 KB |

Load with:

```python
import joblib
model = joblib.load("models/final/climateguard_final_model.joblib")
```

---

## 2. Prediction Task

**Task:** 1-day-ahead heatwave prediction

```
Given weather observations and derived features at date T
    → predict whether tomorrow (T+1) will be a heatwave day
```

**Target:** `heatwave_next_day` — binary {0, 1}
- `1` = heatwave day predicted for T+1
- `0` = normal day predicted for T+1

---

## 3. Input Contract

### 3.1 Required features

The model requires **exactly 110 features** in the **exact order** specified in `models/final/feature_list.json`.

The feature list is reproduced below in index order:

| Index | Feature Name | Group |
|---|---|---|
| 0 | apparent_temperature_max | Current weather |
| 1 | apparent_temperature_mean | Current weather |
| 2 | apparent_temperature_min | Current weather |
| 3 | city_encoded | City |
| 4 | day_of_year | Calendar |
| 5 | doy_cos | Calendar |
| 6 | doy_sin | Calendar |
| 7 | et0_fao_evapotranspiration | Current weather |
| 8 | heatwave_lag1 | Lag (Group 2) |
| 9 | is_coastal | City |
| 10 | latitude | City |
| 11 | longitude | City |
| 12 | month | Calendar |
| 13 | month_cos | Calendar |
| 14 | month_sin | Calendar |
| 15 | precipitation_sum | Current weather |
| 16 | precipitation_sum_lag1 | Lag (Group 2) |
| 17 | precipitation_sum_lag2 | Lag (Group 2) |
| 18 | precipitation_sum_lag3 | Lag (Group 2) |
| 19 | precipitation_sum_lag7 | Lag (Group 2) |
| 20 | precipitation_sum_roll3_max | Rolling (Group 3) |
| 21 | precipitation_sum_roll3_mean | Rolling (Group 3) |
| 22 | precipitation_sum_roll3_min | Rolling (Group 3) |
| 23 | precipitation_sum_roll7_max | Rolling (Group 3) |
| 24 | precipitation_sum_roll7_mean | Rolling (Group 3) |
| 25 | precipitation_sum_roll7_min | Rolling (Group 3) |
| 26 | qualifying_day | Current weather |
| 27 | relative_humidity_2m_max | Current weather |
| 28 | relative_humidity_2m_mean | Current weather |
| 29 | relative_humidity_2m_mean_lag1 | Lag (Group 2) |
| 30 | relative_humidity_2m_mean_lag2 | Lag (Group 2) |
| 31 | relative_humidity_2m_mean_lag3 | Lag (Group 2) |
| 32 | relative_humidity_2m_mean_lag7 | Lag (Group 2) |
| 33 | relative_humidity_2m_mean_roll3_max | Rolling (Group 3) |
| 34 | relative_humidity_2m_mean_roll3_mean | Rolling (Group 3) |
| 35 | relative_humidity_2m_mean_roll3_min | Rolling (Group 3) |
| 36 | relative_humidity_2m_mean_roll7_max | Rolling (Group 3) |
| 37 | relative_humidity_2m_mean_roll7_mean | Rolling (Group 3) |
| 38 | relative_humidity_2m_mean_roll7_min | Rolling (Group 3) |
| 39 | relative_humidity_2m_min | Current weather |
| 40 | season_code | Calendar |
| 41 | shortwave_radiation_sum | Current weather |
| 42 | surface_pressure_mean | Current weather |
| 43 | surface_pressure_mean_lag1 | Lag (Group 2) |
| 44 | surface_pressure_mean_lag2 | Lag (Group 2) |
| 45 | surface_pressure_mean_lag3 | Lag (Group 2) |
| 46 | surface_pressure_mean_lag7 | Lag (Group 2) |
| 47 | surface_pressure_mean_roll3_max | Rolling (Group 3) |
| 48 | surface_pressure_mean_roll3_mean | Rolling (Group 3) |
| 49 | surface_pressure_mean_roll3_min | Rolling (Group 3) |
| 50 | surface_pressure_mean_roll7_max | Rolling (Group 3) |
| 51 | surface_pressure_mean_roll7_mean | Rolling (Group 3) |
| 52 | surface_pressure_mean_roll7_min | Rolling (Group 3) |
| 53 | temperature_2m_max | Current weather |
| 54 | temperature_2m_max_lag1 | Lag (Group 2) |
| 55 | temperature_2m_max_lag2 | Lag (Group 2) |
| 56 | temperature_2m_max_lag3 | Lag (Group 2) |
| 57 | temperature_2m_max_lag7 | Lag (Group 2) |
| 58 | temperature_2m_max_roll3_max | Rolling (Group 3) |
| 59 | temperature_2m_max_roll3_mean | Rolling (Group 3) |
| 60 | temperature_2m_max_roll3_min | Rolling (Group 3) |
| 61 | temperature_2m_max_roll7_max | Rolling (Group 3) |
| 62 | temperature_2m_max_roll7_mean | Rolling (Group 3) |
| 63 | temperature_2m_max_roll7_min | Rolling (Group 3) |
| 64 | temperature_2m_mean | Current weather |
| 65 | temperature_2m_mean_lag1 | Lag (Group 2) |
| 66 | temperature_2m_mean_lag2 | Lag (Group 2) |
| 67 | temperature_2m_mean_lag3 | Lag (Group 2) |
| 68 | temperature_2m_mean_lag7 | Lag (Group 2) |
| 69 | temperature_2m_mean_roll3_max | Rolling (Group 3) |
| 70 | temperature_2m_mean_roll3_mean | Rolling (Group 3) |
| 71 | temperature_2m_mean_roll3_min | Rolling (Group 3) |
| 72 | temperature_2m_mean_roll7_max | Rolling (Group 3) |
| 73 | temperature_2m_mean_roll7_mean | Rolling (Group 3) |
| 74 | temperature_2m_mean_roll7_min | Rolling (Group 3) |
| 75 | temperature_2m_min | Current weather |
| 76 | temperature_2m_min_lag1 | Lag (Group 2) |
| 77 | temperature_2m_min_lag2 | Lag (Group 2) |
| 78 | temperature_2m_min_lag3 | Lag (Group 2) |
| 79 | temperature_2m_min_lag7 | Lag (Group 2) |
| 80 | temperature_2m_min_roll3_max | Rolling (Group 3) |
| 81 | temperature_2m_min_roll3_mean | Rolling (Group 3) |
| 82 | temperature_2m_min_roll3_min | Rolling (Group 3) |
| 83 | temperature_2m_min_roll7_max | Rolling (Group 3) |
| 84 | temperature_2m_min_roll7_mean | Rolling (Group 3) |
| 85 | temperature_2m_min_roll7_min | Rolling (Group 3) |
| 86 | tmax_delta_1d | Trend (Group 4) |
| 87 | tmax_delta_3d | Trend (Group 4) |
| 88 | tmax_delta_7d | Trend (Group 4) |
| 89 | tmax_departure | Current weather |
| 90 | tmax_departure_lag1 | Lag (Group 2) |
| 91 | tmax_departure_lag2 | Lag (Group 2) |
| 92 | tmax_departure_lag3 | Lag (Group 2) |
| 93 | tmax_departure_lag7 | Lag (Group 2) |
| 94 | tmax_departure_zscore | Anomaly (Group 5) |
| 95 | tmax_normal | Current weather |
| 96 | tmax_slope_3d | Trend (Group 4) |
| 97 | tmax_slope_7d | Trend (Group 4) |
| 98 | wind_gusts_10m_max | Current weather |
| 99 | wind_speed_10m_max | Current weather |
| 100 | wind_speed_10m_max_lag1 | Lag (Group 2) |
| 101 | wind_speed_10m_max_lag2 | Lag (Group 2) |
| 102 | wind_speed_10m_max_lag3 | Lag (Group 2) |
| 103 | wind_speed_10m_max_lag7 | Lag (Group 2) |
| 104 | wind_speed_10m_max_roll3_max | Rolling (Group 3) |
| 105 | wind_speed_10m_max_roll3_mean | Rolling (Group 3) |
| 106 | wind_speed_10m_max_roll3_min | Rolling (Group 3) |
| 107 | wind_speed_10m_max_roll7_max | Rolling (Group 3) |
| 108 | wind_speed_10m_max_roll7_mean | Rolling (Group 3) |
| 109 | wind_speed_10m_max_roll7_min | Rolling (Group 3) |

### 3.2 Input shape

```
X : array-like of shape (n_samples, 110)
    dtype: float64
    No missing values (NaN) permitted
    No scaling required — Random Forest is scale-invariant
```

### 3.3 Feature construction requirements

All features must be constructed **using only data at time T or earlier**. See `docs/final_ml_dataset.md` and `results/phase7_feature_groups.json` for full construction rules.

Key rules:
- Lag features: `feature_lag_N = feature.shift(N)` with N ≥ 1 (T-N, not T+1)
- Rolling features: `feature.shift(1).rolling(N).mean/max/min()` — window [T-N, …, T-1]
- `tmax_departure_zscore`: 30-day trailing z-score, excludes current day T
- `qualifying_day`: binary derived from `temperature_2m_max` and `tmax_departure` at T
- `heatwave_lag1`: `heatwave(T-1)` — yesterday's heatwave label (historical, safe)
- `tmax_normal`: city-specific 31-day centred smoothed daily Tmax climatology (1990–2020 baseline)
- `tmax_departure`: `temperature_2m_max(T) − tmax_normal(city, DOY)`

### 3.4 City encoding

| city_key | city_encoded | is_coastal | latitude | longitude |
|---|---|---|---|---|
| delhi | 1 | 0 | 28.6139 | 77.2090 |
| lucknow | 3 | 0 | 26.8467 | 80.9462 |
| nagpur | 4 | 0 | 21.1458 | 79.0882 |
| ahmedabad | 0 | 0 | 23.0225 | 72.5714 |
| mumbai | 2 | 1 | 19.0760 | 72.8777 |

### 3.5 Season encoding

| season_code | Season | Months |
|---|---|---|
| 0 | Winter | Dec, Jan, Feb |
| 1 | Pre-Monsoon | Mar, Apr, May |
| 2 | Monsoon | Jun, Jul, Aug, Sep |
| 3 | Post-Monsoon | Oct, Nov |

---

## 4. Output Contract

### 4.1 Outputs

```python
# Probability output (float in [0.0, 1.0])
prediction_probability = model.predict_proba(X)[:, 1]

# Binary label (int 0 or 1)
prediction_label = (prediction_probability >= 0.70).astype(int)
```

| Output | Type | Description |
|---|---|---|
| `prediction_probability` | float, [0.0, 1.0] | Estimated probability that T+1 is a heatwave day |
| `prediction_label` | int, {0, 1} | `1` = heatwave predicted for T+1; `0` = normal predicted |

### 4.2 Decision threshold

```
threshold = 0.70
prediction_label = 1  if  prediction_probability >= 0.70
prediction_label = 0  if  prediction_probability <  0.70
```

This threshold was fixed during Phase 11 validation. **Do not change it without revalidating on a held-out dataset.**

### 4.3 Minimal usage example

```python
import joblib
import json
import pandas as pd
import numpy as np

# Load model and feature list
model = joblib.load("models/final/climateguard_final_model.joblib")
with open("models/final/feature_list.json") as f:
    feature_list = json.load(f)
feature_names = [f["name"] for f in feature_list]  # ordered list of 110 names

# Prepare input DataFrame — must have exactly these 110 columns in any column order
# (pandas will reindex to correct order)
X = df[feature_names]  # df is your input DataFrame

# Get probabilities and labels
probabilities = model.predict_proba(X.values)[:, 1]
labels        = (probabilities >= 0.70).astype(int)

# Output
results = pd.DataFrame({
    "date": df["date"],           # or your date column
    "city_key": df["city_key"],   # or your city column
    "prediction_probability": probabilities,
    "prediction_label": labels,
})
```

---

## 5. Supported Cities

| city_key | City | Heatwave positives in training | Notes |
|---|---|---|---|
| delhi | New Delhi | 195 (train+val) | Primary target city |
| lucknow | Lucknow | 125 (train+val) | Primary target city |
| nagpur | Nagpur | 115 (train+val) | Primary target city |
| ahmedabad | Ahmedabad | 32 (train only) | Very few positives; predictions may be unreliable |
| mumbai | Mumbai | 0 | No heatwave positives in any split — model may not generalise here |

**Warning:** The model was trained on five Indian cities. Predictions for other cities or regions are outside the training distribution and should be treated with extreme caution.

---

## 6. Known Limitations

1. **ERA5 reanalysis data, not station observations.** The model was trained on ERA5 (Open-Meteo API, 0.25° grid). Real-time inputs from other sources (IMD stations, other APIs) may have systematic differences.

2. **`qualifying_day` is tied to the target definition.** `qualifying_day` encodes the IMD-inspired threshold criteria (Tmax ≥ 40°C + departure ≥ 4.5°C for plains; Tmax ≥ 37°C for coastal). The model does not independently discover the IMD rule — it exploits an operationally designed feature. This is intentional but must be documented for interpretability.

3. **Heatwave definition is IMD-inspired, not official IMD.** The labels are based on ERA5 reanalysis data and an IMD-inspired definition. They are **not** official India Meteorological Department ground-truth labels.

4. **Mumbai and Ahmedabad:** Mumbai has zero heatwave positives in the entire dataset. Ahmedabad has 32 positives, all in the training window. Model predictions for these cities carry high uncertainty.

5. **Class imbalance.** Heatwaves are rare events (0.78% of days). The model is calibrated for detection (recall = 0.87), not precision (precision = 0.58). Approximately 2 in 5 alarms are false positives. This is appropriate for an early-warning system but must be communicated to users.

6. **Temporal coverage.** The model was trained on 1990–2022 data and tested on 2023–2025. It has not been validated beyond this window. Retraining is recommended if deploying significantly past 2025.

7. **No drift detection.** The model does not self-monitor for distribution shift. Monitoring for concept drift is the responsibility of the downstream system (Part 2 / Kshitij's scope).

8. **No SHAP / explainability built in.** Feature importance and SHAP values are the responsibility of Part 2 (Kshitij's scope). The model artifact itself does not include explanations.

---

## 7. Integration Notes for Part 2 (Kshitij) and Part 3 (Pradnesh)

- The model file is a standard scikit-learn `RandomForestClassifier`, serialized with `joblib`. No custom classes are needed to load it.
- Feature construction (lags, rolling, anomaly features) must be performed before passing data to the model. The feature engineering logic is in `feature_engineering.py` (Phase 7).
- The model expects a **NumPy array or pandas DataFrame** with shape `(n_samples, 110)`. Column order must match `feature_list.json`.
- No scaling or normalization is required. Random Forest is scale-invariant.
- The model exposes `.predict_proba()` — use column index `[:, 1]` for the positive-class probability.
- The threshold is fixed at **0.70**. Do not use `.predict()` directly — it applies sklearn's default threshold of 0.50, which was not validated for this use case.
- The `metadata.json` file contains all configuration details, feature names, and final test metrics in machine-readable form.

---

## 8. File Reference

| File | Purpose |
|---|---|
| `models/final/climateguard_final_model.joblib` | Trained model artifact — load with `joblib.load()` |
| `models/final/feature_list.json` | Ordered list of 110 feature names and dtypes |
| `models/final/metadata.json` | Full model configuration, parameters, metrics |
| `results/phase7_feature_groups.json` | Feature group registry and construction rules |
| `feature_engineering.py` | Phase 7 feature construction code |
| `docs/final_model_selection.md` | Full model selection rationale |
| `results/final_model_metrics.json` | Final test metrics and city breakdown |

# Phase 15 — Prediction Interface

**ClimateGuard: Indian Heatwave Prediction**  
**Phase:** 15  
**Date:** 2026-09-02  
**Status:** COMPLETE

---

## 1. Overview

The ClimateGuard prediction interface (`src/prediction/predictor.py`) wraps the Phase 14 final Random Forest model in a clean, importable Python class. It is the single authoritative point of access to the ML model for all downstream project components.

**What it does:**
- Loads the final trained model once per instance
- Validates all 110 input features strictly
- Returns `prediction_probability` and `prediction_label` for any city/date
- Exposes the raw model and feature list for SHAP / explainability

**What it does NOT do:**
- Retrain the model
- Change the threshold
- Fill in missing features
- Silently reorder unexpected columns
- Provide an API, Flask endpoint, or website

---

## 2. Files

| Path | Purpose |
|---|---|
| `src/prediction/__init__.py` | Package init; exports `ClimateGuardPredictor` |
| `src/prediction/predictor.py` | Main predictor class (496 lines) |
| `tests/test_prediction_interface.py` | 18 lightweight tests (458 lines) |
| `examples/predict_example.py` | Minimal working example with real data (157 lines) |
| `models/final/climateguard_final_model.joblib` | Locked final model (1.86 MB) |
| `models/final/feature_list.json` | Authoritative ordered feature list (110 entries) |
| `models/final/metadata.json` | Full model configuration and metrics |

---

## 3. Quick Start

```python
from src.prediction import ClimateGuardPredictor

# Load once — model is cached per instance
predictor = ClimateGuardPredictor()

# Single prediction (dict or 1-row DataFrame)
result = predictor.predict(features_dict)
print(result.prediction_probability)   # float in [0.0, 1.0]
print(result.prediction_label)         # 0 or 1

# Batch prediction
results_df = predictor.predict_batch(features_df)
# results_df has all original columns + prediction_probability + prediction_label

# Probability only (no threshold)
prob = predictor.predict_probability(features_dict)
```

---

## 4. Feature Count and Ordering

**Required:** exactly **110 features** in the **exact order** defined in `models/final/feature_list.json`.

The predictor rejects input with missing features, extra-only columns beyond the 110, or NaN values. It does **not** silently fill or reorder.

Load the canonical feature list programmatically:

```python
import json
with open("models/final/feature_list.json") as f:
    feature_list = json.load(f)
feature_names = [f["name"] for f in feature_list]   # ordered list of 110 names
```

---

## 5. Feature Names (All 110, in Model Order)

| Index | Feature | Group | Description |
|---|---|---|---|
| 0 | apparent_temperature_max | Current weather | Max apparent (feels-like) temperature at T (°C) |
| 1 | apparent_temperature_mean | Current weather | Mean apparent temperature at T (°C) |
| 2 | apparent_temperature_min | Current weather | Min apparent temperature at T (°C) |
| 3 | city_encoded | City | Integer city identifier (see encoding table) |
| 4 | day_of_year | Calendar | Day of year 1–366 |
| 5 | doy_cos | Calendar | Cosine encoding of day_of_year |
| 6 | doy_sin | Calendar | Sine encoding of day_of_year |
| 7 | et0_fao_evapotranspiration | Current weather | Reference evapotranspiration at T (mm/day) |
| 8 | heatwave_lag1 | Lag | Was yesterday (T-1) a heatwave day? Binary 0/1 |
| 9 | is_coastal | City | 1 = coastal city (Mumbai), 0 = plains |
| 10 | latitude | City | City latitude (decimal degrees) |
| 11 | longitude | City | City longitude (decimal degrees) |
| 12 | month | Calendar | Month 1–12 |
| 13 | month_cos | Calendar | Cosine encoding of month |
| 14 | month_sin | Calendar | Sine encoding of month |
| 15 | precipitation_sum | Current weather | Total precipitation at T (mm) |
| 16 | precipitation_sum_lag1 | Lag | Precipitation at T-1 (mm) |
| 17 | precipitation_sum_lag2 | Lag | Precipitation at T-2 (mm) |
| 18 | precipitation_sum_lag3 | Lag | Precipitation at T-3 (mm) |
| 19 | precipitation_sum_lag7 | Lag | Precipitation at T-7 (mm) |
| 20 | precipitation_sum_roll3_max | Rolling | Max daily precip over [T-3, T-1] (mm) |
| 21 | precipitation_sum_roll3_mean | Rolling | Mean daily precip over [T-3, T-1] (mm) |
| 22 | precipitation_sum_roll3_min | Rolling | Min daily precip over [T-3, T-1] (mm) |
| 23 | precipitation_sum_roll7_max | Rolling | Max daily precip over [T-7, T-1] (mm) |
| 24 | precipitation_sum_roll7_mean | Rolling | Mean daily precip over [T-7, T-1] (mm) |
| 25 | precipitation_sum_roll7_min | Rolling | Min daily precip over [T-7, T-1] (mm) |
| 26 | qualifying_day | Current weather | IMD-inspired binary threshold flag at T (0/1) |
| 27 | relative_humidity_2m_max | Current weather | Max 2m relative humidity at T (%) |
| 28 | relative_humidity_2m_mean | Current weather | Mean 2m relative humidity at T (%) |
| 29 | relative_humidity_2m_mean_lag1 | Lag | Mean RH at T-1 (%) |
| 30 | relative_humidity_2m_mean_lag2 | Lag | Mean RH at T-2 (%) |
| 31 | relative_humidity_2m_mean_lag3 | Lag | Mean RH at T-3 (%) |
| 32 | relative_humidity_2m_mean_lag7 | Lag | Mean RH at T-7 (%) |
| 33 | relative_humidity_2m_mean_roll3_max | Rolling | Max mean RH over [T-3, T-1] (%) |
| 34 | relative_humidity_2m_mean_roll3_mean | Rolling | Mean mean RH over [T-3, T-1] (%) |
| 35 | relative_humidity_2m_mean_roll3_min | Rolling | Min mean RH over [T-3, T-1] (%) |
| 36 | relative_humidity_2m_mean_roll7_max | Rolling | Max mean RH over [T-7, T-1] (%) |
| 37 | relative_humidity_2m_mean_roll7_mean | Rolling | Mean mean RH over [T-7, T-1] (%) |
| 38 | relative_humidity_2m_mean_roll7_min | Rolling | Min mean RH over [T-7, T-1] (%) |
| 39 | relative_humidity_2m_min | Current weather | Min 2m relative humidity at T (%) |
| 40 | season_code | Calendar | Season integer (0=Winter, 1=Pre-Monsoon, 2=Monsoon, 3=Post-Monsoon) |
| 41 | shortwave_radiation_sum | Current weather | Daily shortwave radiation at T (MJ/m²) |
| 42 | surface_pressure_mean | Current weather | Mean surface pressure at T (hPa) |
| 43 | surface_pressure_mean_lag1 | Lag | Mean surface pressure at T-1 (hPa) |
| 44 | surface_pressure_mean_lag2 | Lag | Mean surface pressure at T-2 (hPa) |
| 45 | surface_pressure_mean_lag3 | Lag | Mean surface pressure at T-3 (hPa) |
| 46 | surface_pressure_mean_lag7 | Lag | Mean surface pressure at T-7 (hPa) |
| 47 | surface_pressure_mean_roll3_max | Rolling | Max mean pressure over [T-3, T-1] (hPa) |
| 48 | surface_pressure_mean_roll3_mean | Rolling | Mean mean pressure over [T-3, T-1] (hPa) |
| 49 | surface_pressure_mean_roll3_min | Rolling | Min mean pressure over [T-3, T-1] (hPa) |
| 50 | surface_pressure_mean_roll7_max | Rolling | Max mean pressure over [T-7, T-1] (hPa) |
| 51 | surface_pressure_mean_roll7_mean | Rolling | Mean mean pressure over [T-7, T-1] (hPa) |
| 52 | surface_pressure_mean_roll7_min | Rolling | Min mean pressure over [T-7, T-1] (hPa) |
| 53 | temperature_2m_max | Current weather | Max 2m temperature at T (°C) |
| 54 | temperature_2m_max_lag1 | Lag | Max 2m temperature at T-1 (°C) |
| 55 | temperature_2m_max_lag2 | Lag | Max 2m temperature at T-2 (°C) |
| 56 | temperature_2m_max_lag3 | Lag | Max 2m temperature at T-3 (°C) |
| 57 | temperature_2m_max_lag7 | Lag | Max 2m temperature at T-7 (°C) |
| 58 | temperature_2m_max_roll3_max | Rolling | Max daily Tmax over [T-3, T-1] (°C) |
| 59 | temperature_2m_max_roll3_mean | Rolling | Mean daily Tmax over [T-3, T-1] (°C) |
| 60 | temperature_2m_max_roll3_min | Rolling | Min daily Tmax over [T-3, T-1] (°C) |
| 61 | temperature_2m_max_roll7_max | Rolling | Max daily Tmax over [T-7, T-1] (°C) |
| 62 | temperature_2m_max_roll7_mean | Rolling | Mean daily Tmax over [T-7, T-1] (°C) |
| 63 | temperature_2m_max_roll7_min | Rolling | Min daily Tmax over [T-7, T-1] (°C) |
| 64 | temperature_2m_mean | Current weather | Mean 2m temperature at T (°C) |
| 65 | temperature_2m_mean_lag1 | Lag | Mean 2m temperature at T-1 (°C) |
| 66 | temperature_2m_mean_lag2 | Lag | Mean 2m temperature at T-2 (°C) |
| 67 | temperature_2m_mean_lag3 | Lag | Mean 2m temperature at T-3 (°C) |
| 68 | temperature_2m_mean_lag7 | Lag | Mean 2m temperature at T-7 (°C) |
| 69 | temperature_2m_mean_roll3_max | Rolling | Max daily Tmean over [T-3, T-1] (°C) |
| 70 | temperature_2m_mean_roll3_mean | Rolling | Mean daily Tmean over [T-3, T-1] (°C) |
| 71 | temperature_2m_mean_roll3_min | Rolling | Min daily Tmean over [T-3, T-1] (°C) |
| 72 | temperature_2m_mean_roll7_max | Rolling | Max daily Tmean over [T-7, T-1] (°C) |
| 73 | temperature_2m_mean_roll7_mean | Rolling | Mean daily Tmean over [T-7, T-1] (°C) |
| 74 | temperature_2m_mean_roll7_min | Rolling | Min daily Tmean over [T-7, T-1] (°C) |
| 75 | temperature_2m_min | Current weather | Min 2m temperature at T (°C) |
| 76 | temperature_2m_min_lag1 | Lag | Min 2m temperature at T-1 (°C) |
| 77 | temperature_2m_min_lag2 | Lag | Min 2m temperature at T-2 (°C) |
| 78 | temperature_2m_min_lag3 | Lag | Min 2m temperature at T-3 (°C) |
| 79 | temperature_2m_min_lag7 | Lag | Min 2m temperature at T-7 (°C) |
| 80 | temperature_2m_min_roll3_max | Rolling | Max daily Tmin over [T-3, T-1] (°C) |
| 81 | temperature_2m_min_roll3_mean | Rolling | Mean daily Tmin over [T-3, T-1] (°C) |
| 82 | temperature_2m_min_roll3_min | Rolling | Min daily Tmin over [T-3, T-1] (°C) |
| 83 | temperature_2m_min_roll7_max | Rolling | Max daily Tmin over [T-7, T-1] (°C) |
| 84 | temperature_2m_min_roll7_mean | Rolling | Mean daily Tmin over [T-7, T-1] (°C) |
| 85 | temperature_2m_min_roll7_min | Rolling | Min daily Tmin over [T-7, T-1] (°C) |
| 86 | tmax_delta_1d | Trend | temperature_2m_max(T) − temperature_2m_max(T-1) (°C) |
| 87 | tmax_delta_3d | Trend | temperature_2m_max(T) − temperature_2m_max(T-3) (°C) |
| 88 | tmax_delta_7d | Trend | temperature_2m_max(T) − temperature_2m_max(T-7) (°C) |
| 89 | tmax_departure | Current weather | temperature_2m_max(T) − tmax_normal(city, DOY) (°C) |
| 90 | tmax_departure_lag1 | Lag | tmax_departure at T-1 (°C) |
| 91 | tmax_departure_lag2 | Lag | tmax_departure at T-2 (°C) |
| 92 | tmax_departure_lag3 | Lag | tmax_departure at T-3 (°C) |
| 93 | tmax_departure_lag7 | Lag | tmax_departure at T-7 (°C) |
| 94 | tmax_departure_zscore | Anomaly | 30-day trailing z-score of tmax_departure (unitless) |
| 95 | tmax_normal | Current weather | City/DOY climatological daily Tmax normal (°C) |
| 96 | tmax_slope_3d | Trend | Linear slope of Tmax over [T-2, T-1, T] (°C/day) |
| 97 | tmax_slope_7d | Trend | Linear slope of Tmax over [T-6, ..., T] (°C/day) |
| 98 | wind_gusts_10m_max | Current weather | Max wind gusts at 10m at T (km/h) |
| 99 | wind_speed_10m_max | Current weather | Max wind speed at 10m at T (km/h) |
| 100 | wind_speed_10m_max_lag1 | Lag | Max wind speed at T-1 (km/h) |
| 101 | wind_speed_10m_max_lag2 | Lag | Max wind speed at T-2 (km/h) |
| 102 | wind_speed_10m_max_lag3 | Lag | Max wind speed at T-3 (km/h) |
| 103 | wind_speed_10m_max_lag7 | Lag | Max wind speed at T-7 (km/h) |
| 104 | wind_speed_10m_max_roll3_max | Rolling | Max wind speed over [T-3, T-1] (km/h) |
| 105 | wind_speed_10m_max_roll3_mean | Rolling | Mean wind speed over [T-3, T-1] (km/h) |
| 106 | wind_speed_10m_max_roll3_min | Rolling | Min wind speed over [T-3, T-1] (km/h) |
| 107 | wind_speed_10m_max_roll7_max | Rolling | Max wind speed over [T-7, T-1] (km/h) |
| 108 | wind_speed_10m_max_roll7_mean | Rolling | Mean wind speed over [T-7, T-1] (km/h) |
| 109 | wind_speed_10m_max_roll7_min | Rolling | Min wind speed over [T-7, T-1] (km/h) |

---

## 6. City Encoding

| city_key | city_encoded | is_coastal | latitude | longitude |
|---|---|---|---|---|
| ahmedabad | 0 | 0 | 23.0225 | 72.5714 |
| delhi | 1 | 0 | 28.6139 | 77.2090 |
| mumbai | 2 | 1 | 19.0760 | 72.8777 |
| lucknow | 3 | 0 | 26.8467 | 80.9462 |
| nagpur | 4 | 0 | 21.1458 | 79.0882 |

---

## 7. Calendar / Season Encoding

| season_code | Season | Months |
|---|---|---|
| 0 | Winter | December, January, February |
| 1 | Pre-Monsoon | March, April, May |
| 2 | Monsoon | June, July, August, September |
| 3 | Post-Monsoon | October, November |

Cyclic encodings:
- `month_sin = sin(2π × month / 12)`, `month_cos = cos(2π × month / 12)`
- `doy_sin = sin(2π × day_of_year / 365)`, `doy_cos = cos(2π × day_of_year / 365)`

---

## 8. Temporal Feature Construction Rules

All temporal features must be constructed using only data at time T or earlier:

| Group | Construction rule | Window |
|---|---|---|
| Lag N | `feature.shift(N)` | Single day T-N |
| Rolling 3d | `feature.shift(1).rolling(3).agg()` | [T-3, T-2, T-1] |
| Rolling 7d | `feature.shift(1).rolling(7).agg()` | [T-7, …, T-1] |
| Anomaly zscore | 30-day trailing rolling mean/std with `shift(1)` | [T-30, …, T-1] |
| Trend delta | `feature(T) - feature(T-N)` | T and T-N |
| Trend slope | Linear regression over N days ending at T | [T-N+1, …, T] |

The `shift(1)` ensures the current day T is **excluded** from all rolling/lag windows. `qualifying_day` and all Group 1 features use current-day T values (leakage-safe — they do not expose tomorrow's label).

---

## 9. Input / Output Contract

### Input

```
X : dict, pd.Series, or pd.DataFrame
    - Must contain all 110 features from feature_list.json
    - No NaN values permitted
    - All numeric dtypes
    - May optionally contain: city, city_key, date (metadata — not passed to model)
```

### Output — single prediction (`predict()`)

```python
PredictionResult
  .prediction_probability : float in [0.0, 1.0]
  .prediction_label       : int 0 or 1
  .city                   : str or None
  .date                   : str or None
  .threshold              : 0.70
```

### Output — batch prediction (`predict_batch()`)

```
pd.DataFrame
  All original columns preserved
  + prediction_probability : float column
  + prediction_label       : int column (0 or 1)
```

---

## 10. Threshold

**Fixed at 0.70.** Established in Phase 11 on the validation split (2020–2022). Never changed.

```
prediction_label = 1  if  prediction_probability >= 0.70
prediction_label = 0  if  prediction_probability <  0.70
```

Do **not** use `model.predict(X)` directly — sklearn's default threshold is 0.50, which was not validated for this use case.

---

## 11. Error Handling

| Situation | Error raised | Message |
|---|---|---|
| Model file missing | `FileNotFoundError` | Path + has Phase 14 been completed? |
| Feature list missing | `FileNotFoundError` | Path + has Phase 14 been completed? |
| Model load failure | `RuntimeError` | Underlying exception message |
| Feature count wrong | `ValueError` | Expected 110, got N |
| Required feature missing | `ValueError` | Lists missing feature names |
| NaN in any feature | `ValueError` | Lists affected features |
| Non-numeric feature | `ValueError` | Lists affected features |
| Wrong input type | `TypeError` | Expected dict/Series/DataFrame |
| Multi-row input to predict() | `ValueError` | Use predict_batch() |
| Empty DataFrame to predict_batch() | `ValueError` | Empty DataFrame message |

---

## 12. Validation Tests

All 18 tests pass. Run from project root:

```
python tests/test_prediction_interface.py
```

| Test | What it verifies |
|---|---|
| test_model_loads | Predictor constructs and model attribute is set |
| test_feature_list_loads | feature_names populated, all strings |
| test_exactly_110_features | n_features=110, len(feature_names)=110, model.n_features_in_=110 |
| test_valid_sample_prediction | Real data row returns PredictionResult |
| test_probability_in_range | Probability in [0, 1] |
| test_prediction_binary | All labels 0 or 1 (10 rows) |
| test_threshold_applied | label = (prob >= 0.70) for 20 rows |
| test_missing_feature_raises | ValueError on missing column |
| test_nan_input_raises | ValueError on NaN value |
| test_batch_prediction | Returns DataFrame with 2 new columns, original preserved |
| test_input_not_modified | Original DataFrame unchanged after predict_batch() |
| test_feature_ordering | feature_names matches feature_list.json order exactly |
| test_target_not_in_features | heatwave_next_day absent from feature list |
| test_wrong_type_raises | TypeError on list/dict to predict_batch() |
| test_info_method | info() returns all required keys |
| test_get_feature_matrix | Returns (n, 110) DataFrame in model order |
| test_predict_single_row_only | ValueError on multi-row input to predict() |
| test_empty_batch_raises | ValueError on empty DataFrame |

---

## 13. Limitations

1. The model was trained on five Indian cities (Delhi, Lucknow, Nagpur, Ahmedabad, Mumbai) using ERA5 reanalysis data. It should not be applied to other cities without retraining.
2. Mumbai has zero heatwave positives — predictions for Mumbai carry high uncertainty.
3. The model precision is 0.58 on the test set — approximately 42% of raised alarms are false positives. This is intentional for an early-warning system.
4. All 110 features must be pre-computed before calling the predictor. Feature engineering logic is in `feature_engineering.py` (Phase 7).
5. No scaling is required (Random Forest is scale-invariant), but NaN values must be resolved before calling.

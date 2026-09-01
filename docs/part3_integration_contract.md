# Part 3 Integration Contract

**ClimateGuard: Indian Heatwave Prediction**  
**From:** Adrian (Part 1 — Dataset & ML)  
**To:** Pradnesh (Part 3 — Expert Module, ETL, Integration)  
**Date:** 2026-09-02  
**Status:** COMPLETE — ready for Part 3 consumption

---

## 1. What Part 1 Delivers to Part 3

Part 3 (ETL and Integration) is responsible for providing real weather data to the model and forwarding predictions to the backend or downstream systems. Part 1 delivers:

| Deliverable | Path | Description |
|---|---|---|
| Final model | `models/final/climateguard_final_model.joblib` | Trained final model (1.86 MB) |
| Feature list | `models/final/feature_list.json` | Ordered list of 110 required features |
| Metadata | `models/final/metadata.json` | Full model config, parameters, test metrics |
| Prediction interface | `src/prediction/predictor.py` | `ClimateGuardPredictor` importable class |
| Example script | `examples/predict_example.py` | Working call pattern with real data |
| Feature contract | `docs/final_model_contract.md` | Complete feature specification |
| Interface docs | `docs/prediction_interface.md` | Full feature table, encoding rules |

---

## 2. How to Call the Prediction Interface

The predictor is a standard Python class. It does **not** require the Kiro CLI, a web server, or any special runtime.

### 2.1 Install requirements

The predictor depends only on:
- `scikit-learn` (for the model)
- `joblib` (model loading)
- `pandas` (DataFrame handling)
- `numpy` (array operations)

All are standard data-science packages already used in the project.

### 2.2 Import pattern

```python
# Add project root to sys.path if calling from outside the project
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/path/to/climate guard")))

from src.prediction import ClimateGuardPredictor

predictor = ClimateGuardPredictor()   # loads model once
```

### 2.3 Single-row prediction (one city, one day)

```python
# features: dict or 1-row DataFrame with all 110 feature keys
result = predictor.predict(features)

print(result.prediction_probability)   # e.g. 0.8342
print(result.prediction_label)         # 1 (heatwave tomorrow) or 0 (normal)
print(result.city)                     # "delhi" (if provided)
print(result.date)                     # "2024-05-15" (if provided)
print(result.threshold)                # 0.70
```

### 2.4 Batch prediction (multiple cities/days at once)

```python
import pandas as pd

# features_df: DataFrame with one row per city-day
# must contain all 110 feature columns + optionally city_key, date
results_df = predictor.predict_batch(features_df)

# results_df is a NEW DataFrame (original not modified) with:
#   - all original columns
#   - prediction_probability (float)
#   - prediction_label (0 or 1)
```

### 2.5 Full example pipeline

```python
from pathlib import Path
import sys
import pandas as pd

# Setup
PROJECT_ROOT = Path("/absolute/path/to/climate guard")
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import ClimateGuardPredictor

# 1. Load predictor (once per process / session)
predictor = ClimateGuardPredictor()

# 2. Get today's feature data from your ETL pipeline
#    (must produce all 110 features — see feature contract below)
today_features = your_etl_pipeline.get_features(city="delhi", date="2025-06-01")

# 3. Predict
result = predictor.predict(today_features)

# 4. Forward result downstream
downstream_system.send(
    city=result.city,
    date=result.date,
    probability=result.prediction_probability,
    alert=result.prediction_label == 1,
)
```

---

## 3. What Part 3 Must Provide (Input Requirements)

Part 3's ETL pipeline must produce **exactly 110 features** per city per day, in the exact order specified in `models/final/feature_list.json`.

### 3.1 Feature count

```
Exactly 110 features. No more, no fewer.
```

### 3.2 Feature order

The features must match the order in `feature_list.json`. Load the canonical order:

```python
import json
with open("models/final/feature_list.json") as f:
    feature_list = json.load(f)
feature_names = [f["name"] for f in feature_list]   # 110 ordered names
```

### 3.3 Data types

All features must be numeric (float64 or int64). No strings, no NaN values.

### 3.4 Feature groups and construction

| Group | Features | How constructed |
|---|---|---|
| Current weather (18) | Raw ERA5 variables + qualifying_day | Provided directly from weather API |
| Lag (33) | T-1, T-2, T-3, T-7 lags | Must store at least 7 days of history per city |
| Rolling (42) | 3-day and 7-day rolling stats | Computed from the last 7 days of history |
| Trend (5) | Tmax delta and slope | Computed from current and past Tmax |
| Anomaly (1) | tmax_departure_zscore | 30-day trailing z-score — needs 30+ days of history |
| Calendar (7) | Month, DOY, season, cyclic | Derived from the date |
| City (4) | Encoding, coastal flag, lat/lon | Static lookup per city |

The complete feature engineering logic is in `feature_engineering.py` (Phase 7).

### 3.5 Critical: qualifying_day

`qualifying_day` (feature index 26) must be computed as:

```
Plains cities (Delhi, Lucknow, Nagpur, Ahmedabad):
  qualifying_day = 1  if  temperature_2m_max >= 40°C  AND  tmax_departure >= 4.5°C
               OR  if  temperature_2m_max >= 45°C (absolute override)
               else 0

Coastal cities (Mumbai):
  qualifying_day = 1  if  temperature_2m_max >= 37°C  AND  tmax_departure >= 4.5°C
               else 0
```

Where `tmax_departure = temperature_2m_max(T) − tmax_normal(city, DOY)`.

`tmax_normal` is the city-specific 31-day centred smoothed daily Tmax climatology, computed from the 1990–2020 baseline. The lookup table is embedded in the project's feature engineering pipeline.

### 3.6 City encoding

| city_key | city_encoded | is_coastal | latitude | longitude |
|---|---|---|---|---|
| ahmedabad | 0 | 0 | 23.0225 | 72.5714 |
| delhi | 1 | 0 | 28.6139 | 77.2090 |
| mumbai | 2 | 1 | 19.0760 | 72.8777 |
| lucknow | 3 | 0 | 26.8467 | 80.9462 |
| nagpur | 4 | 0 | 21.1458 | 79.0882 |

---

## 4. Output That Part 3 Receives

| Field | Type | Meaning |
|---|---|---|
| `prediction_probability` | float [0.0, 1.0] | Estimated probability heatwave occurs tomorrow |
| `prediction_label` | int 0 or 1 | **1 = heatwave alert for tomorrow**, 0 = no alert |
| `city` | str or None | City identifier (from input if provided) |
| `date` | str or None | Date of day T (prediction is for T+1) |
| `threshold` | float | Always 0.70 |

---

## 5. Minimal Working Integration

```python
"""
Minimal Part 3 integration example.
"""
import json
import sys
from pathlib import Path
import pandas as pd

# Point to project root
PROJECT_ROOT = Path("/path/to/climate guard")
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import ClimateGuardPredictor

def run_daily_prediction(city_key: str, date_str: str, weather_data: dict) -> dict:
    """
    Run a single daily heatwave prediction.

    Parameters
    ----------
    city_key    : "delhi", "lucknow", "nagpur", "ahmedabad", or "mumbai"
    date_str    : ISO date string for day T, e.g. "2025-06-01"
    weather_data: dict containing all 110 features (floats, no NaN)

    Returns
    -------
    dict with prediction_probability, prediction_label, city, date
    """
    # Add metadata to input (these are NOT passed to the model)
    weather_data["city_key"] = city_key
    weather_data["date"] = date_str

    predictor = ClimateGuardPredictor()    # In production: initialise once, reuse
    result = predictor.predict(weather_data)
    return result.to_dict()


# Example call
if __name__ == "__main__":
    # Load feature names in required order
    with open(PROJECT_ROOT / "models/final/feature_list.json") as f:
        fl = json.load(f)
    feature_names = [f["name"] for f in fl]

    # Simulate a feature dict (replace with real ETL output)
    sample = pd.read_csv(PROJECT_ROOT / "data/splits/temporal/X_test.csv", nrows=1)
    weather_data = sample[feature_names].iloc[0].to_dict()

    output = run_daily_prediction("delhi", "2025-06-01", weather_data)
    print(output)
    # {"prediction_probability": 0.0, "prediction_label": 0,
    #  "threshold": 0.70, "city": "delhi", "date": "2025-06-01"}
```

---

## 6. Error Handling

The predictor raises clear exceptions for all invalid inputs. Part 3 should handle:

| Exception | Cause | Action |
|---|---|---|
| `FileNotFoundError` | Model or feature list missing | Verify Phase 14 artifacts exist |
| `ValueError: Missing N required feature(s)` | ETL did not produce all 110 features | Check ETL pipeline |
| `ValueError: NaN values found` | Missing upstream data | Fill or skip before calling predictor |
| `ValueError: Non-numeric values` | String column passed | Cast to float before calling |
| `TypeError` | Wrong input type | Use dict or pd.DataFrame |

---

## 7. Performance Notes

- Load the predictor **once per process** (e.g., at application startup). Do not re-instantiate per prediction — model loading is the expensive step.
- Batch predictions via `predict_batch(df)` are more efficient than calling `predict()` in a loop for large volumes.
- The model itself is fast (Random Forest, `n_estimators=300`) — individual predictions complete in milliseconds.

---

## 8. What Part 3 Must NOT Do

- **Do not retrain the model.** `models/final/climateguard_final_model.joblib` is locked.
- **Do not silently fill missing features with zeros.** The predictor will raise a `ValueError` — fix the ETL instead.
- **Do not change the threshold.** 0.70 is fixed and validated.
- **Do not modify `feature_list.json` or `metadata.json`.**
- **Do not modify any Phase 1–14 datasets.** See `PROJECT_MEMORY.md`.

---

## 9. Limitations to Carry Forward

1. ERA5 reanalysis data — not real-time station observations. If Part 3 uses a live weather API, check for systematic biases vs ERA5.
2. Mumbai has zero heatwave positives in training — predictions for Mumbai are unreliable.
3. The model precision is 0.58 (~42% of alarms are false positives). This is expected for an early-warning system.
4. The model was trained on 1990–2022 data and tested on 2023–2025. Periodic retraining is recommended.
5. The operational label is IMD-inspired (ERA5-based), not official IMD ground truth.

---

## 10. Quick Reference

```
Model          : models/final/climateguard_final_model.joblib
Features       : 110, ordered per models/final/feature_list.json
Threshold      : 0.70
Import         : from src.prediction import ClimateGuardPredictor
Single predict : predictor.predict(features_dict)  → PredictionResult
Batch predict  : predictor.predict_batch(df)        → pd.DataFrame
Probability    : predictor.predict_probability(x)   → float or np.ndarray
Model access   : predictor.model                    → RandomForestClassifier
Feature names  : predictor.feature_names            → list[str] of 110
```

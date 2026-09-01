# Part 2 Integration Contract

**ClimateGuard: Indian Heatwave Prediction**  
**From:** Adrian (Part 1 — Dataset & ML)  
**To:** Kshitij (Part 2 — Risk, Adaptation, Explainability)  
**Date:** 2026-09-02  
**Status:** COMPLETE — ready for Part 2 consumption

---

## 1. What Part 1 Delivers to Part 2

Part 1 delivers a trained, validated, production-ready prediction model with a clean Python interface. The following are available for immediate use:

| Deliverable | Path | Description |
|---|---|---|
| Final model | `models/final/climateguard_final_model.joblib` | Trained RandomForestClassifier, Phase 14 |
| Feature list | `models/final/feature_list.json` | Ordered list of 110 required feature names |
| Metadata | `models/final/metadata.json` | Full config, parameters, and test metrics |
| Prediction interface | `src/prediction/predictor.py` | `ClimateGuardPredictor` class |
| Interface package | `src/prediction/__init__.py` | Import entrypoint |
| Example | `examples/predict_example.py` | Working usage example with real data |
| Prediction contract | `docs/final_model_contract.md` | Complete input/output specification |
| Selection rationale | `docs/final_model_selection.md` | Why this model was chosen |
| Interface documentation | `docs/prediction_interface.md` | Full feature table, encoding, usage |

---

## 2. How to Import and Call the Predictor

### 2.1 Basic import

```python
from src.prediction import ClimateGuardPredictor

predictor = ClimateGuardPredictor()   # loads model + feature list once
```

### 2.2 Single prediction

```python
# features_dict: must contain all 110 keys from feature_list.json
result = predictor.predict(features_dict)

print(result.prediction_probability)   # float in [0.0, 1.0]
print(result.prediction_label)         # 0 or 1
print(result.city)                     # "delhi" (if provided in input)
print(result.date)                     # "2024-05-15" (if provided in input)
print(result.threshold)                # 0.70
```

### 2.3 Batch prediction

```python
# features_df: pd.DataFrame with all 110 feature columns
# may also include city, city_key, date columns — they are passed through, not fed to model
results_df = predictor.predict_batch(features_df)

# results_df has all original columns plus:
# - prediction_probability (float)
# - prediction_label (0 or 1)
```

### 2.4 Probability only (for custom thresholding or risk scoring)

```python
prob = predictor.predict_probability(features_dict)    # float for single row
probs = predictor.predict_probability(features_df)    # np.ndarray for multiple rows
```

---

## 3. What Part 2 Receives Per Prediction

| Field | Type | Description |
|---|---|---|
| `prediction_probability` | float [0.0, 1.0] | P(heatwave tomorrow given today's weather) |
| `prediction_label` | int {0, 1} | 1 = heatwave predicted for T+1 |
| `city` | str or None | City identifier if present in input |
| `date` | str or None | Date string for day T; prediction is for T+1 |
| `threshold` | float | Always 0.70 |

---

## 4. Explainability Access (SHAP)

Part 2 will need direct access to the model and feature matrix for SHAP. Both are cleanly exposed:

### 4.1 Access the underlying model

```python
predictor = ClimateGuardPredictor()
model = predictor.model                   # sklearn RandomForestClassifier
feature_names = predictor.feature_names   # list of 110 names in model order
```

### 4.2 Get the validated feature matrix

```python
# Returns a DataFrame of shape (n_rows, 110) in exact model column order
X_df = predictor.get_feature_matrix(features_df)

# Now safe to pass to SHAP
import shap
explainer = shap.TreeExplainer(predictor.model)
shap_values = explainer.shap_values(X_df)
```

### 4.3 Alternative: load model directly

```python
import joblib
import json

model = joblib.load("models/final/climateguard_final_model.joblib")

with open("models/final/feature_list.json") as f:
    feature_list = json.load(f)
feature_names = [f["name"] for f in feature_list]
```

---

## 5. Feature Column Requirement

The predictor requires **exactly 110 features** in the exact order from `feature_list.json`. The feature names are:

```python
predictor = ClimateGuardPredictor()
print(predictor.feature_names)   # ordered list of 110 names
```

The complete table with descriptions is in `docs/prediction_interface.md` Section 5.

All features must be float64-compatible. No NaN values are permitted. No scaling required.

---

## 6. Threshold

The decision threshold is **0.70** (locked from Phase 11 validation).

```
prediction_label = 1  if  prediction_probability >= 0.70
prediction_label = 0  if  prediction_probability <  0.70
```

If Part 2 requires a different threshold for risk-scoring purposes, use `predict_probability()` to get raw probabilities and apply your own threshold. **Do not change the Phase 14 model or feature list.**

---

## 7. Path Resolution

The predictor resolves paths relative to the project root automatically:

```python
# Works from any location within the project
from src.prediction import ClimateGuardPredictor
predictor = ClimateGuardPredictor()   # no path arguments needed
```

For explicit paths (e.g., in a different working directory):

```python
from pathlib import Path
predictor = ClimateGuardPredictor(
    model_path=Path("/absolute/path/to/models/final/climateguard_final_model.joblib"),
    feature_list_path=Path("/absolute/path/to/models/final/feature_list.json"),
)
```

---

## 8. Model Information

```python
predictor.info()
# Returns dict with: model_type, n_features, feature_names, threshold,
# target, model_path, feature_list_path, model_params
```

---

## 9. What Part 2 Must NOT Do

- **Do not retrain the model.** The artifact at `models/final/climateguard_final_model.joblib` is locked.
- **Do not change the threshold.** 0.70 was fixed on validation. If a different threshold is needed for risk scoring, apply it downstream using raw probabilities.
- **Do not modify feature_list.json.** This is the authoritative feature definition.
- **Do not modify any Phase 1–14 datasets or artifacts.** See `PROJECT_MEMORY.md` for the complete list of validated, read-only artifacts.

---

## 10. Scientific Limitations (Must Document in Part 2)

Part 2 must carry forward and document these limitations in any risk score or visualisation:

1. The model uses ERA5 reanalysis data — not official IMD station observations.
2. The heatwave label is IMD-inspired, not official IMD ground truth.
3. `qualifying_day` is correlated with the target by construction (same IMD threshold family). The model does not independently discover the IMD rule.
4. Mumbai has **zero** heatwave positives — predictions for Mumbai carry high uncertainty and should be flagged.
5. Ahmedabad has only 32 historical positives, all in 1990–2019. Test-set performance is not available.
6. Model precision is 0.58 on the test set — ~42% of alarms are false positives. This is intentional for an early-warning system.
7. No drift detection is built into the model. Part 2 is responsible for monitoring distribution shift.

---

## 11. Test Reference

Phase 15 interface tests: **18/18 PASS**  
Real-data smoke test: **5/5 rows correct** (Ahmedabad, January 2023 — all normal days)  
Full test suite: `python tests/test_prediction_interface.py`

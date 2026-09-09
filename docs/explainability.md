# Model Explainability

**ClimateGuard — Part 2**  
**Module:** `src/explainability/explainer.py`  
**Owner:** Kshitij (Part 2 — Risk, Adaptation, Explainability)  
**Status:** Complete  
**Date:** 2026-09-10

---

## 1. Overview

The Explainability module provides per-prediction feature importance explanations for the Part 1 Random Forest model.  It does NOT create a second model, retrain the model, or alter any model artifact.

---

## 2. Explanation Methods

### 2.1 SHAP (preferred)

Requires the `shap` package.  Install with:

```bash
pip install shap
```

When SHAP is available, `ClimateGuardExplainer` uses `shap.TreeExplainer` to compute per-prediction SHAP values for the positive class (heatwave prediction).

SHAP values represent each feature's marginal contribution to the model's predicted probability for that specific input row.

Output `direction` values:
- `"increases_risk"` — positive SHAP value; feature pushed the prediction toward heatwave
- `"decreases_risk"` — negative SHAP value; feature pushed the prediction away from heatwave
- `"neutral"` — SHAP value is exactly zero

### 2.2 Global RF Feature Importance (fallback)

When SHAP is not installed, the module falls back to the Random Forest model's built-in `feature_importances_` array (mean-decrease-in-impurity importance computed during training).

**⚠️ CRITICAL DISTINCTION: Global feature importance is NOT the same as SHAP values.**

| Property | SHAP | Global RF Importance |
|---|---|---|
| Scope | Per-prediction (local) | All training data (global) |
| Direction | Positive / negative | Not available (direction = `"global_importance"`) |
| Same for every row? | No — varies by input | Yes — same for all inputs |
| Explains this prediction? | Yes | No — explains the model overall |
| Method label | `"shap"` | `"global_rf_importance"` |

The fallback is clearly labelled in all outputs.  No output incorrectly labels global importance as SHAP.

---

## 3. Scientific Language

The explainability output reflects **model behaviour**, not physical causality.

All explanation outputs include these mandatory notes:

**Causality note:**
> "Explanation reflects model behaviour only.  Feature contributions indicate what drove this model prediction — they do NOT imply causality or prove that these features caused a heatwave."

**qualifying_day note:**
> "qualifying_day encodes the same IMD-inspired threshold criteria used to construct the heatwave label (Tmax >= 40 °C + departure >= 4.5 °C for plains; Tmax >= 37 °C for coastal cities).  A high contribution from qualifying_day reflects this feature engineering choice, NOT an independently discovered physical signal."

---

## 4. qualifying_day Limitation

`qualifying_day` is a binary feature derived from the same operational criteria used to construct the heatwave label:

```
qualifying_day = 1  if:
    (temperature_2m_max >= 40.0°C AND tmax_departure >= 4.5°C)   [plains]
    OR temperature_2m_max >= 45.0°C                               [absolute override]

qualifying_day = 1  if:
    temperature_2m_max >= 37.0°C AND tmax_departure >= 4.5°C      [coastal]
```

When `qualifying_day=1`, heatwave conditions are already very likely by construction.  This feature will almost always appear in the top contributors for high-probability predictions.

This should NOT be described as the model independently discovering the IMD rule.  It is an engineered feature that by definition correlates strongly with the target.

---

## 5. Input / Output

### Input to `explain()`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `features` | pd.DataFrame or dict | Yes | Single feature row (110 columns) |
| `prediction` | int | No | Part 1 prediction label; computed if omitted |
| `probability` | float | No | Part 1 probability; computed if omitted |
| `top_n` | int | No | Number of top features to return (default 10) |

### Output: `ExplainabilityResult`

| Attribute | Type | Description |
|---|---|---|
| `prediction` | int | Binary prediction label |
| `probability` | float | Heatwave probability |
| `method` | str | `"shap"` or `"global_rf_importance"` |
| `top_features` | list of dict | Top N features by absolute contribution |
| `n_top` | int | Number of features returned |
| `shap_available` | bool | Whether SHAP was available |
| `shap_base_value` | float or None | SHAP expected-value baseline (SHAP mode only) |
| `qualifying_day_note` | str | Mandatory qualifying_day disclaimer |
| `causality_note` | str | Mandatory causality disclaimer |

Each entry in `top_features`:
```json
{
  "feature":      "qualifying_day",
  "value":        1.0,
  "contribution": 0.18,
  "direction":    "increases_risk"
}
```

When method is `"global_rf_importance"`, `direction` is `"global_importance"` for all entries.

---

## 6. Usage

```python
from src.prediction import ClimateGuardPredictor
from src.explainability import ClimateGuardExplainer

predictor = ClimateGuardPredictor()
explainer = ClimateGuardExplainer(predictor)

# Check SHAP availability
print(ClimateGuardExplainer.shap_available())   # True or False

# Explain a single row
explanation = explainer.explain(features_df, top_n=10)

print(explanation.method)          # "shap" or "global_rf_importance"
print(explanation.top_features)    # list of feature dicts

# Full dict output (for JSON serialisation)
d = explanation.to_dict()

# Global importance summary (always available, independent of SHAP)
global_top = explainer.explain_global(top_n=20)
```

---

## 7. Feature Contract

The explainer operates on exactly 110 features in the order defined by:

```
models/final/feature_list.json
```

Feature names are taken from `predictor.feature_names`.  The explainer never renames, reorders, adds, or removes features.

---

## 8. SHAP Installation

```bash
# Preferred — exact compatible version
pip install shap

# Verify
python -c "import shap; print(shap.__version__)"
```

SHAP is an optional dependency.  If not installed:
- The explainer falls back to global RF importance automatically.
- No ImportError is raised at module import time.
- `ClimateGuardExplainer.shap_available()` returns `False`.

---

## 9. Limitations

1. **SHAP requires installation**: Global importance is used when SHAP is absent.  Always install SHAP for production use.

2. **Global importance is not local**: The fallback does not explain individual predictions.

3. **qualifying_day structural correlation**: As documented above, qualifying_day will dominate explanations for high-probability predictions by construction.

4. **No interaction effects**: Neither SHAP TreeExplainer nor global importance captures interaction effects between features.

5. **Binary classification scope**: SHAP values are for the positive class (heatwave).  The negative class values are the mirror image.

6. **Model is locked**: The explainer reads the locked final model from Part 1.  No retraining occurs.

---

## 10. Module Reference

| Symbol | Description |
|---|---|
| `ClimateGuardExplainer` | Main class; `explain()`, `explain_global()`, `shap_available()` |
| `ExplainabilityResult` | Output container; `.to_dict()` serialisation |
| `_QUALIFYING_DAY_NOTE` | Module-level qualifying_day disclaimer string |
| `_CAUSALITY_NOTE` | Module-level causality disclaimer string |

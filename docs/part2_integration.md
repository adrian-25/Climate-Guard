# Part 2 Integration

**ClimateGuard — Part 2**  
**Module:** `src/risk_engine/engine.py`  
**Owner:** Kshitij (Part 2 — Risk, Adaptation, Explainability)  
**Status:** Complete  
**Date:** 2026-09-10

---

## 1. Overview

This document describes how Part 2 integrates with Part 1 and provides the unified `ClimateGuardRiskEngine` interface.

Part 2 consumes the output of `ClimateGuardPredictor` (Part 1) and adds:
- **Risk Assessment**: maps probability to an operational risk level
- **Adaptation Recommendations**: generates practical preparedness guidance
- **Explainability**: provides feature-level explanation of the prediction

Part 2 does NOT retrain the model, modify the threshold, change the feature list, or alter any Part 1 artifact.

---

## 2. Architecture

```
Part 1 — ClimateGuardPredictor
    Input:  110 features (exact order, feature_list.json)
    Output: prediction_probability (float [0,1])
            prediction_label (int {0,1}, threshold=0.70)
            city (str or None)
            date (str or None)
            ↓

Part 2 — ClimateGuardRiskEngine
    ┌───────────────────────────────────────────────┐
    │                                               │
    │   RiskAssessor                                │
    │   [0.00,0.30) → LOW                          │
    │   [0.30,0.60) → MODERATE                     │
    │   [0.60,0.80) → HIGH                         │
    │   [0.80,1.00] → EXTREME                      │
    │         ↓                                     │
    │   AdaptationEngine                            │
    │   Generates risk-level-appropriate guidance   │
    │                                               │
    │   ClimateGuardExplainer                       │
    │   SHAP (preferred) or global RF importance    │
    │                                               │
    └────────────────────────────────────────────── ┘
                     ↓
    RiskEngineResult (structured output)
```

---

## 3. Quick Start

```python
from src.risk_engine import ClimateGuardRiskEngine
import pandas as pd

# Load a feature row (110 columns from feature_list.json)
# The engine handles predictor loading automatically
engine = ClimateGuardRiskEngine()

result = engine.analyze(features_df)

print(result.heatwave_probability)   # float [0,1]
print(result.prediction)             # 0 or 1
print(result.risk_level)             # LOW / MODERATE / HIGH / EXTREME
print(result.to_json())              # full JSON output

# Disable explanation for speed
result = engine.analyze(features_df, include_explanation=False)

# Batch analysis
results = engine.analyze_batch(features_df)  # one result per row
```

---

## 4. Integration with Part 1

Part 2 consumes Part 1 exclusively through `ClimateGuardPredictor`.

```python
from src.prediction import ClimateGuardPredictor

# Optional: share a pre-loaded predictor for efficiency
predictor = ClimateGuardPredictor()

from src.risk_engine import ClimateGuardRiskEngine
engine = ClimateGuardRiskEngine(predictor=predictor)
```

The predictor is loaded **once** per instance.  Passing an existing predictor avoids duplicate file I/O.

---

## 5. Full Output Structure

`result.to_dict()` produces:

```json
{
  "city": "delhi",
  "date": "2024-05-20",
  "heatwave_probability": 0.82,
  "prediction": 1,
  "risk_level": "EXTREME",
  "explanation": {
    "prediction": 1,
    "probability": 0.82,
    "explanation_method": "shap",
    "shap_available": true,
    "top_features": [
      {
        "feature": "qualifying_day",
        "value": 1.0,
        "contribution": 0.18,
        "direction": "increases_risk"
      },
      {
        "feature": "temperature_2m_max",
        "value": 43.8,
        "contribution": 0.11,
        "direction": "increases_risk"
      }
    ],
    "notes": {
      "causality": "Explanation reflects model behaviour only...",
      "qualifying_day": "qualifying_day encodes the same IMD-inspired threshold criteria..."
    }
  },
  "recommendations": {
    "risk_level": "EXTREME",
    "disclaimer": "These are project-defined operational recommendations...",
    "recommendations": [
      {"category": "Hydration", "message": "Drink water continuously..."},
      {"category": "Outdoor Exposure", "message": "Minimise all unnecessary outdoor activity..."},
      {"category": "Emergency Preparedness", "message": "Know the location of the nearest hospital..."}
    ]
  }
}
```

---

## 6. Component API Summary

### `ClimateGuardRiskEngine`

| Method | Returns | Description |
|---|---|---|
| `analyze(features, city, date, include_explanation, top_n)` | `RiskEngineResult` | Full pipeline for one row |
| `analyze_batch(features_df, include_explanation, top_n)` | `List[RiskEngineResult]` | Full pipeline for all rows |
| `predictor` (property) | `ClimateGuardPredictor` | Underlying Part 1 predictor |
| `explainer` (property) | `ClimateGuardExplainer` | Explainability module |
| `shap_available` (property) | `bool` | SHAP installation status |
| `info()` | `dict` | Engine configuration summary |

### `RiskEngineResult`

| Method | Returns | Description |
|---|---|---|
| `to_dict()` | `dict` | Full result as plain dictionary |
| `to_json(indent)` | `str` | Full result as JSON string |

---

## 7. Part 2 Integration Contract (What Part 3 Receives)

Part 3 (Pradnesh) can consume `RiskEngineResult` in any of these forms:

1. **Direct Python object**: `result.risk_level`, `result.recommendations`, etc.
2. **Dict**: `result.to_dict()`
3. **JSON string**: `result.to_json()`

Required fields that Part 3 can depend on:
- `heatwave_probability` : float
- `prediction` : int (0 or 1)
- `risk_level` : str
- `recommendations.recommendations` : list of `{category, message}`
- `recommendations.disclaimer` : str

Optional fields (present only when `include_explanation=True`):
- `explanation.top_features` : list
- `explanation.explanation_method` : str
- `explanation.notes` : dict

---

## 8. Dependencies

| Package | Version constraint | Required? |
|---|---|---|
| scikit-learn | ≥ 1.0 | Yes (Part 1 model) |
| joblib | any | Yes (Part 1 model loading) |
| pandas | ≥ 1.3 | Yes |
| numpy | ≥ 1.20 | Yes |
| shap | any | No (optional; fallback if absent) |

Install dependencies:

```bash
pip install scikit-learn joblib pandas numpy
# Optional SHAP
pip install shap
```

---

## 9. Part 1 Contract Compliance

| Constraint | Status |
|---|---|
| Model file unchanged | ✓ Not modified |
| Feature list unchanged | ✓ Not modified |
| Threshold remains 0.70 | ✓ Not changed |
| ClimateGuardPredictor reused | ✓ No duplicate predictor |
| No retraining | ✓ No training code |
| No feature renaming/reordering | ✓ Feature names from predictor.feature_names |
| Limitations documented | ✓ See sections below |

---

## 10. Limitations Inherited from Part 1

1. **ERA5 reanalysis data** — not official IMD station observations.
2. **IMD-inspired label** — not certified IMD ground truth.
3. **Mumbai** — zero heatwave positives; risk assessments carry high uncertainty.
4. **Ahmedabad** — all historical positives in training window; no test-set performance.
5. **qualifying_day** — correlated with the target by construction; not independently discovered.
6. **Precision = 0.58** — ~42% of elevated risk alarms may be false positives.
7. **No drift detection** — model not self-monitoring for distribution shift.

---

## 11. Tests

All Part 2 tests are in `tests/test_part2.py`.

```bash
# Run from project root
python tests/test_part2.py

# Or with pytest
python -m pytest tests/test_part2.py -v
```

Test groups:
| Group | Name | Count |
|---|---|---|
| A | Risk probability validation | 10 |
| B | Risk threshold boundaries | 10 |
| C | Risk level generation | 6 |
| D | Adaptation recommendation generation | 10 |
| E | Adaptation recommendation categories | 7 |
| F | Model access | 6 |
| G | Feature-name consistency | 5 |
| H | Explainability execution | 11 |
| I | Invalid input handling | 9 |
| J | Part 1 → Part 2 integration | 11 |
| K | Real-data smoke test | 4 |

---

## 12. File Reference

| Path | Description |
|---|---|
| `src/risk/risk_assessment.py` | Risk assessment module |
| `src/risk/__init__.py` | Risk package init |
| `src/adaptation/recommendations.py` | Adaptation engine |
| `src/adaptation/__init__.py` | Adaptation package init |
| `src/explainability/explainer.py` | Explainability module |
| `src/explainability/__init__.py` | Explainability package init |
| `src/risk_engine/engine.py` | Unified engine interface |
| `src/risk_engine/__init__.py` | Engine package init |
| `tests/test_part2.py` | Full Part 2 test suite |
| `docs/risk_assessment.md` | Risk assessment documentation |
| `docs/adaptation_recommendations.md` | Adaptation documentation |
| `docs/explainability.md` | Explainability documentation |
| `docs/part2_integration.md` | This file |

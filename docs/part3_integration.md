# ClimateGuard Part 3 — Integration Pipeline

**Module:** `src/integration/`  
**Owner:** Pradnesh (Part 3)  
**Status:** Complete  
**Last updated:** 2026-09-10

---

## 1. Purpose

`ClimateGuardPipeline` is the unified Part 3 orchestrator. It connects all ClimateGuard components — ETL validation, Part 1 ML prediction, Part 2 risk assessment, and Part 3 expert rules — into a single callable end-to-end pipeline.

The pipeline returns a single structured `ClimateGuardResult` object that contains all outputs in a JSON-serialisable format.

---

## 2. Integration Flow

```
Incoming Climate Data
  (dict / pd.Series / pd.DataFrame)
            │
            ▼
┌─────────────────────────────┐
│   Stage 1 — ETL             │
│   src.etl.ETLPipeline       │
│                             │
│  • City validation          │
│  • Date format check        │
│  • Type / NaN checks        │
│  • Range sanity checks      │
│  • 110-feature contract     │
│  • Column reorder + cast    │
└────────────┬────────────────┘
             │ ETLResult.valid == False → return ClimateGuardResult(valid=False)
             │ ETLResult.valid == True  → continue
             ▼
┌─────────────────────────────────────────────────────────┐
│   Stage 2 — Part 2 Risk Engine                          │
│   src.risk_engine.ClimateGuardRiskEngine                │
│   (internally calls Part 1 ClimateGuardPredictor)       │
│                                                         │
│  Part 1:  110-feature vector → probability + label      │
│  Part 2a: probability → risk level (LOW/MOD/HIGH/EXT)   │
│  Part 2b: risk level  → adaptation recommendations      │
│  Part 2c: features    → explainability (SHAP or global) │
└────────────┬────────────────────────────────────────────┘
             │ Exception → return ClimateGuardResult(valid=False, warning=...)
             │ Success   → continue
             ▼
┌─────────────────────────────────┐
│   Stage 3 — Expert Rules        │
│   src.expert_rules.ExpertRuleEngine │
│                                 │
│  All 7 rules evaluated using:   │
│  • feature record               │
│  • risk_level (from Stage 2)    │
│  • heatwave_probability (Stg 2) │
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────────────┐
│   Final Result — ClimateGuardResult     │
│                                         │
│  .valid           bool                  │
│  .input_metadata  dict                  │
│  .validation      dict                  │
│  .prediction      dict | None           │
│  .risk            dict | None           │
│  .recommendations list | None           │
│  .explanation     dict | None           │
│  .expert_rules    list[dict]            │
│  .warnings        list[str]             │
│  .metadata        dict                  │
└─────────────────────────────────────────┘
```

---

## 3. Module Files

| File | Class | Description |
|---|---|---|
| `src/integration/__init__.py` | — | Package exports |
| `src/integration/pipeline.py` | `ClimateGuardPipeline` | End-to-end orchestrator |
| `src/integration/pipeline.py` | `ClimateGuardResult` | Structured output object |

---

## 4. ClimateGuardPipeline

### 4.1 Constructor

```python
from src.integration import ClimateGuardPipeline

pipeline = ClimateGuardPipeline(
    predictor=None,           # Pre-loaded Part 1 predictor (optional)
    include_explanation=True, # Request feature explanation from Part 2
    top_n=10,                 # Number of top features in explanation
)
```

- The Part 1 `ClimateGuardPredictor` is loaded **once** at construction time and reused for all subsequent calls. This avoids repeated model file I/O.
- The Part 2 `ClimateGuardRiskEngine` is constructed sharing the same predictor instance.
- The ETL pipeline and expert rule engine are also instantiated once.
- A pre-loaded predictor can be passed in to share it across multiple pipeline instances.

### 4.2 Single-record analysis

```python
result = pipeline.analyze(
    data,                      # dict, pd.Series, or 1-row pd.DataFrame
    include_explanation=True,  # optional: override constructor default
    top_n=10,                  # optional: override constructor default
)
```

Returns a `ClimateGuardResult`. Always returns — never raises on validation failure.

### 4.3 Batch analysis

```python
results = pipeline.analyze_batch(
    df,                        # pd.DataFrame — one row per city/day
    include_explanation=None,  # optional
    top_n=None,                # optional
)
# Returns list[ClimateGuardResult], one per row
```

Each row is processed independently through the full pipeline. A failure in one row does not affect other rows.

### 4.4 Properties

| Property | Type | Description |
|---|---|---|
| `pipeline.predictor` | `ClimateGuardPredictor` | The Part 1 predictor |
| `pipeline.risk_engine` | `ClimateGuardRiskEngine` | The Part 2 engine |
| `pipeline.expert_engine` | `ExpertRuleEngine` | The Part 3 rule engine |
| `pipeline.shap_available` | `bool` | Whether SHAP is installed |

### 4.5 info()

```python
info = pipeline.info()
# Returns dict with:
#   pipeline, predictor, shap_available, include_explanation,
#   top_n, threshold, n_features, expert_rules (list of names)
```

---

## 5. ClimateGuardResult

The final structured output of the pipeline.

### 5.1 Fields

| Field | Type | Present when | Description |
|---|---|---|---|
| `valid` | `bool` | always | True if full pipeline ran successfully |
| `input_metadata` | `dict` | always | Extracted `city_key` and `date` from input |
| `validation` | `dict` | always | ETL validation outcome (valid, errors, warnings) |
| `prediction` | `dict` | valid=True | Part 1 output: `probability`, `prediction` |
| `risk` | `dict` | valid=True | Part 2 risk: `level`, `score` |
| `recommendations` | `list` | valid=True | Part 2 adaptation recommendations list |
| `explanation` | `dict` or None | valid=True, if enabled | Part 2 feature explanation |
| `expert_rules` | `list[dict]` | valid=True; else `[]` | 7 expert rule results |
| `warnings` | `list[str]` | always | Pipeline-level warnings |
| `metadata` | `dict` | always | Pipeline config: threshold, n_features, etc. |

### 5.2 prediction dict

```json
{
  "probability": 0.7234,
  "prediction":  1
}
```

- `probability`: float in [0, 1] — the model's raw heatwave-tomorrow probability
- `prediction`: 0 or 1 — label at threshold=0.70 (1 = heatwave predicted tomorrow)

### 5.3 risk dict

```json
{
  "level": "HIGH",
  "score": 0.7234
}
```

Risk levels and their project-defined probability bands:

| Level | Probability range |
|---|---|
| LOW | [0.00, 0.30) |
| MODERATE | [0.30, 0.60) |
| HIGH | [0.60, 0.80) |
| EXTREME | [0.80, 1.00] |

These are **project-defined operational categories**. NOT official IMD risk categories.

### 5.4 recommendations

A list of recommendation strings from Part 2's `AdaptationEngine`.  
Count varies by risk level: LOW=3, MODERATE=5, HIGH=6, EXTREME=7.

### 5.5 explanation

Either a SHAP-based per-prediction explanation or a global RF importance fallback, depending on whether the `shap` package is installed.

```json
{
  "method": "global_rf_importance",
  "top_features": [
    {"feature": "qualifying_day", "importance": 0.142, "direction": "global_importance"},
    ...
  ],
  "notes": ["...", "..."],
  "disclaimer": "Explanation reflects model behaviour only. ..."
}
```

When `shap` is installed, `method` becomes `"shap"` and each feature has `"direction": "increases_risk"` or `"decreases_risk"`.

Both methods are handled without crashing. If explanation is disabled (`include_explanation=False`), this field is `None`.

### 5.6 expert_rules

A list of 7 dicts, one per rule (RULE_01 through RULE_07):

```json
[
  {
    "rule_id":     "RULE_01",
    "name":        "Extreme Temperature",
    "triggered":   false,
    "severity":    "CRITICAL",
    "description": "Fires when today's maximum temperature exceeds..."
  },
  {
    "rule_id":   "RULE_07",
    "name":      "Hydration and Cooling Reminder",
    "triggered": true,
    "severity":  "INFO",
    "description": "...",
    "message":   "Elevated heat conditions (Tmax=36.5°C)..."
  }
]
```

`message` is only present when `triggered=true`.

### 5.7 metadata dict

```json
{
  "pipeline_version":        "1.0.0",
  "part1_threshold":         0.70,
  "part1_n_features":        110,
  "shap_available":          false,
  "explanation_enabled":     true,
  "expert_rules_disclaimer": "Expert rule results are project-defined..."
}
```

### 5.8 Serialisation

```python
# Full dict (JSON-serialisable)
d = result.to_dict()

# JSON string
s = result.to_json(indent=2)

# Parsed back
import json
parsed = json.loads(s)
```

All NumPy types and pandas objects are converted to native Python types before serialisation.

---

## 6. Error Handling

The pipeline never raises on user-input errors. All failure modes return a `ClimateGuardResult` with `valid=False` and an explanation in `validation["errors"]` or `warnings`.

| Failure mode | Outcome |
|---|---|
| Invalid city | ETL error, `valid=False`, prediction=None |
| Missing required field | ETL error, `valid=False` |
| Malformed date | ETL error, `valid=False` |
| NaN in feature | ETL error, `valid=False` |
| Non-numeric feature | ETL error, `valid=False` |
| Missing features (wrong count) | ETL contract error, `valid=False` |
| Wrong feature order | ETL contract error, `valid=False` |
| Part 2 engine error | Warning added, `valid=False`, prediction=None |
| Explanation failure | Warning added, explanation=None, pipeline continues |

In all `valid=False` cases:
- `prediction` is `None`
- `risk` is `None`
- `recommendations` is `None`
- `expert_rules` is `[]`

---

## 7. Part 1 and Part 2 Integration

The pipeline uses the **existing** Part 1 and Part 2 APIs without modification.

### Part 1 usage

```python
# ClimateGuardPredictor is loaded inside ClimateGuardRiskEngine
# The ETL feature_df is passed directly to it via the risk engine
engine_result = self._risk_engine.analyze(features=feature_df, ...)
```

Part 1 contract respected:
- 110 features, exact order, float64, no NaN
- Threshold 0.70 — never altered
- Model artifact never modified

### Part 2 usage

```python
engine_result = self._risk_engine.analyze(
    features=feature_df,
    city=city_key,
    date=date,
    include_explanation=True,
    top_n=10,
)
```

Returns `RiskEngineResult` with:
- `heatwave_probability` — passed to expert rules
- `risk_level` — passed to expert rules
- `recommendations` — included in final result
- `explanation` — included in final result (SHAP or global importance fallback)

The Part 2 engine handles both explanation methods internally. Part 3 does not need to know which method is active.

---

## 8. Input Requirements

The pipeline accepts records that already contain the **110 engineered features** plus metadata columns.

### Minimum required fields

| Field | Type | Description |
|---|---|---|
| `city_key` | str | One of: `delhi`, `lucknow`, `nagpur`, `ahmedabad`, `mumbai` |
| `date` | str | ISO format: `YYYY-MM-DD` |
| `temperature_2m_max` | float | Today's maximum temperature (°C) |
| *all 110 feature columns* | float | See `models/final/feature_list.json` |

### Where to get the 110 features

The 110 features are produced by the Phase 7 feature engineering pipeline (`feature_engineering.py`) and are available in:
- `data/splits/temporal/X_test.csv` (test split, 4,865 rows)
- `data/splits/temporal/X_train.csv` (train split)
- `data/splits/temporal/X_val.csv` (validation split)
- `data/features/ml_temporal.csv` (full dataset)

For new data, the features must be constructed using the same Phase 7 methodology (lag features, rolling means, anomaly z-scores, calendar encodings, city encodings). The ETL module does not construct these features.

---

## 9. Usage Examples

### Basic single-record analysis

```python
from src.integration import ClimateGuardPipeline
import pandas as pd

pipeline = ClimateGuardPipeline()

# Load a feature row from the test split
X    = pd.read_csv("data/splits/temporal/X_test.csv")
meta = pd.read_csv("data/splits/temporal/meta_test.csv")

row = X.iloc[0].to_dict()
row["city_key"] = meta.iloc[0]["city_key"]
row["date"]     = meta.iloc[0]["date"]

result = pipeline.analyze(row)

print(result.valid)
print(result.prediction["probability"])
print(result.risk["level"])
print(len(result.recommendations), "recommendations")
triggered = [r for r in result.expert_rules if r["triggered"]]
print(len(triggered), "expert rules triggered")
print(result.to_json(indent=2))
```

### Batch analysis

```python
import pandas as pd
from src.integration import ClimateGuardPipeline

pipeline = ClimateGuardPipeline()

X    = pd.read_csv("data/splits/temporal/X_test.csv")
meta = pd.read_csv("data/splits/temporal/meta_test.csv")

# Take first 10 rows
batch = X.iloc[:10].copy()
batch["city_key"] = meta.iloc[:10]["city_key"].values
batch["date"]     = meta.iloc[:10]["date"].values

results = pipeline.analyze_batch(batch)

for r in results:
    city  = r.input_metadata.get("city_key")
    date  = r.input_metadata.get("date")
    level = r.risk["level"] if r.valid else "INVALID"
    prob  = r.prediction["probability"] if r.valid else "N/A"
    print(f"{date} {city}: prob={prob:.3f}, risk={level}")
```

### Error handling

```python
record = {"city_key": "invalid_city", "date": "2024-05-01", "temperature_2m_max": 40.0}
result = pipeline.analyze(record)

assert not result.valid
print(result.validation["errors"])
# ["Unrecognised city: 'invalid_city'. Supported city_key values: ..."]
```

---

## 10. SHAP Availability

SHAP per-prediction explanations require the `shap` package:

```bash
pip install shap
```

When SHAP is not installed, the explainer automatically falls back to global Random Forest feature importance (`method = "global_rf_importance"`). This fallback produces the same result for every prediction regardless of the input values, and is clearly labelled in the output.

Both modes are handled transparently — the pipeline never crashes due to SHAP availability.

Check at runtime:

```python
print(pipeline.shap_available)  # True / False
```

---

## 11. Model Integrity

The Part 3 pipeline does not modify any Part 1 or Part 2 artifacts:

| Artifact | Status |
|---|---|
| `models/final/climateguard_final_model.joblib` | Read-only, not modified |
| `models/final/feature_list.json` | Read-only, not modified |
| Threshold 0.70 | Preserved — never altered |
| 110-feature contract | Enforced — ETL rejects non-conforming input |
| Part 1 prediction logic | Not duplicated |
| Part 2 risk logic | Not duplicated |

---

## 12. Limitations

1. **Pre-built features required.** The pipeline does not compute features from raw ERA5 weather observations. The caller must supply all 110 engineered features.

2. **Five cities only.** Inputs for cities outside the training set are rejected by the ETL layer.

3. **Single-row `analyze()`.** For batch processing, use `analyze_batch()`. Passing a multi-row DataFrame to `analyze()` returns an error.

4. **SHAP not installed by default.** The global RF importance fallback is used unless `pip install shap` has been run. The fallback explanation is the same for all inputs and is not per-prediction.

5. **No temporal context between calls.** Each pipeline call is independent. The pipeline does not maintain state between calls (e.g., it does not track consecutive heatwave days across calls to improve RULE_02).

6. **Model training horizon 1990–2022.** Predictions for dates significantly beyond 2025 may be less reliable as climate patterns evolve past the training distribution.

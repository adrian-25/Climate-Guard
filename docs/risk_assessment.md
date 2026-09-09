# Risk Assessment

**ClimateGuard — Part 2**  
**Module:** `src/risk/risk_assessment.py`  
**Owner:** Kshitij (Part 2 — Risk, Adaptation, Explainability)  
**Status:** Complete  
**Date:** 2026-09-10

---

## 1. Overview

The Risk Assessment module translates the continuous heatwave probability produced by the Part 1 Random Forest model into a discrete operational risk level that can be communicated to downstream consumers.

The module does NOT alter the ML probability or prediction label.  It adds a categorical overlay — risk level — as a communication aid.

---

## 2. Architecture Position

```
ClimateGuardPredictor (Part 1)
        ↓
  probability  +  prediction_label
        ↓
  RiskAssessor  (Part 2 — this module)
        ↓
  RiskAssessmentResult
        ↓
  AdaptationEngine  →  Recommendations
```

---

## 3. Risk Thresholds

**⚠️ IMPORTANT: These are project-defined operational categories for ClimateGuard.  They are NOT official India Meteorological Department (IMD) risk categories.**

| Risk Level | Lower Bound (inclusive) | Upper Bound (exclusive) |
|---|---|---|
| LOW      | 0.00 | 0.30 |
| MODERATE | 0.30 | 0.60 |
| HIGH     | 0.60 | 0.80 |
| EXTREME  | 0.80 | 1.00 (inclusive) |

The mapping is **deterministic**: identical probability always produces identical risk level.

Boundary behaviour:
- 0.30 → MODERATE (not LOW)
- 0.60 → HIGH (not MODERATE)
- 0.80 → EXTREME (not HIGH)
- 1.00 → EXTREME

---

## 4. Input / Output

### Input

| Parameter | Type | Required | Description |
|---|---|---|---|
| `probability` | float [0.0, 1.0] | Yes | Raw ML probability from ClimateGuardPredictor |
| `prediction` | int {0, 1} | Yes | Binary prediction label from ClimateGuardPredictor |
| `city` | str | No | City identifier (e.g. "delhi") |
| `date` | str | No | Date string for day T |

### Output: `RiskAssessmentResult`

| Attribute | Type | Description |
|---|---|---|
| `heatwave_probability` | float | Original ML probability (preserved verbatim) |
| `prediction` | int | Original prediction label (preserved verbatim) |
| `risk_level` | str | "LOW" \| "MODERATE" \| "HIGH" \| "EXTREME" |
| `city` | str or None | Passed through from input |
| `date` | str or None | Passed through from input |

---

## 5. Usage

```python
from src.risk import RiskAssessor

assessor = RiskAssessor()

result = assessor.assess(
    probability=0.82,
    prediction=1,
    city="delhi",
    date="2024-05-20",
)

print(result.risk_level)           # EXTREME
print(result.heatwave_probability) # 0.82 (preserved)
print(result.prediction)           # 1    (preserved)

# Dict output
d = result.to_dict()
# {
#   "heatwave_probability": 0.82,
#   "prediction": 1,
#   "risk_level": "EXTREME",
#   "city": "delhi",
#   "date": "2024-05-20"
# }
```

---

## 6. Error Handling

| Situation | Error |
|---|---|
| `probability < 0.0` | `ValueError` |
| `probability > 1.0` | `ValueError` |
| `probability` is NaN | `ValueError` or `TypeError` |
| `probability` is a non-numeric type | `TypeError` |

---

## 7. Design Decisions

1. **Probability preserved**: The original ML probability is never replaced or adjusted. The risk level is an additive layer, not a substitution.

2. **Prediction label preserved**: The Part 1 threshold of 0.70 is not changed. Risk level is derived from the raw probability independently.

3. **No external data required**: Risk assessment is stateless. No model file, no database, no network call.

4. **Threshold documentation**: The four thresholds are project-defined for ClimateGuard. They have no external authoritative source and must not be presented as official IMD guidance.

5. **`qualifying_day` note**: The Part 1 model uses `qualifying_day`, a feature derived from the same threshold criteria as the heatwave label. High-risk predictions often correlate with `qualifying_day=1`. This is an artefact of feature engineering, not independent discovery.

---

## 8. Limitations

1. **Not IMD risk categories.** The LOW/MODERATE/HIGH/EXTREME mapping is project-defined. It is not derived from official IMD heat-health guidance.

2. **Single-threshold mapping.** The risk level does not account for duration of the heatwave, city-specific vulnerability, or population-level factors.

3. **No uncertainty quantification.** The boundary between two adjacent risk levels (e.g., 0.595 vs 0.601) represents a large jump in label but a tiny change in probability. Users should treat risk levels as approximate categories.

4. **Downstream model limitations inherited.** All limitations of the Part 1 model apply:
   - Mumbai: zero heatwave positives; predictions carry high uncertainty.
   - Ahmedabad: all historical positives fall in the training window.
   - Precision = 0.58: ~42% of MODERATE/HIGH/EXTREME alarms may be false positives.

---

## 9. Module Reference

| Symbol | Description |
|---|---|
| `RiskLevel` | Enum with values LOW, MODERATE, HIGH, EXTREME |
| `probability_to_risk_level(p)` | Maps float probability to RiskLevel |
| `RiskAssessor` | Main class; `assess(probability, prediction, city, date)` |
| `RiskAssessmentResult` | Output dataclass; `.to_dict()` serialisation |
| `RISK_THRESHOLDS` | Module-level constant: list of `(lower_bound, RiskLevel)` tuples |

# Known Limitations and Research Caveats

**ClimateGuard: Indian Heatwave Prediction**  
**Phase:** 16 — Documentation  
**Date:** 2026-09-02  
**Status:** COMPLETE

---

## 1. Overview

This document catalogues the known limitations of the ClimateGuard model, dataset, and heatwave label. These limitations are not bugs or oversights — most are inherent to the data source, the problem domain, or explicit design choices made and documented across Phases 1–14. All downstream uses of the model (Part 2, Part 3) must carry forward and communicate these limitations.

---

## 2. Data Limitations

### 2.1 ERA5 Reanalysis, Not Station Observations

**Severity: High — affects all cities**

The entire dataset is derived from the [ERA5 reanalysis model](https://open-meteo.com/) via the Open-Meteo Historical Weather API, at 0.25° spatial resolution. ERA5 is a modelled estimate of historical weather conditions, not direct measurements from weather stations.

Differences from IMD station observations:
- ERA5 averages conditions over a ~28 km grid cell; station data reflects point measurements
- ERA5 can underestimate or overestimate extreme daily temperature maxima compared to in-situ readings
- ERA5 surface-layer humidity and pressure fields may deviate from station observations, especially during monsoon transitions

**Implication:** The model's real-world performance on actual station-observed heatwave days could differ from the test results reported here. Systematic warm or cool biases in ERA5 relative to IMD stations would shift the effective decision boundary.

**Disclosure requirement:** Any public-facing product built on this model must state: *"This model was trained on ERA5 reanalysis data, not IMD weather station observations."*

### 2.2 Open-Meteo API Version Consistency

The data was downloaded in September 2026. Open-Meteo periodically updates ERA5 reanalysis with newer version numbers (ERA5-Land, ERA5 backextension). Minor version differences could produce slightly different values if the data were re-downloaded in the future.

---

## 3. Label Limitations

### 3.1 IMD-Inspired Label, Not Official IMD Ground Truth

**Severity: High — affects label validity**

The heatwave label was constructed using an operational definition **inspired by** published IMD criteria, applied to ERA5 data. It has **not** been validated against official IMD heatwave declarations.

The label is named "IMD-Inspired Operational Heatwave Label (ERA5-based)" — it must not be called "official IMD labels" or "certified heatwave events."

Published IMD criteria used as reference:
- Plains: Tmax ≥ 40°C AND departure ≥ 4.5°C from normal, OR Tmax ≥ 45°C (absolute override)
- Coastal: Tmax ≥ 37°C AND departure ≥ 4.5°C from normal
- Duration: ≥ 2 consecutive qualifying days constitute a heatwave event

The ERA5 Tmax values used are not identical to IMD station Tmax. A day classified as "qualifying" under this label may not correspond to an official IMD heatwave declaration at the nearest station.

### 3.2 Climatological Normal Baseline (1990–2020)

The departure is calculated relative to a 31-day centred smoothed climatological normal computed from the **1990–2020 ERA5 baseline** (31 years). Using a different baseline period or a different smoothing window would shift departure values and alter which days are classified as qualifying.

---

## 4. City-Level Limitations

### 4.1 Mumbai — Zero Positive Examples

**Severity: High — city-specific**

Mumbai has **zero heatwave positive examples** in the entire 35-year dataset (1990–2025) under the coastal city definition (Tmax ≥ 37°C AND departure ≥ 4.5°C).

- The model has never been trained on a positive example for Mumbai's city encoding
- Mumbai's maritime climate produces narrow Tmax range (std ≈ 2.1°C) and almost never exceeds the 37°C threshold at the ERA5 grid point
- **Predictions for Mumbai are unreliable** — the model will almost always return probability ≈ 0 for Mumbai, regardless of input conditions

**Implication:** Do not use this model to generate heatwave alerts for Mumbai without additional city-specific retraining or a fundamentally different definition.

### 4.2 Ahmedabad — No Test-Set Positives

**Severity: Medium — affects evaluation validity**

Ahmedabad has 32 positive examples in the dataset, but all 32 fall within the training window (1990–2019). The validation and test splits contain zero Ahmedabad positives.

- The model cannot be evaluated for Ahmedabad generalisation
- It is unknown whether the model has learned patterns that transfer to Ahmedabad heatwaves or is overfitting to the training years

### 4.3 Five-City Scope

The model was trained on five specific Indian cities. It must not be applied to other cities without retraining. The city encoding is ordinal (ahmedabad=0, delhi=1, mumbai=2, lucknow=3, nagpur=4) — there is no generalisation to unlisted cities.

---

## 5. Feature Limitations

### 5.1 qualifying_day Feature Coupling

**Severity: Medium — scientific transparency issue**

`qualifying_day` (feature index 26) is computed from the same IMD-inspired threshold logic used to construct the target variable `heatwave_next_day`. Specifically:

- `qualifying_day(T)` = 1 if today T meets the IMD temperature/departure thresholds
- `heatwave_next_day(T)` = 1 if tomorrow T+1 is part of a heatwave event

Because heatwave events require ≥ 2 consecutive qualifying days, `qualifying_day(T)` is a strong predictor of `heatwave(T+1)` (which requires qualifying_day(T+1)=1 as well). The feature is **leakage-safe** (it uses only current-day T values, not tomorrow's), but it embeds explicit domain knowledge about the labeling rule.

**Consequence:** The model is not discovering the meteorological pattern independently — it is exploiting the encoded IMD rule. This is a design choice (qualifying_day was retained because it substantially improved F1), but it must be disclosed. SHAP values for `qualifying_day` will show very high feature importance, which reflects the label construction rather than a purely empirical finding.

### 5.2 Temporal Feature History Requirement

Lag and rolling features require **at least 7 days of prior weather history** per city to construct correctly. The anomaly feature `tmax_departure_zscore` requires **30+ days** of history for a stable z-score (the rolling standard deviation requires at least 10 prior observations for a non-NaN result).

**Implication:** The predictor cannot produce valid predictions for the first 30 days of data for any city. Part 3's ETL pipeline must buffer at least 30 days of prior observations before calling the model.

---

## 6. Model Limitations

### 6.1 Precision vs Recall Trade-off

**Severity: Medium — known design choice**

The model was deliberately tuned for **recall** over precision, using a 0.70 probability threshold to suppress false positives while retaining most true positives.

Test-set results (2023–2025):
- Recall = 0.8684 — detects 33 of 38 heatwave days
- Precision = 0.5789 — approximately 42% of raised alarms are false positives

This is appropriate for a public-health early-warning system where missing a heatwave event carries higher risk than a false alarm. However, users expecting high precision should be clearly informed that roughly 2 in 5 alerts will not correspond to observed heatwave conditions.

### 6.2 Threshold Fixed on 3-Year Validation Window

The decision threshold of 0.70 was fixed on the Phase 11 validation set (2020–2022, 39 positives across 5 cities). This is a relatively small positive-event sample for threshold calibration. The optimal threshold could shift on a different time period or under climate change.

### 6.3 No Drift Detection or Retraining Pipeline

The model does not monitor for distribution shift. Climate patterns are not stationary — rising temperature trends observed in EDA (Tmin rising at +0.030–0.040°C/yr across all 5 cities) will eventually move the model's input distribution out of its training range.

Retraining is recommended if deploying past 2025 or if systematic performance degradation is observed (Precision < 0.40 or Recall < 0.70 over a 12-month evaluation window are suggested retraining triggers).

### 6.4 ROC-AUC Is Misleading

The reported ROC-AUC of 0.9979 should not be used as the primary performance metric. At a 1:128 class imbalance ratio, a classifier that predicts all-negative achieves ROC-AUC ≈ 0.50, while the precision-recall curve (PR-AUC = 0.8339) better reflects discrimination on the minority class.

### 6.5 2023 Year-Level Performance

Year-level test results show that 2023 performance was substantially weaker than 2024:

| Year | Heatwave days | F1 |
|---|---|---|
| 2023 | 4 total across 5 cities | 0.36–0.53 (model-dependent) |
| 2024 | 34 total across 5 cities | 0.78–0.83 |
| 2025 (Jan–Aug) | 0 total | N/A (all alarms are FP) |

2024 was an exceptionally strong heatwave year in India. Performance in years with few or no heatwave events (like 2023 and 2025 Jan–Aug) is structurally limited by positive event scarcity.

---

## 7. Scope Limitations

### 7.1 No SHAP or Explainability Built In

Explainability (SHAP values, feature attribution, risk explanations) is out of scope for Part 1 and is Kshitij's (Part 2) responsibility. The model artifact (`climateguard_final_model.joblib`) exposes a standard scikit-learn RandomForestClassifier, which is compatible with SHAP TreeExplainer. See `docs/part2_integration_contract.md` for access instructions.

### 7.2 No API, Backend, or Real-Time Integration

No web server, REST API, or real-time data pipeline exists in this repository. ETL integration and backend deployment are Pradnesh's (Part 3) responsibility. See `docs/part3_integration_contract.md` for the integration specification.

### 7.3 No Ensemble or Uncertainty Estimates

The model outputs a single probability value per prediction. No prediction intervals, ensemble variance, or calibration curves are provided. Probability calibration was not explicitly optimised — PR-AUC reflects ranking quality, not calibration quality.

---

## 8. Summary Table

| Limitation | Severity | Affects |
|---|---|---|
| ERA5 reanalysis vs IMD station data | High | All cities |
| IMD-inspired label, not certified ground truth | High | All labels |
| Mumbai: zero positives, unreliable predictions | High | Mumbai only |
| Ahmedabad: no test-set positives | Medium | Ahmedabad evaluation |
| qualifying_day feature coupling | Medium | Feature interpretation |
| 30+ day history required for tmax_departure_zscore | Medium | ETL bootstrap |
| Precision = 0.58, ~42% false alarm rate | Medium | All predictions |
| No drift detection or retraining pipeline | Medium | Long-term deployment |
| ROC-AUC inflated by class imbalance | Low | Metric reporting |
| 2023 low performance (sparse positives) | Low | Year-level analysis |
| SHAP/explainability not in Part 1 | Scope | Part 2 responsibility |
| No API/backend | Scope | Part 3 responsibility |

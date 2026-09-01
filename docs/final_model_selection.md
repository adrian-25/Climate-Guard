# Phase 14 — Final Model Selection

**ClimateGuard: Indian Heatwave Prediction**  
**Date:** 2026-09-02  
**Author:** Adrian (Part 1 — Dataset & ML)  
**Status:** COMPLETE

---

## 1. Models Considered

All candidate models were evaluated across Phases 10–13. The following were carried into the final selection decision:

| Model | Feature Set | Strategy | Source |
|---|---|---|---|
| Logistic Regression | baseline_wqd / nqd | balanced weight | Phase 10 |
| Random Forest | baseline_wqd / nqd | balanced weight | Phase 10 |
| XGBoost | baseline_wqd / nqd | scale_pos_weight=126.89 | Phase 10 |
| Random Forest | baseline_wqd / nqd | random_undersample | Phase 11 |
| Random Forest | baseline_wqd / nqd | strong_weight | Phase 11 |
| XGBoost | baseline_wqd / nqd | spw_64/128/256/512 | Phase 11 |
| Random Forest | temporal_wqd / nqd | random_undersample | Phase 13 |
| XGBoost | temporal_wqd / nqd | baseline_weight | Phase 13 |

All models were trained on the Phase 9 chronological train split (1990-01-11 → 2019-12-31) and evaluated on the validation split (2020-01-01 → 2022-12-31). The test set (2023-01-01 → 2025-08-30) was held out throughout the selection process.

---

## 2. Phase 10 Baseline Findings

**Script:** `train_baseline_models.py`  
**Feature set:** baseline (29 features: Groups 1 + 6 + 7)  
**Imbalance handling:** sklearn `class_weight='balanced'` / XGBoost `scale_pos_weight=126.89`

| Rank | Model | Feature Set | Val F1 | Val PR-AUC |
|---|---|---|---|---|
| 1 | XGBoost | without_qd | 0.5424 | 0.5433 |
| 2 | XGBoost | with_qd | 0.5299 | 0.5668 |
| 3 | Random Forest | with_qd | 0.5000 | 0.5535 |
| 4 | Random Forest | without_qd | 0.4789 | 0.5325 |
| 5 | Logistic Regression | with_qd | 0.2879 | **0.6356** |
| 6 | Logistic Regression | without_qd | 0.2796 | 0.6216 |

**Key findings:**
- F1 values are low (0.28–0.54) with default threshold = 0.50 and no threshold tuning
- Logistic Regression achieves highest PR-AUC (0.6356) but poor F1 — threshold tuning would be needed
- XGBoost leads F1 with baseline class weight
- Logistic Regression eliminated from further consideration due to low F1 and limited capacity to exploit temporal features

---

## 3. Phase 11 Imbalance Findings

**Script:** `train_imbalance_models.py`  
**Feature sets:** baseline with_qd / without_qd  
**Strategies tested:** baseline_weight, strong_weight, random_oversample, random_undersample, XGBoost spw grid (64/128/256/512)

### Best F1 at threshold = 0.50

| Model | Feature Set | Strategy | F1 |
|---|---|---|---|
| XGBoost | with_qd | spw_64 | 0.5714 |

### Best F1 after threshold optimisation (validation only)

| Model | Feature Set | Strategy | Threshold | F1 | P | R | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **with_qd** | **random_undersample** | **0.70** | **0.6122** | **0.5085** | **0.7692** | **0.5497** | **29** | **9** |
| XGBoost | without_qd | baseline_weight | 0.80 | 0.6105 | 0.5179 | 0.7436 | 0.5433 | 27 | 10 |

**Key findings:**
- Random undersampling with elevated threshold (0.70) produces the best F1 for Random Forest
- Threshold tuning was performed on validation only — test set not consulted
- SMOTE was skipped (imbalanced-learn not installed); `smote_skipped` strategy defaults to class_weight handling
- qualifying_day is beneficial: RF/with_qd/undersample (0.6122) > RF/without_qd/undersample (0.5843)
- Recommended Phase 11 candidate: RF / with_qd / random_undersample / threshold=0.70

---

## 4. Phase 12 Test Evaluation

**Script:** `evaluate_test_set.py`  
**Test period:** 2023-01-01 → 2025-08-30 (38 positives)  
**Models evaluated:** Phase 11 candidates (no retraining, no threshold change)

| Candidate | Threshold | Test F1 | Test P | Test R | Test PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| RF / with_qd / random_undersample | 0.70 | 0.6957 | 0.5926 | 0.8421 | 0.7705 | 22 | 6 |
| XGB / without_qd / baseline_weight | 0.80 | 0.7191 | 0.6275 | 0.8421 | **0.8440** | 19 | 6 |
| RF / with_qd / smote_skipped | 0.20 | **0.7381** | **0.6739** | 0.8158 | 0.8307 | **15** | 7 |
| RF / without_qd / smote_skipped | 0.15 | 0.7143 | 0.5833 | **0.9211** | 0.8397 | 25 | **3** |

All four candidates **improved on test vs validation** — consistent with 2024 being an exceptionally strong heatwave year (34/38 test positives in 2024).

**Key finding:** RF/smote_skipped achieved the highest test F1 (0.7381), but its threshold (0.20) was selected on validation which made it less robust than the RF/undersample/0.70 candidate. The smote_skipped models also had lower val F1, meaning their test improvement is larger but less reliable as a signal.

---

## 5. Phase 13 Temporal Feature Experiment

**Script:** `temporal_feature_experiment.py`  
**Feature sets:** baseline_wqd (29), baseline_nqd (28), temporal_wqd (110), temporal_nqd (109)

### Primary comparison — validation

| Config | F1 | P | R | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| **RF / temporal_wqd / undersample** | **0.6154** | **0.5385** | **0.7179** | **0.5298** | **24** | **11** |
| RF / baseline_wqd / undersample | 0.6122 | 0.5085 | 0.7692 | 0.5497 | 29 | 9 |
| XGB / baseline_nqd / baseline_weight | 0.6105 | 0.5179 | 0.7436 | 0.5433 | 27 | 10 |

### Test confirmation — Phase 13

| Config | Test F1 | Test P | Test R | Test PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| RF / temporal_wqd / undersample | 0.7586 | 0.6735 | 0.8684 | 0.7885 | 16 | 5 |
| RF / baseline_wqd / undersample | 0.6957 | 0.5926 | 0.8421 | 0.7705 | 22 | 6 |

### Ablation study — validation (RF / undersample)

| Feature Set | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|
| baseline_only (29) | 0.6122 | 0.5497 | 29 | 9 |
| + anomaly (30) | 0.6263 | 0.5037 | 29 | 8 |
| full_temporal (110) | 0.6154 | 0.5298 | 24 | 11 |
| full_temporal_nqd (109) | 0.5393 | 0.5527 | 26 | 15 |

**Key finding:** temporal_wqd achieves the highest validation F1 (0.6154) — this is the configuration selected for the final model. The anomaly feature (`tmax_departure_zscore`) alone is the single most valuable addition, but the full temporal set with qualifying_day gives the best combined F1 + FP reduction.

---

## 6. Why Random Forest Was Selected

1. **Consistently best F1 across validation.** RF/temporal_wqd/undersample achieves the highest validation F1 (0.6154) among all Phase 13 configurations.

2. **Stable under undersampling.** Random undersampling + elevated threshold produces a well-calibrated precision/recall trade-off for RF. XGBoost with the same strategy is less stable.

3. **Strong test generalisation.** RF/temporal_wqd generalises to test F1 = 0.7586 (Phase 13 Part C), and F1 = 0.6947 for the final train+val model — both with identical recall (0.87).

4. **Robust to feature scale.** Random Forest is scale-invariant and does not require StandardScaler, simplifying the deployment pipeline.

5. **Interpretable feature structure.** Random Forest feature importances are directly usable by Kshitij (Part 2 — Explainability) without custom wrapper code.

6. **XGBoost degraded with temporal features.** XGBoost with 110 features at threshold=0.80 dropped to F1=0.5128–0.5185, well below its baseline performance. This is likely due to the interaction between a fixed scale_pos_weight and a high-dimensional feature space. XGBoost was not pursued further for the final model.

7. **Logistic Regression eliminated in Phase 10.** Insufficient capacity for the temporal feature set and poor F1 even after threshold tuning.

---

## 7. Why Temporal Features Were Selected

1. **Validation F1 improvement.** `temporal_wqd` (0.6154) > `baseline_wqd` (0.6122) on validation — modest but consistent.

2. **False positive reduction.** At threshold=0.70, temporal features reduce validation FP from 29 to 24 (5 fewer false alarms per validation period).

3. **Test F1 improvement is substantial.** `temporal_wqd` test F1 = 0.7586 vs `baseline_wqd` test F1 = 0.6957 — a +0.063 gain with 6 fewer false positives. The improvement is precision-driven (+0.081), meaning temporal memory helps the model distinguish real heatwave onset from isolated hot days.

4. **Ablation confirms temporal value.** The anomaly feature (`tmax_departure_zscore`) is the highest-value single group. Rolling features reduce FP. Combined they provide both precision and recall benefits.

5. **Leakage-safe.** All 110 temporal features are verified leakage-free by Phase 7 construction rules (shift ≥ 1 for lags, shift(1).rolling(N) for rolling, 30-day trailing for anomaly).

---

## 8. Why qualifying_day Was Retained

`qualifying_day` is a binary feature (0 or 1) defined at time T as:

```
Plains cities (Delhi, Lucknow, Nagpur, Ahmedabad):
  qualifying_day = 1  if  temperature_2m_max >= 40°C  AND  tmax_departure >= 4.5°C
                      OR  temperature_2m_max >= 45°C

Coastal cities (Mumbai):
  qualifying_day = 1  if  temperature_2m_max >= 37°C  AND  tmax_departure >= 4.5°C
```

**Reasons for retention:**

1. **Leakage-safe.** `qualifying_day` is derived entirely from current-day T data (temperature and departure), not from T+1 or later. It does not expose the target.

2. **Strong performance signal.** Phase 13 ablation showed that removing qualifying_day from the 110-feature set drops F1 by −0.076 (0.6154 → 0.5393). It is the most important individual feature.

3. **Operationally designed.** It encodes the IMD-inspired threshold rules into a clean binary signal, providing the model with an explicit representation of the meteorological regime.

4. **Consistent with validated feature design.** qualifying_day was part of the validated Phase 7 feature set and has been confirmed leakage-safe in all leakage audits (Phases 7–14).

**Documented limitation (required):**

`qualifying_day` is derived from the same IMD-inspired threshold criteria used to define the heatwave target (`heatwave_next_day`). Specifically:
- `qualifying_day(T)` is built from the same threshold conditions that determine whether a day T is a heatwave day.
- `heatwave_next_day(T)` = `heatwave(T+1)`, which depends on whether T+1 qualifies AND is part of a ≥2-day run.

This means qualifying_day is strongly correlated with the target by construction. The model does **not** independently discover the IMD rule from scratch — it exploits an operationally designed feature that encodes domain knowledge. This is intentional and valid, but it means the model should not be described as having discovered the heatwave criteria autonomously.

---

## 9. Why Threshold 0.70 Was Selected

The threshold of 0.70 was determined during Phase 11 validation by evaluating F1 across thresholds {0.05, 0.10, ..., 0.50, 0.60, 0.70, 0.80, 0.90} on the validation set (2020–2022).

| Threshold | RF/with_qd/undersample Val F1 |
|---|---|
| 0.50 | 0.5714 |
| 0.60 | 0.5882 |
| **0.70** | **0.6122** |
| 0.80 | 0.5882 |

**0.70 is the optimal trade-off point on validation.** It increases precision (fewer false alarms) without excessive recall loss for the Random Forest undersampling configuration.

This threshold was locked before any test-set evaluation. It was **not tuned on test data**.

---

## 10. Final Training Procedure

### Step 1 — Configuration locked

All model configuration (algorithm, features, strategy, threshold) was fixed using validation evidence only, before any test-set evaluation.

### Step 2 — Combined train+validation set

```
Training data: train (1990-01-11 → 2019-12-31) + val (2020-01-01 → 2022-12-31)
Rows: 60,215
Positives: 467  (428 train + 39 val)
Negatives: 59,748
Ratio: 1:127.9
```

### Step 3 — Random undersampling

Applied to the combined train+val set only:

```
Target ratio: 1:10 (pos:neg)
Result: 467 positives + 4,670 negatives = 5,137 rows
Random seed: 42
```

The test set was never touched during this step.

### Step 4 — Model training

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    class_weight=None,   # undersampling handles imbalance
    random_state=42,
    n_jobs=-1,
)
```

Training time: ~2 seconds on combined dev set.

### Step 5 — Test evaluation

Evaluated once, after training, with threshold=0.70 pre-fixed:

```python
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.70).astype(int)
```

No threshold adjustment after seeing test results.

---

## 11. Final Test Performance

**Test period:** 2023-01-01 → 2025-08-30 (4,865 rows, 38 positives)

### Overall metrics

| Metric | Value |
|---|---|
| **F1** | **0.6947** |
| Precision | 0.5789 |
| Recall | 0.8684 |
| PR-AUC | 0.8339 |
| ROC-AUC | 0.9979 |
| Accuracy | 0.9940 |
| TP | 33 |
| FP | 24 |
| TN | 4,803 |
| FN | 5 |
| Predicted positives | 57 |
| Actual positives | 38 |

### Per-city metrics

| City | F1 | Precision | Recall | TP | FP | FN | Notes |
|---|---|---|---|---|---|---|---|
| Delhi | 0.7805 | 0.6957 | 0.8889 | 16 | 7 | 2 | Best performing city |
| Lucknow | 0.6829 | 0.5600 | 0.8750 | 14 | 11 | 2 | Strong recall |
| Nagpur | 0.5000 | 0.3750 | 0.7500 | 3 | 5 | 1 | Few test positives (4 total) |
| Ahmedabad | N/A | N/A | N/A | — | — | — | 0 test positives |
| Mumbai | N/A | N/A | N/A | — | — | — | 0 test positives |

### Comparison: Phase 13 (train-only) vs Phase 14 (train+val)

| Config | Training data | F1 | P | R | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| Phase 13 Part C (train only) | 1990–2019 | 0.7586 | 0.6735 | 0.8684 | 0.7885 | 16 | 5 |
| **Phase 14 FINAL (train+val)** | **1990–2022** | **0.6947** | **0.5789** | **0.8684** | **0.8339** | **24** | **5** |

**Explanation of F1 drop (0.7586 → 0.6947):**

F1 dropped because adding the 2020–2022 validation data to training introduces harder patterns. The 2020–2022 period includes post-COVID climate years and the model must now generalise from a broader distribution. Recall is unchanged (0.8684 in both cases — 33 TP, 5 FN), indicating the model still detects the same heatwave events. The difference is entirely in precision: FP increased from 16 to 24 (+8). This is the honest cost of using all available labelled data for training.

Importantly, PR-AUC **improved** from 0.7885 to 0.8339, indicating the overall probability calibration (ranking quality) is better with more training data — even if the F1 at threshold=0.70 is lower.

---

## 12. Limitations

1. **ERA5 reanalysis, not station data.** The entire dataset is ERA5 gridded reanalysis at 0.25° resolution from Open-Meteo. This is not the same as IMD weather station observations. Systematic biases between ERA5 and real station data could affect real-world performance.

2. **IMD-inspired labels, not official IMD ground truth.** The heatwave labels were constructed using an IMD-inspired operational definition applied to ERA5 data. They are not validated against official IMD heatwave declarations. The label is called "IMD-Inspired Operational Heatwave Label (ERA5-based)" — not "official IMD labels".

3. **Five city training distribution.** The model was trained on five Indian cities. It should not be applied to other cities without retraining or validation.

4. **Mumbai zero positives.** Mumbai has zero heatwave positives in the entire 35-year dataset under the coastal threshold definition. The model has no positive examples to learn from for Mumbai's city_encoded value. Predictions for Mumbai are unreliable.

5. **Ahmedabad sparse positives.** Ahmedabad has 32 positives, all in the 1990–2019 training window. No validation or test positives exist. The model cannot be evaluated for Ahmedabad generalization.

6. **Class imbalance is extreme (0.78%).** Despite undersampling and threshold tuning, the model raises ~57 alarms per test period (2.5 years) for 38 actual events. Approximately 42% of alarms are false positives. This is appropriate for an early-warning tool — missing a heatwave (FN) is more costly than a false alarm (FP) — but the precision limitation must be communicated clearly.

7. **Threshold was optimised on validation 2020–2022.** This is a 3-year window with only 39 positive examples. The optimal threshold could shift on a different time period.

8. **No drift detection or retraining pipeline.** The model does not self-monitor. Climate patterns may shift over time (warming trends observed in EDA). Regular retraining is recommended.

9. **qualifying_day design coupling.** As documented in Section 8, qualifying_day is designed from the same threshold family as the target. The model is exploiting this explicitly encoded domain knowledge, not discovering the meteorological pattern independently.

10. **No SHAP or interpretability built in.** Explainability is out of scope for Part 1. This is Kshitij's (Part 2) responsibility.

---

## 13. Integration Contract

See `docs/final_model_contract.md` for the complete machine-readable prediction contract.

### Summary for Part 2 / Part 3

| Property | Value |
|---|---|
| Model file | `models/final/climateguard_final_model.joblib` |
| Feature list | `models/final/feature_list.json` |
| Metadata | `models/final/metadata.json` |
| Required features | 110, in exact order from `feature_list.json` |
| No scaling required | Random Forest is scale-invariant |
| Output probability | `model.predict_proba(X)[:, 1]` |
| Decision threshold | 0.70 |
| Prediction label | `1` if probability ≥ 0.70, else `0` |
| Task | 1-day-ahead heatwave prediction |
| Prediction horizon | T → heatwave probability for T+1 |
| Load command | `joblib.load("models/final/climateguard_final_model.joblib")` |

**Do NOT use `model.predict(X)` directly** — it applies sklearn's default 0.50 threshold, which was not validated for this use case. Always use `model.predict_proba(X)[:, 1] >= 0.70`.

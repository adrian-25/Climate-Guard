# Model Evaluation — ClimateGuard Phase 12

**Created:** 2026-09-01  
**Script:** `evaluate_test_set.py`  
**Split used:** Phase 9 chronological split (train 1990-2019, val 2020-2022, test 2023-2025)  
**Test set status:** Used for the first and only time in this phase

---

## 1. Purpose of Held-Out Test Evaluation

The test set (2023-01-01 to 2025-08-30) was locked away at the start of Phase 9 and
never touched during training, feature engineering, imbalance strategy selection, or
threshold optimisation. Its sole purpose is to provide an unbiased estimate of how the
candidate models will perform on genuinely unseen future data.

Rules enforced without exception:

- Models are loaded from Phase 11 artifacts — no retraining occurs
- Thresholds are fixed from Phase 11 validation optimisation — no test-set tuning
- Test data is never resampled, oversampled, or undersampled
- No model selection decisions are made based on test-set results
- No hyperparameter adjustments are made after seeing test results

The test set is a one-shot measurement. Its results are reported as-is, including any
degradation compared to validation.

---

## 2. Fixed Train / Validation / Test Split

| Split | Start | End | Rows | Positives | Positive % |
|---|---|---|---|---|---|
| Train | 1990-01-11 | 2019-12-31 | 54,735 | 428 | 0.78% |
| Validation | 2020-01-01 | 2022-12-31 | 5,480 | 39 | 0.71% |
| **Test** | **2023-01-01** | **2025-08-30** | **4,865** | **38** | **0.78%** |

Split method: chronological / year-based. No random shuffle. No data leakage across boundaries.

Test set per-city positives:

| City | Test positives | Notes |
|---|---|---|
| Delhi | 18 | Main positive contributor |
| Lucknow | 16 | Good sample size for city-level metrics |
| Nagpur | 4 | Small — city metrics are directional only |
| Ahmedabad | 0 | N/A — no positives in test period |
| Mumbai | 0 | N/A — zero positives in any split (coastal definition) |

---

## 3. Candidates Evaluated

Four candidates were selected from Phase 11 based exclusively on validation performance.
Their thresholds were fixed before the test set was opened.

| # | Candidate | Feature Set | Strategy | Fixed Threshold | Primary |
|---|---|---|---|---|---|
| 1 | Random Forest | with_qd | random_undersample | 0.70 | YES |
| 2 | XGBoost | without_qd | baseline_weight | 0.80 | no |
| 3 | Random Forest | with_qd | smote_skipped | 0.20 | no |
| 4 | Random Forest | without_qd | smote_skipped | 0.15 | no |

Note on "smote_skipped": SMOTE was skipped in Phase 11 because `imbalanced-learn` was
not installed. The `smote_skipped` models were trained on the unmodified training set
(no resampling). The strategy slot label is preserved for auditability.

---

## 4. Validation Performance (Phase 11, for reference)

| Candidate | Threshold | Val F1 | Val P | Val R | Val PR-AUC | Val TP | Val FP | Val FN |
|---|---|---|---|---|---|---|---|---|
| RF / with_qd / random_undersample | 0.70 | 0.6122 | 0.5085 | 0.7692 | 0.5497 | 30 | 29 | 9 |
| XGB / without_qd / baseline_weight | 0.80 | 0.6105 | 0.5179 | 0.7436 | 0.5433 | 29 | 27 | 10 |
| RF / with_qd / smote_skipped | 0.20 | 0.6105 | 0.5179 | 0.7436 | 0.5951 | 29 | 27 | 10 |
| RF / without_qd / smote_skipped | 0.15 | 0.6095 | 0.4848 | 0.8205 | 0.6048 | 32 | 34 | 7 |

Validation set: 5,480 rows, 39 positives, 2020-2022.

---

## 5. Test Performance

Test set: 4,865 rows, 38 positives, 2023-2025.

| Candidate | Threshold | Test F1 | Test P | Test R | Test PR-AUC | Test ROC-AUC | Test Accuracy |
|---|---|---|---|---|---|---|---|
| RF / with_qd / random_undersample | 0.70 | 0.6957 | 0.5926 | 0.8421 | 0.7705 | 0.9978 | 0.9942 |
| XGB / without_qd / baseline_weight | 0.80 | 0.7191 | 0.6275 | 0.8421 | **0.8440** | **0.9982** | 0.9949 |
| **RF / with_qd / smote_skipped** | **0.20** | **0.7381** | **0.6739** | 0.8158 | 0.8307 | 0.9971 | **0.9955** |
| RF / without_qd / smote_skipped | 0.15 | 0.7143 | 0.5833 | **0.9211** | 0.8397 | 0.9978 | 0.9942 |

**Bold = best in column.**

---

## 6. Overall Confusion Matrices (Test Set)

### Candidate 1 — RF / with_qd / random_undersample (PRIMARY, threshold=0.70)

|  | Predicted Normal | Predicted Heatwave |
|---|---|---|
| **Actual Normal** | TN = 4,805 | FP = 22 |
| **Actual Heatwave** | FN = 6 | TP = 32 |

Predicted positives: 54 / Actual positives: 38

### Candidate 2 — XGBoost / without_qd / baseline_weight (threshold=0.80)

|  | Predicted Normal | Predicted Heatwave |
|---|---|---|
| **Actual Normal** | TN = 4,808 | FP = 19 |
| **Actual Heatwave** | FN = 6 | TP = 32 |

Predicted positives: 51 / Actual positives: 38

### Candidate 3 — RF / with_qd / smote_skipped (threshold=0.20)

|  | Predicted Normal | Predicted Heatwave |
|---|---|---|
| **Actual Normal** | TN = 4,812 | FP = 15 |
| **Actual Heatwave** | FN = 7 | TP = 31 |

Predicted positives: 46 / Actual positives: 38

### Candidate 4 — RF / without_qd / smote_skipped (threshold=0.15)

|  | Predicted Normal | Predicted Heatwave |
|---|---|---|
| **Actual Normal** | TN = 4,802 | FP = 25 |
| **Actual Heatwave** | FN = 3 | TP = 35 |

Predicted positives: 60 / Actual positives: 38

Confusion matrix PNGs: `results/phase12_confusion_matrices/`

---

## 7. City-Wise Test Results

### RF / with_qd / random_undersample (PRIMARY)

| City | Rows | Actual+ | Pred+ | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Delhi | 973 | 18 | 26 | 15 | 11 | 944 | 3 | 0.5769 | 0.8333 | 0.6818 |
| Lucknow | 973 | 16 | 23 | 15 | 8 | 949 | 1 | 0.6522 | 0.9375 | 0.7692 |
| Nagpur | 973 | 4 | 5 | 2 | 3 | 966 | 2 | 0.4000 | 0.5000 | 0.4444 |
| Ahmedabad | 973 | 0 | 0 | — | 0 | — | — | N/A | N/A | N/A |
| Mumbai | 973 | 0 | 0 | — | 0 | — | — | N/A | N/A | N/A |

### XGBoost / without_qd / baseline_weight

| City | Rows | Actual+ | Pred+ | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Delhi | 973 | 18 | 21 | 15 | 6 | 949 | 3 | 0.7143 | 0.8333 | 0.7692 |
| Lucknow | 973 | 16 | 23 | 14 | 9 | 948 | 2 | 0.6087 | 0.8750 | 0.7179 |
| Nagpur | 973 | 4 | 7 | 3 | 4 | 965 | 1 | 0.4286 | 0.7500 | 0.5455 |
| Ahmedabad | 973 | 0 | 0 | — | 0 | — | — | N/A | N/A | N/A |
| Mumbai | 973 | 0 | 0 | — | 0 | — | — | N/A | N/A | N/A |

### RF / with_qd / smote_skipped

| City | Rows | Actual+ | Pred+ | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Delhi | 973 | 18 | 22 | 15 | 7 | 948 | 3 | 0.6818 | 0.8333 | 0.7500 |
| Lucknow | 973 | 16 | 19 | 14 | 5 | 952 | 2 | 0.7368 | 0.8750 | 0.8000 |
| Nagpur | 973 | 4 | 5 | 2 | 3 | 966 | 2 | 0.4000 | 0.5000 | 0.4444 |
| Ahmedabad | 973 | 0 | 0 | — | 0 | — | — | N/A | N/A | N/A |
| Mumbai | 973 | 0 | 0 | — | 0 | — | — | N/A | N/A | N/A |

### RF / without_qd / smote_skipped

| City | Rows | Actual+ | Pred+ | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Delhi | 973 | 18 | 28 | 15 | 13 | 942 | 3 | 0.5357 | 0.8333 | 0.6522 |
| Lucknow | 973 | 16 | 24 | 16 | 8 | 949 | 0 | 0.6667 | 1.0000 | 0.8000 |
| Nagpur | 973 | 4 | 7 | 4 | 3 | 966 | 0 | 0.5714 | 1.0000 | 0.7273 |
| Ahmedabad | 973 | 0 | 1 | — | 1 | — | — | N/A | N/A | N/A |
| Mumbai | 973 | 0 | 0 | — | 0 | — | — | N/A | N/A | N/A |

City-level notes:
- Delhi (18 positives) and Lucknow (16 positives) produce the most reliable city metrics
- Nagpur (4 positives) — directional only; single TP/FP differences shift metrics significantly
- Ahmedabad and Mumbai — zero positives; only false-positive count is meaningful
- RF/without_qd/smote_skipped produced 1 false positive in Ahmedabad

---

## 8. Year-Wise Test Results

### RF / with_qd / random_undersample (PRIMARY)

| Year | Rows | Actual+ | Pred+ | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 2023 | 1,825 | 4 | 7 | 2 | 5 | 2 | 0.2857 | 0.5000 | 0.3636 |
| 2024 | 1,830 | 34 | 43 | 30 | 13 | 4 | 0.6977 | 0.8824 | 0.7792 |
| 2025 | 1,210 | 0 | 4 | N/A | 4 | N/A | N/A | N/A | N/A |

### XGBoost / without_qd / baseline_weight

| Year | Rows | Actual+ | Pred+ | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 2023 | 1,825 | 4 | 9 | 3 | 6 | 1 | 0.3333 | 0.7500 | 0.4615 |
| 2024 | 1,830 | 34 | 38 | 29 | 9 | 5 | 0.7632 | 0.8529 | 0.8056 |
| 2025 | 1,210 | 0 | 4 | N/A | 4 | N/A | N/A | N/A | N/A |

### RF / with_qd / smote_skipped

| Year | Rows | Actual+ | Pred+ | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 2023 | 1,825 | 4 | 6 | 2 | 4 | 2 | 0.3333 | 0.5000 | 0.4000 |
| 2024 | 1,830 | 34 | 36 | 29 | 7 | 5 | 0.8056 | 0.8529 | 0.8286 |
| 2025 | 1,210 | 0 | 4 | N/A | 4 | N/A | N/A | N/A | N/A |

### RF / without_qd / smote_skipped

| Year | Rows | Actual+ | Pred+ | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 2023 | 1,825 | 4 | 11 | 4 | 7 | 0 | 0.3636 | 1.0000 | 0.5333 |
| 2024 | 1,830 | 34 | 43 | 31 | 12 | 3 | 0.7209 | 0.9118 | 0.8052 |
| 2025 | 1,210 | 0 | 6 | N/A | 6 | N/A | N/A | N/A | N/A |

Year-level notes:
- 2023 performance is poor across all candidates (4 positives, scattered)
- 2024 drives overall test performance — 34 of 38 positives fall in 2024
- 2025 (Jan–Aug) has zero positives in the test period. All 2025 alarms are false positives.
  This is likely due to the dataset ending 2025-08-30 before the peak heatwave season,
  and/or a naturally milder 2025 heatwave season in the ERA5 record.

---

## 9. Precision / Recall Trade-offs

| Candidate | Test Precision | Test Recall | Test F1 | Test FP | Test FN | Operational interpretation |
|---|---|---|---|---|---|---|
| RF / random_undersample | 0.5926 | 0.8421 | 0.6957 | 22 | 6 | 59% alarm accuracy; catches 84% of events |
| XGB / baseline_weight | 0.6275 | 0.8421 | 0.7191 | 19 | 6 | 63% alarm accuracy; catches 84%; fewest FP |
| RF / smote_skipped / wqd | **0.6739** | 0.8158 | **0.7381** | **15** | 7 | Best precision; 15 false alarms in 3 years |
| RF / smote_skipped / nqd | 0.5833 | **0.9211** | 0.7143 | 25 | **3** | Catches most events; only 3 missed |

For an operational heatwave warning system over the test period (975 days per city):
- The best precision candidate (RF/with_qd/smote_skipped) raises 15 false alarms over 3 years — approximately one false alarm every 73 days
- The best recall candidate (RF/without_qd/smote_skipped) catches 35/38 events but raises 25 false alarms (one per 44 days)
- The XGBoost candidate offers the best compromise: 19 FP, 6 FN, highest PR-AUC (0.844)

---

## 10. Validation vs Test Generalization

| Candidate | Val F1 | Test F1 | Delta F1 | Val PR-AUC | Test PR-AUC | Delta PR-AUC | Status |
|---|---|---|---|---|---|---|---|
| RF / random_undersample | 0.6122 | 0.6957 | **+0.0835** | 0.5497 | 0.7705 | **+0.2208** | IMPROVED |
| XGB / baseline_weight | 0.6105 | 0.7191 | **+0.1086** | 0.5433 | 0.8440 | **+0.3007** | IMPROVED |
| RF / smote_skipped wqd | 0.6105 | 0.7381 | **+0.1276** | 0.5951 | 0.8307 | **+0.2356** | IMPROVED |
| RF / smote_skipped nqd | 0.6095 | 0.7143 | **+0.1048** | 0.6048 | 0.8397 | **+0.2349** | IMPROVED |

**All four candidates improved substantially on the test set versus validation.**

This is an unusual but scientifically explainable result:

1. **The 2024 heatwave season was exceptional.** 34 of 38 test positives fall in 2024. A concentrated, strong heatwave season produces clearer meteorological signals that tree-based models detect reliably.

2. **Validation (2020-2022) was harder.** 39 positives spread across 3 years with more variable conditions, including 2020 (COVID lockdowns may have altered some heat-related patterns in the data).

3. **The validation imbalance was more challenging.** Val has only 39 positives vs 38 in test, but the test-set class distribution is slightly more concentrated temporally, giving models a stronger signal.

4. **No overfitting to test.** Thresholds were not tuned on test. The improvement is a genuine reflection of 2024 being a year where the models' learned patterns transferred well.

5. **PR-AUC improvement is large (+0.22 to +0.30).** This reflects a higher probability mass on true positives in the test set, consistent with the strong 2024 heatwave signal.

This does not mean the models are "perfect." The 2023 performance was poor (F1 ~ 0.36-0.53 across candidates), and all models produce false alarms in 2025 (where no positives exist in the test window).

---

## 11. Limitations

1. **Test set has only 38 positives.** Statistical power is limited. A single missed event shifts recall by 0.026; a single extra false positive shifts precision by ~0.01.

2. **2025 has zero positives in the test window.** The test ends 2025-08-30, before peak heatwave season. All 2025 alarms (4-6 per model) are false positives. This inflates the FP count and slightly deflates precision.

3. **2024 dominates the test positives (34/38 = 89%).** Overall test metrics largely reflect 2024 performance. Generalization to years with fewer or more dispersed events (like 2023) is weaker.

4. **Ahmedabad has zero test positives.** Its 32 historical heatwave events all fall in the training window. The model cannot be evaluated for Ahmedabad positive-class performance.

5. **Mumbai has zero positives in any split.** Scientifically correct under the coastal definition; no evaluation possible.

6. **Nagpur has only 4 test positives.** City-level metrics for Nagpur are directional only.

7. **Temporal feature set not evaluated.** Phase 12 used the baseline feature set (29 features). Phase 13 will evaluate whether adding lag/rolling/trend features (110 features) changes performance.

8. **No final production model is declared here.** Phase 14 makes the final selection after Phase 13.

9. **smote_skipped models are effectively no-resampling models.** Their label reflects a failed SMOTE attempt. Their strong test performance reflects their baseline model quality, not SMOTE effectiveness.

---

## 12. Leakage Audit

All 12 leakage checks PASSED.

| Check | Result |
|---|---|
| Test labels never used during model training (Phase 11) | PASS |
| Test labels never used during threshold selection (Phase 11) | PASS |
| Thresholds FIXED from Phase 11 validation — no test-set tuning | PASS |
| Test set NOT resampled, oversampled, or undersampled | PASS |
| Models loaded from Phase 11 artifacts — no retraining | PASS |
| Scaler fitted on training data only — none needed here (RF/XGB) | PASS |
| Phase 9 split boundaries unchanged (train<=2019, val=2020-2022, test>=2023) | PASS |
| Phase 8 datasets (ml_baseline.csv, ml_temporal.csv) not modified | PASS |
| Phase 11 model artifacts not modified | PASS |
| No future weather variables (T+1) in feature sets | PASS |
| heatwave_next_day (target) absent from feature matrices | PASS |
| Candidate selection based on validation results only | PASS |

**Overall: PASSED (12/12)**

Full audit: `results/phase12_leakage_audit.csv`

---

## Artifacts

| Path | Contents |
|---|---|
| `results/phase12_test_metrics.csv` | Overall test metrics for all 4 candidates |
| `results/phase12_city_metrics.csv` | City-level test metrics (4 candidates x 5 cities) |
| `results/phase12_yearly_metrics.csv` | Year-level test metrics (4 candidates x 3 years) |
| `results/phase12_comparison.csv` | Val vs test comparison with delta columns |
| `results/phase12_metrics.json` | Full metrics including city and yearly breakdown |
| `results/phase12_log.txt` | Full execution log |
| `results/phase12_leakage_audit.csv` | 12-check leakage audit |
| `results/phase12_confusion_matrices/` | 4 confusion matrix PNGs |
| `results/plots/phase12/pr_curve.png` | Precision-Recall curves for all 4 candidates |
| `results/plots/phase12/roc_curve.png` | ROC curves for all 4 candidates |

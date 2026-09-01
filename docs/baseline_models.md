# Baseline ML Models — ClimateGuard Phase 10

**Created:** 2026-09-01  
**Script:** `train_baseline_models.py`  
**Split used:** Phase 9 chronological split (train 1990–2019, val 2020–2022, test 2023–2025)  
**Test set status:** Held out — NOT used in Phase 10

---

## 1. Models Trained

Three classifiers were trained, each on two feature sets:

| Model | Implementation | Imbalance handling |
|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `class_weight='balanced'` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `class_weight='balanced'` |
| XGBoost | `xgboost.XGBClassifier` | `scale_pos_weight=126.89` |

Total models trained: **6** (3 models × 2 feature sets).

---

## 2. Features Used

### Feature Set A — with qualifying_day (29 features)
All 29 baseline features from `data/splits/baseline/X_train.csv`, including `qualifying_day`.

### Feature Set B — without qualifying_day (28 features)
Same as Set A with `qualifying_day` removed. Used for the qualifying_day controlled experiment.

The baseline feature set (Groups 1 + 6 + 7 from Phase 7) contains:
- 18 current-day weather variables (including `tmax_departure`, `tmax_normal`, `qualifying_day`)
- 7 calendar features (month, day_of_year, season_code, sin/cos encodings)
- 4 city features (city_encoded, is_coastal, latitude, longitude)

---

## 3. Preprocessing

**Logistic Regression only:**
- `StandardScaler` fitted exclusively on `X_train`
- Applied (transform only) to `X_val` and `X_test` using the fitted scaler
- Scaler saved to `models/phase10/logistic_regression/<feat_set>/scaler.joblib`

**Random Forest and XGBoost:**
- No scaling applied (tree-based models are scale-invariant)

**No preprocessing was fitted using validation or test data.**

---

## 4. Class Imbalance Handling at Baseline Stage

| Model | Method | Value |
|---|---|---|
| Logistic Regression | `class_weight='balanced'` | Weights inversely proportional to class frequency |
| Random Forest | `class_weight='balanced'` | Same |
| XGBoost | `scale_pos_weight` | 126.89 (= 54,307 negatives / 428 positives) |

These are conservative, well-established imbalance accommodations built into each model's training objective. They do **not** alter the dataset — no rows are added, removed, or synthetic examples created.

**SMOTE, undersampling, oversampling, and manual threshold tuning are deferred to Phase 11.**

---

## 5. Hyperparameters

### Logistic Regression
```python
C=1.0, max_iter=1000, solver='lbfgs',
class_weight='balanced', random_state=42
```

### Random Forest
```python
n_estimators=300, max_depth=10, min_samples_leaf=10,
class_weight='balanced', random_state=42, n_jobs=-1
```

### XGBoost
```python
n_estimators=300, max_depth=5, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
scale_pos_weight=126.89, eval_metric='logloss',
random_state=42, verbosity=0
```

These are conservative baseline parameters. No hyperparameter search was performed. Phase 11+ will tune these.

---

## 6. Validation Methodology

- **Split:** Chronological — all models evaluated on 2020–2022 (5,480 rows, 39 positives)
- **No reshuffling** of training or validation data at any point
- **Test set (2023–2025) was not used** for model selection, threshold selection, or any comparison
- Model selection and qualifying_day decision are based on validation metrics only

---

## 7. Metrics

Because the positive class is rare (~0.78%), accuracy is not a meaningful metric. The primary metrics are:

- **Precision** — of predicted positives, how many are true heatwave days
- **Recall** — of actual heatwave days, how many are detected
- **F1-score** — harmonic mean of precision and recall
- **PR-AUC (Average Precision)** — area under the precision-recall curve; most informative for rare classes
- **ROC-AUC** — area under the ROC curve
- Confusion matrix (TP, FP, TN, FN)
- Predicted positives vs actual positives

---

## 8. Validation Results

### Full comparison table

| Model | Feature Set | Precision | Recall | F1 | PR-AUC | ROC-AUC | Pred+ | Act+ |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | with_qd | 0.1689 | 0.9744 | 0.2879 | **0.6356** | 0.9942 | 225 | 39 |
| Logistic Regression | without_qd | 0.1625 | 1.0000 | 0.2796 | 0.6216 | 0.9936 | 240 | 39 |
| Random Forest | with_qd | 0.3505 | 0.8718 | 0.5000 | 0.5535 | 0.9948 | 97 | 39 |
| Random Forest | without_qd | 0.3301 | 0.8718 | 0.4789 | 0.5325 | 0.9944 | 103 | 39 |
| XGBoost | with_qd | 0.3974 | 0.7949 | 0.5299 | 0.5668 | 0.9943 | 78 | 39 |
| XGBoost | without_qd | 0.4051 | 0.8205 | **0.5424** | 0.5433 | 0.9940 | 79 | 39 |

**Best F1 on validation:** XGBoost without_qd — 0.5424  
**Best PR-AUC on validation:** Logistic Regression with_qd — 0.6356  
**ROC-AUC:** All models cluster around 0.994 — ROC-AUC is high across the board but misleading under extreme imbalance (dominated by the large number of true negatives)

**Confusion matrices:** `results/phase10_confusion_matrices/`

### Interpretation

All models achieve extremely high ROC-AUC (≈0.994) because ROC-AUC is inflated by the large negative class. PR-AUC is the more informative metric here.

Logistic Regression achieves the highest PR-AUC (0.6356) but at the cost of very low precision — it predicts 225 positives for 39 actual (81% are false alarms). This reflects a high-recall, low-precision strategy that is a natural consequence of `class_weight='balanced'` on a very imbalanced problem.

Random Forest and XGBoost produce a better precision-recall trade-off: predicting ~78–103 positives for 39 actual, with F1 scores of 0.48–0.54.

---

## 9. qualifying_day Experiment

`qualifying_day(T)` = 1 when today's Tmax meets the heatwave threshold and departure criterion. It is **leakage-safe** (computed from T, not T+1), but it is a direct precursor to the target.

| Model | F1 with_qd | F1 without_qd | Delta F1 | PR-AUC with_qd | PR-AUC without_qd | Delta PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.2879 | 0.2796 | +0.0083 | 0.6356 | 0.6216 | +0.014 |
| Random Forest | 0.5000 | 0.4789 | +0.0211 | 0.5535 | 0.5325 | +0.021 |
| XGBoost | 0.5299 | 0.5424 | -0.0125 | 0.5668 | 0.5433 | +0.0235 |

**Finding:** The effect of `qualifying_day` is **small but not negligible** for Random Forest (+0.02 F1, +0.02 PR-AUC). For XGBoost, removing it slightly improves F1 while the PR-AUC drops slightly. For Logistic Regression, it provides a marginal benefit in PR-AUC.

**Conclusion:** `qualifying_day` does not dramatically inflate model performance. The models are not simply learning a near-direct copy of the target. The feature provides marginal value and its inclusion/exclusion does not change the ranking of models. The final decision on whether to include it in the production feature set is deferred to Phase 12/14.

---

## 10. City-Wise Validation Results (with_qd, 2020–2022)

| City | Actual pos | Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| Delhi | 28 | Logistic Regression | 0.2727 | 0.9643 | 0.4252 | 0.6464 |
| Delhi | 28 | Random Forest | 0.4182 | 0.8214 | 0.5542 | 0.5728 |
| Delhi | 28 | XGBoost | 0.4773 | 0.7500 | 0.5833 | 0.5536 |
| Lucknow | 7 | Logistic Regression | 0.0959 | 1.0000 | 0.1750 | 0.8803 |
| Lucknow | 7 | Random Forest | 0.2800 | 1.0000 | 0.4375 | 0.8279 |
| Lucknow | 7 | XGBoost | 0.3333 | 1.0000 | 0.5000 | 0.8605 |
| Nagpur | 4 | Logistic Regression | 0.1176 | 1.0000 | 0.2105 | 0.4789 |
| Nagpur | 4 | Random Forest | 0.2500 | 1.0000 | 0.4000 | 0.4792 |
| Nagpur | 4 | XGBoost | 0.2500 | 0.7500 | 0.3750 | 0.4704 |
| Ahmedabad | 0 | — | N/A — no positive ground-truth examples | | | |
| Mumbai | 0 | — | N/A — no positive ground-truth examples | | | |

Notes:
- Delhi has the most positives in validation (28) and shows the most stable per-city metrics
- Lucknow metrics are noisy (only 7 positives) — the high PR-AUC there should be treated with caution
- Nagpur has only 4 positives — metrics are unreliable
- Ahmedabad and Mumbai have 0 positives in validation — city-level positive-class metrics are undefined

---

## 11. Mumbai Limitation

Mumbai has **zero positive heatwave events** under the IMD-inspired operational definition in all splits (train, validation, test). This is scientifically correct — Mumbai's maritime climate does not produce sustained heatwaves under the coastal threshold.

Mumbai was kept in training data. It contributes 10,947 negative-class training rows. Its presence means the model learns that Mumbai-like meteorological conditions (maritime, narrow Tmax range) are associated with non-heatwave outcomes.

No synthetic positives were created for Mumbai. Its zero-positive status is not an error.

---

## 12. Model Comparison Summary

Ranked by validation F1 (primary), PR-AUC (secondary):

| Rank | Model | Feature Set | Val F1 | Val PR-AUC |
|---|---|---|---|---|
| 1 | XGBoost | without_qd | 0.5424 | 0.5433 |
| 2 | XGBoost | with_qd | 0.5299 | 0.5668 |
| 3 | Random Forest | with_qd | 0.5000 | 0.5535 |
| 4 | Random Forest | without_qd | 0.4789 | 0.5325 |
| 5 | Logistic Regression | with_qd | 0.2879 | 0.6356 |
| 6 | Logistic Regression | without_qd | 0.2796 | 0.6216 |

If ranked by PR-AUC, Logistic Regression leads — but this comes at the cost of very low precision (high false alarm rate). For an operational heatwave warning system, precision matters: unnecessary alerts erode trust.

**No final model is declared in Phase 10.** These are baseline results. Phase 11 will address class imbalance more systematically, which will likely change rankings.

---

## 13. What Remains for Phase 11

Phase 11 (Class Imbalance) will:
- Systematically evaluate SMOTE, SMOTE-Tomek, class weights, and threshold tuning on the temporal feature set
- Test whether more aggressive imbalance handling improves precision without collapsing recall
- Establish the best imbalance strategy for each model family before moving to full evaluation in Phase 12

The test set remains untouched and is reserved for Phase 12/14 final evaluation only.

---

## Artifacts

| Path | Contents |
|---|---|
| `models/phase10/logistic_regression/with_qd/model.joblib` | Trained LR model |
| `models/phase10/logistic_regression/with_qd/scaler.joblib` | Fitted StandardScaler |
| `models/phase10/logistic_regression/with_qd/metadata.json` | Params, features, metrics |
| `models/phase10/logistic_regression/without_qd/` | Same for Set B |
| `models/phase10/random_forest/with_qd/` | RF artifacts |
| `models/phase10/random_forest/without_qd/` | RF artifacts |
| `models/phase10/xgboost/with_qd/` | XGBoost artifacts |
| `models/phase10/xgboost/without_qd/` | XGBoost artifacts |
| `results/phase10_metrics.json` | Full metrics for all 6 models |
| `results/phase10_model_comparison.csv` | Comparison table |
| `results/phase10_log.txt` | Full execution log |
| `results/phase10_confusion_matrices/` | 6 confusion matrix PNGs |

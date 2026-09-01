# Train / Validation / Test Split — ClimateGuard Phase 9

**Created:** 2026-09-01  
**Script:** `time_series_split.py`  
**Inputs:** `data/features/ml_baseline.csv`, `data/features/ml_temporal.csv` (Phase 8 validated artifacts, MD5 verified untouched)

---

## 1. Why Random Splitting Is Inappropriate

This project predicts **heatwave_next_day(T) = heatwave(T+1)** — a one-day-ahead forecast in a time series. Using random shuffling (e.g., `train_test_split(shuffle=True)`) would cause temporal leakage:

- Future observations would appear in the training set
- A model trained on shuffled data sees weather patterns from 2023 before being evaluated on 2020 — an unrealistic scenario with no real-world analogue
- Lag and rolling features (e.g., `temperature_2m_max_lag1`, `temperature_2m_max_roll7_mean`) are derived from preceding days; mixing split membership would make those features encode information from the wrong time period
- Evaluation metrics would be overly optimistic and would not reflect real-world deployment performance

For time-series forecasting, evaluation must always be **forward-looking**: train on the past, evaluate on the future. The model must never see future data during training.

---

## 2. Exact Date Boundaries

| Split | Start | End | Years covered |
|---|---|---|---|
| Train | 1990-01-11 | 2019-12-31 | 30 years |
| Validation | 2020-01-01 | 2022-12-31 | 3 years |
| Test | 2023-01-01 | 2025-08-30 | ~2.7 years |

The split is **year-based and inclusive**. All five cities share identical date boundaries.

The dataset starts on 1990-01-11 (not 1990-01-01) because Phase 7 dropped the first 10 rows per city to ensure complete lag and zscore windows, and Phase 8 dropped 3 additional rows per city for the zscore min_periods requirement.

---

## 3. Train Period

**1990-01-11 to 2019-12-31 — 30 years of historical data**

The longest available historical window gives the model the most diverse examples of heatwave and non-heatwave conditions across different climate regimes, ENSO cycles, and long-term warming trends. Ending before 2020 cleanly separates training data from the COVID-19 period, during which reduced anthropogenic activity produced unusual atmospheric conditions.

---

## 4. Validation Period

**2020-01-01 to 2022-12-31 — 3 years**

Strictly after the training period. Used for hyperparameter tuning, model selection, and early stopping without contaminating the final test evaluation. Deliberately kept to 3 years to preserve as much recent data as possible for the held-out test set.

---

## 5. Test Period

**2023-01-01 to 2025-08-30 — ~2.7 years**

The most recent available data, held out completely until final model evaluation. Represents genuine out-of-sample future data — the closest proxy to real-world deployment conditions. The 2025 end date reflects the extent of the ERA5 download (2025-08-30).

---

## 6. Number of Rows in Each Split

### Baseline dataset (`data/splits/baseline/`)

| Split | X shape | y shape | meta shape |
|---|---|---|---|
| Train | (54,735, 29) | (54,735, 1) | (54,735, 10) |
| Validation | (5,480, 29) | (5,480, 1) | (5,480, 10) |
| Test | (4,865, 29) | (4,865, 1) | (4,865, 10) |

### Temporal dataset (`data/splits/temporal/`)

| Split | X shape | y shape | meta shape |
|---|---|---|---|
| Train | (54,735, 110) | (54,735, 1) | (54,735, 10) |
| Validation | (5,480, 110) | (5,480, 1) | (5,480, 10) |
| Test | (4,865, 110) | (4,865, 1) | (4,865, 10) |

Meta files contain: `city`, `city_key`, `date`, `state`, `region_type`, `heatwave`, `hw_event_id`, `hw_event_start`, `hw_event_end`, `hw_event_length`. These are identifiers and event reference columns, not ML features.

---

## 7. City Distribution

All five cities have identical date ranges and identical row counts within each split.

| City | Train rows | Val rows | Test rows |
|---|---|---|---|
| Delhi | 10,947 | 1,096 | 973 |
| Lucknow | 10,947 | 1,096 | 973 |
| Nagpur | 10,947 | 1,096 | 973 |
| Ahmedabad | 10,947 | 1,096 | 973 |
| Mumbai | 10,947 | 1,096 | 973 |
| **Total** | **54,735** | **5,480** | **4,865** |

---

## 8. Class Distribution

### Overall

| Split | Total | Positive | Negative | Positive % | Imbalance |
|---|---|---|---|---|---|
| Train | 54,735 | 428 | 54,307 | 0.78% | 1:127 |
| Validation | 5,480 | 39 | 5,441 | 0.71% | 1:139 |
| Test | 4,865 | 38 | 4,827 | 0.78% | 1:127 |

The positive rate is consistent across splits (~0.7–0.8%), as expected for a temporal split where the underlying heatwave frequency does not change dramatically between periods.

### Per City — Train

| City | Total | Positive | Positive % |
|---|---|---|---|
| Delhi | 10,947 | 167 | 1.53% |
| Lucknow | 10,947 | 118 | 1.08% |
| Nagpur | 10,947 | 111 | 1.01% |
| Ahmedabad | 10,947 | 32 | 0.29% |
| Mumbai | 10,947 | 0 | 0.00% |

### Per City — Validation

| City | Total | Positive | Positive % |
|---|---|---|---|
| Delhi | 1,096 | 28 | 2.55% |
| Lucknow | 1,096 | 7 | 0.64% |
| Nagpur | 1,096 | 4 | 0.36% |
| Ahmedabad | 1,096 | 0 | 0.00% |
| Mumbai | 1,096 | 0 | 0.00% |

### Per City — Test

| City | Total | Positive | Positive % |
|---|---|---|---|
| Delhi | 973 | 18 | 1.85% |
| Lucknow | 973 | 16 | 1.64% |
| Nagpur | 973 | 4 | 0.41% |
| Ahmedabad | 973 | 0 | 0.00% |
| Mumbai | 973 | 0 | 0.00% |

Ahmedabad has 0 positives in validation and test — all 32 of its heatwave-next-day events fall within the 1990–2019 training window. This is a genuine property of the data, not an error.

---

## 9. Leakage Checks

All checks were performed programmatically and recorded in `results/phase9_leakage_audit.csv`.

| Check | Result |
|---|---|
| max(train_date) < min(val_date) globally | PASS: 2019-12-31 < 2020-01-01 |
| max(val_date) < min(test_date) globally | PASS: 2022-12-31 < 2023-01-01 |
| Per-city train_end < val_start (5 cities x 2 datasets) | PASS (10/10) |
| Per-city val_end < test_start (5 cities x 2 datasets) | PASS (10/10) |
| city+date duplicates across splits (baseline) | PASS: 0 duplicates |
| city+date duplicates across splits (temporal) | PASS: 0 duplicates |
| target (heatwave_next_day) NOT in X | PASS (all 6 splits x 2 datasets) |
| baseline and temporal city+date index match | PASS |
| Phase 8 source MD5 unchanged after split | PASS (both files) |

All checks passed. No issues found.

---

## 10. Why No Scaling or SMOTE Was Performed

**Scaling** (StandardScaler, MinMaxScaler, etc.) must be fit on training data only and then applied to validation and test sets. Fitting a scaler on the full dataset before splitting would allow test-set statistics to influence the training pipeline — a form of preprocessing leakage. Scaling belongs inside the model training pipeline (Phase 10+).

**SMOTE and resampling** address the 1:128 class imbalance but must be applied only to the training split — never to validation or test data. Applying SMOTE before splitting would contaminate evaluation with synthetic examples. This is addressed in Phase 11.

No encoders, imputers, or transformers of any kind were fitted in Phase 9.

---

## 11. qualifying_day — Note for Future Phases

`qualifying_day(T) = 1` when today's Tmax meets the heatwave threshold and departure criterion. It uses only same-day observations at T and is therefore **leakage-safe** — it does not encode any information about T+1.

However, it is a direct same-day precursor to the target and is likely strongly correlated with `heatwave_next_day`. Its inclusion may inflate model performance metrics compared to what would be achievable in a purely operational setting.

`qualifying_day` has **not been removed**. The decision to include or exclude it from the final model should be made during Phase 10 (Baseline Models) or Phase 12 (Model Evaluation) by comparing performance with and without this feature.

---

## 12. Limitations

| Limitation | Detail |
|---|---|
| Short val and test periods | Val: 3 years (39 positives), Test: ~2.7 years (38 positives). Small positive counts make metric estimates noisy with wide confidence intervals |
| Ahmedabad has 0 positives in val/test | All 32 Ahmedabad positives fall in the training window. City-level positive-class evaluation for Ahmedabad is not possible in val/test |
| Mumbai has 0 positives in all splits | Mumbai contributes only negative examples throughout. Positive-class evaluation for Mumbai is impossible |
| Fixed year boundaries | Year-based split is simple and reproducible but does not optimise for class balance. Walk-forward cross-validation would give more robust estimates at higher implementation cost |
| No multi-fold temporal CV | A single val/test split is used. Multiple temporal folds would reduce metric variance but were not implemented in this phase |

---

## Files Produced by Phase 9

| File | Description |
|---|---|
| `data/splits/baseline/X_train.csv` | Baseline features — train (54,735 x 29) |
| `data/splits/baseline/X_val.csv` | Baseline features — validation (5,480 x 29) |
| `data/splits/baseline/X_test.csv` | Baseline features — test (4,865 x 29) |
| `data/splits/baseline/y_train.csv` | Target — train |
| `data/splits/baseline/y_val.csv` | Target — validation |
| `data/splits/baseline/y_test.csv` | Target — test |
| `data/splits/baseline/meta_train.csv` | Identifiers + event metadata — train |
| `data/splits/baseline/meta_val.csv` | Identifiers + event metadata — validation |
| `data/splits/baseline/meta_test.csv` | Identifiers + event metadata — test |
| `data/splits/temporal/X_train.csv` | Temporal features — train (54,735 x 110) |
| `data/splits/temporal/X_val.csv` | Temporal features — validation (5,480 x 110) |
| `data/splits/temporal/X_test.csv` | Temporal features — test (4,865 x 110) |
| `data/splits/temporal/y_train.csv` | Target — train |
| `data/splits/temporal/y_val.csv` | Target — validation |
| `data/splits/temporal/y_test.csv` | Target — test |
| `data/splits/temporal/meta_train.csv` | Identifiers + event metadata — train |
| `data/splits/temporal/meta_val.csv` | Identifiers + event metadata — validation |
| `data/splits/temporal/meta_test.csv` | Identifiers + event metadata — test |
| `results/phase9_split_report.json` | Full split statistics (machine-readable) |
| `results/phase9_split_log.txt` | Full execution log |
| `results/phase9_leakage_audit.csv` | Per-city, per-dataset leakage audit table |
| `time_series_split.py` | Reproducible split script |

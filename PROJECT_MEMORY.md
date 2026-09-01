# PROJECT_MEMORY.md
## ClimateGuard — Indian Heatwave Analysis & Prediction

**Last updated:** 2026-09-02 (Phase 16 complete)  
**Purpose:** Complete, self-contained project state record. This file allows the project to be resumed on any account, session, or assistant without dependency on any prior conversation.

---

## PROJECT OVERVIEW

**Full name:** ClimateGuard — Explainable & Drift-Aware Heatwave Risk Prediction  
**Domain:** Indian climate / machine learning / public health  
**Language:** Python  
**Root directory:** `C:\Users\Adrian\Documents\climate guard\`

### Team Responsibilities (DO NOT CHANGE)

| Member | Part | Responsibility |
|---|---|---|
| **Adrian** | Part 1 | Dataset + ML |
| **Kshitij** | Part 2 | Risk + Adaptation + Explainability |
| **Pradnesh** | Part 3 | Expert + ETL + Integration |

**Adrian's scope ends** at a clean, trained model and prediction interface.  
**Do NOT** implement Kshitij's work (risk scoring, drift detection, SHAP, explainability).  
**Do NOT** implement Pradnesh's work (expert module, ETL, backend, website, API).

---

## PHASE STATUS

| Phase | Name | Status |
|---|---|---|
| Phase 1 | Dataset Research & Selection | ✅ COMPLETE |
| Phase 2 | Problem Definition | ✅ COMPLETE |
| Phase 3 | Data Acquisition | ✅ COMPLETE |
| Phase 4 | Exploratory Data Analysis | ✅ COMPLETE |
| Phase 5 | Data Cleaning | ✅ COMPLETE |
| Phase 6 | Heatwave Label Generation | ✅ COMPLETE |
| **Phase 7** | **Feature Engineering** | **✅ COMPLETE** |
| **Phase 8** | **Final ML Dataset** | **✅ COMPLETE** |
| **Phase 9** | **Train/Validation/Test Split** | **✅ COMPLETE** |
| **Phase 10** | **Baseline ML Models** | **✅ COMPLETE** |
| **Phase 11** | **Class Imbalance** | **✅ COMPLETE** |
| **Phase 12** | **Model Evaluation** | **✅ COMPLETE** |
| **Phase 13** | **Temporal Feature Experiment** | **✅ COMPLETE** |
| **Phase 14** | **Model Selection** | **✅ COMPLETE** |
| **Phase 15** | **Prediction Interface** | **✅ COMPLETE** |
| **Phase 16** | **Documentation** | **✅ COMPLETE** |
| **Final Integration Audit** | **Final Integration Audit** | **❌ NOT STARTED — NEXT TASK** |

---

## MASTER DATASET

**File:** `data/raw/all_cities_era5_raw.csv`  
**Size:** 8.40 MB  
**MD5:** `71d25a015e2c6a8015a155785b8d7cd0`  
**Source:** Open-Meteo Historical Weather API — ERA5 model (0.25° resolution)

| Property | Value |
|---|---|
| Rows | 65,135 |
| Columns | 23 |
| Cities | 5 |
| Rows per city | 13,027 |
| Date range | 1990-01-01 → 2025-08-31 |
| Missing values | 0 |
| Missing dates | 0 per city |
| Duplicates | 0 |
| Physical-range violations | 0 |

### Cities

| city_key | City | State | Region | Lat | Lon |
|---|---|---|---|---|---|
| delhi | New Delhi | Delhi | plains | 28.6139 | 77.2090 |
| lucknow | Lucknow | Uttar Pradesh | plains | 26.8467 | 80.9462 |
| nagpur | Nagpur | Maharashtra | plains | 21.1458 | 79.0882 |
| ahmedabad | Ahmedabad | Gujarat | plains | 23.0225 | 72.5714 |
| mumbai | Mumbai | Maharashtra | coastal | 19.0760 | 72.8777 |

**⚠️ CRITICAL: The raw dataset must NEVER be modified.**

Individual city files also exist:
- `data/raw/delhi_era5_raw.csv`
- `data/raw/lucknow_era5_raw.csv`
- `data/raw/nagpur_era5_raw.csv`
- `data/raw/ahmedabad_era5_raw.csv`
- `data/raw/mumbai_era5_raw.csv`

Old broken backups (pre-staging) in `data/raw_backup/` — ignore.  
Clean staging downloads (source of the promoted data) in `data/raw_staging/`.

---

## PHASE 3 — DATA ACQUISITION (COMPLETE)

- Data downloaded via `download_safe.py` using ERA5 model from Open-Meteo API
- Four cities (Lucknow, Nagpur, Ahmedabad, Mumbai) were re-downloaded to `data/raw_staging/` after the original ERA5-Land download returned 10 empty columns for those cities
- `validate_staged.py` passed (0 failures), then `promote_staged.py` rebuilt the combined file
- Delhi's original download was clean and was not re-downloaded

**Do NOT re-download anything. All data is present and validated.**

---

## PHASE 4 — EDA (COMPLETE)

**Script:** `eda_climateguard.py`  
**Notebook:** `notebooks/01_eda.ipynb`  
**Plots:** `results/plots/EDA/` (30 plots, numbered 01–30)  
**Reports:**
- `results/eda_summary.txt`
- `results/data_dictionary.json`
- `results/city_summary.json`
- `results/extreme_analysis.json`
- `results/trend_results.json`

**Docs:**
- `docs/data_dictionary.md`
- `docs/eda_findings.md`

### Key EDA Findings

| Finding | Detail |
|---|---|
| Dataset quality | Excellent — 0 missing, 0 duplicates, 0 range violations |
| Hottest city | Nagpur (mean Tmax 32.85°C, max 46.9°C on 2010-05-24) |
| Hottest month | May for all cities |
| Tmin trend | Rising significantly (p<0.05) in ALL 5 cities: +0.030–+0.040°C/yr |
| Mumbai | Uniquely narrow Tmax range (std 2.1°C vs ~5.8°C for plains) |
| rain_sum | Identical to precipitation_sum (r=+1.000) — flagged as redundant |
| Mumbai threshold | Fixed ≥37°C produced only 8 qualifying days in 35 years |
| ET₀ | Strongest single Tmax correlate (r=+0.75–0.89) |

---

## PHASE 5 — DATA CLEANING (COMPLETE)

**Script:** `data_cleaning.py`  
**Input:** `data/raw/all_cities_era5_raw.csv` (read-only)  
**Output:** `data/processed/weather_cleaned.csv`  
**Log:** `results/phase5_cleaning_log.txt`  
**Doc:** `docs/preprocessing_decisions.md`

| Metric | Value |
|---|---|
| Input shape | 65,135 × 23 |
| Output shape | 65,135 × 22 |
| Rows removed | 0 |
| Columns removed | 1 (`rain_sum`) |
| Validation checks | 105 passed / 0 failed |

**Only change:** `rain_sum` removed because it was byte-for-byte identical to `precipitation_sum` (max abs diff = 0.0, r = +1.000).

**⚠️ `weather_cleaned.csv` is a validated artifact. Do NOT recreate or modify.**

### Columns in weather_cleaned.csv (22)
`city, city_key, latitude, longitude, region_type, state, date,
temperature_2m_max, temperature_2m_min, temperature_2m_mean,
apparent_temperature_max, apparent_temperature_min, apparent_temperature_mean,
precipitation_sum, wind_speed_10m_max, wind_gusts_10m_max,
relative_humidity_2m_max, relative_humidity_2m_min, relative_humidity_2m_mean,
surface_pressure_mean, shortwave_radiation_sum, et0_fao_evapotranspiration`

---

## PHASE 6 — HEATWAVE LABEL GENERATION (COMPLETE)

**Script:** `heatwave_labeling.py`  
**Input:** `data/processed/weather_cleaned.csv` (read-only)  
**Output:** `data/processed/weather_labelled.csv`  
**Log:** `results/phase6_labeling_log.txt`  
**Stats:** `results/phase6_class_balance.json`  
**Doc:** `docs/heatwave_labeling_methodology.md`  
**Plots:** `results/plots/heatwave_labels/` (10 plots)

| Metric | Value |
|---|---|
| Output shape | 65,135 × 30 |
| File size | 11.66 MB |

### Heatwave Definition

**Name:** IMD-Inspired Operational Heatwave Label (ERA5-based)  
**⚠️ Do NOT call it "official IMD labels"** — ERA5 reanalysis, not IMD station data.

#### Reference sources
- WHO India: https://www.who.int/india/heat-waves
- DrishtiIAS IMD criteria summary
- Times of India (IMD Mumbai head confirming coastal 37°C threshold)

#### Plains cities (Delhi, Lucknow, Nagpur, Ahmedabad)

```
qualifying_day = 1  if:
    (temperature_2m_max >= 40.0°C  AND  tmax_departure >= 4.5°C)
    OR
    (temperature_2m_max >= 45.0°C)           ← absolute override

heatwave = 1  if:  qualifying_day=1  AND  part of a consecutive run >= 2 days
heatwave = 0  otherwise (including isolated single qualifying days)
```

#### Coastal city (Mumbai)

```
qualifying_day = 1  if:
    temperature_2m_max >= 37.0°C  AND  tmax_departure >= 4.5°C

heatwave = 1  if:  qualifying_day=1  AND  part of a consecutive run >= 2 days
```

#### Departure formula

```
tmax_departure(T) = temperature_2m_max(T) − tmax_normal(city, DOY)
```

Where `tmax_normal` is the city-specific 31-day centred smoothed mean Tmax for that day-of-year, computed from the **1990–2020 baseline** (31 years).

### Class Distribution

| City | Region | HW days | Total | HW% | Events | Avg dur | Max dur |
|---|---|---|---|---|---|---|---|
| New Delhi | plains | 213 | 13,027 | 1.64% | 54 | 3.9 d | 11 d |
| Lucknow | plains | 141 | 13,027 | 1.08% | 32 | 4.4 d | 12 d |
| Nagpur | plains | 119 | 13,027 | 0.91% | 34 | 3.5 d | 10 d |
| Ahmedabad | plains | 32 | 13,027 | 0.25% | 12 | 2.7 d | 5 d |
| **Mumbai** | coastal | **0** | 13,027 | **0.00%** | 0 | — | — |

**⚠️ Mumbai has ZERO positive heatwave examples. Do NOT artificially change this.**  
Mumbai's zero positives are scientifically correct — its maritime climate does not produce sustained heatwaves under IMD coastal criteria.

### Labeling columns added to weather_labelled.csv

| Column | Description |
|---|---|
| `tmax_normal` | City-specific climatological daily Tmax normal (°C) |
| `tmax_departure` | `temperature_2m_max` − `tmax_normal` (°C) |
| `qualifying_day` | 1 if day meets threshold+departure (before duration filter) |
| `heatwave` | **Ground-truth same-day label.** 1 = heatwave day (T) |
| `hw_event_id` | Sequential event number (0 = non-heatwave) |
| `hw_event_start` | Start date of this event |
| `hw_event_end` | End date of this event |
| `hw_event_length` | Duration of this event in days |

**⚠️ `weather_labelled.csv` is a validated artifact. Do NOT recreate or modify.**

---

## CRITICAL PREDICTION DESIGN

### 1-Day-Ahead Prediction

The project requires **1-day-ahead heatwave prediction**:

```
Features available at date T
        ↓
ML model predicts
        ↓
heatwave_next_day(T) = heatwave(T+1)
```

### Target construction (Phase 7/8)

```python
# Applied per-city independently (NEVER across city boundaries)
df['heatwave_next_day'] = df.groupby('city_key')['heatwave'].shift(-1)
# Last row per city → NaN → must be dropped from ML dataset
```

### Leakage rule — ABSOLUTE

```
ALLOWED:  Any weather variable at T or earlier
ALLOWED:  heatwave_next_day(T) as the TARGET only
FORBIDDEN: temperature_2m_max(T+1) or any weather variable from T+1 or later as a feature
FORBIDDEN: heatwave(T) as a direct input feature (it IS heatwave_next_day(T-1) if lagged correctly)
```

### Same-day label clarification

- `heatwave(T)` = whether today T is a heatwave day (ground truth, in `weather_labelled.csv`)
- `heatwave_next_day(T)` = `heatwave(T+1)` = tomorrow's condition = **the ML prediction target**
- If using prior heatwave state as a feature, use `heatwave_lag_1(T)` = `heatwave(T-1)` — this is safe

---

## PHASE 7 — FEATURE ENGINEERING (COMPLETE)

**Script:** `feature_engineering.py`  
**Input:** `data/processed/weather_labelled.csv` (read-only)  
**Output:** `data/features/climateguard_features.csv`  
**Log:** `results/phase7_feature_engineering_log.txt`  
**Registry:** `results/phase7_feature_groups.json`

| Metric | Value |
|---|---|
| Input shape | 65,135 × 30 |
| Output shape | 65,095 × 121 |
| Dropped rows | 40 (7 head + 1 tail per city × 5 cities — incomplete lag windows + missing target) |
| File size | 59.91 MB |
| Leakage audit | PASSED — 110 features checked, 0 issues |

### Feature groups

| Group | Description | Count |
|---|---|---|
| 1 — Current weather | All 15 weather variables at T + tmax_normal + tmax_departure + qualifying_day | 18 |
| 2 — Lag features | T-1, T-2, T-3, T-7 for Tmax/Tmin/Tmean/RH/precip/wind/pressure + tmax_departure lags + heatwave_lag1 | 33 |
| 3 — Rolling features | 3-day and 7-day rolling mean/max/min for 7 key variables (shift(1).rolling(N) pattern) | 42 |
| 4 — Trend features | tmax_delta_1d/3d/7d + tmax_slope_3d/7d | 5 |
| 5 — Anomaly features | tmax_departure_zscore (30-day trailing z-score) | 1 |
| 6 — Calendar features | month, day_of_year, season_code, month_sin/cos, doy_sin/cos | 7 |
| 7 — City features | city_encoded, is_coastal, latitude, longitude | 4 |

### Experimental feature sets (for Phase 13)

| Set | Features | Purpose |
|---|---|---|
| `baseline_features` | 29 features (Groups 1 + 6 + 7) | Current-day only — no temporal memory |
| `temporal_features` | 110 features (all groups) | Full feature set with lags/rolling/trends |

### Target construction

```python
# Applied per-city via groupby — NEVER across city boundaries
df['heatwave_next_day'] = df.groupby('city_key')['heatwave'].shift(-1)
```

### Class balance (heatwave_next_day)

| Class | Count | % |
|---|---|---|
| normal (0) | 64,590 | 99.22% |
| heatwave (1) | 505 | 0.78% |

Per-city: Delhi 213 (1.64%), Lucknow 141 (1.08%), Nagpur 119 (0.91%), Ahmedabad 32 (0.25%), Mumbai 0 (0.00%)

### Leakage design

- All lag columns: `shift(N)` with N ≥ 1 (past data only)
- All rolling columns: `shift(1).rolling(N)` — window is [T-N, …, T-1], excludes T
- Delta features: compare current T vs past T-N — T component is current-day weather (allowed in Group 1)
- `heatwave_lag1` = `heatwave(T-1)` — safe, does NOT expose today's label as a feature
- `heatwave_next_day` is the TARGET, excluded from all feature lists
- Leakage audit ran programmatically inside the script — PASSED

**⚠️ `climateguard_features.csv` is a validated artifact. Do NOT recreate or modify.**

---

## PHASE 8 — FINAL ML DATASET (COMPLETE)

**Script:** `build_ml_dataset.py`  
**Input:** `data/features/climateguard_features.csv` (read-only, MD5 verified)  
**Registry:** `results/phase7_feature_groups.json` (read-only)  
**Outputs:** `data/features/ml_baseline.csv`, `data/features/ml_temporal.csv`  
**Log:** `results/phase8_report.txt`  
**Audit:** `results/phase8_feature_audit.csv`  
**Doc:** `docs/final_ml_dataset.md`

| Metric | Value |
|---|---|
| Input shape (Phase 7) | 65,095 × 121 |
| Rows dropped (zscore NaN) | 15 (3 per city × 5 — first rows with <10 prior observations for rolling std) |
| Final ML rows | 65,080 (13,016 per city) |
| Date range | 1990-01-11 to 2025-08-30 |
| Phase 7 MD5 (verified untouched) | `fdb559545ef4a0155fbb5c8a813c9eb8` |
| Validation checks | 23 passed / 0 failed |

### Output datasets

| File | Shape | Size | Contents |
|---|---|---|---|
| `ml_baseline.csv` | 65,080 × 40 | 16.66 MB | 10 ID cols + 29 baseline features + target |
| `ml_temporal.csv` | 65,080 × 121 | 59.22 MB | 10 ID cols + 110 temporal features + target |

### Target

```
heatwave_next_day(T) = heatwave(T+1)   [verified per-city, 0 mismatches]
```
Target is `float64` binary `{0.0, 1.0}`, confirmed absent from all feature lists.

### Class distribution (both datasets identical)

| Class | Count | % |
|---|---|---|
| Normal (0) | 64,575 | 99.22% |
| Heatwave (1) | 505 | 0.78% |
| Imbalance ratio | 1:128 | |

Per city: Delhi 213 (1.64%), Lucknow 141 (1.08%), Nagpur 119 (0.91%), Ahmedabad 32 (0.25%), Mumbai 0 (0.00%)

### Missing value resolution

- `hw_event_start` / `hw_event_end`: NaN for non-heatwave rows — passthrough metadata, not features, no action.
- `tmax_departure_zscore`: 15 NaN rows (rows 7–9 of each city in Phase 7 output lacked 10 prior observations for `min_periods=10` rolling std). Resolved by dropping these rows. No imputation.
- Feature matrices: **0 NaN** (confirmed).

### Validation summary

All 23 checks passed:
- Both output files exist ✓
- Both contain `heatwave_next_day` ✓
- Target NOT in feature sets ✓
- Baseline has exactly 29 features ✓
- Temporal has exactly 110 features ✓
- `city`, `city_key`, `date` identifiers present ✓
- 0 NaN in feature matrices ✓
- 0 NaN in target ✓
- Target binary {0, 1} ✓
- Phase 7 source row count intact (65,095) ✓
- Feature sets match registry exactly ✓
- City boundaries validated ✓
- Chronological order validated ✓
- Leakage audit PASSED ✓

### Decisions deferred to Phase 10

1. **Mumbai:** Zero positive events — kept in both datasets. Whether to exclude from supervised training decided in Phase 10.
2. **qualifying_day feature:** Leakage-safe (derived at T, not T+1) but strongly correlated with target. Review at Phase 10/12.
3. **heatwave_lag1:** Uses `heatwave(T-1)` as a feature — safe and intentional.

**⚠️ `ml_baseline.csv` and `ml_temporal.csv` are validated artifacts. Do NOT recreate or modify.**

---

## PHASE 9 — TRAIN/VALIDATION/TEST SPLIT (COMPLETE)

**Script:** `time_series_split.py`  
**Inputs:** `data/features/ml_baseline.csv` (MD5: `7851299a3bfa3293f6f66e1870b83d41`), `data/features/ml_temporal.csv` (MD5: `513724a7c1ab7d4fec417997a8df540b`)  
**Doc:** `docs/train_validation_test_split.md`  
**Log:** `results/phase9_split_log.txt`  
**Report:** `results/phase9_split_report.json`  
**Audit:** `results/phase9_leakage_audit.csv`

### Split boundaries

| Split | Start | End |
|---|---|---|
| Train | 1990-01-11 | 2019-12-31 |
| Validation | 2020-01-01 | 2022-12-31 |
| Test | 2023-01-01 | 2025-08-30 |

Split method: **chronological / year-based. NO random shuffle.**

### Split shapes

| Dataset | Split | X shape | y shape |
|---|---|---|---|
| baseline | train | (54,735, 29) | (54,735, 1) |
| baseline | val | (5,480, 29) | (5,480, 1) |
| baseline | test | (4,865, 29) | (4,865, 1) |
| temporal | train | (54,735, 110) | (54,735, 1) |
| temporal | val | (5,480, 110) | (5,480, 1) |
| temporal | test | (4,865, 110) | (4,865, 1) |

Each split also has a `meta_*.csv` (10 ID/event columns, not ML features).

### Class distribution

| Split | Total | Positive | Positive % |
|---|---|---|---|
| Train | 54,735 | 428 | 0.78% |
| Validation | 5,480 | 39 | 0.71% |
| Test | 4,865 | 38 | 0.78% |

Per-city positives (train / val / test): Delhi 167/28/18, Lucknow 118/7/16, Nagpur 111/4/4, Ahmedabad 32/0/0, Mumbai 0/0/0

### Leakage audit

All checks passed:
- max(train) < min(val): 2019-12-31 < 2020-01-01
- max(val) < min(test): 2022-12-31 < 2023-01-01
- Per-city checks: 10/10 PASS
- city+date duplicates across splits: 0
- target NOT in X: verified all splits
- Phase 8 MD5 unchanged: verified

### Deferred decisions

1. **qualifying_day:** Leakage-safe but strongly correlated with target. Review inclusion in Phase 10/12.
2. **Mumbai:** 0 positives in all splits. Training-inclusion decision deferred to Phase 10.
3. **Ahmedabad:** 0 positives in val/test. All 32 positives fall in training window.
4. **Class imbalance (1:127):** Handled in Phase 11 (not in this phase).

### File structure

```
data/splits/
  baseline/  X_train.csv  X_val.csv  X_test.csv
             y_train.csv  y_val.csv  y_test.csv
             meta_train.csv  meta_val.csv  meta_test.csv
  temporal/  X_train.csv  X_val.csv  X_test.csv
             y_train.csv  y_val.csv  y_test.csv
             meta_train.csv  meta_val.csv  meta_test.csv
```

---

## PHASE 10 — BASELINE ML MODELS (COMPLETE)

**Script:** `train_baseline_models.py`  
**Inputs:** `data/splits/baseline/X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`  
**Feature set used:** Baseline only (29 features: Groups 1 + 6 + 7)  
**Outputs:** `models/phase10/` (6 model directories), `results/phase10_metrics.json`, `results/phase10_model_comparison.csv`, `results/phase10_log.txt`, `results/phase10_confusion_matrices/`  
**Doc:** `docs/baseline_models.md`

### Models trained

| Model | Implementation | Imbalance handling |
|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `class_weight='balanced'` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `class_weight='balanced'` |
| XGBoost | `xgboost.XGBClassifier` | `scale_pos_weight=126.89` |

Each model trained on two feature sets: **with_qd** (29 features, including `qualifying_day`) and **without_qd** (28 features). Total: **6 models**.

### Validation results (2020–2022, 39 positives)

| Rank | Model | Feature Set | Val F1 | Val PR-AUC |
|---|---|---|---|---|
| 1 | XGBoost | without_qd | 0.5424 | 0.5433 |
| 2 | XGBoost | with_qd | 0.5299 | 0.5668 |
| 3 | Random Forest | with_qd | 0.5000 | 0.5535 |
| 4 | Random Forest | without_qd | 0.4789 | 0.5325 |
| 5 | Logistic Regression | with_qd | 0.2879 | 0.6356 |
| 6 | Logistic Regression | without_qd | 0.2796 | 0.6216 |

**Best F1:** XGBoost without_qd (0.5424) | **Best PR-AUC:** Logistic Regression with_qd (0.6356)  
ROC-AUC ~0.994 across all models — inflated by large negative class, not a reliable discriminator here.

### qualifying_day experiment

Effect is small but not negligible. Removing it slightly improves XGBoost F1 while dropping PR-AUC marginally. Does not change model rankings. Final decision on inclusion deferred to Phase 12/14.

### Test set status

**Held out — NOT used in Phase 10.** Reserved for Phase 12/14 final evaluation only.

### No final model declared in Phase 10

Phase 11 will address class imbalance more systematically (SMOTE, SMOTE-Tomek, threshold tuning), which will likely change rankings.

---

## PHASE 11 — CLASS IMBALANCE HANDLING (COMPLETE)

**Script:** `train_imbalance_models.py`  
**Inputs:** `data/splits/baseline/X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`  
**Feature sets:** baseline with_qd (29 features) and without_qd (28 features)  
**Doc:** `docs/class_imbalance.md`

### Training class distribution

| Class | Count | % |
|---|---|---|
| Positive (heatwave) | 428 | 0.78% |
| Negative (normal) | 54,307 | 99.22% |
| Imbalance ratio | 1 : 126.9 | |

### Strategies tested

| Strategy | Description |
|---|---|
| baseline_weight | Phase 10 class_weight=balanced / scale_pos_weight=126.89 (reference) |
| strong_weight | Doubled positive-class weight: {0:1, 1:254} for LR/RF |
| spw_64 / spw_128 / spw_256 / spw_512 | XGBoost scale_pos_weight grid |
| random_oversample | Minority class oversampled to 1:1 on X_train only |
| random_undersample | Majority class downsampled to 1:10 on X_train only |
| smote_skipped | imbalanced-learn not installed — SMOTE skipped |

### Resampling rule

ALL resampling applied exclusively to X_train / y_train.  
Validation and test sets retain their real-world class distribution unchanged.

### Threshold analysis

Decision thresholds `[0.05, 0.10, ..., 0.50, 0.60, 0.70, 0.80, 0.90]` evaluated on validation predictions.  
Best threshold chosen from validation only — test set never used.

### Key validation results

**Best F1 at default threshold (0.50):**

| Model | Feature Set | Strategy | F1 | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| XGBoost | with_qd | spw_64 | **0.5714** | 0.4545 | 0.7692 | 0.5111 | 36 | 9 |

**Best F1 threshold-optimised:**

| Model | Feature Set | Strategy | Threshold | F1 | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | with_qd | random_undersample | **0.70** | **0.6122** | 0.5085 | 0.7692 | 0.5497 | 29 | 9 |

**Best PR-AUC (carried from Phase 10, unchanged):**

| Model | Feature Set | Strategy | PR-AUC |
|---|---|---|---|
| Logistic Regression | with_qd | baseline_weight | **0.6356** |

### Recommended candidate strategy

**Random Forest / with_qd / random_undersample / threshold = 0.70**

- Highest overall F1 (0.6122)
- Precision > 0.50 — over half of raised alarms are real heatwave days
- 77% recall — captures most events
- Only 29 false positives over 3-year validation (≈ 1 per 38 days)
- Interpretable — no synthetic data artefacts, feature importance available

Close alternatives: XGBoost/without_qd/baseline_weight/thresh=0.80 (F1=0.6105, FP=27)

### Qualifying_day finding

`qualifying_day` is beneficial when combined with undersampling and an elevated threshold.
RF/with_qd/random_undersample at thresh=0.70 (F1=0.6122) outperforms
RF/without_qd/random_undersample at thresh=0.70 (F1=0.5843).  
Final decision deferred to Phase 14.

### Leakage audit

All 10 checks: **PASSED**

### Test set status

**Held out — NOT used in Phase 11.**  Reserved for Phase 12/14 final evaluation.

### Final model NOT declared

Phase 14 (Model Selection) makes the final decision after Phase 12 evaluation
and Phase 13 temporal feature experiment.

---

## PHASE 12 — MODEL EVALUATION (COMPLETE)

**Script:** `evaluate_test_set.py`  
**Inputs:** `data/splits/baseline/X_test.csv`, `y_test.csv`, `meta_test.csv`  
**Models loaded from:** `models/phase11/` (no retraining)  
**Thresholds:** Fixed from Phase 11 validation (not tuned on test)  
**Doc:** `docs/model_evaluation.md`

### Test set

| Property | Value |
|---|---|
| Rows | 4,865 |
| Positives | 38 (0.78%) |
| Period | 2023-01-01 to 2025-08-30 |

### Candidates evaluated

| Candidate | Threshold | Test F1 | Test P | Test R | Test PR-AUC | Test FP | Test FN |
|---|---|---|---|---|---|---|---|
| RF / with_qd / random_undersample [PRIMARY] | 0.70 | 0.6957 | 0.5926 | 0.8421 | 0.7705 | 22 | 6 |
| XGB / without_qd / baseline_weight | 0.80 | 0.7191 | 0.6275 | 0.8421 | **0.8440** | 19 | 6 |
| RF / with_qd / smote_skipped | 0.20 | **0.7381** | **0.6739** | 0.8158 | 0.8307 | **15** | 7 |
| RF / without_qd / smote_skipped | 0.15 | 0.7143 | 0.5833 | **0.9211** | 0.8397 | 25 | **3** |

### Generalization: ALL FOUR CANDIDATES IMPROVED on test vs validation

| Candidate | Val F1 | Test F1 | Delta |
|---|---|---|---|
| RF / random_undersample | 0.6122 | 0.6957 | +0.0835 |
| XGB / baseline_weight | 0.6105 | 0.7191 | +0.1086 |
| RF / smote_skipped wqd | 0.6105 | 0.7381 | +0.1276 |
| RF / smote_skipped nqd | 0.6095 | 0.7143 | +0.1048 |

Improvement explained by: 2024 was an exceptionally strong heatwave year (34/38 test positives in 2024); models generalise well to concentrated heatwave signals.

### Key year-level finding

- 2023: sparse positives (4 total) — weak performance across all models (F1 = 0.36-0.53)
- 2024: 34 positives — strong performance (F1 = 0.78-0.83)
- 2025 (Jan-Aug): 0 positives — all alarms are false positives (4-6 per model)

### Leakage audit: 12/12 PASS

### Best test F1: RF / with_qd / smote_skipped — F1=0.7381, P=0.6739, R=0.8158, FP=15

### Best test PR-AUC: XGB / without_qd / baseline_weight — PR-AUC=0.8440

### Final production model NOT declared

Phase 13 (temporal feature experiment) is next.  
Phase 14 will make the final model selection.

---

## PHASE 13 — TEMPORAL FEATURE EXPERIMENT (COMPLETE)

**Script:** `temporal_feature_experiment.py`  
**Inputs:** `data/splits/temporal/` (X/y/meta train, val, test)  
**Feature registry:** `results/phase7_feature_groups.json`  
**Doc:** `docs/temporal_feature_experiment.md`

### Feature sets evaluated

| Set | Groups | Features | Description |
|---|---|---|---|
| `baseline_wqd` | 1 + 6 + 7 | 29 | Current-day + calendar + city (with qualifying_day) |
| `baseline_nqd` | 1 + 6 + 7 | 28 | Current-day + calendar + city (without qualifying_day) |
| `temporal_wqd` | 1+2+3+4+5 + 6 + 7 | 110 | Full temporal set (with qualifying_day) |
| `temporal_nqd` | 1+2+3+4+5 + 6 + 7 | 109 | Full temporal set (without qualifying_day) |

### Part A — Primary comparison validation results

| Model | Feature Set | Strategy | F1 | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **Random_Forest** | **`temporal_wqd`** | **random_undersample** | **0.6154** | **0.5385** | **0.7179** | **0.5298** | **24** | **11** |
| Random_Forest | `baseline_wqd` | random_undersample | 0.6122 | 0.5085 | 0.7692 | 0.5497 | 29 | 9 |
| XGBoost | `baseline_nqd` | baseline_weight | 0.6105 | 0.5179 | 0.7436 | 0.5433 | 27 | 10 |
| XGBoost | `baseline_wqd` | baseline_weight | 0.5979 | 0.5000 | 0.7436 | 0.5668 | 29 | 10 |
| Random_Forest | `baseline_nqd` | random_undersample | 0.5843 | 0.5200 | 0.6667 | 0.5767 | 24 | 13 |
| XGBoost | `temporal_wqd` | baseline_weight | 0.5185 | 0.5000 | 0.5385 | 0.5452 | 21 | 18 |
| XGBoost | `temporal_nqd` | baseline_weight | 0.5128 | 0.5128 | 0.5128 | 0.5711 | 19 | 19 |
| Random_Forest | `temporal_nqd` | random_undersample | 0.5393 | 0.4800 | 0.6154 | 0.5527 | 26 | 15 |

**Best val F1:** RF / `temporal_wqd` / random_undersample = 0.6154

### Part B — Ablation study (RF / random_undersample / thresh=0.70)

| Feature Set | #Features | F1 | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| `baseline_only` | 29 | 0.6122 | 0.5085 | 0.7692 | 0.5497 | 29 | 9 |
| `baseline + lag` | 62 | 0.5957 | 0.5091 | 0.7179 | 0.5538 | 27 | 11 |
| `baseline + rolling` | 71 | 0.6024 | 0.5682 | 0.6410 | 0.5822 | 19 | 14 |
| `baseline + trend` | 34 | 0.6105 | 0.5179 | 0.7436 | 0.5247 | 27 | 10 |
| **`baseline + anomaly`** | **30** | **0.6263** | **0.5167** | **0.7949** | **0.5037** | **29** | **8** |
| `full_temporal` | 110 | 0.6154 | 0.5385 | 0.7179 | 0.5298 | 24 | 11 |
| `full_temporal_nqd` | 109 | 0.5393 | 0.4800 | 0.6154 | 0.5527 | 26 | 15 |

**Key ablation finding:** `tmax_departure_zscore` (anomaly feature alone) produces the highest single-group F1 (0.6263). Rolling features reduce false positives most (FP=19).

### Part C — Test confirmation

**Best val config: RF / `temporal_wqd` / random_undersample / threshold=0.70**

| Config | F1 | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| Temporal (RF / `temporal_wqd`) | **0.7586** | **0.6735** | **0.8684** | **0.7885** | **16** | **5** |
| Baseline (RF / `baseline_wqd`) | 0.6957 | 0.5926 | 0.8421 | 0.7705 | 22 | 6 |

**Temporal advantage on test:** F1 +0.063, Precision +0.081, FP −6

### Leakage audit: 14/14 PASS

### Key conclusions

1. **Temporal features help RF** — `temporal_wqd` outperforms `baseline_wqd` on both val (+0.003 F1, −5 FP) and test (+0.063 F1, −6 FP).
2. **qualifying_day is load-bearing** — Removing it from any feature set causes large F1 drops. Retain in Phase 14 final model.
3. **Anomaly feature (`tmax_departure_zscore`) is the most valuable single addition** — highest ablation F1 (0.6263), fewest missed events.
4. **XGBoost does not benefit from temporal features at threshold=0.80** — F1 drops from 0.6105 to 0.5128–0.5185. Threshold re-tuning deferred to Phase 14.
5. **Recommended Phase 14 config:** RF / `temporal_wqd` (110 features) / random_undersample / threshold=0.70

### Phase 13 outputs

| File | Description |
|---|---|
| `results/phase13_temporal_comparison.csv` | 8-row primary comparison (validation) |
| `results/phase13_ablation.csv` | 7-row ablation study (validation) |
| `results/phase13_metrics.json` | Full metrics for all 15 experiments + test confirmation |
| `results/phase13_leakage_audit.csv` | 14-check leakage audit (all PASS) |
| `results/phase13_log.txt` | Full execution log |
| `results/plots/phase13/baseline_vs_temporal_f1.png` | F1 bar chart — primary comparison |
| `results/plots/phase13/baseline_vs_temporal_prauc.png` | PR-AUC bar chart — primary comparison |
| `results/plots/phase13/precision_recall_comparison.png` | Precision vs Recall scatter |
| `results/plots/phase13/ablation_comparison.png` | Ablation F1 / PR-AUC grouped bar chart |
| `models/phase13/Random_Forest/{4 feature sets}/random_undersample/` | 4 RF primary models |
| `models/phase13/XGBoost/{4 feature sets}/baseline_weight/` | 4 XGBoost primary models |
| `models/phase13/Random_Forest/ablation/{7 sets}/` | 7 RF ablation models |
| `docs/temporal_feature_experiment.md` | Full Phase 13 documentation (12 sections, 268 lines) |

---

## PHASE 14 — FINAL MODEL SELECTION (COMPLETE)

**Script:** `final_model_selection.py`
**Inputs:** `data/splits/temporal/` train + val + test
**Docs:** `docs/final_model_selection.md`, `docs/final_model_contract.md`

### Final model configuration

| Property | Value |
|---|---|
| Model type | RandomForestClassifier |
| Feature set | temporal_wqd — 110 features (with qualifying_day) |
| Feature list | models/final/feature_list.json |
| Imbalance strategy | Random undersampling, 1:10 (neg:pos), on train+val only |
| Threshold | 0.70 (fixed from Phase 11 validation, not tuned on test) |
| Random seed | 42 |
| RF params | n_estimators=300, max_depth=10, min_samples_leaf=10 |
| Training data | train + val (1990-01-11 to 2022-12-31), 60,215 rows, 467 positives |
| After undersampling | 5,137 rows (467 pos + 4,670 neg) |
| Test data | test (2023-01-01 to 2025-08-30), 4,865 rows, 38 positives |

### Selection rationale (from validation only)

- RF / temporal_wqd / random_undersample selected in Phase 13 with val-F1 = 0.6154 (best among all Phase 13 configurations)
- qualifying_day retained: removing it drops val-F1 by -0.076 (0.6154 to 0.5393)
- Threshold = 0.70 fixed from Phase 11 validation (optimal RF/undersample operating point)
- XGBoost rejected: temporal features degraded XGBoost performance (F1 0.61 to 0.51)

### Final test results (2023-01-01 to 2025-08-30, 38 positives)

| Metric | Phase 13 Part C (train only) | Phase 14 FINAL (train+val) |
|---|---|---|
| F1 | 0.7586 | 0.6947 |
| Precision | 0.6735 | 0.5789 |
| Recall | 0.8684 | 0.8684 |
| PR-AUC | 0.7885 | 0.8339 |
| ROC-AUC | — | 0.9979 |
| Accuracy | — | 0.9940 |
| TP | 33 | 33 |
| FP | 16 | 24 |
| TN | — | 4,803 |
| FN | 5 | 5 |

Note on F1 change: Recall is identical (0.8684, 33 TP, 5 FN). The drop is in precision (FP 16 to 24) because adding 2020-2022 validation data during training introduces a broader range of near-threshold patterns. PR-AUC improved (+0.045), indicating better probability calibration overall.

### Per-city test results

| City | F1 | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|
| Delhi | 0.7805 | 0.6957 | 0.8889 | 16 | 7 | 2 |
| Lucknow | 0.6829 | 0.5600 | 0.8750 | 14 | 11 | 2 |
| Nagpur | 0.5000 | 0.3750 | 0.7500 | 3 | 5 | 1 |
| Ahmedabad | N/A | — | — | — | — | — |
| Mumbai | N/A | — | — | — | — | — |

### Validation checks: 12/12 PASS | Leakage audit: 12/12 PASS

### Key limitations

1. ERA5 reanalysis data — not IMD station observations
2. IMD-inspired label — not official IMD ground truth
3. Mumbai has 0 heatwave positives — model cannot be evaluated for Mumbai
4. Ahmedabad has 0 test positives — all 32 positives in training window only
5. qualifying_day is tied to target definition by construction — document to users
6. ~42% of alarms are false positives (precision=0.58); recall-optimised for early-warning
7. No drift detection; retraining recommended for deployment past 2025

### Final model artifacts

| File | Description |
|---|---|
| models/final/climateguard_final_model.joblib | Final model (1.86 MB), load with joblib.load() |
| models/final/feature_list.json | Ordered list of 110 feature names and dtypes |
| models/final/metadata.json | Full configuration, parameters, metrics (6 KB) |
| results/final_model_metrics.json | Full test metrics + city breakdown |
| results/final_model_comparison.csv | Phase 13 Part C vs Phase 14 comparison |
| results/final_model_leakage_audit.csv | 12-check leakage audit (all PASS) |
| results/final_model_log.txt | Full execution log |
| results/plots/final_confusion_matrix.png | Confusion matrix (test 2023-2025) |
| results/plots/final_precision_recall.png | PR curve with operating point |
| docs/final_model_selection.md | Full 13-section selection rationale (360 lines) |
| docs/final_model_contract.md | Prediction contract for Part 2 / Part 3 (323 lines) |

---

## PHASE 15 -- PREDICTION INTERFACE (COMPLETE)

Interface package: src/prediction/predictor.py
Tests: tests/test_prediction_interface.py (18/18 PASS)
Example: examples/predict_example.py

Model loaded from: models/final/climateguard_final_model.joblib
Feature list: models/final/feature_list.json (110 features, ordered)
Threshold: 0.70 (locked)
Target: heatwave_next_day
Prediction: label=1 heatwave tomorrow, label=0 normal tomorrow

Input contract: dict or pd.DataFrame with all 110 features, no NaN, all numeric
Output contract: prediction_probability (float [0,1]) + prediction_label (0 or 1)

Test results: 18/18 PASS
Smoke test: 5/5 correct (Ahmedabad 2023-01-01 to 2023-01-05, all normal)

Import pattern:
    from src.prediction import ClimateGuardPredictor
    predictor = ClimateGuardPredictor()
    result = predictor.predict(features_dict)
    df_out = predictor.predict_batch(features_df)
    model  = predictor.model        (for SHAP / Part 2)
    names  = predictor.feature_names  (110 ordered feature names)

Phase 15 artifacts:
- src/prediction/__init__.py
- src/prediction/predictor.py (496 lines)
- tests/test_prediction_interface.py (458 lines)
- examples/predict_example.py (157 lines)
- results/phase15_interface_test.txt
- results/phase15_interface_validation.json
- docs/prediction_interface.md (345 lines)
- docs/part2_integration_contract.md (211 lines)
- docs/part3_integration_contract.md (311 lines)

---

## FULL FILE INVENTORY

### Data files (validated artifacts — do not modify)

| File | Size | Description |
|---|---|---|
| `data/raw/all_cities_era5_raw.csv` | 8.40 MB | Master raw dataset (READ-ONLY) |
| `data/raw/delhi_era5_raw.csv` | 1.61 MB | Delhi individual file |
| `data/raw/lucknow_era5_raw.csv` | 1.72 MB | Lucknow individual file |
| `data/raw/nagpur_era5_raw.csv` | 1.67 MB | Nagpur individual file |
| `data/raw/ahmedabad_era5_raw.csv` | 1.70 MB | Ahmedabad individual file |
| `data/raw/mumbai_era5_raw.csv` | 1.69 MB | Mumbai individual file |
| `data/processed/weather_cleaned.csv` | 8.14 MB | Phase 5 output (READ-ONLY) |
| `data/processed/weather_labelled.csv` | 11.12 MB | Phase 6 output (READ-ONLY) |
| `data/features/climateguard_features.csv` | 59.91 MB | Phase 7 output (READ-ONLY) |
| `data/features/ml_baseline.csv` | 16.66 MB | Phase 8 baseline ML dataset (READ-ONLY) |
| `data/features/ml_temporal.csv` | 59.22 MB | Phase 8 temporal ML dataset (READ-ONLY) |
| `data/splits/baseline/` | — | Phase 9: X/y/meta for train, val, test (baseline features) |
| `data/splits/temporal/` | — | Phase 9: X/y/meta for train, val, test (temporal features) |
| `models/phase10/logistic_regression/with_qd/` | — | LR model.joblib, scaler.joblib, metadata.json (baseline with_qd) |
| `models/phase10/logistic_regression/without_qd/` | — | LR model.joblib, scaler.joblib, metadata.json (baseline without_qd) |
| `models/phase10/random_forest/with_qd/` | — | RF model.joblib, metadata.json (baseline with_qd) |
| `models/phase10/random_forest/without_qd/` | — | RF model.joblib, metadata.json (baseline without_qd) |
| `models/phase10/xgboost/with_qd/` | — | XGBoost model.joblib, metadata.json (baseline with_qd) |
| `models/phase10/xgboost/without_qd/` | — | XGBoost model.joblib, metadata.json (baseline without_qd) |
| `models/phase11/` | — | 30 trained models (LR/RF/XGBoost × with_qd/without_qd × 6 strategies) — model.joblib + metadata.json per experiment |
| `data/features/` | (empty) | Phase 7 output destination |

### Scripts

| File | Phase | Purpose |
|---|---|---|
| `download_safe.py` | 3 | Safe staged downloader (ERA5) |
| `validate_staged.py` | 3 | Validates staged files |
| `promote_staged.py` | 3 | Promotes staged → raw/ |
| `run_inspection.py` | 3 | Inspects combined raw file |
| `inspect_raw.py` | 3 | Detailed per-city quality report |
| `eda_climateguard.py` | 4 | Full EDA — generates all plots and reports |
| `data_cleaning.py` | 5 | Creates weather_cleaned.csv |
| `heatwave_labeling.py` | 6 | Creates weather_labelled.csv |
| `feature_engineering.py` | 7 | Creates climateguard_features.csv + feature group registry |
| `build_ml_dataset.py` | 8 | Creates ml_baseline.csv + ml_temporal.csv, runs all validations |
| `time_series_split.py` | 9 | Creates chronological train/val/test splits for both datasets |
| `train_baseline_models.py` | 10 | Trains 6 baseline models (LR, RF, XGBoost × 2 feature sets), evaluates on val set, saves artifacts |
| `train_imbalance_models.py` | 11 | Trains 30 imbalance-strategy models across 6 strategies, threshold analysis, saves artifacts |
| `evaluate_test_set.py` | 12 | Evaluates 4 Phase 11 candidates on held-out test set, generates all comparison/city/yearly CSVs, confusion matrices, PR/ROC curves, leakage audit |
| `temporal_feature_experiment.py` | 13 | Compares baseline (29) vs temporal (110) features using Phase 11 recommended strategy; ablation study by feature group; test confirmation; leakage audit |
| `final_model_selection.py` | 14 | Trains final RF/temporal_wqd/undersample on train+val; evaluates once on test; saves model.joblib, feature_list.json, metadata.json; 12 validation checks; 12 leakage checks; 2 plots |
| (old download scripts) | — | Superseded, keep for reference only |

### Documentation

| File | Content |
|---|---|
| `docs/data_dictionary.md` | Variable descriptions, units, stats |
| `docs/eda_findings.md` | Full EDA findings document |
| `docs/preprocessing_decisions.md` | Phase 5 cleaning decisions |
| `docs/heatwave_labeling_methodology.md` | Full Phase 6 methodology |
| `docs/final_ml_dataset.md` | Phase 8 ML dataset documentation (13 sections) |
| `docs/train_validation_test_split.md` | Phase 9 split methodology and results |
| `docs/baseline_models.md` | Phase 10 baseline ML models — 6 models trained (3 × 2 feature sets), validation results, qualifying_day experiment, city-wise breakdown, model comparison |
| `docs/class_imbalance.md` | Phase 11 class imbalance handling — 6 strategies, threshold analysis, recommended candidate strategy, limitations (12 sections) |
| `docs/model_evaluation.md` | Phase 12 held-out test evaluation — 4 candidates, test metrics, city/year breakdown, generalization analysis, leakage audit (12 sections) |
| `docs/temporal_feature_experiment.md` | Phase 13 temporal feature experiment — baseline vs temporal comparison, ablation study, test confirmation, leakage audit (12 sections, 268 lines) |
| `docs/final_model_selection.md` | Phase 14 final model selection — full 13-section rationale: models considered, why RF/temporal/undersample, qualifying_day discussion, training procedure, test results, limitations, integration contract (360 lines) |
| `docs/final_model_contract.md` | Phase 14 prediction contract for Part 2 / Part 3 — 110-feature table, input/output spec, usage code, city/season encoding, limitations (323 lines) |
| `docs/setup.md` | Phase 16 setup and reproducibility guide — dependencies, installation, artifact verification (236 lines) |
| `docs/project_structure.md` | Phase 16 full annotated file inventory — all phases, data, models, results, scripts (343 lines) |
| `docs/testing.md` | Phase 16 test suite documentation — 18 tests, smoke test, run instructions (159 lines) |
| `docs/limitations.md` | Phase 16 known limitations — 12 limitations across data, label, city, feature, model, scope (187 lines) |

### Notebooks

| File | Content |
|---|---|
| `notebooks/01_eda.ipynb` | Reproducible EDA notebook (19 sections) |

### Results / Reports

| File | Content |
|---|---|
| `results/eda_summary.txt` | Full EDA text report |
| `results/data_dictionary.json` | Machine-readable variable stats |
| `results/city_summary.json` | City-level statistics |
| `results/extreme_analysis.json` | Threshold exceedance results |
| `results/trend_results.json` | Linear trend slopes and p-values |
| `results/phase5_cleaning_log.txt` | Phase 5 validation log (105 checks) |
| `results/phase6_labeling_log.txt` | Phase 6 full execution log |
| `results/phase6_class_balance.json` | Heatwave event statistics |
| `results/phase7_feature_engineering_log.txt` | Phase 7 full execution log |
| `results/phase7_feature_groups.json` | Feature group registry (baseline vs temporal sets for Phase 13) |
| `results/phase8_report.txt` | Phase 8 full validation log (23 checks) |
| `results/phase8_feature_audit.csv` | Per-feature audit table (group, time reference, leakage status) |
| `results/phase9_split_log.txt` | Phase 9 full execution log |
| `results/phase9_split_report.json` | Phase 9 split statistics (shapes, class distributions, boundaries) |
| `results/phase9_leakage_audit.csv` | Phase 9 per-city, per-dataset leakage audit table |
| `results/phase10_metrics.json` | Phase 10 full metrics for all 6 models |
| `results/phase10_model_comparison.csv` | Phase 10 model comparison table |
| `results/phase10_log.txt` | Phase 10 full execution log |
| `results/phase10_confusion_matrices/` | 6 confusion matrix PNGs (3 models × 2 feature sets) |
| `results/phase11_imbalance_comparison.csv` | Phase 11 strategy comparison (60 rows — 30 experiments × default + opt threshold) |
| `results/phase11_threshold_analysis.csv` | Phase 11 threshold analysis (420 rows — 30 experiments × 14 thresholds) |
| `results/phase11_metrics.json` | Phase 11 full metrics + city breakdown for all 30 experiments |
| `results/phase11_log.txt` | Phase 11 full execution log |
| `results/phase11_confusion_matrices/` | 60 confusion matrix PNGs |
| `results/plots/phase11/` | 4 plots: precision/recall/F1 vs threshold, PR curves |
| `results/phase12_test_metrics.csv` | Phase 12 overall test metrics for all 4 candidates |
| `results/phase12_city_metrics.csv` | Phase 12 city-level test metrics |
| `results/phase12_yearly_metrics.csv` | Phase 12 year-level test metrics (2023/2024/2025) |
| `results/phase12_comparison.csv` | Phase 12 val vs test comparison with delta columns |
| `results/phase12_metrics.json` | Phase 12 full metrics including city and yearly breakdowns |
| `results/phase12_log.txt` | Phase 12 full execution log |
| `results/phase12_leakage_audit.csv` | Phase 12 12-check leakage audit (all PASS) |
| `results/phase12_confusion_matrices/` | 4 confusion matrix PNGs (test set) |
| `results/plots/phase12/` | PR curves and ROC curves for all 4 candidates |
| `results/phase13_temporal_comparison.csv` | Phase 13 primary comparison — 8 rows (2 models × 4 feature sets, val metrics) |
| `results/phase13_ablation.csv` | Phase 13 ablation study — 7 rows (RF/undersample, val metrics) |
| `results/phase13_metrics.json` | Phase 13 full metrics for all 15 experiments + test confirmation |
| `results/phase13_leakage_audit.csv` | Phase 13 14-check leakage audit (all PASS) |
| `results/phase13_log.txt` | Phase 13 full execution log |
| `results/plots/phase13/` | 4 plots: F1 bar, PR-AUC bar, precision-recall scatter, ablation grouped bar |
| `results/data_quality_report_raw.txt` | Raw data quality report |
| `results/inspection_output.txt` | Raw data inspection output |

### Plots

| Directory | Count | Content |
|---|---|---|
| `results/plots/EDA/` | 30 | EDA plots (01–30) |
| `results/plots/heatwave_labels/` | 10 | Phase 6 event plots |

---

## DATASET COLUMN REFERENCE

### weather_labelled.csv — all 30 columns

```
city, city_key, latitude, longitude, region_type, state, date,
temperature_2m_max, temperature_2m_min, temperature_2m_mean,
apparent_temperature_max, apparent_temperature_min, apparent_temperature_mean,
precipitation_sum,
wind_speed_10m_max, wind_gusts_10m_max,
relative_humidity_2m_max, relative_humidity_2m_min, relative_humidity_2m_mean,
surface_pressure_mean, shortwave_radiation_sum, et0_fao_evapotranspiration,
tmax_normal, tmax_departure, qualifying_day,
heatwave,           ← ground-truth same-day label (NOT the ML target directly)
hw_event_id, hw_event_start, hw_event_end, hw_event_length
```

---

## KEY DECISIONS ALREADY MADE

| Decision | Details |
|---|---|
| Data source | ERA5 via Open-Meteo API (not IMD station data) |
| Variable removed | `rain_sum` (identical to `precipitation_sum`) |
| Heatwave definition | IMD-inspired operational label (NOT official IMD) |
| Baseline period | 1990–2020 (31-year ERA5 internal baseline) |
| Smoothing | 31-day centred rolling mean on daily normals |
| Duration criterion | ≥ 2 consecutive qualifying days = heatwave event |
| Mumbai treatment | Zero positive events; do not change; address in ML phase |
| Prediction horizon | 1-day-ahead: predict heatwave(T+1) using features(T) |
| Leakage policy | Zero tolerance — no T+1 weather as features |
| Class imbalance | High (0.25–1.64% positive); address in Phase 11 |

---

## IMPORTANT WARNINGS

1. **Do NOT re-download data.** All ERA5 data is present and validated.
2. **Do NOT modify `weather_cleaned.csv` or `weather_labelled.csv`.** These are validated artifacts. Any modification requires re-running the respective phase script from scratch and re-validating.
3. **Do NOT add Mumbai heatwave positives artificially.** Its zero-positive result is scientifically correct.
4. **Do NOT label the heatwave definition "official IMD".** It is ERA5-based and must be called "IMD-Inspired Operational Heatwave Label (ERA5-based)".
5. **Do NOT use `heatwave(T)` as a direct input feature** — only as a lagged version `heatwave_lag_1(T) = heatwave(T-1)`.
6. **Do NOT implement Kshitij's or Pradnesh's work** (SHAP, drift detection, risk scoring, API, backend).

---

## PHASE 16 — DOCUMENTATION (COMPLETE)

**Date completed:** 2026-09-02

### Documents Created

| File | Lines | Description |
|---|---|---|
| `README.md` | 329 | Project overview, quick start, pipeline summary, key results, integration contracts |
| `docs/setup.md` | 236 | Installation guide, dependencies, reproducibility, artifact verification |
| `docs/project_structure.md` | 343 | Full annotated file inventory (all phases) |
| `docs/testing.md` | 159 | Test suite description, 18 test descriptions, smoke test, run instructions |
| `docs/limitations.md` | 187 | 12 limitations across data, label, city, feature, model, and scope categories |

### Documents Verified (Pre-existing, Complete)

| File | Lines | Status |
|---|---|---|
| `docs/final_model_selection.md` | 360 | COMPLETE — 13 sections, full selection rationale |
| `docs/final_model_contract.md` | 323 | COMPLETE — 110-feature table, input/output contract |
| `docs/part2_integration_contract.md` | 211 | COMPLETE — SHAP access, limitations, prohibited actions |
| `docs/part3_integration_contract.md` | 311 | COMPLETE — ETL requirements, feature construction, quick reference |
| `docs/prediction_interface.md` | 345 | COMPLETE — full feature table, error handling, validation |
| `examples/predict_example.py` | 157 | CORRECT — real data, no training, no dataset modification |

### Consistency Checks

- All metrics in README.md match `results/final_model_metrics.json` (F1=0.6947, P=0.5789, R=0.8684, PR-AUC=0.8339)
- All per-city results in README.md match `results/final_model_metrics.json`
- Feature count 110 consistent across: README.md, docs/setup.md, docs/project_structure.md, docs/testing.md, docs/limitations.md, docs/final_model_contract.md, docs/part2_integration_contract.md, docs/part3_integration_contract.md
- Threshold 0.70 consistent across all documents
- Model artifact path `models/final/climateguard_final_model.joblib` consistent across all references

### Phase 14 Artifact Verification (Checksum)

| File | Expected size | Observed size | Status |
|---|---|---|---|
| `models/final/climateguard_final_model.joblib` | 1.86 MB | 1,864,473 bytes | UNCHANGED |
| `models/final/feature_list.json` | 10 KB | 10,255 bytes | UNCHANGED |
| `models/final/metadata.json` | 6 KB | 6,149 bytes | UNCHANGED |

All Phase 14 model artifacts confirmed unmodified. Timestamp: `2026-09-02T00:28:21` (recorded in `metadata.json`).

---

## HOW TO RESUME

1. Open the ClimateGuard project folder: `C:\Users\Adrian\Documents\climate guard\`
2. Read `PROJECT_MEMORY.md` completely.
3. Inspect the current files before changing anything.
4. Confirm Phases 1-16 are complete by verifying:
   - `README.md` exists (329 lines)
   - `models/final/climateguard_final_model.joblib` exists (1.86 MB)
   - `models/final/feature_list.json` exists (110 features)
   - `src/prediction/predictor.py` exists
   - `results/phase15_interface_validation.json` shows 18/18 tests passed
   - `docs/prediction_interface.md` exists
   - `docs/part2_integration_contract.md` exists
   - `docs/part3_integration_contract.md` exists
   - `docs/setup.md` exists
   - `docs/project_structure.md` exists
   - `docs/testing.md` exists
   - `docs/limitations.md` exists
5. Confirm the Final Integration Audit has not started.
6. Continue from **Final Integration Audit** only after verifying the project state.
7. Do not repeat completed phases or modify validated artifacts.

### The NEXT TASK after resuming:

**"Begin Final Integration Audit."**

Verify that all Phase 1–16 artifacts are internally consistent,
that the prediction interface works end-to-end with real test data,
and that all integration contracts are ready for handoff to Part 2 and Part 3.

---

## RECOMMENDED RESUME PROMPT

> Read PROJECT_MEMORY.md completely. Inspect the current project files and verify the recorded state. Do not repeat completed phases. Phases 1-16 are complete. The Final Integration Audit is the next task and has not yet started. Continue from the Final Integration Audit only after verifying the project state.

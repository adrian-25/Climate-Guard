# Project Structure

**ClimateGuard: Indian Heatwave Prediction**  
**Phase:** 16 — Documentation  
**Date:** 2026-09-02  
**Status:** COMPLETE

---

## 1. Top-Level Overview

```
climate guard/
├── README.md                          ← Project overview and quick start
├── PROJECT_MEMORY.md                  ← Complete project state record (all phases)
│
├── src/                               ← Importable Python packages
├── tests/                             ← Test suite
├── examples/                          ← Working usage examples
├── docs/                              ← All project documentation
├── models/                            ← Trained model artifacts
├── data/                              ← All datasets (validated, read-only)
├── results/                           ← Metrics, logs, plots from all phases
├── notebooks/                         ← Jupyter notebook (EDA)
├── plots/                             ← Legacy plot directory (from Phase 4)
│
└── (pipeline scripts)                 ← Phase 3–14 execution scripts
```

---

## 2. Source Code

```
src/
├── __init__.py                        ← Package root init
├── data_download.py                   ← ERA5 download helper (Phase 3, reference only)
└── prediction/                        ← Phase 15 prediction interface package
    ├── __init__.py                    ← Exports ClimateGuardPredictor
    └── predictor.py                   ← Main predictor class (496 lines)
```

### src/prediction/predictor.py

The primary deliverable for Part 2 and Part 3. Provides:
- `ClimateGuardPredictor` class — loads model once, validates input, returns predictions
- `PredictionResult` dataclass — structured output with probability, label, city, date, threshold
- Input validation: 110 features, no NaN, all numeric, correct column presence
- Exposes `predictor.model` (sklearn RandomForest) and `predictor.feature_names` for SHAP

---

## 3. Tests

```
tests/
└── test_prediction_interface.py       ← 18 unit tests for Phase 15 interface (458 lines)
```

All 18 tests pass. See `docs/testing.md` for full test descriptions and run instructions.

---

## 4. Examples

```
examples/
└── predict_example.py                 ← Working example with real Phase 9 test data (157 lines)
```

Runs from project root: `python examples/predict_example.py`  
Demonstrates single prediction, batch prediction, probability-only call, and explainability access.

---

## 5. Documentation

```
docs/
│
│  ── Phase 16 (this phase) ──
├── setup.md                           ← Installation and reproducibility guide
├── project_structure.md               ← This file — annotated file inventory
├── limitations.md                     ← Known limitations and research caveats
├── testing.md                         ← Test suite documentation
│
│  ── Phase 15 ──
├── prediction_interface.md            ← Full interface docs (345 lines)
├── part2_integration_contract.md      ← Integration contract for Part 2 / Kshitij (211 lines)
├── part3_integration_contract.md      ← Integration contract for Part 3 / Pradnesh (311 lines)
│
│  ── Phase 14 ──
├── final_model_selection.md           ← Selection rationale, 13 sections (360 lines)
├── final_model_contract.md            ← Prediction input/output specification (323 lines)
│
│  ── Phase 13 ──
├── temporal_feature_experiment.md     ← Baseline vs 110-feature comparison (268 lines)
│
│  ── Phase 12 ──
├── model_evaluation.md                ← Held-out test evaluation, 4 candidates (12 sections)
│
│  ── Phase 11 ──
├── class_imbalance.md                 ← 30 strategy experiments, threshold analysis (12 sections)
│
│  ── Phase 10 ──
├── baseline_models.md                 ← 6 baseline models, validation results
│
│  ── Phase 9 ──
├── train_validation_test_split.md     ← Split methodology, boundaries, leakage audit
│
│  ── Phase 8 ──
├── final_ml_dataset.md                ← ML dataset documentation, 13 sections
│
│  ── Phase 7 ──
│  (see results/phase7_feature_groups.json for machine-readable feature registry)
│
│  ── Phase 6 ──
├── heatwave_labeling_methodology.md   ← IMD-inspired label definition, full methodology
│
│  ── Phase 5 ──
├── preprocessing_decisions.md         ← Data cleaning decisions (Phase 5)
│
│  ── Phase 4 ──
├── eda_findings.md                    ← EDA findings (16 sections)
├── data_dictionary.md                 ← All variable definitions and units
```

---

## 6. Models

```
models/
│
├── final/                             ← ⚠️ LOCKED — Phase 14 final artifacts
│   ├── climateguard_final_model.joblib  ← Trained RF model (1.86 MB)
│   ├── feature_list.json              ← Ordered list of 110 features
│   └── metadata.json                  ← Full config, parameters, test metrics (6 KB)
│
├── phase13/                           ← Phase 13 experimental models (reference only)
│   ├── Random_Forest/
│   │   ├── temporal_wqd/random_undersample/  ← Best Phase 13 RF model
│   │   ├── temporal_nqd/random_undersample/
│   │   ├── baseline_wqd/random_undersample/
│   │   ├── baseline_nqd/random_undersample/
│   │   └── ablation/{7 feature sets}/
│   └── XGBoost/
│       ├── temporal_wqd/baseline_weight/
│       ├── temporal_nqd/baseline_weight/
│       ├── baseline_wqd/baseline_weight/
│       └── baseline_nqd/baseline_weight/
│
├── phase11/                           ← Phase 11 strategy models (reference only)
│   ├── Random_Forest/{6 strategy variants}/
│   ├── XGBoost/{6 strategy variants}/
│   └── Logistic_Regression/{6 strategy variants}/
│
└── phase10/                           ← Phase 10 baseline models (reference only)
    ├── logistic_regression/{with_qd, without_qd}/
    ├── random_forest/{with_qd, without_qd}/
    └── xgboost/{with_qd, without_qd}/
```

**Only `models/final/` is used in production.** All other model directories are historical experiment artifacts for reference.

---

## 7. Data

```
data/
│
├── raw/                               ← ⚠️ READ-ONLY — ERA5 raw data
│   ├── all_cities_era5_raw.csv        ← Master combined file (8.40 MB, MD5: 71d25a...)
│   ├── delhi_era5_raw.csv             ← Delhi individual file (1.61 MB)
│   ├── lucknow_era5_raw.csv           ← Lucknow individual file (1.72 MB)
│   ├── nagpur_era5_raw.csv            ← Nagpur individual file (1.67 MB)
│   ├── ahmedabad_era5_raw.csv         ← Ahmedabad individual file (1.70 MB)
│   └── mumbai_era5_raw.csv            ← Mumbai individual file (1.69 MB)
│
├── processed/                         ← ⚠️ READ-ONLY — cleaned and labelled data
│   ├── weather_cleaned.csv            ← Phase 5 output (8.14 MB, 65,135 × 22)
│   └── weather_labelled.csv           ← Phase 6 output (11.12 MB, 65,135 × 30)
│
├── features/                          ← ⚠️ READ-ONLY — feature-engineered datasets
│   ├── climateguard_features.csv      ← Phase 7 output (59.91 MB, 65,095 × 121)
│   ├── ml_baseline.csv                ← Phase 8 baseline ML data (16.66 MB, 65,080 × 40)
│   └── ml_temporal.csv                ← Phase 8 temporal ML data (59.22 MB, 65,080 × 121)
│
├── splits/                            ← ⚠️ READ-ONLY — train/val/test splits
│   ├── baseline/
│   │   ├── X_train.csv (54,735 × 29)  ← Training features (29 baseline)
│   │   ├── X_val.csv (5,480 × 29)     ← Validation features
│   │   ├── X_test.csv (4,865 × 29)    ← Test features (2023–2025)
│   │   ├── y_train.csv / y_val.csv / y_test.csv   ← Target labels
│   │   └── meta_train.csv / meta_val.csv / meta_test.csv  ← ID columns
│   │
│   └── temporal/
│       ├── X_train.csv (54,735 × 110) ← Training features (110 temporal)
│       ├── X_val.csv (5,480 × 110)
│       ├── X_test.csv (4,865 × 110)   ← Used by predict_example.py
│       ├── y_train.csv / y_val.csv / y_test.csv
│       └── meta_train.csv / meta_val.csv / meta_test.csv
│
├── raw_staging/                       ← Phase 3 staging area (ignore — historical)
└── raw_backup/                        ← Old broken backups (ignore — historical)
```

### Data Summary

| File | Rows | Columns | Date Range | Description |
|---|---|---|---|---|
| `all_cities_era5_raw.csv` | 65,135 | 23 | 1990-01-01 to 2025-08-31 | Master raw data |
| `weather_cleaned.csv` | 65,135 | 22 | 1990-01-01 to 2025-08-31 | `rain_sum` removed |
| `weather_labelled.csv` | 65,135 | 30 | 1990-01-01 to 2025-08-31 | + heatwave labels |
| `climateguard_features.csv` | 65,095 | 121 | 1990-01-11 to 2025-08-30 | + 91 engineered features |
| `ml_baseline.csv` | 65,080 | 40 | 1990-01-11 to 2025-08-30 | 29 features + target |
| `ml_temporal.csv` | 65,080 | 121 | 1990-01-11 to 2025-08-30 | 110 features + target |

---

## 8. Results

```
results/
│
├── (phase-level logs and metrics)
│   ├── eda_summary.txt                ← Phase 4 full EDA text report
│   ├── data_dictionary.json
│   ├── city_summary.json
│   ├── extreme_analysis.json
│   ├── trend_results.json
│   ├── phase5_cleaning_log.txt        ← 105 validation checks
│   ├── phase6_labeling_log.txt
│   ├── phase6_class_balance.json
│   ├── phase7_feature_engineering_log.txt
│   ├── phase7_feature_groups.json     ← Machine-readable feature registry
│   ├── phase8_report.txt              ← 23 validation checks
│   ├── phase8_feature_audit.csv
│   ├── phase9_split_log.txt
│   ├── phase9_split_report.json
│   ├── phase9_leakage_audit.csv
│   ├── phase10_metrics.json
│   ├── phase10_model_comparison.csv
│   ├── phase10_log.txt
│   ├── phase11_imbalance_comparison.csv
│   ├── phase11_threshold_analysis.csv ← 420 rows (30 experiments × 14 thresholds)
│   ├── phase11_metrics.json
│   ├── phase11_log.txt
│   ├── phase12_test_metrics.csv
│   ├── phase12_city_metrics.csv
│   ├── phase12_yearly_metrics.csv
│   ├── phase12_comparison.csv
│   ├── phase12_metrics.json
│   ├── phase12_log.txt
│   ├── phase12_leakage_audit.csv
│   ├── phase13_temporal_comparison.csv
│   ├── phase13_ablation.csv
│   ├── phase13_metrics.json
│   ├── phase13_leakage_audit.csv
│   ├── phase13_log.txt
│   ├── final_model_metrics.json       ← Phase 14 final test metrics + city breakdown
│   ├── final_model_comparison.csv
│   ├── final_model_leakage_audit.csv  ← 12/12 PASS
│   ├── final_model_log.txt
│   ├── phase15_interface_test.txt
│   └── phase15_interface_validation.json  ← 18/18 tests PASS
│
├── plots/
│   ├── EDA/                           ← 30 Phase 4 EDA plots (numbered 01–30)
│   ├── heatwave_labels/               ← 10 Phase 6 event plots
│   ├── phase11/                       ← 4 Phase 11 imbalance analysis plots
│   ├── phase12/                       ← PR/ROC curves for 4 candidate models
│   ├── phase13/                       ← 4 Phase 13 temporal comparison plots
│   ├── final_confusion_matrix.png     ← Phase 14 test confusion matrix
│   └── final_precision_recall.png     ← Phase 14 test PR curve with operating point
│
├── phase10_confusion_matrices/        ← 6 Phase 10 confusion matrices
├── phase11_confusion_matrices/        ← 60 Phase 11 confusion matrices
└── phase12_confusion_matrices/        ← 4 Phase 12 test confusion matrices
```

---

## 9. Pipeline Scripts

```
(project root)
├── download_safe.py                   ← Phase 3: ERA5 staged download
├── validate_staged.py                 ← Phase 3: Validates staged data files
├── promote_staged.py                  ← Phase 3: Promotes staging → raw/
├── run_inspection.py                  ← Phase 3: Combined raw file inspection
├── inspect_raw.py                     ← Phase 3: Per-city quality report
├── eda_climateguard.py                ← Phase 4: Full EDA (30 plots, 5 reports)
├── data_cleaning.py                   ← Phase 5: Creates weather_cleaned.csv
├── heatwave_labeling.py               ← Phase 6: Creates weather_labelled.csv
├── feature_engineering.py             ← Phase 7: Creates climateguard_features.csv
├── build_ml_dataset.py                ← Phase 8: Creates ml_baseline.csv + ml_temporal.csv
├── time_series_split.py               ← Phase 9: Creates chronological splits
├── train_baseline_models.py           ← Phase 10: Trains 6 baseline models
├── train_imbalance_models.py          ← Phase 11: Trains 30 imbalance strategy models
├── evaluate_test_set.py               ← Phase 12: Evaluates 4 candidates on test set
├── temporal_feature_experiment.py     ← Phase 13: Temporal feature comparison + ablation
├── final_model_selection.py           ← Phase 14: Trains and saves final model
├── run_phase15_tests.py               ← Phase 15: Test runner
│
└── (legacy download scripts)          ← Superseded by download_safe.py
    ├── climateguard_download_four_cities.py
    ├── download_final.py
    ├── download_remaining.py
    ├── redownload_all.py
    ├── check_download.py
    ├── download_missing2.py
    ├── download_missing.py
    ├── diagnose_zscore_nans.py
    └── inspect_dates.py
```

---

## 10. Notebooks

```
notebooks/
└── 01_eda.ipynb                       ← Phase 4 EDA notebook (19 sections)
```

The notebook mirrors `eda_climateguard.py`. All EDA outputs are already saved in `results/`. The notebook is for interactive exploration only.

---

## 11. Artifact Protection Rules

| Status | Files | Rule |
|---|---|---|
| **LOCKED** | `models/final/climateguard_final_model.joblib` | Never retrain or replace |
| **LOCKED** | `models/final/feature_list.json` | Never modify |
| **LOCKED** | `models/final/metadata.json` | Never modify |
| **READ-ONLY** | All files in `data/raw/` | Never modify |
| **READ-ONLY** | All files in `data/processed/` | Never modify |
| **READ-ONLY** | All files in `data/features/` | Never modify |
| **READ-ONLY** | All files in `data/splits/` | Never modify |
| **Reference** | `models/phase10/`, `models/phase11/`, `models/phase13/` | Historical only |

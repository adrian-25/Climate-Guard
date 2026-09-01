# Testing Documentation

**ClimateGuard: Indian Heatwave Prediction**  
**Phase:** 15–16  
**Date:** 2026-09-02  
**Status:** 18/18 tests passing

---

## 1. Overview

The ClimateGuard test suite validates the Phase 15 prediction interface (`src/prediction/predictor.py`). Tests verify the complete prediction pipeline: model loading, input validation, probability bounds, threshold logic, batch prediction, error handling, and SHAP-related access.

No pipeline or model training is tested — the test suite operates exclusively against the locked Phase 14 artifacts.

---

## 2. Test File

```
tests/test_prediction_interface.py    ← 18 tests (458 lines)
```

---

## 3. Running the Tests

### Without pytest (standard library only)

```bash
# Run from project root
python tests/test_prediction_interface.py
```

Expected output:

```
......................
----------------------------------------------------------------------
Ran 18 tests in X.XXXs

OK
```

### With pytest

```bash
# Run from project root
python -m pytest tests/test_prediction_interface.py -v
```

Verbose output shows each test name and result:

```
tests/test_prediction_interface.py::TestClimateGuardPredictor::test_model_loads PASSED
tests/test_prediction_interface.py::TestClimateGuardPredictor::test_feature_list_loads PASSED
...
```

---

## 4. Test Results

**Phase 15 test run (2026-09-02):** 18/18 PASS  
Results saved in `results/phase15_interface_validation.json`

---

## 5. Test Descriptions

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_model_loads` | Predictor constructs without error; `predictor.model` is set |
| 2 | `test_feature_list_loads` | `predictor.feature_names` is populated with 110 strings |
| 3 | `test_exactly_110_features` | `n_features=110`, `len(feature_names)=110`, `model.n_features_in_=110` |
| 4 | `test_valid_sample_prediction` | Real data row returns a `PredictionResult` object |
| 5 | `test_probability_in_range` | `prediction_probability` is a float in `[0.0, 1.0]` |
| 6 | `test_prediction_binary` | All `prediction_label` values are 0 or 1 (tested on 10 rows) |
| 7 | `test_threshold_applied` | `label == (prob >= 0.70)` holds for all 20 test rows |
| 8 | `test_missing_feature_raises` | `ValueError` raised when a required feature column is missing |
| 9 | `test_nan_input_raises` | `ValueError` raised when input contains a NaN value |
| 10 | `test_batch_prediction` | `predict_batch()` returns DataFrame with `prediction_probability` and `prediction_label` columns; all original columns preserved |
| 11 | `test_input_not_modified` | The original input DataFrame is unchanged after `predict_batch()` |
| 12 | `test_feature_ordering` | `predictor.feature_names` matches `feature_list.json` order exactly |
| 13 | `test_target_not_in_features` | `heatwave_next_day` is absent from the feature list |
| 14 | `test_wrong_type_raises` | `TypeError` raised when passing a plain list or wrong type to `predict_batch()` |
| 15 | `test_info_method` | `predictor.info()` returns a dict with all required keys |
| 16 | `test_get_feature_matrix` | `predictor.get_feature_matrix()` returns a DataFrame of shape `(n, 110)` in model column order |
| 17 | `test_predict_single_row_only` | `predict()` raises `ValueError` when passed a multi-row DataFrame |
| 18 | `test_empty_batch_raises` | `predict_batch()` raises `ValueError` on empty DataFrame |

---

## 6. Real-Data Smoke Test

In addition to unit tests, a smoke test runs predictions on the first 5 rows of the Phase 9 held-out test split:

```
Rows tested: 5 (Ahmedabad, 2023-01-01 to 2023-01-05)
Expected: all 5 normal days (heatwave_next_day = 0)
Result:   5/5 correct (probability < 0.70 for all rows)
```

This smoke test is run as part of `examples/predict_example.py`:

```bash
python examples/predict_example.py
```

---

## 7. What the Tests Do NOT Cover

These items are intentionally out of scope for the Phase 15 unit tests:

1. **Pipeline correctness** (data cleaning, feature engineering, model training) — validated by per-phase logs and leakage audits in `results/`
2. **Numerical accuracy of predictions** — the Phase 14 final test evaluation (`results/final_model_metrics.json`) is the ground truth for prediction quality
3. **SHAP value correctness** — explainability is Part 2 (Kshitij's) scope
4. **ETL integration** — end-to-end ETL from live data is Part 3 (Pradnesh's) scope
5. **Drift detection** — not implemented in Part 1

---

## 8. Dependencies

The test file uses only:
- Python standard library (`unittest`, `json`, `math`, `pathlib`, `sys`)
- `numpy`
- `pandas`
- `src.prediction` (the Phase 15 interface under test)

No additional testing framework is required beyond Python's built-in `unittest`. `pytest` is optional.

---

## 9. Constants Tested

The tests import and verify these constants directly from `predictor.py`:

| Constant | Value | Verified by |
|---|---|---|
| `THRESHOLD` | `0.70` | `test_threshold_applied` |
| `N_FEATURES` | `110` | `test_exactly_110_features` |
| `TARGET_NAME` | `"heatwave_next_day"` | `test_target_not_in_features` |

---

## 10. Leakage Audit (Not Unit Tests)

Leakage checks are not part of the unit test suite — they were run programmatically during Phase 14 training and again during Phase 15 interface validation. Results:

| Audit | Result |
|---|---|
| Phase 14 model validation (12 checks) | 12/12 PASS |
| Phase 15 interface validation (18 tests) | 18/18 PASS |

Results stored in:
- `results/final_model_leakage_audit.csv`
- `results/phase15_interface_validation.json`

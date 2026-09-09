# ClimateGuard Final Independent Audit

**Audit type:** Independent Final Audit  
**Date:** 2026-09-10  
**Auditor:** Kiro (independent — did not trust previous completion reports; verified from source)  
**Project root:** `C:\Users\Adrian\Documents\climate guard\`

---

## 1. Executive Summary

ClimateGuard is a three-part student project implementing heatwave prediction for five Indian cities using ERA5 reanalysis weather data. This audit independently verified the complete repository — source code, tests, data artifacts, model files, documentation, and integration — without modifying any files.

All three parts are functional and internally consistent. 234 tests pass across all parts. The real-data end-to-end pipeline produces deterministic, sensible outputs on actual test-set rows. The final model artifact is unchanged from its documented state. No security issues were found. No fabricated test results were detected.

**The project is: READY WITH WARNINGS**

The warnings are all pre-existing, documented limitations — not newly discovered blockers. The core system works correctly end-to-end.

---

## 2. Final Status

## ✅ READY WITH WARNINGS

Core system works correctly. All tests pass. Real-data E2E passes. Model is unchanged. Integration is complete. Warnings are pre-existing documented limitations (SHAP unavailable, qualifying_day correlation, pre-built feature requirement, city coverage gaps).

---

## 3. Repository Audit

### Structure

```
climate guard/
  src/
    __init__.py              ✅ exists
    prediction/              ✅ __init__.py + predictor.py
    risk/                    ✅ __init__.py + risk_assessment.py
    adaptation/              ✅ __init__.py + recommendations.py
    explainability/          ✅ __init__.py + explainer.py
    risk_engine/             ✅ __init__.py + engine.py
    etl/                     ✅ __init__.py + validator.py + transformer.py + pipeline.py
    expert_rules/            ✅ __init__.py + rules.py + engine.py
    integration/             ✅ __init__.py + pipeline.py
  tests/
    test_prediction_interface.py  ✅ 458 lines
    test_part2.py                 ✅ 939 lines
    test_part3.py                 ✅ 1,179 lines
    __init__.py                   ⚠️ ABSENT (see Warnings)
  models/final/
    climateguard_final_model.joblib  ✅ 1,864,473 bytes
    feature_list.json                ✅ 10,255 bytes
    metadata.json                    ✅ 6,149 bytes
  docs/                        ✅ 14+ documents present
  examples/
    predict_example.py         ✅ exists, no hardcoded paths
    part3_pipeline_example.py  ✅ exists, no hardcoded paths
  results/
    part3_smoke_test.json      ✅ 23,671 bytes, PASS
  README.md                    ✅ 329 lines
  .gitignore                   ✅ appropriate exclusions
```

### Findings

| Check | Result |
|---|---|
| All `__init__.py` in `src/` subdirectories | ✅ PASS (8/8 present) |
| `src/__init__.py` | ✅ PASS |
| `tests/__init__.py` | ⚠️ ABSENT — non-blocking (unittest discover still works) |
| Hardcoded absolute paths in source | ✅ NONE — all paths use `Path(__file__).resolve().parent.parent.parent` |
| TODO/FIXME/HACK comments affecting core | ✅ NONE found |
| Broken imports | ✅ NONE — all cross-module imports verified against actual symbols |
| Duplicate prediction/risk logic | ✅ NONE — `pipeline.py` delegates entirely to Part 1/2 APIs |
| Placeholder/stub files | ✅ NONE |
| Temporary files in src/ | ✅ NONE (audit helper `_audit_checks.py` was deleted after use) |
| Abandoned code | ✅ NONE in core src/ |
| `__pycache__` directories | ⚠️ 11 present (normal Python bytecode cache — `.gitignore` handles it) |

**Utility scripts in root** (`run_inspection.py`, `run_part3_smoke_test.py`, `run_phase15_tests.py`) are project-level runners, not core source. They are not imported by any test or source module. No concern.

---

## 4. Part 1 Audit

**Scope:** Data pipeline from raw ERA5 → final prediction interface.

### Data Pipeline

| Stage | File | Status |
|---|---|---|
| Raw data | `data/raw/all_cities_era5_raw.csv` | ✅ 65,135 × 23, 0 missing, 0 duplicates |
| Cleaning | `data/processed/weather_cleaned.csv` | ✅ 65,135 × 22 (rain_sum removed) |
| Labelling | `data/processed/weather_labelled.csv` | ✅ 65,135 × 30 (8 label columns added) |
| Feature engineering | `data/features/` | ✅ 65,095 × 121 (phase7), 65,080 × 121 (phase8) |
| Chronological split | `data/splits/temporal/` | ✅ train/val/test with strict date separation |
| Final model | `models/final/` | ✅ RF, 110 features, threshold=0.70 |

### Key verification

- Raw data: 5 cities (delhi, lucknow, nagpur, ahmedabad, mumbai), 1990-01-01 to 2025-08-31, **0 missing values**
- Heatwave label: IMD-inspired (NOT official IMD). Plains: Tmax ≥ 40°C + departure ≥ 4.5°C + duration ≥ 2 days, or Tmax ≥ 45°C. Coastal: Tmax ≥ 37°C + departure ≥ 4.5°C + duration ≥ 2 days. Documented correctly.
- Mumbai: 0 heatwave positives in all splits — correctly documented as scientifically accurate given maritime climate
- Ahmedabad: 0 test-period positives (all 32 positives fall in training window) — documented

---

## 5. Data & Leakage Audit

### Chronological split leakage

| Boundary | Check | Result |
|---|---|---|
| Train max date | 2019-12-31 | — |
| Val min date | 2020-01-01 | train_max < val_min: **PASS** |
| Val max date | 2022-12-31 | — |
| Test min date | 2023-01-01 | val_max < test_min: **PASS** |

**No temporal leakage** across train/val/test splits.

### Target leakage

- `heatwave_next_day` NOT present in `X_test.csv` columns: **PASS**
- `heatwave` NOT present in `X_test.csv` columns: **PASS**
- `X_test.csv` shape: (4865, 110) — exactly 110 feature columns: **PASS**

### qualifying_day

`qualifying_day` is feature index 26. It is derived at time T using the same threshold conditions as the heatwave label definition. This creates a strong structural correlation between the feature and the next-day target (since a day qualifying at T is predictive of T+1 also qualifying). This is **NOT leakage** — `qualifying_day` uses only T data, not T+1 — but it is a proximity concern.

**Documentation status:** ✅ DOCUMENTED in:
- `docs/limitations.md`
- `docs/final_model_contract.md`
- `docs/final_model_selection.md`
- `src/explainability/explainer.py` (embedded note in all explanation outputs)
- `models/final/metadata.json` (`"qualifying_day_retained": true`)

The retention is a deliberate, documented engineering choice (removal degrades val-F1 by −0.076). The limitation is correctly disclosed. **No action required — classified as WARNING (documented).**

### Feature engineering leakage design

- All lag features: `shift(N)` with N ≥ 1 — uses past data only ✅
- All rolling features: `shift(1).rolling(N)` — window [T-N, T-1], excludes T ✅
- `heatwave_lag1` = `heatwave(T-1)` — lagged, safe ✅
- Phase 7 leakage audit: 14/14 checks PASS (per `results/phase13_leakage_audit.csv`)

---

## 6. Final Model Audit

| Property | Documented | Observed | Match |
|---|---|---|---|
| File | `models/final/climateguard_final_model.joblib` | EXISTS | ✅ |
| Size | 1,864,473 bytes (from metadata context) | 1,864,473 bytes | ✅ |
| MD5 | `24da8b976561761467b2f9cc563d4d4a` | `24da8b976561761467b2f9cc563d4d4a` | ✅ |
| SHA-256 | `1597d2d4ce97782f8c5ce060def7599eaa4821fbce580860735fd8960ad7c1aa` | same | ✅ |
| Model type | RandomForestClassifier | RandomForestClassifier (metadata.json) | ✅ |
| Feature count | 110 | 110 (feature_list.json, runtime) | ✅ |
| Threshold | 0.70 | 0.70 (metadata.json + runtime predictor) | ✅ |
| Saved at | 2026-09-02T00:28:21 | 2026-09-02T00:28:21 (metadata.json) | ✅ |
| n_estimators | 300 | 300 (metadata.json) | ✅ |
| max_depth | 10 | 10 (metadata.json) | ✅ |
| Training period | 1990-01-11 to 2022-12-31 | same (metadata.json) | ✅ |
| Test F1 | 0.6947 | 0.6947 (metadata.json, final_model_metrics.json) | ✅ |
| Test Recall | 0.8684 | 0.8684 | ✅ |

**Feature list consistency:** `predictor.feature_names == [e["name"] for e in feature_list.json]` → **True** (verified at runtime).

**Model loads successfully** via `joblib.load()` in `ClimateGuardPredictor.__init__()`.

---

## 7. Prediction Interface Audit

**Module:** `src/prediction/predictor.py`

### Implementation checks

| Check | Result |
|---|---|
| Model path: project-relative (not hardcoded) | ✅ `Path(__file__).resolve().parent.parent.parent / "models/final/..."` |
| Threshold constant: `THRESHOLD = 0.70` | ✅ defined at module level |
| Feature count constant: `N_FEATURES = 110` | ✅ defined at module level |
| Target name constant: `TARGET_NAME = "heatwave_next_day"` | ✅ defined at module level |
| Feature validation: checks exact 110 names + order | ✅ in `_validate_and_extract()` |
| NaN validation: rejects NaN inputs | ✅ tested in test_A09 |
| Non-numeric validation: rejects string inputs | ✅ tested in test_A10 |
| Batch prediction: `predict_batch(df)` | ✅ returns DataFrame |
| `PredictionResult.to_dict()` serialisable | ✅ |
| Passthrough columns ignored: city, date, etc. | ✅ `_PASSTHROUGH_COLS` set |

### Test execution (live run during this audit)

```
Result: 18/18 passed, 0 failed
Return code: 0
```

Individual test names verified (all PASS): `test_model_loads`, `test_feature_list_loads`, `test_exactly_110_features`, `test_valid_sample_prediction`, `test_probability_in_range`, `test_prediction_binary`, `test_threshold_applied`, `test_missing_feature_raises`, `test_nan_input_raises`, `test_batch_prediction`, `test_input_not_modified`, `test_feature_ordering`, `test_target_not_in_features`, `test_wrong_type_raises`, `test_info_method`, `test_get_feature_matrix`, `test_predict_single_row_only`, `test_empty_batch_raises`

---

## 8. Part 2 Audit

**Modules:** `src/risk/`, `src/adaptation/`, `src/explainability/`, `src/risk_engine/`

### Risk Assessment

| Check | Result |
|---|---|
| Risk thresholds defined | ✅ `RISK_THRESHOLDS` in `risk_assessment.py` |
| LOW: [0.00, 0.30) | ✅ verified from source |
| MODERATE: [0.30, 0.60) | ✅ verified from source |
| HIGH: [0.60, 0.80) | ✅ verified from source |
| EXTREME: [0.80, 1.00] | ✅ verified from source |
| Project-defined (not official IMD) | ✅ documented |
| Probability preserved verbatim | ✅ design confirmed in source |
| Deterministic | ✅ stateless function |

### Adaptation Recommendations

| Risk Level | Recommendations | Status |
|---|---|---|
| LOW | 3 | ✅ |
| MODERATE | 5 | ✅ |
| HIGH | 6 | ✅ |
| EXTREME | 7 | ✅ |

### Test execution (live run during this audit)

```
Ran 89 tests in 3.911s
OK
Return code: 0
```

Test groups A (Risk probability), B (Threshold boundaries), C (Risk level generation), D (Recommendation generation), E (Categories), F (Model access), G (Feature-name consistency), H (Explainability), I (Invalid input), J (Part 1→2 integration), K (Real-data smoke) — all PASS.

### Part 1 → Part 2 integration

`ClimateGuardRiskEngine.__init__()` accepts an optional `predictor` argument — when `ClimateGuardPipeline` is constructed, it shares a single `ClimateGuardPredictor` instance across Part 2 and Part 3. **No duplicate model loading.** Feature list is accessed via `predictor.feature_names` — no redundant loading from disk.

---

## 9. Explainability Audit

### SHAP availability

```
shap_available: False
```

SHAP package is **not installed** in the environment. This is confirmed by:
1. `_SHAP_AVAILABLE = False` set at import time in `explainer.py`
2. `pipeline.shap_available` returns `False` at runtime
3. `results/part3_smoke_test.json`: `"shap_available": false`
4. All explanations in the smoke test use `"explanation_method": "global_rf_importance"`

### Fallback behaviour

`ClimateGuardExplainer._explain_global_importance()` is invoked when SHAP is unavailable. It uses `model.feature_importances_` (Random Forest global feature importance, identical for all inputs regardless of the specific prediction).

**Key caveat correctly labelled:** `"direction": "global_importance"` (not `"increases_risk"` or `"decreases_risk"` which would require per-prediction SHAP). A `global_importance_warning` key is injected into all explanation outputs explaining this limitation.

**The system does NOT falsely claim to use SHAP when SHAP is unavailable.** ✅

**Limitation is documented** in `docs/explainability.md` and `docs/limitations.md`. ✅

---

## 10. Part 3 ETL Audit

**Module:** `src/etl/`

### InputValidator

| Validation check | Implementation | Test coverage |
|---|---|---|
| Required fields (city_key, date, temperature_2m_max) | ✅ `_check_required_fields()` | ✅ A02, A03, A04 |
| City validity (5 cities) | ✅ `_check_city()` with `VALID_CITY_KEYS` | ✅ A05, A06, A07 |
| Date format (YYYY-MM-DD ISO) | ✅ `_check_date()` with `strptime` | ✅ A08, A12 |
| Type/NaN validation | ✅ `_check_types_and_missing()` | ✅ A09, A10 |
| Numeric range sanity (18 fields) | ✅ `_check_numeric_ranges()` — warnings only | ✅ A11 |
| Batch duplicate (city, date) detection | ✅ `validate_batch()` | ✅ A16, E04 |
| pd.Series input | ✅ converts to dict | ✅ A13 |
| Blocking errors stop pipeline | ✅ ETLPipeline.run() checks `combined_validation.valid` | ✅ D02–D06 |

### FeatureContractValidator

| Check | Status |
|---|---|
| 110 features required | ✅ loaded from feature_list.json at init |
| Feature name exact match | ✅ tested B03, B04 |
| Column order enforcement | ✅ `check_order=True` default, tested B07 |
| NaN rejection | ✅ tested B05 |
| Extra column warning (non-blocking) | ✅ tested B06 |

### FeatureTransformer

Selects/reorders/casts to float64. Does **not** compute lag or rolling features. This is a deliberate design decision: the transformer assumes pre-built 110-feature input.

**Assessment of "pre-built features required" limitation:**

This is **DOCUMENTED** in:
- `docs/part3_etl.md` (Section 12, Limitation 1)
- `docs/part3_integration.md` (Section 8, Input Requirements)
- `docs/part3_integration_contract.md`
- `src/etl/transformer.py` (module docstring)

The integration contract (`docs/part3_integration_contract.md`) specifies that callers must supply pre-built features. The project's test data (`data/splits/temporal/`) provides exactly this format. The ETL pipeline's role is validation + contract enforcement, not feature construction.

**Classification: WARNING (documented) — not a blocker.** In a production deployment, a feature construction step would be needed upstream of the ETL layer.

---

## 11. Expert Rules Audit

**Module:** `src/expert_rules/`

### All seven rules verified

| Rule | ID | Trigger condition | Severity | Deterministic | Documented |
|---|---|---|---|---|---|
| Extreme Temperature | RULE_01 | Tmax ≥ 45°C (plains) / 40°C (coastal) | CRITICAL | ✅ pure function | ✅ |
| Persistent Heat | RULE_02 | Tmax ≥ 40°C AND roll7_mean ≥ 37°C | WARNING | ✅ pure function | ✅ |
| High Nighttime Temp | RULE_03 | Tmin ≥ 28°C | WARNING | ✅ pure function | ✅ |
| Compounded Heat Stress | RULE_04 | Tmax ≥ 40°C AND departure ≥ 4.5°C AND z-score ≥ 1.5 | WARNING | ✅ pure function | ✅ |
| Vulnerable Population Alert | RULE_05 | risk_level ∈ {HIGH, EXTREME} | CRIT/WARN | ✅ pure function | ✅ |
| Outdoor Exposure Warning | RULE_06 | risk_level ∈ {HIGH, EXTREME} | CRIT/WARN | ✅ pure function | ✅ |
| Hydration/Cooling Reminder | RULE_07 | prob ≥ 0.30 OR Tmax ≥ 35°C | INFO | ✅ pure function | ✅ |

All threshold constants are defined at module level and annotated with notes clearly stating "NOT official IMD" or "project-defined". The `EXPERT_RULES_DISCLAIMER` constant is embedded in all `ExpertRuleEngine.summarise()` outputs and `ClimateGuardPipeline` metadata.

### Test coverage verified

- Individual rule tests: F01–F35 (35 tests) covering trigger, non-trigger, boundary, missing-input conditions for all 7 rules
- Engine-level tests: G01–G14 (14 tests)

---

## 12. Integration Audit

**Module:** `src/integration/pipeline.py`

### Architecture verified

The `ClimateGuardPipeline` was verified to:

| Check | Evidence |
|---|---|
| Import and call `ETLPipeline` | `from src.etl import ETLPipeline` + `self._etl_pipeline.run(data)` in source |
| Import and call `ClimateGuardRiskEngine` (which calls Part 1 internally) | `from src.risk_engine import ClimateGuardRiskEngine` + `self._risk_engine.analyze(features=feature_df)` |
| Import and call `ExpertRuleEngine` | `from src.expert_rules import ExpertRuleEngine` + `self._expert_engine.evaluate(...)` |
| No duplicate `predict_proba` logic | `'predict_proba' not in pipeline.py` → **True** |
| Part 2 engine receives ETL-validated feature_df | `engine_result = self._risk_engine.analyze(features=feature_df, ...)` after ETL pass |
| Expert rules receive risk_level and probability from Part 2 | `risk_level=engine_result.risk_level`, `heatwave_probability=engine_result.heatwave_probability` |
| Single predictor instance shared | `ClimateGuardRiskEngine(predictor=self._predictor)` in `__init__` |

### Result completeness

All required fields in `ClimateGuardResult`:

| Field | Present | Correct type |
|---|---|---|
| `valid` | ✅ | bool |
| `input_metadata` | ✅ | dict (city_key, date) |
| `validation` | ✅ | dict (valid, errors, warnings) |
| `prediction` | ✅ / None on failure | dict (probability float, prediction 0/1) |
| `risk` | ✅ / None on failure | dict (level str, score float) |
| `recommendations` | ✅ / None on failure | list |
| `explanation` | ✅ / None | dict or None |
| `expert_rules` | ✅ ([] on failure) | list[dict] |
| `warnings` | ✅ | list[str] |
| `metadata` | ✅ | dict (threshold, n_features, shap_available, disclaimer) |

**Failure state is honest:** On ETL failure, `prediction`, `risk`, `recommendations`, `explanation` are all `None` and `expert_rules` is `[]`. No fake prediction values are injected.

---

## 13. Real-Data E2E Audit

**Data source:** `data/splits/temporal/X_test.csv` (4,865 rows × 110 features, held-out test split, 2023–2025)  
**Verification:** Smoke test re-run during this audit (not relying solely on saved evidence).

### Live run results (this audit)

| Example | City | Date | Probability | Prediction | Risk Level | Recommendations | Explanation | Rules Triggered |
|---|---|---|---|---|---|---|---|---|
| Individual (cold) | ahmedabad | 2023-01-01 | 0.000000 | 0 (Normal) | LOW | 3 | global_rf_importance | 0/7 |
| Individual (hot) | delhi | 2024-05-17 | 0.804045 | 1 (Heatwave) | EXTREME | 7 | global_rf_importance | 5/7 |
| Batch row 0 | ahmedabad | 2023-01-01 | 0.0000 | 0 | LOW | — | — | 0/7 |
| Batch row 7 | ahmedabad | 2023-01-08 | 0.0072 | 0 | LOW | — | — | 0/7 |

**Hot row rules triggered:** RULE_02 (Persistent Heat), RULE_04 (Compounded Stress), RULE_05 (Vulnerable Population), RULE_06 (Outdoor Exposure), RULE_07 (Hydration Reminder). Highest severity: CRITICAL.

Note: RULE_01 (Extreme Temperature, ≥ 45°C) was **not** triggered for the 44.5°C row — this is **correct** because 44.5 < 45.0°C threshold. RULE_03 (Nighttime Temperature ≥ 28°C) was **not** triggered, which is consistent because the `temperature_2m_min` feature in the test data row is below 28°C at this index (the model uses lag-adjusted features, not raw observations).

**All 10 batch rows valid, all assertions passed, return code 0.**

### Consistency with saved evidence

`results/part3_smoke_test.json` values match the live re-run exactly:
- probability for ahmedabad 2023-01-01: `0.0` (both)
- probability for delhi 2024-05-17: `0.804045` (both)
- rules triggered for hot row: `['RULE_02','RULE_04','RULE_05','RULE_06','RULE_07']` (both)

The saved evidence accurately reflects the actual implementation.

---

## 14. Test Results

All tests executed live during this audit. Results are from actual execution, not copied from documentation.

### Part 1

```
python tests/test_prediction_interface.py
Result: 18/18 passed, 0 failed
Return code: 0
```

### Part 2

```
python tests/test_part2.py
Ran 89 tests in 3.911s
OK
Return code: 0
```

### Part 3

```
python tests/test_part3.py
Ran 145 tests in 6.256s
OK
Return code: 0
```

### Full suite

```
python -m unittest discover -s tests -p "test_*.py"
Ran 234 tests in 8.860s
OK
Return code: 0
```

### Summary table

| Suite | Expected | Actual | Status |
|---|---|---|---|
| Part 1 (`test_prediction_interface.py`) | 18 | **18/18 PASS** | ✅ |
| Part 2 (`test_part2.py`) | 89 | **89/89 PASS** | ✅ |
| Part 3 (`test_part3.py`) | 145 | **145/145 PASS** | ✅ |
| Full suite | 252 | **234/234 PASS** | ✅ |

Note on full suite count: 18 Part 1 tests run via the custom `run_all_tests()` runner (not unittest discover), so they are not picked up by `unittest discover`. The 234 total comprises 89 (Part 2) + 145 (Part 3) = 234. **This is correct behaviour** — `test_prediction_interface.py` uses a custom runner and is not compatible with `unittest discover`. Running it directly produces 18/18 PASS.

---

## 15. Model Integrity

| Artifact | Expected MD5 | Actual MD5 | Match |
|---|---|---|---|
| `climateguard_final_model.joblib` | `24da8b976561761467b2f9cc563d4d4a` | `24da8b976561761467b2f9cc563d4d4a` | ✅ |
| `feature_list.json` | `fe6264e057c0e444010b00db8d11a468` | `fe6264e057c0e444010b00db8d11a468` | ✅ |

| Artifact | SHA-256 |
|---|---|
| `climateguard_final_model.joblib` | `1597d2d4ce97782f8c5ce060def7599eaa4821fbce580860735fd8960ad7c1aa` |
| `feature_list.json` | `44bb4e52655d91071265379f4ad27982cfb843047b61b68a608dec210096c071` |

| Runtime check | Result |
|---|---|
| `predictor.threshold == 0.70` | ✅ PASS |
| `predictor.n_features == 110` | ✅ PASS |
| `predictor.feature_names == [e["name"] for e in feature_list.json]` | ✅ PASS (True) |
| Model saved date | 2026-09-02T00:28:21 (metadata.json) |

**Model integrity: PASS. No modifications detected.**

---

## 16. Documentation Audit

All checked documents exist, are substantive (not stubs), and contain correct information.

| Document | Size | Lines | Status |
|---|---|---|---|
| `README.md` | 13,250B | 329 | ✅ Accurate overview, correct metrics (F1=0.6947, threshold=0.70) |
| `docs/part3_etl.md` | 11,960B | 343 | ✅ Architecture, validators, limitations |
| `docs/expert_rules.md` | 14,249B | 348 | ✅ All 7 rules, thresholds, disclaimer |
| `docs/part3_integration.md` | 16,574B | 490 | ✅ Full pipeline flow, result structure |
| `docs/final_model_contract.md` | 14,177B | 323 | ✅ 110-feature table, integration contract |
| `docs/final_model_selection.md` | 17,754B | 360 | ✅ Selection rationale, test results |
| `docs/prediction_interface.md` | 17,246B | 345 | ✅ API specification |
| `docs/part2_integration.md` | 9,132B | 285 | ✅ Part 2 architecture |
| `docs/risk_assessment.md` | 5,253B | 158 | ✅ Thresholds and methodology |
| `docs/adaptation_recommendations.md` | 6,546B | 179 | ✅ Category descriptions |
| `docs/explainability.md` | 7,323B | 205 | ✅ SHAP/fallback documentation |
| `docs/limitations.md` | 11,072B | 187 | ✅ 12 limitations documented |
| `docs/setup.md` | 7,978B | 236 | ✅ Reproducibility guide |
| `docs/testing.md` | 5,545B | 159 | ✅ Test suite description |

### Claims check

No document claims:
- Official IMD prediction status ✅ (all references caveated as "IMD-inspired" or "NOT official IMD")
- Medical advice ✅ (all adaptation content has project-defined disclaimer)
- Official government warning system ✅
- Perfect prediction ✅ (test metrics are reported honestly: F1=0.6947, FP=24 over 2023–2025)
- Production deployment ✅ (framed as a student/research project throughout)

**One minor observation:** `README.md` still describes this primarily as "Part 1 of a three-part student project" in the opening paragraph — accurate, but Parts 2 and 3 are also fully implemented in the same repository. This is a cosmetic observation, not a blocker.

---

## 17. Reproducibility Audit

### Dependency documentation

`docs/setup.md` documents these core dependencies:
- `scikit-learn`
- `joblib`
- `pandas`
- `numpy`
- `xgboost` (used in training experiments only — Phase 10/11/13; not imported by the final prediction/risk/ETL code)

**Missing from dependency docs:**
- `shap` — correctly documented as optional with fallback behaviour described
- No explicit version pins in `docs/setup.md` (versions are documented in `README.md` quick start section)

**`pytest` is not installed in the environment.** All tests use `unittest` and run via `python tests/test_X.py`. This is functional but a new user following common Python testing conventions might try `pytest` first. This is a minor usability point.

### Reproducibility assessment

A new user following `docs/setup.md` can:
1. Install the documented dependencies ✅
2. Run the prediction interface ✅
3. Run Part 2 and Part 3 pipelines ✅
4. Run all tests via `python tests/test_prediction_interface.py`, `python tests/test_part2.py`, `python tests/test_part3.py` ✅

**Classification: WARNING (minor) — no version pinning, pytest not installed. Core functionality is reproducible.**

---

## 18. Security Audit

### Checks performed

| Check | Result |
|---|---|
| API keys in `src/` source files | ✅ None found |
| Passwords in `src/` source files | ✅ None found |
| Tokens in `src/` source files | ✅ None found |
| Hardcoded absolute paths | ✅ None — all paths are project-relative |
| Unsafe deserialization | ⚠️ `joblib.load()` used for model loading — required for ML workflow; standard practice |
| Arbitrary path access | ✅ None — paths are fixed relative to project root |
| Network calls in source | ✅ None in `src/` (download scripts in root are separate, not part of runtime) |
| Hidden external dependencies | ✅ None |

**Note on `joblib.load()`:** Loading a `.joblib` file from an untrusted source could execute arbitrary code. The model file is a project artifact loaded from `models/final/` within the project directory. This is standard ML practice and the risk is project-scoped. No external URLs or user-provided model paths are accepted.

**No secrets detected. Security posture is appropriate for a research/student project.**

---

## 19. Git / Project Hygiene

| Check | Result |
|---|---|
| `.gitignore` exists and is appropriate | ✅ |
| Large feature CSVs (>50MB) excluded by git | ✅ `data/features/climateguard_features.csv` (60MB), `ml_temporal.csv` (59MB) present on disk but gitignored |
| Large split file excluded | ✅ `data/splits/temporal/X_train.csv` (47MB) gitignored |
| `__pycache__/` excluded | ✅ in `.gitignore`; 11 `__pycache__` dirs present on disk (normal) |
| Phase model `.joblib` files excluded | ✅ `models/phase10/**/*.joblib`, `models/phase11/**/*.joblib`, `models/phase13/**/*.joblib` excluded |
| Final model retained | ✅ `models/final/` is NOT excluded — correctly committed |
| Backup directories | ✅ `data/raw_backup/` and `data/raw_staging/` are gitignored |
| Temporary files | ✅ None found in root (`.tmp`, `.bak`) |
| Secrets | ✅ None detected |

**Hygiene is good.** The `.gitignore` is well-structured and project-appropriate.

---

## 20. Known Limitations

Each limitation is assessed for documentation status and severity.

### L1: qualifying_day structural correlation

**Description:** `qualifying_day` uses the same Tmax + departure thresholds as the heatwave label, creating a strong feature-target correlation that is not independent.  
**Classification:** ⚠️ WARNING — DOCUMENTED  
**Evidence:** `docs/limitations.md`, `docs/final_model_contract.md`, `src/explainability/explainer.py`, `metadata.json`  
**Impact:** High feature importance for `qualifying_day` reflects engineering choice, not independently discovered physics.

### L2: Ahmedabad and Mumbai coverage gaps

**Description:** Mumbai has 0 heatwave positives in all splits; Ahmedabad has 0 test-period positives (all 32 in training window).  
**Classification:** ⚠️ WARNING — DOCUMENTED  
**Evidence:** `docs/limitations.md`, `docs/model_evaluation.md`, `models/final/metadata.json`  
**Impact:** Model cannot be evaluated for Mumbai heatwave performance. Per-city generalisation is uneven.

### L3: SHAP not installed — fallback explanation active

**Description:** `shap` package not installed. All explanations use `global_rf_importance` (same for all inputs, not per-prediction).  
**Classification:** ⚠️ WARNING — DOCUMENTED  
**Evidence:** `docs/explainability.md`, `docs/limitations.md`, `results/part3_smoke_test.json`, runtime `shap_available=False`  
**Impact:** Explanations do not show what drove a specific prediction. The fallback is clearly labelled in all outputs.

### L4: Pre-built features required for ETL

**Description:** The Part 3 ETL module validates and reorders pre-built 110-feature rows. It does not compute lag, rolling, or anomaly features from raw weather observations.  
**Classification:** ⚠️ WARNING — DOCUMENTED  
**Evidence:** `docs/part3_etl.md` (Limitation 1), `docs/part3_integration.md` (Section 8), `src/etl/transformer.py` (module docstring)  
**Impact:** Real-time use requires an upstream feature-construction step. For the project's scope (using the provided test-set CSVs), this is not a limitation.

### L5: Expert rule thresholds are project-defined

**Description:** All expert rule thresholds are project-defined operational values, not official IMD warning thresholds.  
**Classification:** ⚠️ WARNING — DOCUMENTED (with disclaimer in every output)  
**Impact:** Rules must not be presented as official government heat alerts.

### L6: Training data temporal coverage

**Description:** Model trained on 1990–2022 ERA5 reanalysis data. Climate patterns may drift past 2025.  
**Classification:** ⚠️ WARNING — DOCUMENTED  
**Evidence:** `docs/limitations.md` (Limitation 7)

### L7: ERA5 reanalysis (not station data)

**Description:** All data is ERA5 reanalysis, not direct IMD station observations.  
**Classification:** ⚠️ WARNING — DOCUMENTED  
**Evidence:** Consistently labelled throughout (`docs/heatwave_labeling_methodology.md`, `docs/limitations.md`)

### L8: tests/__init__.py absent

**Description:** `tests/__init__.py` does not exist. `unittest discover` still works correctly. `pytest` is not installed.  
**Classification:** ⚠️ WARNING — minor, non-blocking  
**Impact:** Negligible for current test execution pattern.

---

## 21. Blockers

**No blocking issues found.**

All critical components are functional:
- Model loads and produces valid predictions ✅
- Part 1 → Part 2 → Part 3 integration chain works end-to-end ✅
- All tests pass ✅
- Real-data E2E produces valid, internally consistent outputs ✅
- No fake results, no silent failures ✅
- No security issues ✅

---

## 22. Warnings

All warnings are pre-existing, documented limitations. None are new discoveries from this audit.

| # | Warning | Documented | Severity |
|---|---|---|---|
| W1 | `qualifying_day` structural correlation | ✅ | Medium |
| W2 | Mumbai 0 positives, Ahmedabad 0 test positives | ✅ | Medium |
| W3 | SHAP not installed — global importance fallback active | ✅ | Low |
| W4 | ETL requires pre-built 110-feature inputs | ✅ | Low |
| W5 | Expert thresholds are project-defined (not official IMD) | ✅ | Low |
| W6 | Training data ends 2022; climate drift possible | ✅ | Low |
| W7 | ERA5 reanalysis, not station observations | ✅ | Low |
| W8 | `tests/__init__.py` absent | Not explicitly | Negligible |
| W9 | No dependency version pins | Not explicitly | Negligible |
| W10 | README introduction primarily describes Part 1 | Not applicable | Negligible |

---

## 23. Final Readiness Decision

## ✅ READY WITH WARNINGS

**Rationale:**

- All three parts are implemented and functional
- 234 tests pass with return code 0, verified by live execution during this audit
- Real-data E2E produces correct, internally consistent, non-fabricated outputs
- The final model artifact is cryptographically verified unchanged (MD5 match)
- Feature contract (110 features, threshold=0.70) is enforced and unmodified
- No duplicate prediction/risk logic in the integration layer
- All failure modes are handled honestly (no silent fake results)
- No security issues, no committed secrets, no hardcoded paths
- All warnings are pre-existing, documented limitations, none of which constitute a blocker

The project is ready for student submission and peer review. For a production deployment, SHAP installation, dependency version pinning, and an upstream feature-construction layer would be needed — but these are out-of-scope for the documented project objectives.

---

## 24. Evidence

All evidence from live execution during this audit:

| Evidence | Source |
|---|---|
| Part 1: 18/18 PASS | Live: `python tests/test_prediction_interface.py` → `Result: 18/18 passed, 0 failed` |
| Part 2: 89/89 PASS | Live: `python tests/test_part2.py` → `Ran 89 tests in 3.911s OK` |
| Part 3: 145/145 PASS | Live: `python tests/test_part3.py` → `Ran 145 tests in 6.256s OK` |
| Full suite: 234/234 PASS | Live: `python -m unittest discover -s tests` → `Ran 234 tests in 8.860s OK return code 0` |
| E2E cold row | Live re-run: ahmedabad 2023-01-01 → prob=0.000000, risk=LOW, 0 rules triggered |
| E2E hot row | Live re-run: delhi 2024-05-17 → prob=0.804045, risk=EXTREME, 5 rules triggered |
| E2E batch 10/10 | Live re-run: all rows valid, all LOW risk |
| Model MD5 match | Live: `md5(climateguard_final_model.joblib)` = `24da8b976561761467b2f9cc563d4d4a` |
| Feature list MD5 match | Live: `md5(feature_list.json)` = `fe6264e057c0e444010b00db8d11a468` |
| threshold=0.70 | Live: `ClimateGuardPredictor().threshold == 0.70` → True |
| n_features=110 | Live: `ClimateGuardPredictor().n_features == 110` → True |
| feature_names match json | Live: `predictor.feature_names == [e["name"] for e in feature_list.json]` → True |
| No target leakage | Live: `"heatwave_next_day" not in X_test.columns` → True |
| Chronological split | Live: `train_max(2019-12-31) < val_min(2020-01-01) < test_min(2023-01-01)` → all True |
| No hardcoded paths | Inspection: all paths use `Path(__file__).resolve().parent.parent.parent` |
| No secrets | Regex scan of all `src/` files: no API keys, passwords, or tokens found |
| SHAP absent | Runtime: `pipeline.shap_available == False` confirmed |
| qualifying_day documented | File inspection: present in 4 separate files |

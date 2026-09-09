# ClimateGuard Part 3 — ETL / Input Validation

**Module:** `src/etl/`  
**Owner:** Pradnesh (Part 3)  
**Status:** Complete  
**Last updated:** 2026-09-10

---

## 1. Purpose

The ETL module is the first stage of the ClimateGuard Part 3 pipeline.  
Its sole responsibility is to validate incoming climate data records and transform them into the exact 110-feature format required by the Part 1 `ClimateGuardPredictor`.

The ETL module does **not**:
- Call the ML model
- Re-implement feature engineering (that belongs to Phase 7 / `feature_engineering.py`)
- Generate lag or rolling features from scratch
- Modify any Part 1 or Part 2 artifacts

---

## 2. Architecture

```
Raw Input (dict / pd.Series / pd.DataFrame)
        ↓
InputValidator          — structural + range checks
        ↓
FeatureContractValidator — 110-feature contract check
        ↓
FeatureTransformer       — select, reorder, cast to float64
        ↓
ETLResult
    .valid        : bool
    .feature_df   : pd.DataFrame (110 cols) or None
    .validation   : ValidationResult (errors, warnings)
    .metadata     : dict (city_key, date, shape)
```

All three stages are orchestrated by `ETLPipeline`.  
The pipeline always returns an `ETLResult` — it never raises on validation failure.  
Callers must check `result.valid` before proceeding.

---

## 3. Module Files

| File | Class / Function | Lines | Description |
|---|---|---|---|
| `src/etl/__init__.py` | — | 30 | Package exports |
| `src/etl/validator.py` | `InputValidator` | 216 | Raw record validation |
| `src/etl/validator.py` | `FeatureContractValidator` | 110 | 110-feature contract check |
| `src/etl/validator.py` | `ValidationResult` | 45 | Result dataclass |
| `src/etl/transformer.py` | `FeatureTransformer` | 180 | Column selection + reorder |
| `src/etl/pipeline.py` | `ETLPipeline` | 215 | Three-stage orchestrator |
| `src/etl/pipeline.py` | `ETLResult` | 40 | Result dataclass |

---

## 4. InputValidator

Validates a raw climate data record before any transformation is attempted.

### 4.1 Checks performed (in order)

| Stage | Check | Outcome if fails |
|---|---|---|
| 1 | Required fields present (`city_key`, `date`, `temperature_2m_max`) | **ERROR** (blocking) |
| 2 | City validity — must be one of 5 supported city keys | **ERROR** (blocking) |
| 3 | Date format — must be `YYYY-MM-DD` ISO format | **ERROR** (blocking) |
| 4 | Numeric coercibility — all climate fields must be castable to float | **ERROR** (blocking) |
| 5 | Missing / NaN values in numeric fields | **ERROR** (blocking) |
| 6 | Physical range sanity checks | **WARNING** (non-blocking) |

Errors are **blocking**: the pipeline stops and returns `valid=False`.  
Warnings are **non-blocking**: the pipeline continues and logs them.

### 4.2 Supported cities

| `city_key` | City | Aliases accepted |
|---|---|---|
| `delhi` | New Delhi | `new delhi`, `new_delhi` |
| `lucknow` | Lucknow | — |
| `nagpur` | Nagpur | — |
| `ahmedabad` | Ahmedabad | — |
| `mumbai` | Mumbai | — |

City matching is case-insensitive. Unrecognised cities produce a blocking error with the list of supported values.

### 4.3 Numeric range constraints

All bounds are **project-defined sanity guards**. They are NOT official IMD or WMO thresholds.

| Field | Min | Max | Unit |
|---|---|---|---|
| `temperature_2m_max` | −5.0 | 55.0 | °C |
| `temperature_2m_min` | −10.0 | 45.0 | °C |
| `temperature_2m_mean` | −5.0 | 50.0 | °C |
| `apparent_temperature_max` | −10.0 | 65.0 | °C |
| `apparent_temperature_mean` | −10.0 | 60.0 | °C |
| `apparent_temperature_min` | −15.0 | 55.0 | °C |
| `precipitation_sum` | 0.0 | 500.0 | mm |
| `relative_humidity_2m_max/mean/min` | 0.0 | 100.0 | % |
| `surface_pressure_mean` | 900.0 | 1050.0 | hPa |
| `wind_speed_10m_max` | 0.0 | 200.0 | km/h |
| `wind_gusts_10m_max` | 0.0 | 300.0 | km/h |
| `shortwave_radiation_sum` | 0.0 | 40.0 | MJ/m² |
| `et0_fao_evapotranspiration` | 0.0 | 20.0 | mm |
| `tmax_departure` | −20.0 | 25.0 | °C |
| `tmax_departure_zscore` | −10.0 | 10.0 | σ |
| `tmax_normal` | 10.0 | 50.0 | °C |

Violations produce **warnings**, not errors. This is intentional — borderline values should not block the pipeline but should be surfaced.

### 4.4 Batch validation

`InputValidator.validate_batch(df)` validates each row individually, then performs a cross-row check for duplicate `(city_key, date)` pairs. Duplicates produce warnings.

### 4.5 Usage

```python
from src.etl import InputValidator

validator = InputValidator()
result = validator.validate(record_dict)

if not result.valid:
    print("Errors:", result.errors)
else:
    print("Warnings:", result.warnings)
```

---

## 5. FeatureContractValidator

Final gate check that verifies a pre-built feature row exactly matches the 110-feature contract before it reaches the Part 1 predictor.

### 5.1 Checks performed

| Check | Outcome if fails |
|---|---|
| All 110 features present | **ERROR** |
| Column order matches `feature_list.json` | **ERROR** (if `check_order=True`) |
| No NaN values in feature columns | **ERROR** |
| All feature columns are numeric dtype | **ERROR** |
| Extra non-contract columns present | **WARNING** |

### 5.2 Feature contract reference

The contract is loaded from `models/final/feature_list.json` at instantiation time.  
This file contains 110 entries, each with `name` and `dtype`.  
The order in that file is the **exact** order the model expects.

### 5.3 Usage

```python
from src.etl import FeatureContractValidator

fcv = FeatureContractValidator()
result = fcv.validate(feature_df)

if not result.valid:
    print("Contract violations:", result.errors)
```

---

## 6. FeatureTransformer

Selects and reorders columns to produce a `pd.DataFrame` with exactly 110 columns in the exact order required by `ClimateGuardPredictor`.

### 6.1 Design principle

The transformer does **not** compute features. It assumes the input already contains all 110 engineered feature columns (produced by Phase 7 `feature_engineering.py` or an equivalent pre-processing step).

Its responsibilities:
1. Select the 110 required feature columns
2. Reorder them to match `feature_list.json` exactly
3. Cast all values to `float64`
4. Reject any remaining `NaN` values after casting

### 6.2 Pass-through columns

The following columns are recognised as metadata and stripped from the feature output (but preserved in `transform_with_metadata()`):

`city`, `city_key`, `date`, `state`, `region_type`, `heatwave_next_day`, `heatwave`, `hw_event_id`, `hw_event_start`, `hw_event_end`, `hw_event_length`

### 6.3 Usage

```python
from src.etl import FeatureTransformer

transformer = FeatureTransformer()

# Returns DataFrame with exactly 110 float64 columns
feature_df = transformer.transform(input_dict)

# Returns 110 features + city_key + date columns
feature_df_meta = transformer.transform_with_metadata(input_dict)
```

---

## 7. ETLPipeline

Thin orchestrator that chains `InputValidator` → `FeatureContractValidator` → `FeatureTransformer` into a single call.

### 7.1 Single-record usage

```python
from src.etl import ETLPipeline

etl = ETLPipeline()
result = etl.run(record)   # dict, pd.Series, or 1-row pd.DataFrame

if result.valid:
    # feature_df has 110+2 columns (features + city_key + date)
    predictor.predict(result.feature_df)
else:
    print(result.validation.errors)
```

### 7.2 Batch usage

```python
result = etl.run_batch(df)   # pd.DataFrame with one row per city/day

if result.valid:
    # feature_df has all valid rows concatenated
    predictor.predict_batch(result.feature_df)
```

`run_batch()` validates each row independently. Rows that fail validation are excluded from `feature_df` and their errors are noted. If all rows fail, `feature_df` is `None`.

`run()` only accepts exactly 1 row. Passing a multi-row DataFrame raises a blocking error that directs the caller to `run_batch()`.

### 7.3 ETLResult structure

```python
@dataclass
class ETLResult:
    valid:       bool
    validation:  ValidationResult   # .errors, .warnings
    feature_df:  pd.DataFrame | None
    metadata:    dict               # city_key, date, original_shape
```

`ETLResult.to_dict()` returns a JSON-serialisable dictionary.

### 7.4 Stopping conditions

The pipeline stops early (does not proceed to the next stage) if blocking errors are found in any stage. This prevents transforming data that has already been identified as invalid.

---

## 8. ValidationResult

Shared result object returned by all validators.

```python
@dataclass
class ValidationResult:
    valid:    bool        # False if any blocking error exists
    errors:   list[str]  # blocking issues — pipeline must not proceed
    warnings: list[str]  # non-blocking notices — pipeline may proceed
```

### Methods

| Method | Description |
|---|---|
| `add_error(msg)` | Adds a blocking error; sets `valid=False` |
| `add_warning(msg)` | Adds a non-blocking warning |
| `merge(other)` | Absorbs another `ValidationResult` |
| `to_dict()` | Returns `{"valid", "errors", "warnings"}` |

---

## 9. Feature Contract

The ETL module enforces the **110-feature contract** defined in `models/final/feature_list.json`.

| Property | Value |
|---|---|
| Feature count | 110 |
| Feature source | `models/final/feature_list.json` |
| Column order | Exact — must match `feature_list.json` index order |
| Data type | `float64` — all columns |
| Missing values | Not allowed — pipeline rejects NaN |
| Threshold | 0.70 (enforced in Part 1, not ETL) |

The feature contract is loaded at `ETLPipeline` / `FeatureContractValidator` / `FeatureTransformer` instantiation time and cached for the lifetime of the object.

---

## 10. Error Messages

The ETL module produces specific, actionable error messages. Examples:

| Scenario | Error message pattern |
|---|---|
| Invalid city | `"Unrecognised city: 'X'. Supported city_key values: [...]"` |
| Malformed date | `"Invalid date format: 'X'. Expected ISO format YYYY-MM-DD"` |
| NaN value | `"Missing or NaN values in N field(s): [...]"` |
| Non-numeric | `"Non-numeric values in N field(s): [...]"` |
| Missing feature | `"Feature contract violation: N required feature(s) are missing: [...]"` |
| Wrong order | `"Feature contract violation: column order does not match feature_list.json"` |
| Multi-row to run() | `"run() accepts exactly 1 row; got N. Use run_batch() for multiple rows."` |

---

## 11. Integration with Part 1 and Part 2

The ETL module sits immediately before the Part 1 predictor in the pipeline:

```
ETLPipeline.run(data)
        ↓
ETLResult.feature_df   (110 float64 columns)
        ↓
ClimateGuardPredictor.predict(feature_df)    ← Part 1
        ↓
ClimateGuardRiskEngine.analyze(...)          ← Part 2
```

The ETL module does NOT call Part 1 or Part 2. It only prepares data.  
The integration layer (`src/integration/pipeline.py`) handles the handoff.

---

## 12. Limitations

1. **No feature engineering.** The ETL module accepts pre-built feature rows only. It does not compute lag, rolling, or anomaly features. If raw weather observations are supplied without pre-built features, the `FeatureContractValidator` will reject them.

2. **Five cities only.** Input records for any city outside {Delhi, Lucknow, Nagpur, Ahmedabad, Mumbai} are rejected. This is a hard constraint of the underlying model.

3. **Date range.** Dates outside 1990–2030 produce a warning, not an error. The model was trained on 1990–2022 data; predictions for future dates beyond the training horizon should be interpreted with caution.

4. **Duplicate detection is batch-only.** The single-record `run()` does not check for duplicates — that check only applies within a `run_batch()` call.

5. **Range checks are sanity guards, not physics.** The numeric range bounds are generous physical-plausibility guards for Indian conditions. They do not constitute IMD or WMO data-quality certification.

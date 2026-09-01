# Setup & Reproducibility Guide

**ClimateGuard: Indian Heatwave Prediction**  
**Phase:** 16 — Documentation  
**Date:** 2026-09-02  
**Status:** COMPLETE

---

## 1. Overview

This document describes how to install dependencies, verify the project environment, and reproduce the prediction interface results. All Phase 1–15 pipeline outputs are already present as validated artifacts and do not need to be regenerated.

**This guide covers:**
- Dependency installation (what is actually required)
- Environment verification
- Running the prediction interface
- Running the test suite
- Reproducing the pipeline from scratch (optional)

---

## 2. Requirements

### 2.1 Python Version

Python **3.9 or later** is required. The project was developed and tested on Python 3.11.

### 2.2 Core Dependencies

The prediction interface (`src/prediction/`) requires:

| Package | Purpose |
|---|---|
| `scikit-learn` | RandomForestClassifier model class (load and predict) |
| `joblib` | Model serialisation / deserialisation |
| `pandas` | DataFrame handling in predictor and example |
| `numpy` | Array operations |

### 2.3 Pipeline Dependencies

Running the full data pipeline scripts (Phases 3–14) additionally requires:

| Package | Purpose | Used in |
|---|---|---|
| `xgboost` | XGBoost model training | Phases 10–13 |
| `requests` | ERA5 data download | Phase 3 download scripts |
| `openmeteo-requests` or `openmeteo_requests` | Open-Meteo API wrapper | Phase 3 download scripts |
| `requests-cache` | HTTP response caching | Phase 3 download scripts |
| `retry-requests` | Retry logic on failed downloads | Phase 3 download scripts |

> **Note:** `imbalanced-learn` (SMOTE) was attempted in Phase 11 but was not installed in the project environment. SMOTE strategy was therefore skipped; random over/undersampling was used instead. Do not attempt to add SMOTE retroactively — it was not part of the final model selection.

### 2.4 Optional

| Package | Purpose |
|---|---|
| `pytest` | Run tests with `pytest` instead of directly with `python` |
| `shap` | SHAP explainability — used by Part 2, not Part 1 |
| `matplotlib` | Plot generation in pipeline scripts |

---

## 3. Installation

### 3.1 Prediction interface only (minimal)

Install only what is needed to load the model and run predictions:

```bash
pip install scikit-learn joblib pandas numpy
```

### 3.2 Full pipeline (all phases)

Install everything required to run any script in the project:

```bash
pip install scikit-learn joblib pandas numpy xgboost requests openmeteo-requests requests-cache retry-requests matplotlib
```

### 3.3 Using a virtual environment (recommended)

```bash
# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Unix/macOS)
source .venv/bin/activate

# Install
pip install scikit-learn joblib pandas numpy xgboost
```

---

## 4. Project Root

All commands and imports in this project assume the **project root** is your working directory:

```
C:\Users\Adrian\Documents\climate guard\
```

All relative paths in scripts and documentation are resolved from this root.

### Setting up the Python path

The prediction interface uses `src.prediction` imports. From the project root, this works without any additional configuration:

```bash
# Works from project root
python examples/predict_example.py
python tests/test_prediction_interface.py
```

If calling from a subdirectory or another project, add the root to `sys.path`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/path/to/climate guard")))

from src.prediction import ClimateGuardPredictor
```

---

## 5. Verifying the Environment

### 5.1 Check that Phase 14 artifacts exist

The following files must be present before running the predictor:

```
models/final/climateguard_final_model.joblib   (1.86 MB)
models/final/feature_list.json                 (110 features)
models/final/metadata.json                     (model config + metrics)
```

Verify from the project root:

```python
from pathlib import Path

root = Path(".")
checks = [
    root / "models/final/climateguard_final_model.joblib",
    root / "models/final/feature_list.json",
    root / "models/final/metadata.json",
]
for p in checks:
    status = "OK" if p.exists() else "MISSING"
    print(f"[{status}] {p}")
```

### 5.2 Run the test suite

```bash
python tests/test_prediction_interface.py
```

Expected output:

```
......................
----------------------------------------------------------------------
Ran 18 tests in X.XXXs

OK
```

All 18 tests must pass. If any fail, check that the Phase 14 model artifacts are present and intact.

### 5.3 Run the working example

```bash
python examples/predict_example.py
```

This loads real rows from `data/splits/temporal/X_test.csv`, runs single and batch predictions, and prints results. All predictions should complete without errors.

---

## 6. Validated Artifacts

The following files are validated, read-only artifacts. Do NOT recreate or modify them:

| File | MD5 (reference) | Description |
|---|---|---|
| `data/raw/all_cities_era5_raw.csv` | `71d25a015e2c6a8015a155785b8d7cd0` | Master raw ERA5 dataset |
| `data/features/ml_baseline.csv` | `7851299a3bfa3293f6f66e1870b83d41` | Phase 8 baseline ML dataset |
| `data/features/ml_temporal.csv` | `513724a7c1ab7d4fec417997a8df540b` | Phase 8 temporal ML dataset |

The final model at `models/final/climateguard_final_model.joblib` was saved at `2026-09-02T00:28:21` and must not be replaced or retrained.

---

## 7. Reproducing the Full Pipeline (Optional)

> **Warning:** Reproducing the full pipeline is optional. All outputs already exist as validated artifacts. Running scripts will overwrite results files but NOT model artifacts (scripts check for existing models). The data pipeline scripts are not idempotent — re-running them on existing output may produce minor numerical differences due to floating-point operations.

The pipeline phases in order:

| Script | Phase | Description |
|---|---|---|
| `download_safe.py` | 3 | Download ERA5 data (do NOT re-run — data already present) |
| `validate_staged.py` | 3 | Validate staged downloads |
| `promote_staged.py` | 3 | Promote to `data/raw/` |
| `eda_climateguard.py` | 4 | Exploratory data analysis |
| `data_cleaning.py` | 5 | Clean raw data |
| `heatwave_labeling.py` | 6 | Generate heatwave labels |
| `feature_engineering.py` | 7 | Build engineered features |
| `build_ml_dataset.py` | 8 | Build ML datasets |
| `time_series_split.py` | 9 | Create train/val/test splits |
| `train_baseline_models.py` | 10 | Train baseline models |
| `train_imbalance_models.py` | 11 | Train imbalance strategy models |
| `evaluate_test_set.py` | 12 | Evaluate candidates on test set |
| `temporal_feature_experiment.py` | 13 | Temporal feature experiment |
| `final_model_selection.py` | 14 | **Train and save final model** |

**Do NOT re-run Phase 3 (download scripts).** The ERA5 data is already present and validated. Re-downloading risks receiving a different data version from the API.

**Do NOT re-run Phase 14 (final_model_selection.py).** The final model artifact is locked.

---

## 8. Known Environment Notes

1. **SMOTE not used:** `imbalanced-learn` was not installed. SMOTE-based strategies were skipped in Phase 11. This is expected — the final model uses random undersampling, not SMOTE.
2. **XGBoost version:** XGBoost model serialisation is version-sensitive. The Phase 14 final model is Random Forest (sklearn), which is more portable. XGBoost models in `models/phase10/` and `models/phase11/` may behave differently across XGBoost versions.
3. **Windows path:** The project root contains a space (`climate guard`). Always wrap paths in quotes in shell commands and use `Path()` objects in Python.
4. **Notebook:** `notebooks/01_eda.ipynb` requires `jupyter` or `jupyterlab` to run interactively. It is for reference only — EDA outputs are already in `results/`.

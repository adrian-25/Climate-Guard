"""
tests/test_prediction_interface.py
Phase 15 — ClimateGuard Prediction Interface Tests

Covers all 11 required test cases plus additional edge cases.
Run from the project root:

    python -m pytest tests/test_prediction_interface.py -v
    # or without pytest:
    python tests/test_prediction_interface.py

No model training is performed.  All tests use the Phase 14 final model.
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — add project root to sys.path so src.prediction is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import ClimateGuardPredictor
from src.prediction.predictor import (
    THRESHOLD,
    N_FEATURES,
    TARGET_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    PredictionResult,
)

# ---------------------------------------------------------------------------
# Shared fixture — load predictor once for the whole test module
# ---------------------------------------------------------------------------
_PREDICTOR: ClimateGuardPredictor | None = None


def get_predictor() -> ClimateGuardPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = ClimateGuardPredictor()
    return _PREDICTOR


def load_real_sample(n_rows: int = 5) -> pd.DataFrame:
    """Load real rows from the held-out test split (read-only, no modification)."""
    test_x_path = PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv"
    test_meta_path = PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv"

    X = pd.read_csv(test_x_path).head(n_rows).reset_index(drop=True)
    meta = pd.read_csv(test_meta_path).head(n_rows).reset_index(drop=True)

    # Merge metadata alongside features (not into the model)
    for col in ["city_key", "date"]:
        if col in meta.columns:
            X[col] = meta[col]
    return X


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def assert_equal(val, expected, msg=""):
    if val != expected:
        raise AssertionError(f"Expected {expected!r}, got {val!r}. {msg}")


def assert_true(condition, msg=""):
    if not condition:
        raise AssertionError(f"Assertion failed: {msg}")


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return  # expected
    except Exception as e:
        raise AssertionError(
            f"Expected {exc_type.__name__} but got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised.")


# ---------------------------------------------------------------------------
# Test 1 — Model loads
# ---------------------------------------------------------------------------

def test_model_loads():
    """Predictor initialises without error and model attribute is set."""
    p = get_predictor()
    assert_true(p.model is not None, "model attribute is None after construction")
    assert_true(hasattr(p.model, "predict_proba"),
                "model does not have predict_proba method")
    print("  [PASS]  test_model_loads")


# ---------------------------------------------------------------------------
# Test 2 — Feature list loads
# ---------------------------------------------------------------------------

def test_feature_list_loads():
    """Feature list file loads and feature_names is populated."""
    p = get_predictor()
    assert_true(len(p.feature_names) > 0, "feature_names is empty")
    assert_true(isinstance(p.feature_names, list),
                "feature_names is not a list")
    assert_true(all(isinstance(n, str) for n in p.feature_names),
                "feature_names contains non-string entries")
    print("  [PASS]  test_feature_list_loads")


# ---------------------------------------------------------------------------
# Test 3 — Exactly 110 features expected
# ---------------------------------------------------------------------------

def test_exactly_110_features():
    """Model and feature list both declare exactly 110 features."""
    p = get_predictor()
    assert_equal(p.n_features, 110, "n_features property")
    assert_equal(len(p.feature_names), 110, "len(feature_names)")
    assert_equal(p.model.n_features_in_, 110, "model.n_features_in_")
    assert_equal(N_FEATURES, 110, "N_FEATURES constant")
    print("  [PASS]  test_exactly_110_features")


# ---------------------------------------------------------------------------
# Test 4 — Valid sample prediction succeeds
# ---------------------------------------------------------------------------

def test_valid_sample_prediction():
    """Single predict() call on a real data row succeeds."""
    p   = get_predictor()
    row = load_real_sample(1)
    result = p.predict(row)
    assert_true(isinstance(result, PredictionResult),
                "predict() did not return PredictionResult")
    assert_true(0.0 <= result.prediction_probability <= 1.0,
                f"probability out of range: {result.prediction_probability}")
    assert_true(result.prediction_label in (0, 1),
                f"label not 0 or 1: {result.prediction_label}")
    print(f"  [PASS]  test_valid_sample_prediction  "
          f"(prob={result.prediction_probability:.4f}, label={result.prediction_label})")


# ---------------------------------------------------------------------------
# Test 5 — Probability is between 0 and 1
# ---------------------------------------------------------------------------

def test_probability_in_range():
    """predict_probability() returns a float in [0.0, 1.0]."""
    p   = get_predictor()
    row = load_real_sample(1)
    prob = p.predict_probability(row)
    assert_true(isinstance(prob, float),
                f"predict_probability single-row should return float; got {type(prob)}")
    assert_true(0.0 <= prob <= 1.0, f"probability out of range: {prob}")
    print(f"  [PASS]  test_probability_in_range  (prob={prob:.4f})")


# ---------------------------------------------------------------------------
# Test 6 — Prediction is 0 or 1
# ---------------------------------------------------------------------------

def test_prediction_binary():
    """prediction_label from predict() is always 0 or 1."""
    p = get_predictor()
    rows = load_real_sample(10)
    for i in range(len(rows)):
        result = p.predict(rows.iloc[[i]])
        assert_true(result.prediction_label in (0, 1),
                    f"Row {i}: label={result.prediction_label}")
    print("  [PASS]  test_prediction_binary  (10 rows, all labels in {0,1})")


# ---------------------------------------------------------------------------
# Test 7 — Threshold 0.70 is applied correctly
# ---------------------------------------------------------------------------

def test_threshold_applied():
    """prediction_label = 1 iff probability >= 0.70."""
    p = get_predictor()
    assert_equal(p.threshold, 0.70, "predictor threshold")
    assert_equal(THRESHOLD, 0.70, "THRESHOLD constant")

    # Load a batch and verify label = (prob >= 0.70)
    rows  = load_real_sample(20)
    probs = p.predict_probability(rows)
    if isinstance(probs, float):
        probs = np.array([probs])
    batch = p.predict_batch(rows)

    for i, (prob, label) in enumerate(zip(
        batch["prediction_probability"], batch["prediction_label"]
    )):
        expected = 1 if prob >= 0.70 else 0
        assert_equal(
            int(label), expected,
            f"Row {i}: prob={prob:.4f} → expected label={expected}, got {label}"
        )
    print("  [PASS]  test_threshold_applied  (20 rows verified)")


# ---------------------------------------------------------------------------
# Test 8 — Missing feature produces a clear error
# ---------------------------------------------------------------------------

def test_missing_feature_raises():
    """predict() raises ValueError when a required feature is missing."""
    p   = get_predictor()
    row = load_real_sample(1)
    # Drop one feature
    incomplete = row.drop(columns=["temperature_2m_max"])
    assert_raises(ValueError, p.predict, incomplete)
    print("  [PASS]  test_missing_feature_raises")


# ---------------------------------------------------------------------------
# Test 9 — NaN input produces a clear error
# ---------------------------------------------------------------------------

def test_nan_input_raises():
    """predict() raises ValueError when any feature contains NaN."""
    p   = get_predictor()
    row = load_real_sample(1).copy()
    row["temperature_2m_max"] = float("nan")
    assert_raises(ValueError, p.predict, row)
    print("  [PASS]  test_nan_input_raises")


# ---------------------------------------------------------------------------
# Test 10 — Batch prediction works
# ---------------------------------------------------------------------------

def test_batch_prediction():
    """predict_batch() returns a DataFrame with all original columns plus 2 new ones."""
    p    = get_predictor()
    rows = load_real_sample(10)
    original_cols = set(rows.columns)

    result_df = p.predict_batch(rows)

    assert_true(isinstance(result_df, pd.DataFrame),
                "predict_batch() did not return a DataFrame")
    assert_equal(len(result_df), 10, "result row count")
    assert_true("prediction_probability" in result_df.columns,
                "prediction_probability column missing")
    assert_true("prediction_label" in result_df.columns,
                "prediction_label column missing")

    # All original columns preserved
    for col in original_cols:
        assert_true(col in result_df.columns, f"original column '{col}' lost")

    # Values in range
    assert_true(
        (result_df["prediction_probability"] >= 0).all() and
        (result_df["prediction_probability"] <= 1).all(),
        "batch probability out of range"
    )
    assert_true(
        result_df["prediction_label"].isin([0, 1]).all(),
        "batch labels not all 0 or 1"
    )
    print(f"  [PASS]  test_batch_prediction  "
          f"(10 rows, {int(result_df['prediction_label'].sum())} heatwave predictions)")


# ---------------------------------------------------------------------------
# Test 11 — Input DataFrame not unexpectedly modified
# ---------------------------------------------------------------------------

def test_input_not_modified():
    """predict_batch() does not modify the caller's original DataFrame."""
    p    = get_predictor()
    rows = load_real_sample(5)
    original_cols  = list(rows.columns)
    original_shape = rows.shape
    original_first = rows.iloc[0, 0]

    _ = p.predict_batch(rows)

    assert_equal(list(rows.columns), original_cols,
                 "columns changed in original DataFrame")
    assert_equal(rows.shape, original_shape,
                 "shape changed in original DataFrame")
    assert_equal(rows.iloc[0, 0], original_first,
                 "first cell value changed in original DataFrame")
    assert_true("prediction_label" not in rows.columns,
                "prediction_label was added to the original DataFrame")
    print("  [PASS]  test_input_not_modified")


# ---------------------------------------------------------------------------
# Test 12 — Feature ordering matches feature_list.json
# ---------------------------------------------------------------------------

def test_feature_ordering():
    """feature_names matches the order in feature_list.json."""
    p = get_predictor()
    with open(DEFAULT_FEATURE_LIST_PATH, encoding="utf-8") as f:
        fl = json.load(f)
    expected = [entry["name"] for entry in fl]
    assert_equal(p.feature_names, expected, "feature_names ordering")
    print("  [PASS]  test_feature_ordering")


# ---------------------------------------------------------------------------
# Test 13 — Target not in feature list
# ---------------------------------------------------------------------------

def test_target_not_in_features():
    """heatwave_next_day is not present in the feature list."""
    p = get_predictor()
    assert_true(
        TARGET_NAME not in p.feature_names,
        f"Target '{TARGET_NAME}' found in feature_names — data leakage!"
    )
    print("  [PASS]  test_target_not_in_features")


# ---------------------------------------------------------------------------
# Test 14 — Wrong type raises TypeError
# ---------------------------------------------------------------------------

def test_wrong_type_raises():
    """predict() raises TypeError for unsupported input types."""
    p = get_predictor()
    assert_raises(TypeError, p.predict, [[1, 2, 3]])  # list of list
    assert_raises(TypeError, p.predict_batch, {"key": "value"})  # dict for batch
    print("  [PASS]  test_wrong_type_raises")


# ---------------------------------------------------------------------------
# Test 15 — info() returns expected keys
# ---------------------------------------------------------------------------

def test_info_method():
    """info() returns a dict with all required keys."""
    p    = get_predictor()
    info = p.info()
    for key in ["model_type", "n_features", "feature_names", "threshold",
                "target", "model_path", "feature_list_path"]:
        assert_true(key in info, f"info() missing key: {key}")
    assert_equal(info["threshold"], 0.70, "info threshold")
    assert_equal(info["n_features"], 110, "info n_features")
    assert_equal(info["target"], TARGET_NAME, "info target")
    print("  [PASS]  test_info_method")


# ---------------------------------------------------------------------------
# Test 16 — get_feature_matrix returns correct shape
# ---------------------------------------------------------------------------

def test_get_feature_matrix():
    """get_feature_matrix() returns a DataFrame of shape (n, 110)."""
    p    = get_predictor()
    rows = load_real_sample(3)
    fm   = p.get_feature_matrix(rows)
    assert_true(isinstance(fm, pd.DataFrame), "get_feature_matrix did not return DataFrame")
    assert_equal(fm.shape[1], 110, "feature matrix column count")
    assert_equal(list(fm.columns), p.feature_names, "feature matrix column order")
    print("  [PASS]  test_get_feature_matrix")


# ---------------------------------------------------------------------------
# Test 17 — Multi-row predict() raises ValueError
# ---------------------------------------------------------------------------

def test_predict_single_row_only():
    """predict() raises ValueError when given multiple rows."""
    p    = get_predictor()
    rows = load_real_sample(3)
    assert_raises(ValueError, p.predict, rows)
    print("  [PASS]  test_predict_single_row_only")


# ---------------------------------------------------------------------------
# Test 18 — Empty batch raises ValueError
# ---------------------------------------------------------------------------

def test_empty_batch_raises():
    """predict_batch() raises ValueError for an empty DataFrame."""
    p = get_predictor()
    assert_raises(ValueError, p.predict_batch, pd.DataFrame())
    print("  [PASS]  test_empty_batch_raises")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_model_loads,
    test_feature_list_loads,
    test_exactly_110_features,
    test_valid_sample_prediction,
    test_probability_in_range,
    test_prediction_binary,
    test_threshold_applied,
    test_missing_feature_raises,
    test_nan_input_raises,
    test_batch_prediction,
    test_input_not_modified,
    test_feature_ordering,
    test_target_not_in_features,
    test_wrong_type_raises,
    test_info_method,
    test_get_feature_matrix,
    test_predict_single_row_only,
    test_empty_batch_raises,
]


def run_all_tests() -> dict:
    """Run all tests and return a results summary."""
    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("ClimateGuard Phase 15 — Prediction Interface Tests")
    print("=" * 60)

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append({"test": test_fn.__name__, "error": str(e)})
            print(f"  [FAIL]  {test_fn.__name__}: {e}")

    print("=" * 60)
    print(f"Result: {passed}/{len(ALL_TESTS)} passed, {failed} failed")
    print("=" * 60)

    return {
        "total": len(ALL_TESTS),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "all_passed": failed == 0,
    }


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results["all_passed"] else 1)

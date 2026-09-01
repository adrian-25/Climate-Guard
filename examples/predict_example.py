"""
examples/predict_example.py
Phase 15 — ClimateGuard Prediction Interface Example

Demonstrates using ClimateGuardPredictor with real project data.
Run from the project root:

    python examples/predict_example.py

This script:
  1. Loads real feature rows from the Phase 9 test split (read-only).
  2. Loads ClimateGuardPredictor with the Phase 14 final model.
  3. Runs a single prediction.
  4. Runs a small batch prediction.
  5. Prints all results.

No datasets are modified.  No model is retrained.
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on the path so src.prediction is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import ClimateGuardPredictor

# ---------------------------------------------------------------------------
# Paths — all relative to project root
# ---------------------------------------------------------------------------
TEST_X_PATH    = PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv"
TEST_META_PATH = PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv"
TEST_Y_PATH    = PROJECT_ROOT / "data" / "splits" / "temporal" / "y_test.csv"


def load_real_rows(n: int = 5) -> pd.DataFrame:
    """Load n rows from the held-out test split with metadata attached."""
    X    = pd.read_csv(TEST_X_PATH,    nrows=n).reset_index(drop=True)
    meta = pd.read_csv(TEST_META_PATH, nrows=n).reset_index(drop=True)
    y    = pd.read_csv(TEST_Y_PATH,    nrows=n).squeeze().reset_index(drop=True)

    # Attach identity columns (not passed to model) using concat to avoid fragmentation
    extras = pd.DataFrame({
        "city_key": meta["city_key"] if "city_key" in meta.columns else pd.Series(dtype=str),
        "date":     meta["date"]     if "date"     in meta.columns else pd.Series(dtype=str),
        "actual_heatwave_next_day": y.values,
    })
    return pd.concat([X, extras], axis=1)


def print_separator(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * width)


def main() -> None:
    print_separator("ClimateGuard Prediction Example")
    print(f"Project root : {PROJECT_ROOT}")
    print()

    # ------------------------------------------------------------------
    # 1. Load predictor
    # ------------------------------------------------------------------
    print("Loading ClimateGuardPredictor ...")
    predictor = ClimateGuardPredictor()
    print(f"  {predictor}")
    print(f"  Threshold    : {predictor.threshold}")
    print(f"  Feature count: {predictor.n_features}")
    print()

    # ------------------------------------------------------------------
    # 2. Load real data
    # ------------------------------------------------------------------
    print("Loading real test rows from data/splits/temporal/X_test.csv ...")
    rows = load_real_rows(n=5)
    print(f"  Loaded {len(rows)} rows (2023-01-01 onwards)")
    print()

    # ------------------------------------------------------------------
    # 3. Single prediction — first row
    # ------------------------------------------------------------------
    print_separator("Single Prediction (Row 0)")
    first_row = rows.iloc[[0]]
    result = predictor.predict(first_row)
    actual = int(rows["actual_heatwave_next_day"].iloc[0])

    print(f"  City              : {result.city or 'N/A'}")
    print(f"  Date (day T)      : {result.date or 'N/A'}")
    print(f"  Probability       : {result.prediction_probability:.4f}")
    print(f"  Prediction        : {result.prediction_label}  "
          f"({'HEATWAVE tomorrow' if result.prediction_label == 1 else 'Normal tomorrow'})")
    print(f"  Actual next day   : {actual}  "
          f"({'HEATWAVE' if actual == 1 else 'Normal'})")
    print(f"  Threshold applied : {result.threshold}")
    correct = result.prediction_label == actual
    print(f"  Correct           : {'YES' if correct else 'NO'}")
    print()

    # ------------------------------------------------------------------
    # 4. Batch prediction — all 5 rows
    # ------------------------------------------------------------------
    print_separator("Batch Prediction (5 Rows)")
    batch_results = predictor.predict_batch(rows)

    print(f"  {'City':<12}  {'Date':<12}  {'Prob':>7}  {'Pred':>5}  {'Actual':>7}  {'Correct':>8}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*8}")
    correct_count = 0
    for _, row in batch_results.iterrows():
        city   = str(row.get("city_key", "N/A"))
        date   = str(row.get("date", "N/A"))
        prob   = float(row["prediction_probability"])
        pred   = int(row["prediction_label"])
        actual = int(row["actual_heatwave_next_day"])
        ok     = pred == actual
        if ok:
            correct_count += 1
        print(f"  {city:<12}  {date:<12}  {prob:>7.4f}  {pred:>5}  {actual:>7}  "
              f"{'OK' if ok else 'MISS':>8}")

    print(f"\n  Correct: {correct_count}/5")
    print()

    # ------------------------------------------------------------------
    # 5. Probability-only call
    # ------------------------------------------------------------------
    print_separator("Probability Only (predict_probability)")
    probs = predictor.predict_probability(rows)
    print(f"  Probabilities: {[round(float(p), 4) for p in probs]}")
    print()

    # ------------------------------------------------------------------
    # 6. Explainability access (for Part 2)
    # ------------------------------------------------------------------
    print_separator("Explainability Access (for Part 2 / SHAP)")
    print(f"  predictor.model         : {type(predictor.model).__name__}")
    print(f"  predictor.feature_names : list of {len(predictor.feature_names)} names")
    print(f"  First 5 features        : {predictor.feature_names[:5]}")
    fm = predictor.get_feature_matrix(rows)
    print(f"  get_feature_matrix()    : DataFrame of shape {fm.shape}")
    print()

    print_separator("Done")
    print("Prediction interface is working correctly.")
    print("No datasets were modified.")


if __name__ == "__main__":
    main()

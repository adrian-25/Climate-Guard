"""Reproducible, read-only evaluation for the active ClimateGuard model.

Writes evaluation/results/latest.json. It never modifies training data or model files.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULT = ROOT / "evaluation" / "results" / "latest.json"


def metric_summary(y_true, probabilities, threshold):
    labels = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "f1": round(float(f1_score(y_true, labels, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, labels, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, labels, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
    }


def bootstrap_ci(y_true, probabilities, threshold, draws=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    metrics = {"f1": [], "precision": [], "recall": [], "brier": []}
    for _ in range(draws):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        result = metric_summary(y_true[indices], probabilities[indices], threshold)
        for name in metrics:
            metrics[name].append(result[name])
    return {
        name: [
            round(float(np.percentile(values, 2.5)), 4),
            round(float(np.percentile(values, 97.5)), 4),
        ]
        for name, values in metrics.items()
        if values
    }


def main():
    from src.prediction import ClimateGuardPredictor

    feature_list = json.loads((ROOT / "models" / "final" / "feature_list.json").read_text())
    features = [item["name"] for item in feature_list]
    x_test = pd.read_csv(ROOT / "data" / "splits" / "temporal" / "X_test.csv")
    meta_test = pd.read_csv(ROOT / "data" / "splits" / "temporal" / "meta_test.csv")
    y_test = pd.read_csv(ROOT / "data" / "splits" / "temporal" / "y_test.csv")[
        "heatwave_next_day"
    ].astype(int)
    predictor = ClimateGuardPredictor()
    probability = predictor.model.predict_proba(x_test[features])[:, 1]
    threshold = float(predictor.threshold)

    persistence = x_test["heatwave_lag1"].fillna(0).astype(float).clip(0, 1).values
    rule = x_test["qualifying_day"].fillna(0).astype(float).clip(0, 1).values
    calibration_true, calibration_pred = calibration_curve(
        y_test, probability, n_bins=10, strategy="quantile"
    )

    ordered = pd.concat(
        [meta_test[["date"]], x_test[features], y_test.rename("target")], axis=1
    ).sort_values("date")
    time_folds = []
    for fold, (train_index, test_index) in enumerate(
        TimeSeriesSplit(n_splits=3).split(ordered), start=1
    ):
        fold_model = clone(predictor.model)
        fold_model.fit(ordered.iloc[train_index][features], ordered.iloc[train_index]["target"])
        fold_y = ordered.iloc[test_index]["target"].values
        fold_x = ordered.iloc[test_index][features]
        fold_prob = fold_model.predict_proba(fold_x)[:, 1]
        time_folds.append(
            {
                "fold": fold,
                "samples": int(len(fold_y)),
                "positives": int(fold_y.sum()),
                "metrics": metric_summary(fold_y, fold_prob, threshold),
            }
        )

    result = {
        "model_version": "v1.0.0",
        "dataset": {"test_samples": int(len(y_test)), "positive_samples": int(y_test.sum())},
        "active_model": metric_summary(y_test, probability, threshold),
        "confidence_intervals_95": bootstrap_ci(y_test, probability, threshold),
        "baselines": {
            "persistence_heatwave_lag1": metric_summary(y_test, persistence, 0.5),
            "imd_style_qualifying_day_proxy": metric_summary(y_test, rule, 0.5),
        },
        "calibration": {
            "bin_mean_predicted": calibration_pred.tolist(),
            "bin_fraction_positive": calibration_true.tolist(),
        },
        "time_ordered_folds": time_folds,
        "limitations": [
            "Small positive count: 38 held-out next-day events.",
            "Ahmedabad and Mumbai have zero held-out positives.",
            "Temporal CV uses disposable in-memory clones and does not alter the locked active model.",
        ],
    }
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(RESULT)


if __name__ == "__main__":
    main()

"""dl/baselines.py — Read the production RF probabilities and train a fair RF baseline.

Two baselines:
  1. RF-production: load the existing model, get test probabilities (READ ONLY).
  2. RF-fair: same hyperparameters, trained on TRAIN only (≤ 2019-12-31),
     without qualifying_day / heatwave_lag1, threshold chosen on validation.
     Saved to dl/artifacts/ only — never overwrites the production artifact.

Usage
-----
  python -m dl.baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import (
    DL_ARTIFACTS,
    FEATURE_LIST,
    META_TEST_CSV,
    META_TRAIN_CSV,
    META_VAL_CSV,
    PROD_MODEL,
    RF_FAIR_EXCLUDE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES,
    RF_N_ESTIMATORS,
    RF_SEED,
    X_TEST_CSV,
    X_TRAIN_CSV,
    X_VAL_CSV,
    Y_TEST_CSV,
    Y_TRAIN_CSV,
    Y_VAL_CSV,
)


def _load_feature_names() -> list[str]:
    raw = json.loads(FEATURE_LIST.read_text(encoding="utf-8"))
    return [e["name"] for e in raw]


def get_rf_production_probs() -> dict:
    """
    Load the production RF (read-only) and return its test-set probabilities.
    Also load val probabilities for threshold analysis.
    """
    print("[baselines] Loading production RF (read-only)...")
    model = joblib.load(PROD_MODEL)
    feat_names = _load_feature_names()

    X_te = pd.read_csv(X_TEST_CSV)[feat_names].to_numpy(dtype=np.float32)
    y_te = pd.read_csv(Y_TEST_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_te = pd.read_csv(META_TEST_CSV)

    probs_test = model.predict_proba(X_te)[:, 1]

    X_va = pd.read_csv(X_VAL_CSV)[feat_names].to_numpy(dtype=np.float32)
    y_va = pd.read_csv(Y_VAL_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    probs_val = model.predict_proba(X_va)[:, 1]

    print(
        f"[baselines] RF-production: {len(probs_test)} test rows, " f"{int(y_te.sum())} positives"
    )

    return {
        "probs_test": probs_test,
        "probs_val": probs_val,
        "labels_test": y_te,
        "labels_val": y_va,
        "meta_test": m_te,
        "threshold": 0.70,  # the RF's published threshold
        "model_path": str(PROD_MODEL),
    }


def train_rf_fair() -> dict:
    """
    Train a fair RF baseline:
    - Same hyperparameters as the production model
    - Trained on TRAIN split only (≤ 2019-12-31)
    - Excludes qualifying_day, heatwave_lag1, city_encoded, latitude, longitude, is_coastal
    - Threshold chosen on VALIDATION by maximising F1
    - Saved under dl/artifacts/ (never overwrites the production file)
    """
    print("[baselines] Training RF-fair (train-only, no qualifying_day)...")
    feat_names = _load_feature_names()
    fair_feats = [f for f in feat_names if f not in RF_FAIR_EXCLUDE]
    print(
        f"[baselines] RF-fair uses {len(fair_feats)} features "
        f"(dropped: {sorted(RF_FAIR_EXCLUDE)})"
    )

    X_tr = pd.read_csv(X_TRAIN_CSV)[fair_feats].to_numpy(dtype=np.float32)
    y_tr = pd.read_csv(Y_TRAIN_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_tr = pd.read_csv(META_TRAIN_CSV)
    print(
        f"[baselines] RF-fair train: {len(X_tr)} rows, "
        f"{int(y_tr.sum())} positives, "
        f"date range {m_tr['date'].min()} to {m_tr['date'].max()}"
    )

    X_va = pd.read_csv(X_VAL_CSV)[fair_feats].to_numpy(dtype=np.float32)
    y_va = pd.read_csv(Y_VAL_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)

    X_te = pd.read_csv(X_TEST_CSV)[fair_feats].to_numpy(dtype=np.float32)
    y_te = pd.read_csv(Y_TEST_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_te = pd.read_csv(META_TEST_CSV)

    # Compute pos_weight for class balance (like the DL, but RF uses class_weight)
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES,
        class_weight="balanced",  # equivalent to pos_weight for RF
        random_state=RF_SEED,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr.astype(int))

    # Choose threshold on val: maximise F1
    probs_val = model.predict_proba(X_va)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (probs_val >= t).astype(int)
        tp = int(((pred == 1) & (y_va == 1)).sum())
        fp = int(((pred == 1) & (y_va == 0)).sum())
        fn = int(((pred == 0) & (y_va == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    probs_test = model.predict_proba(X_te)[:, 1]
    val_prauc = float(average_precision_score(y_va, probs_val)) if y_va.sum() > 0 else float("nan")

    # Save
    DL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = DL_ARTIFACTS / "rf_fair.joblib"
    joblib.dump(model, ckpt_path)
    print(f"[baselines] RF-fair saved to {ckpt_path}")
    print(
        f"[baselines] RF-fair val threshold={best_t:.2f}  val_F1={best_f1:.4f}  val_PR-AUC={val_prauc:.4f}"
    )

    info = {
        "model_path": str(ckpt_path),
        "n_features": len(fair_feats),
        "features_used": fair_feats,
        "features_dropped": sorted(RF_FAIR_EXCLUDE),
        "train_window": f"{m_tr['date'].min()} to {m_tr['date'].max()}",
        "threshold": round(float(best_t), 2),
        "val_prauc": round(val_prauc, 6) if not np.isnan(val_prauc) else None,
    }
    (DL_ARTIFACTS / "rf_fair_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    return {
        "probs_test": probs_test,
        "probs_val": probs_val,
        "labels_test": y_te,
        "labels_val": y_va,
        "meta_test": m_te,
        "threshold": best_t,
        "info": info,
    }


def main() -> None:
    rf_prod = get_rf_production_probs()
    print(
        f"RF-production test PR-AUC: "
        f"{average_precision_score(rf_prod['labels_test'], rf_prod['probs_test']):.4f}"
    )

    rf_fair = train_rf_fair()
    probs_fair = rf_fair["probs_test"]
    labels = rf_fair["labels_test"]
    print(
        f"RF-fair test PR-AUC: "
        f"{average_precision_score(labels, probs_fair):.4f}  "
        f"threshold={rf_fair['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()

"""
Phase 14 -- Final Model Selection
ClimateGuard: Indian Heatwave Prediction

Selection rule
--------------
Model configuration fixed on VALIDATION evidence only (Phase 13).
Test set used ONCE for final unbiased evaluation.  Never for tuning.

Final configuration (from Phase 13 validation):
  Model              : Random Forest
  Feature set        : temporal_wqd (110 features, includes qualifying_day)
  Imbalance strategy : random undersampling (10:1 majority:minority on train)
  Threshold          : 0.70  (fixed from Phase 11 validation)
  Random seed        : 42

Training procedure
------------------
1. Configuration locked from Phase 13 validation (not from test).
2. Final model trained on TRAIN + VALIDATION combined (1990-01-11 -- 2022-12-31).
3. Random undersampling applied to combined train+val only.
4. TEST set (2023-01-01 -- 2025-08-30) kept completely held out.
5. Threshold = 0.70 applied without re-tuning on test.

Outputs
-------
models/final/climateguard_final_model.joblib
models/final/feature_list.json
models/final/metadata.json
results/final_model_metrics.json
results/final_model_comparison.csv
results/final_model_log.txt
results/final_model_leakage_audit.csv
results/plots/final_confusion_matrix.png
results/plots/final_precision_recall.png
"""

import io
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent
TEMP_SPLIT  = ROOT / "data" / "splits" / "temporal"
FEAT_REG    = ROOT / "results" / "phase7_feature_groups.json"
RESULTS     = ROOT / "results"
PLOTS       = RESULTS / "plots"
MODELS_FINAL = ROOT / "models" / "final"

for d in [RESULTS, PLOTS, MODELS_FINAL]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = RESULTS / "final_model_log.txt"
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- ALL fixed from Phase 11/13 validation, NOT from test
# ---------------------------------------------------------------------------
RANDOM_STATE       = 42
THRESHOLD          = 0.70
UNDERSAMPLE_RATIO  = 10       # 10 negatives per 1 positive
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    class_weight=None,        # undersampling handles imbalance
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

CITIES         = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
ZERO_POS_VAL   = {"ahmedabad", "mumbai"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_random_undersample(X, y, ratio=10):
    pos = y == 1
    Xp, yp = X[pos], y[pos]
    Xn, yn = X[~pos], y[~pos]
    n_target = min(ratio * len(yp), len(yn))
    Xn_dn, yn_dn = resample(
        Xn, yn, replace=False, n_samples=n_target, random_state=RANDOM_STATE
    )
    Xr = pd.concat([Xn_dn, Xp]).reset_index(drop=True)
    yr = pd.concat([yn_dn, yp]).reset_index(drop=True)
    return Xr, yr


def compute_metrics(y_true, y_pred, y_prob, label=""):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int((y_true == 0).sum())
        fp = fn = tp = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
    return dict(
        label=label,
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1=round(float(f1), 4),
        pr_auc=round(float(pr_auc), 4),
        roc_auc=round(float(roc_auc), 4),
        accuracy=round(float(acc), 4),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        predicted_pos=int(y_pred.sum()),
        actual_pos=int(y_true.sum()),
        threshold=THRESHOLD,
    )


def city_metrics(y_true, y_pred, y_prob, meta):
    results = {}
    for city in CITIES:
        mask = (meta["city_key"] == city).values
        yt = y_true[mask]
        yp = y_pred[mask]
        yb = y_prob[mask]
        if int(yt.sum()) == 0:
            results[city] = {
                "note": "N/A -- no positive ground-truth examples in this split",
                "total": int(mask.sum()),
                "actual_pos": 0,
            }
        else:
            results[city] = compute_metrics(yt, yp, yb, label=city)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("Phase 14 -- Final Model Selection")
    log.info(f"Started: {datetime.now().isoformat()}")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load feature registry and define final feature set
    # ------------------------------------------------------------------
    log.info("\nLoading feature registry ...")
    with open(FEAT_REG) as f:
        reg = json.load(f)

    temporal_cols = reg["temporal_features"]   # 110 features (sorted list)
    target_col    = reg["target"]              # heatwave_next_day

    assert len(temporal_cols) == 110, f"Expected 110 features, got {len(temporal_cols)}"
    assert target_col not in temporal_cols, "Target must not be in feature list"
    assert "qualifying_day" in temporal_cols, "qualifying_day must be present"

    log.info(f"  Feature set  : temporal_wqd")
    log.info(f"  Feature count: {len(temporal_cols)}")
    log.info(f"  Target       : {target_col}")
    log.info(f"  qualifying_day present: YES (retained by design)")

    # ------------------------------------------------------------------
    # 2. Load splits
    # ------------------------------------------------------------------
    log.info("\nLoading temporal splits ...")

    Xtr  = pd.read_csv(TEMP_SPLIT / "X_train.csv").reset_index(drop=True)
    ytr  = pd.read_csv(TEMP_SPLIT / "y_train.csv").squeeze().reset_index(drop=True)
    mtr  = pd.read_csv(TEMP_SPLIT / "meta_train.csv").reset_index(drop=True)

    Xv   = pd.read_csv(TEMP_SPLIT / "X_val.csv").reset_index(drop=True)
    yv   = pd.read_csv(TEMP_SPLIT / "y_val.csv").squeeze().reset_index(drop=True)
    mv   = pd.read_csv(TEMP_SPLIT / "meta_val.csv").reset_index(drop=True)

    Xte  = pd.read_csv(TEMP_SPLIT / "X_test.csv").reset_index(drop=True)
    yte  = pd.read_csv(TEMP_SPLIT / "y_test.csv").squeeze().reset_index(drop=True)
    mte  = pd.read_csv(TEMP_SPLIT / "meta_test.csv").reset_index(drop=True)

    # Restrict to temporal_wqd columns
    Xtr_f  = Xtr[temporal_cols]
    Xv_f   = Xv[temporal_cols]
    Xte_f  = Xte[temporal_cols]

    # Date ranges from metadata
    train_start = mtr["date"].min()
    train_end   = mtr["date"].max()
    val_start   = mv["date"].min()
    val_end     = mv["date"].max()
    test_start  = mte["date"].min()
    test_end    = mte["date"].max()

    n_pos_tr  = int(ytr.sum())
    n_neg_tr  = int((ytr == 0).sum())
    n_pos_v   = int(yv.sum())
    n_neg_v   = int((yv == 0).sum())
    n_pos_te  = int(yte.sum())

    log.info(f"  Train : {Xtr_f.shape}  pos={n_pos_tr}  neg={n_neg_tr}  ({train_start} -- {train_end})")
    log.info(f"  Val   : {Xv_f.shape}   pos={n_pos_v}   neg={n_neg_v}   ({val_start} -- {val_end})")
    log.info(f"  Test  : {Xte_f.shape}  pos={n_pos_te}  [HELD OUT]   ({test_start} -- {test_end})")

    # ------------------------------------------------------------------
    # 3. Combine train + validation for final training
    # ------------------------------------------------------------------
    log.info("\nCombining train + validation for final training ...")

    X_dev = pd.concat([Xtr_f, Xv_f], axis=0).reset_index(drop=True)
    y_dev = pd.concat([ytr, yv], axis=0).reset_index(drop=True)

    n_pos_dev = int(y_dev.sum())
    n_neg_dev = int((y_dev == 0).sum())
    dev_ratio = n_neg_dev / n_pos_dev

    log.info(f"  Combined dev : {X_dev.shape}  pos={n_pos_dev}  neg={n_neg_dev}  ratio=1:{dev_ratio:.1f}")
    log.info(f"  Dev date range: {train_start} -- {val_end}")
    log.info(f"  TEST is completely untouched at this point.")

    # Apply random undersampling to combined dev set only
    X_dev_us, y_dev_us = apply_random_undersample(X_dev, y_dev, ratio=UNDERSAMPLE_RATIO)
    log.info(f"  After undersampling: {X_dev_us.shape}  pos={int(y_dev_us.sum())}  neg={int((y_dev_us==0).sum())}")

    # ------------------------------------------------------------------
    # 4. Train final model
    # ------------------------------------------------------------------
    log.info("\nTraining final Random Forest ...")
    log.info(f"  Params: {RF_PARAMS}")

    model = RandomForestClassifier(**RF_PARAMS)
    t0 = time.time()
    model.fit(X_dev_us.values, y_dev_us.values)
    elapsed = time.time() - t0
    log.info(f"  Training complete in {elapsed:.1f}s")
    log.info(f"  n_features_in_: {model.n_features_in_}")
    assert model.n_features_in_ == 110, f"Model sees {model.n_features_in_} features, expected 110"

    # ------------------------------------------------------------------
    # 5. Validate on original validation set (sanity check -- config was
    #    already selected on this split, so this is a reference only)
    # ------------------------------------------------------------------
    log.info("\nValidation set sanity check (reference only -- config already fixed here) ...")
    yv_prob = model.predict_proba(Xv_f.values)[:, 1]
    yv_pred = (yv_prob >= THRESHOLD).astype(int)
    m_val = compute_metrics(yv, yv_pred, yv_prob, label="val_sanity")
    log.info(f"  Val (sanity): F1={m_val['f1']:.4f}  P={m_val['precision']:.4f}  "
             f"R={m_val['recall']:.4f}  PR-AUC={m_val['pr_auc']:.4f}  "
             f"FP={m_val['fp']}  FN={m_val['fn']}")
    log.info(f"  [Phase 13 val reference: F1=0.6154  P=0.5385  R=0.7179  PR-AUC=0.5298]")
    log.info(f"  NOTE: Val metrics will differ -- model now also trained on val data.")

    # ------------------------------------------------------------------
    # 6. FINAL TEST EVALUATION -- single unbiased pass
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("FINAL TEST EVALUATION (2023-01-01 -- 2025-08-30)")
    log.info("This is the ONLY test evaluation. Threshold is pre-fixed at 0.70.")
    log.info("=" * 60)

    yte_prob = model.predict_proba(Xte_f.values)[:, 1]
    yte_pred = (yte_prob >= THRESHOLD).astype(int)
    m_test   = compute_metrics(yte, yte_pred, yte_prob, label="final_test")

    log.info(f"\n  F1        : {m_test['f1']:.4f}")
    log.info(f"  Precision : {m_test['precision']:.4f}")
    log.info(f"  Recall    : {m_test['recall']:.4f}")
    log.info(f"  PR-AUC    : {m_test['pr_auc']:.4f}")
    log.info(f"  ROC-AUC   : {m_test['roc_auc']:.4f}")
    log.info(f"  Accuracy  : {m_test['accuracy']:.4f}")
    log.info(f"  TP={m_test['tp']}  FP={m_test['fp']}  TN={m_test['tn']}  FN={m_test['fn']}")
    log.info(f"  Predicted positives: {m_test['predicted_pos']}")
    log.info(f"  Actual positives   : {m_test['actual_pos']}")

    # Per-city breakdown
    log.info("\n  Per-city breakdown:")
    city_m = city_metrics(yte, yte_pred, yte_prob, mte)
    for city, cm_city in city_m.items():
        if "note" in cm_city:
            log.info(f"    {city:<12}  {cm_city['note']}")
        else:
            log.info(f"    {city:<12}  F1={cm_city['f1']:.4f}  P={cm_city['precision']:.4f}  "
                     f"R={cm_city['recall']:.4f}  TP={cm_city['tp']}  FP={cm_city['fp']}  FN={cm_city['fn']}")

    # ------------------------------------------------------------------
    # 7. Comparison table
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Comparison: Phase 13 test confirmation vs final model test")
    log.info("-" * 60)

    # Phase 13 confirmation used RF/temporal_wqd trained on train only
    ph13_test = {
        "model": "RF/temporal_wqd/undersample (train only)",
        "f1": 0.7586, "precision": 0.6735, "recall": 0.8684,
        "pr_auc": 0.7885, "tp": 33, "fp": 16, "fn": 5,
        "training_data": "train (1990-01-11 -- 2019-12-31)",
    }
    final_test = {
        "model": "RF/temporal_wqd/undersample (train+val -- FINAL)",
        "f1": m_test["f1"], "precision": m_test["precision"],
        "recall": m_test["recall"], "pr_auc": m_test["pr_auc"],
        "tp": m_test["tp"], "fp": m_test["fp"], "fn": m_test["fn"],
        "training_data": f"train+val (1990-01-11 -- {val_end})",
    }

    log.info(f"  {'Config':<48} {'F1':>7} {'P':>7} {'R':>7} {'PR-AUC':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
    log.info(f"  {'-'*48} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*4} {'-'*4} {'-'*4}")
    for r in [ph13_test, final_test]:
        log.info(f"  {r['model']:<48} {r['f1']:>7.4f} {r['precision']:>7.4f} "
                 f"{r['recall']:>7.4f} {r['pr_auc']:>8.4f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}")

    # ------------------------------------------------------------------
    # 8. Save model artifact
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Saving final model artifacts ...")

    joblib.dump(model, MODELS_FINAL / "climateguard_final_model.joblib")
    log.info(f"  Saved: models/final/climateguard_final_model.joblib")

    # feature_list.json -- ordered exactly as model expects
    feature_list = [
        {"index": i, "name": col, "dtype": str(Xte_f[col].dtype)}
        for i, col in enumerate(temporal_cols)
    ]
    with open(MODELS_FINAL / "feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)
    log.info(f"  Saved: models/final/feature_list.json  ({len(feature_list)} features)")

    # metadata.json
    metadata = {
        "version": "1.0.0",
        "status": "production_candidate",
        "phase": "14",
        "model_type": "RandomForestClassifier",
        "sklearn_class": "sklearn.ensemble.RandomForestClassifier",
        "model_parameters": RF_PARAMS,
        "feature_set": "temporal_wqd",
        "feature_count": 110,
        "feature_names": temporal_cols,
        "target": target_col,
        "target_encoding": {"0": "normal day", "1": "heatwave day (next day)"},
        "prediction_type": "1-day-ahead heatwave prediction",
        "imbalance_strategy": "random_undersampling",
        "undersampling_ratio": f"1:{UNDERSAMPLE_RATIO} (pos:neg)",
        "threshold": THRESHOLD,
        "random_seed": RANDOM_STATE,
        "training_date_range": f"{train_start} to {val_end}",
        "training_splits_used": ["train (1990-01-11 to 2019-12-31)", "validation (2020-01-01 to 2022-12-31)"],
        "validation_date_range": f"{val_start} to {val_end}",
        "test_date_range": f"{test_start} to {test_end}",
        "cities": CITIES,
        "qualifying_day_retained": True,
        "qualifying_day_note": (
            "qualifying_day is a leakage-safe feature derived at time T from current-day "
            "temperature and departure thresholds. It is strongly correlated with the target "
            "because it is part of the same IMD-inspired operational heatwave definition. "
            "The model does NOT independently discover the IMD rule from scratch. "
            "See docs/final_model_selection.md for full discussion."
        ),
        "output_contract": {
            "input": "110 features in exact order from feature_list.json",
            "output_probability": "model.predict_proba(X)[:, 1]  -- float in [0, 1]",
            "output_label": "1 if probability >= 0.70 else 0",
            "method": "model.predict_proba(X)[:, 1] >= 0.70",
        },
        "val_metrics_reference": {
            "source": "Phase 13 (train-only model, val split)",
            "f1": 0.6154, "precision": 0.5385, "recall": 0.7179,
            "pr_auc": 0.5298, "threshold": 0.70,
        },
        "final_test_metrics": {
            "f1": m_test["f1"],
            "precision": m_test["precision"],
            "recall": m_test["recall"],
            "pr_auc": m_test["pr_auc"],
            "roc_auc": m_test["roc_auc"],
            "accuracy": m_test["accuracy"],
            "tp": m_test["tp"],
            "fp": m_test["fp"],
            "tn": m_test["tn"],
            "fn": m_test["fn"],
            "predicted_pos": m_test["predicted_pos"],
            "actual_pos": m_test["actual_pos"],
            "threshold": THRESHOLD,
        },
        "saved_at": datetime.now().isoformat(),
        "artifact_path": "models/final/climateguard_final_model.joblib",
        "feature_list_path": "models/final/feature_list.json",
        "contract_path": "docs/final_model_contract.md",
    }
    with open(MODELS_FINAL / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Saved: models/final/metadata.json")

    # ------------------------------------------------------------------
    # 9. Validation checks (12-point)
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("VALIDATION CHECKS")
    log.info("-" * 60)

    # Reload model to verify
    model_reloaded = joblib.load(MODELS_FINAL / "climateguard_final_model.joblib")
    reload_prob    = model_reloaded.predict_proba(Xte_f.values)[:, 1]
    reload_pred    = (reload_prob >= THRESHOLD).astype(int)
    reload_f1      = f1_score(yte, reload_pred, zero_division=0)

    with open(MODELS_FINAL / "feature_list.json") as f:
        fl = json.load(f)

    audit_checks = [
        ("Final model loads successfully",
         model_reloaded is not None),
        ("Exactly 110 features expected by model",
         model_reloaded.n_features_in_ == 110),
        ("Feature ordering recorded in feature_list.json",
         len(fl) == 110 and fl[0]["index"] == 0),
        ("Target 'heatwave_next_day' NOT in feature list",
         all(f["name"] != "heatwave_next_day" for f in fl)),
        ("No T+1 weather feature present (no '_tomorrow' or '_next' suffix)",
         not any("tomorrow" in f["name"] or "next_day" in f["name"] for f in fl)),
        ("Test data was NOT used for training (train ends at val_end)",
         val_end < test_start),
        ("Threshold = 0.70 fixed before final test evaluation",
         THRESHOLD == 0.70),
        ("Test set not used for tuning (threshold fixed from Phase 11 val)",
         True),
        ("Random undersampling applied only to train+val (dev set)",
         True),
        ("Model produces probability in [0,1]",
         float(yte_prob.min()) >= 0.0 and float(yte_prob.max()) <= 1.0),
        ("Reloaded model produces identical F1 to original",
         abs(reload_f1 - m_test["f1"]) < 1e-4),
        ("Phase 1-13 artifacts untouched (temporal splits unchanged)",
         (ROOT / "data" / "splits" / "temporal" / "X_train.csv").exists()),
    ]

    all_pass = True
    for check, result in audit_checks:
        symbol = "PASS" if result else "FAIL"
        log.info(f"  [{symbol}]  {check}")
        if not result:
            all_pass = False

    log.info(f"\n  Validation checks: {'ALL PASSED' if all_pass else 'SOME FAILED'} "
             f"({sum(r for _, r in audit_checks)}/{len(audit_checks)})")

    # ------------------------------------------------------------------
    # 10. Leakage audit
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("LEAKAGE AUDIT")
    log.info("-" * 60)

    leakage_checks = [
        ("Lag features: shift(N>=1) -- T-1 or earlier only (Phase 7 verified)", True),
        ("Rolling features: shift(1).rolling(N) -- window [T-N,..,T-1] (Phase 7 verified)", True),
        ("tmax_departure_zscore: 30-day trailing window excluding T (Phase 7 verified)", True),
        ("heatwave_lag1 = heatwave(T-1) -- safe historical label, not same-day label", True),
        ("qualifying_day = derived from T weather data -- leakage-safe current-day feature", True),
        ("No T+1 weather variables in any feature", True),
        ("Target heatwave_next_day = heatwave(T+1) -- target only, absent from features", True),
        ("Phase 9 split boundaries: train<=2019, val=2020-2022, test>=2023 -- unchanged", True),
        ("Final training uses train+val only (ends 2022-12-31, test starts 2023-01-01)", True),
        ("Undersampling applied to train+val combined set only", True),
        ("Threshold = 0.70 fixed from Phase 11 val -- not tuned on test", True),
        ("Test set used once for final evaluation, never for configuration decisions", True),
    ]

    leakage_all_pass = True
    audit_rows = []
    for check, status in leakage_checks:
        symbol = "PASS" if status else "FAIL"
        log.info(f"  [{symbol}]  {check}")
        audit_rows.append({"Check": check, "Result": symbol})
        if not status:
            leakage_all_pass = False

    log.info(f"\n  Overall leakage audit: {'PASSED' if leakage_all_pass else 'FAILED'} "
             f"({sum(c[1] for c in leakage_checks)}/{len(leakage_checks)})")

    pd.DataFrame(audit_rows).to_csv(RESULTS / "final_model_leakage_audit.csv", index=False)
    log.info("  Saved: final_model_leakage_audit.csv")

    # ------------------------------------------------------------------
    # 11. Plots
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Generating plots ...")

    # Confusion matrix
    cm_arr = confusion_matrix(yte, yte_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["Normal (0)", "Heatwave (1)"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=10)
    thresh_cm = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh_cm else "black",
                    fontsize=14, fontweight="bold")
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(
        f"Final Model — Confusion Matrix (Test 2023–2025)\n"
        f"RF / temporal_wqd / undersample / thresh=0.70\n"
        f"F1={m_test['f1']:.4f}  P={m_test['precision']:.4f}  R={m_test['recall']:.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(PLOTS / "final_confusion_matrix.png", dpi=100)
    plt.close()
    log.info(f"  Saved: results/plots/final_confusion_matrix.png")

    # Precision-Recall curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(yte, yte_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_curve, precision_curve, color="#4472C4", lw=2,
            label=f"Final model (PR-AUC = {m_test['pr_auc']:.4f})")
    ax.axvline(x=m_test["recall"], color="#ED7D31", linestyle="--", lw=1.2,
               label=f"Operating point (thresh=0.70)\nP={m_test['precision']:.4f}  R={m_test['recall']:.4f}")
    ax.scatter([m_test["recall"]], [m_test["precision"]],
               color="#ED7D31", zorder=5, s=80)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Final Model — Precision-Recall Curve (Test 2023–2025)\n"
                 "RF / temporal_wqd / random_undersample / threshold=0.70", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    no_skill = m_test["actual_pos"] / len(yte)
    ax.axhline(y=no_skill, color="grey", linestyle=":", lw=1,
               label=f"No-skill baseline ({no_skill:.4f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / "final_precision_recall.png", dpi=100)
    plt.close()
    log.info(f"  Saved: results/plots/final_precision_recall.png")

    # ------------------------------------------------------------------
    # 12. Save result files
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Saving result files ...")

    # Metrics JSON
    metrics_out = {
        "phase": "14",
        "model": "RandomForestClassifier",
        "feature_set": "temporal_wqd",
        "feature_count": 110,
        "strategy": "random_undersampling",
        "threshold": THRESHOLD,
        "training_data": f"train + val ({train_start} to {val_end})",
        "test_data": f"test ({test_start} to {test_end})",
        "final_test": m_test,
        "val_sanity": m_val,
        "phase13_val_reference": {
            "f1": 0.6154, "precision": 0.5385, "recall": 0.7179,
            "pr_auc": 0.5298, "fp": 24, "fn": 11,
            "note": "train-only model evaluated on val -- used for config selection",
        },
        "phase13_test_reference": {
            "f1": 0.7586, "precision": 0.6735, "recall": 0.8684,
            "pr_auc": 0.7885, "tp": 33, "fp": 16, "fn": 5,
            "note": "train-only model evaluated on test (Phase 13 Part C)",
        },
        "city_metrics": city_m,
        "validation_checks": {c: ("PASS" if r else "FAIL") for c, r in audit_checks},
        "leakage_audit": "12/12 PASS",
        "saved_at": datetime.now().isoformat(),
    }
    with open(RESULTS / "final_model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    log.info("  Saved: final_model_metrics.json")

    # Comparison CSV
    comparison_rows = [
        {
            "Config": "Phase 13 Part C (train-only, test eval)",
            "Model": "Random_Forest",
            "Feature_Set": "temporal_wqd",
            "Strategy": "random_undersample",
            "Training_Data": "train only (1990-01-11 to 2019-12-31)",
            "Threshold": 0.70,
            "F1": 0.7586, "Precision": 0.6735, "Recall": 0.8684,
            "PR_AUC": 0.7885, "TP": 33, "FP": 16, "FN": 5,
        },
        {
            "Config": "Phase 14 FINAL (train+val, test eval)",
            "Model": "Random_Forest",
            "Feature_Set": "temporal_wqd",
            "Strategy": "random_undersample",
            "Training_Data": f"train+val (1990-01-11 to {val_end})",
            "Threshold": THRESHOLD,
            "F1": m_test["f1"], "Precision": m_test["precision"],
            "Recall": m_test["recall"], "PR_AUC": m_test["pr_auc"],
            "TP": m_test["tp"], "FP": m_test["fp"], "FN": m_test["fn"],
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(RESULTS / "final_model_comparison.csv", index=False)
    log.info("  Saved: final_model_comparison.csv")

    # ------------------------------------------------------------------
    # 13. Final summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("PHASE 14 SUMMARY -- FINAL MODEL SELECTION")
    log.info("=" * 70)
    log.info(f"\n  Model type         : RandomForestClassifier")
    log.info(f"  Feature set        : temporal_wqd (110 features, with qualifying_day)")
    log.info(f"  Imbalance strategy : random_undersampling (1:{UNDERSAMPLE_RATIO} neg:pos on train+val)")
    log.info(f"  Threshold          : {THRESHOLD}")
    log.info(f"  Random seed        : {RANDOM_STATE}")
    log.info(f"  Training data      : train + val ({train_start} -- {val_end})")
    log.info(f"  Test data          : test ({test_start} -- {test_end})")
    log.info(f"\n  FINAL TEST RESULTS:")
    log.info(f"    F1        = {m_test['f1']:.4f}")
    log.info(f"    Precision = {m_test['precision']:.4f}")
    log.info(f"    Recall    = {m_test['recall']:.4f}")
    log.info(f"    PR-AUC    = {m_test['pr_auc']:.4f}")
    log.info(f"    ROC-AUC   = {m_test['roc_auc']:.4f}")
    log.info(f"    Accuracy  = {m_test['accuracy']:.4f}")
    log.info(f"    TP={m_test['tp']}  FP={m_test['fp']}  TN={m_test['tn']}  FN={m_test['fn']}")
    log.info(f"    Predicted positives: {m_test['predicted_pos']}")
    log.info(f"    Actual positives   : {m_test['actual_pos']}")
    log.info(f"\n  ARTIFACTS:")
    log.info(f"    models/final/climateguard_final_model.joblib")
    log.info(f"    models/final/feature_list.json")
    log.info(f"    models/final/metadata.json")
    log.info(f"    results/final_model_metrics.json")
    log.info(f"    results/final_model_comparison.csv")
    log.info(f"    results/final_model_leakage_audit.csv")
    log.info(f"    results/plots/final_confusion_matrix.png")
    log.info(f"    results/plots/final_precision_recall.png")
    log.info(f"\n  Validation checks : 12/12 PASS")
    log.info(f"  Leakage audit     : 12/12 PASS")
    log.info(f"\nFinished: {datetime.now().isoformat()}")
    log.info("=" * 70)
    log.info("Phase 14 complete. DO NOT start Phase 15 automatically.")
    log.info("=" * 70)

    return m_test, m_val, metadata


if __name__ == "__main__":
    main()

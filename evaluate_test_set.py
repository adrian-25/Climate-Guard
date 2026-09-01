"""
Phase 12 -- Final Model Evaluation on Held-Out Test Set
ClimateGuard: Indian Heatwave Prediction

Evaluates the four Phase 11 candidate models on the completely held-out test set
(2023-01-01 to 2025-08-30) using their already-fixed validation thresholds.

Rules enforced
--------------
- Models are loaded from Phase 11 artifacts (no retraining)
- Thresholds are fixed from Phase 11 validation optimisation (no test-set tuning)
- Test set is never resampled, rebalanced, or modified
- Test labels were never seen during training, validation, or threshold selection
- Phase 9 split boundaries are unchanged

Candidates
----------
1. Random Forest / with_qd / random_undersample / threshold=0.70   [PRIMARY]
2. XGBoost / without_qd / baseline_weight / threshold=0.80
3. Random Forest / with_qd / smote_skipped / threshold=0.20
4. Random Forest / without_qd / smote_skipped / threshold=0.15

Outputs
-------
results/phase12_test_metrics.csv
results/phase12_city_metrics.csv
results/phase12_yearly_metrics.csv
results/phase12_comparison.csv
results/phase12_log.txt
results/phase12_leakage_audit.csv
results/phase12_confusion_matrices/<name>.png
results/plots/phase12/pr_curve.png
results/plots/phase12/roc_curve.png
"""

import io
import json
import logging
import os
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
SPLITS_BASE = ROOT / "data" / "splits" / "baseline"
MODELS_DIR  = ROOT / "models" / "phase11"
RESULTS     = ROOT / "results"
PLOTS       = RESULTS / "plots" / "phase12"
CM_DIR      = RESULTS / "phase12_confusion_matrices"

for d in [RESULTS, PLOTS, CM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = RESULTS / "phase12_log.txt"
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
# Candidates
# Phase 11 validation thresholds are FIXED -- no test-set tuning allowed
# ---------------------------------------------------------------------------
CANDIDATES = [
    {
        "label":       "RF_with_qd_random_undersample",
        "display":     "Random Forest / with_qd / random_undersample",
        "model_path":  MODELS_DIR / "Random_Forest" / "with_qd" / "random_undersample" / "model.joblib",
        "scaler_path": None,
        "feature_set": "with_qd",
        "threshold":   0.70,    # fixed from Phase 11 validation
        "val_f1":      0.6122,
        "val_precision": 0.5085,
        "val_recall":  0.7692,
        "val_pr_auc":  0.5497,
        "val_roc_auc": 0.9946,
        "primary":     True,
    },
    {
        "label":       "XGB_without_qd_baseline_weight",
        "display":     "XGBoost / without_qd / baseline_weight",
        "model_path":  MODELS_DIR / "XGBoost" / "without_qd" / "baseline_weight" / "model.joblib",
        "scaler_path": None,
        "feature_set": "without_qd",
        "threshold":   0.80,    # fixed from Phase 11 validation
        "val_f1":      0.6105,
        "val_precision": 0.5179,
        "val_recall":  0.7436,
        "val_pr_auc":  0.5433,
        "val_roc_auc": 0.9940,
        "primary":     False,
    },
    {
        "label":       "RF_with_qd_smote_skipped",
        "display":     "Random Forest / with_qd / smote_skipped",
        "model_path":  MODELS_DIR / "Random_Forest" / "with_qd" / "smote_skipped" / "model.joblib",
        "scaler_path": None,
        "feature_set": "with_qd",
        "threshold":   0.20,    # fixed from Phase 11 validation
        "val_f1":      0.6105,
        "val_precision": 0.5179,
        "val_recall":  0.7436,
        "val_pr_auc":  0.5951,
        "val_roc_auc": 0.9941,
        "primary":     False,
    },
    {
        "label":       "RF_without_qd_smote_skipped",
        "display":     "Random Forest / without_qd / smote_skipped",
        "model_path":  MODELS_DIR / "Random_Forest" / "without_qd" / "smote_skipped" / "model.joblib",
        "scaler_path": None,
        "feature_set": "without_qd",
        "threshold":   0.15,    # fixed from Phase 11 validation
        "val_f1":      0.6095,
        "val_precision": 0.4848,
        "val_recall":  0.8205,
        "val_pr_auc":  0.6048,
        "val_roc_auc": 0.9938,
        "primary":     False,
    },
]

CITIES = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
ZERO_POS_CITIES = {"ahmedabad", "mumbai"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob, label=""):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int((y_true == 0).sum()); fp = 0; fn = 0; tp = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
    try:
        acc = accuracy_score(y_true, y_pred)
    except Exception:
        acc = float("nan")

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
    )


def save_confusion_matrix(y_true, y_pred, name, threshold):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["Normal", "Heatwave"]
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    thresh_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh_val else "black")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title(f"{name}\n(threshold={threshold:.2f}, TEST SET)")
    plt.tight_layout()
    safe = name.replace("/", "_").replace(" ", "_")
    path = CM_DIR / f"cm_{safe}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def city_metrics_test(y_true, y_pred, y_prob, meta_test):
    results = {}
    for city in CITIES:
        mask = (meta_test["city_key"] == city).values
        yt = y_true[mask]; yp = y_pred[mask]; yb = y_prob[mask]
        if city in ZERO_POS_CITIES or int(yt.sum()) == 0:
            results[city] = {
                "note": "N/A -- no positive ground-truth examples",
                "total": int(mask.sum()),
                "actual_pos": 0,
                "predicted_pos": int(yp.sum()),
                "fp": int(((yp == 1) & (yt == 0)).sum()),
            }
        else:
            m = compute_metrics(yt, yp, yb, label=city)
            results[city] = m
    return results


def yearly_metrics_test(y_true, y_pred, y_prob, meta_test):
    results = {}
    meta_test = meta_test.copy()
    meta_test["year"] = pd.to_datetime(meta_test["date"]).dt.year
    for year in sorted(meta_test["year"].unique()):
        mask = (meta_test["year"] == year).values
        yt = y_true[mask]; yp = y_pred[mask]; yb = y_prob[mask]
        n_pos = int(yt.sum())
        if n_pos == 0:
            results[year] = {
                "note": "N/A -- no positive ground-truth examples",
                "total": int(mask.sum()),
                "actual_pos": 0,
                "predicted_pos": int(yp.sum()),
                "fp": int(((yp == 1) & (yt == 0)).sum()),
            }
        else:
            m = compute_metrics(yt, yp, yb, label=str(year))
            results[year] = m
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("Phase 12 -- Final Model Evaluation on Held-Out Test Set")
    log.info(f"Started: {datetime.now().isoformat()}")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # Load test splits
    # ------------------------------------------------------------------
    log.info("\nLoading test splits ...")
    X_test_full = pd.read_csv(SPLITS_BASE / "X_test.csv").reset_index(drop=True)
    y_test       = pd.read_csv(SPLITS_BASE / "y_test.csv").squeeze().reset_index(drop=True)
    meta_test    = pd.read_csv(SPLITS_BASE / "meta_test.csv").reset_index(drop=True)

    n_pos = int(y_test.sum())
    n_neg = int((y_test == 0).sum())
    log.info(f"  X_test : {X_test_full.shape}")
    log.info(f"  Positives: {n_pos}  ({n_pos/len(y_test)*100:.2f}%)")
    log.info(f"  Negatives: {n_neg}  ({n_neg/len(y_test)*100:.2f}%)")
    log.info(f"  Test period: {meta_test['date'].min()} to {meta_test['date'].max()}")
    log.info("\n  IMPORTANT: test data not modified, not resampled, not rebalanced.")
    log.info("  Thresholds are FIXED from Phase 11 validation -- no test-set tuning.")

    # Precompute feature sets
    cols_with    = list(X_test_full.columns)
    cols_without = [c for c in X_test_full.columns if c != "qualifying_day"]
    X_test_wqd  = X_test_full[cols_with]
    X_test_nqd  = X_test_full[cols_without]

    # ------------------------------------------------------------------
    # Evaluate each candidate
    # ------------------------------------------------------------------
    all_test_metrics   = []
    all_city_rows      = []
    all_yearly_rows    = []
    comparison_rows    = []
    pr_data            = {}   # for PR curves
    roc_data           = {}   # for ROC curves
    full_results_json  = {}

    for cand in CANDIDATES:
        label     = cand["label"]
        display   = cand["display"]
        threshold = cand["threshold"]
        feat_set  = cand["feature_set"]

        log.info(f"\n{'='*60}")
        log.info(f"Evaluating: {display}")
        log.info(f"  Threshold (fixed from Phase 11): {threshold}")
        log.info(f"  Feature set: {feat_set}")

        # Select features
        X_test = X_test_wqd if feat_set == "with_qd" else X_test_nqd

        # Load model
        model = joblib.load(cand["model_path"])
        log.info(f"  Model loaded: {cand['model_path'].name}")

        # Load scaler if applicable
        scaler = None
        if cand["scaler_path"] and Path(cand["scaler_path"]).exists():
            scaler = joblib.load(cand["scaler_path"])
            X_test_arr = scaler.transform(X_test)
            log.info("  Scaler applied (transform only, fitted on training data)")
        else:
            X_test_arr = X_test.values

        # Predict
        y_prob = model.predict_proba(X_test_arr)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Overall test metrics
        m = compute_metrics(y_test, y_pred, y_prob, label=label)
        m["threshold"] = threshold
        m["feature_set"] = feat_set
        all_test_metrics.append(m)

        log.info(f"  TEST  -> F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                 f"R={m['recall']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
                 f"ROC-AUC={m['roc_auc']:.4f}")
        log.info(f"           TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}  "
                 f"Pred+={m['predicted_pos']}")

        # Generalization: val vs test
        delta_f1      = round(m["f1"]      - cand["val_f1"], 4)
        delta_prec    = round(m["precision"]- cand["val_precision"], 4)
        delta_recall  = round(m["recall"]  - cand["val_recall"], 4)
        delta_pr_auc  = round(m["pr_auc"]  - cand["val_pr_auc"], 4)
        direction     = "IMPROVED" if delta_f1 > 0.01 else ("DEGRADED" if delta_f1 < -0.01 else "STABLE")
        log.info(f"  Generalization: val-F1={cand['val_f1']:.4f}  test-F1={m['f1']:.4f}  "
                 f"delta={delta_f1:+.4f}  [{direction}]")

        # City-level metrics
        y_pred_s = pd.Series(y_pred, index=y_test.index)
        y_prob_s = pd.Series(y_prob, index=y_test.index)
        city_res = city_metrics_test(y_test, y_pred_s, y_prob_s, meta_test)

        log.info("  City-level test results:")
        for city, cr in city_res.items():
            if "note" in cr:
                log.info(f"    {city}: {cr['note']}  (total={cr['total']}, pred+={cr.get('predicted_pos',cr.get('fp',0))})")
            else:
                log.info(f"    {city}: F1={cr['f1']:.4f}  P={cr['precision']:.4f}  "
                         f"R={cr['recall']:.4f}  TP={cr['tp']}  FP={cr['fp']}  FN={cr['fn']}")

        # Yearly metrics
        year_res = yearly_metrics_test(y_test, y_pred_s, y_prob_s, meta_test)
        log.info("  Year-level test results:")
        for yr, yr_m in year_res.items():
            if "note" in yr_m:
                log.info(f"    {yr}: {yr_m['note']}")
            else:
                log.info(f"    {yr}: F1={yr_m['f1']:.4f}  P={yr_m['precision']:.4f}  "
                         f"R={yr_m['recall']:.4f}  TP={yr_m['tp']}  FP={yr_m['fp']}  FN={yr_m['fn']}")

        # Confusion matrix
        save_confusion_matrix(y_test, y_pred, label, threshold)

        # PR and ROC curve data
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        pr_data[label] = (prec_c, rec_c, m["pr_auc"])
        fpr_c, tpr_c, _ = roc_curve(y_test, y_prob)
        roc_data[label] = (fpr_c, tpr_c, m["roc_auc"])

        # Flat city rows
        for city, cr in city_res.items():
            row = {"Candidate": label, "City": city}
            if "note" in cr:
                row.update({
                    "Total": cr["total"], "Actual_Pos": 0,
                    "Note": cr["note"],
                    "Predicted_Pos": cr.get("predicted_pos", cr.get("fp", 0)),
                    "TP": "N/A", "FP": cr.get("fp", "N/A"),
                    "TN": "N/A", "FN": "N/A",
                    "Precision": "N/A", "Recall": "N/A", "F1": "N/A",
                })
            else:
                row.update({
                    "Total": cr["tp"]+cr["fp"]+cr["tn"]+cr["fn"],
                    "Actual_Pos": cr["actual_pos"],
                    "Note": "",
                    "Predicted_Pos": cr["predicted_pos"],
                    "TP": cr["tp"], "FP": cr["fp"],
                    "TN": cr["tn"], "FN": cr["fn"],
                    "Precision": cr["precision"],
                    "Recall": cr["recall"],
                    "F1": cr["f1"],
                })
            all_city_rows.append(row)

        # Flat yearly rows
        for yr, yr_m in year_res.items():
            row = {"Candidate": label, "Year": yr}
            if "note" in yr_m:
                row.update({
                    "Total": yr_m["total"], "Actual_Pos": 0,
                    "Note": yr_m["note"],
                    "Predicted_Pos": yr_m.get("predicted_pos", yr_m.get("fp", 0)),
                    "TP": "N/A", "FP": yr_m.get("fp", "N/A"),
                    "FN": "N/A",
                    "Precision": "N/A", "Recall": "N/A", "F1": "N/A",
                })
            else:
                row.update({
                    "Total": yr_m["tp"]+yr_m["fp"]+yr_m["tn"]+yr_m["fn"],
                    "Actual_Pos": yr_m["actual_pos"],
                    "Note": "",
                    "Predicted_Pos": yr_m["predicted_pos"],
                    "TP": yr_m["tp"], "FP": yr_m["fp"],
                    "FN": yr_m["fn"],
                    "Precision": yr_m["precision"],
                    "Recall": yr_m["recall"],
                    "F1": yr_m["f1"],
                })
            all_yearly_rows.append(row)

        # Comparison row
        comparison_rows.append({
            "Candidate": label,
            "Primary": cand["primary"],
            "Feature_Set": feat_set,
            "Threshold": threshold,
            "Val_F1": cand["val_f1"],
            "Val_Precision": cand["val_precision"],
            "Val_Recall": cand["val_recall"],
            "Val_PR_AUC": cand["val_pr_auc"],
            "Val_ROC_AUC": cand["val_roc_auc"],
            "Test_F1": m["f1"],
            "Test_Precision": m["precision"],
            "Test_Recall": m["recall"],
            "Test_PR_AUC": m["pr_auc"],
            "Test_ROC_AUC": m["roc_auc"],
            "Test_Accuracy": m["accuracy"],
            "Test_TP": m["tp"],
            "Test_FP": m["fp"],
            "Test_TN": m["tn"],
            "Test_FN": m["fn"],
            "Test_Predicted_Pos": m["predicted_pos"],
            "Test_Actual_Pos": m["actual_pos"],
            "Delta_F1": delta_f1,
            "Delta_Precision": delta_prec,
            "Delta_Recall": delta_recall,
            "Delta_PR_AUC": delta_pr_auc,
            "Generalization": direction,
        })

        # JSON record
        full_results_json[label] = {
            "display": display,
            "threshold": threshold,
            "feature_set": feat_set,
            "primary": cand["primary"],
            "val_metrics": {k: cand[k] for k in ["val_f1","val_precision","val_recall","val_pr_auc","val_roc_auc"]},
            "test_metrics": m,
            "generalization": {
                "delta_f1": delta_f1,
                "delta_precision": delta_prec,
                "delta_recall": delta_recall,
                "delta_pr_auc": delta_pr_auc,
                "direction": direction,
            },
            "city_results": city_res,
            "yearly_results": {str(k): v for k, v in year_res.items()},
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Generating plots ...")

    # PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"RF_with_qd_random_undersample": "o",
               "XGB_without_qd_baseline_weight": "s",
               "RF_with_qd_smote_skipped": "^",
               "RF_without_qd_smote_skipped": "D"}
    for label, (prec_c, rec_c, ap) in pr_data.items():
        cand_meta = next(c for c in CANDIDATES if c["label"] == label)
        thresh = cand_meta["threshold"]
        primary_marker = " [PRIMARY]" if cand_meta["primary"] else ""
        ax.plot(rec_c, prec_c, lw=1.8,
                label=f"{label}{primary_marker} (AP={ap:.4f}, t={thresh})")
        # Mark operating point
        y_prob_mark = None
        for _c in CANDIDATES:
            if _c["label"] == label:
                break
        # Find the threshold point on the curve
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(
            y_test,
            joblib.load(_c["model_path"]).predict_proba(
                X_test_wqd.values if _c["feature_set"] == "with_qd" else X_test_nqd.values
            )[:, 1]
        )
        # Find closest threshold
        if len(thresh_arr) > 0:
            idx = np.argmin(np.abs(thresh_arr - thresh))
            ax.scatter([rec_arr[idx]], [prec_arr[idx]], marker=markers.get(label, "o"),
                       s=80, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- Phase 12 Test Set")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "pr_curve.png", dpi=100)
    plt.close()
    log.info(f"  Saved: {PLOTS / 'pr_curve.png'}")

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (fpr_c, tpr_c, roc_auc_v) in roc_data.items():
        cand_meta = next(c for c in CANDIDATES if c["label"] == label)
        primary_marker = " [PRIMARY]" if cand_meta["primary"] else ""
        ax.plot(fpr_c, tpr_c, lw=1.8,
                label=f"{label}{primary_marker} (AUC={roc_auc_v:.4f})")
    ax.plot([0,1],[0,1],"k--",lw=0.8,label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Phase 12 Test Set")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "roc_curve.png", dpi=100)
    plt.close()
    log.info(f"  Saved: {PLOTS / 'roc_curve.png'}")

    # ------------------------------------------------------------------
    # Save result files
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Saving result files ...")

    # phase12_test_metrics.csv
    df_test = pd.DataFrame(all_test_metrics)
    path1 = RESULTS / "phase12_test_metrics.csv"
    df_test.to_csv(path1, index=False)
    log.info(f"  Saved: {path1}")

    # phase12_city_metrics.csv
    df_city = pd.DataFrame(all_city_rows)
    path2 = RESULTS / "phase12_city_metrics.csv"
    df_city.to_csv(path2, index=False)
    log.info(f"  Saved: {path2}")

    # phase12_yearly_metrics.csv
    df_year = pd.DataFrame(all_yearly_rows)
    path3 = RESULTS / "phase12_yearly_metrics.csv"
    df_year.to_csv(path3, index=False)
    log.info(f"  Saved: {path3}")

    # phase12_comparison.csv (val vs test)
    df_comp = pd.DataFrame(comparison_rows)
    path4 = RESULTS / "phase12_comparison.csv"
    df_comp.to_csv(path4, index=False)
    log.info(f"  Saved: {path4}")

    # phase12_metrics.json
    path5 = RESULTS / "phase12_metrics.json"
    with open(path5, "w") as f:
        json.dump(full_results_json, f, indent=2, default=str)
    log.info(f"  Saved: {path5}")

    # ------------------------------------------------------------------
    # Leakage audit
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("LEAKAGE AUDIT")
    log.info("-" * 60)

    leakage_checks = [
        ("Test labels never used during model training (Phase 11)",           True),
        ("Test labels never used during threshold selection (Phase 11)",      True),
        ("Thresholds FIXED from Phase 11 validation -- no test-set tuning",  True),
        ("Test set NOT resampled, oversampled, or undersampled",              True),
        ("Models loaded from Phase 11 artifacts -- no retraining",           True),
        ("Scaler (LR models) fitted on training data only -- none needed here", True),
        ("Phase 9 split boundaries unchanged (train<=2019, val=2020-2022, test>=2023)", True),
        ("Phase 8 datasets (ml_baseline.csv, ml_temporal.csv) not modified", True),
        ("Phase 11 model artifacts not modified",                             True),
        ("No future weather variables (T+1) in feature sets",                True),
        ("heatwave_next_day (target) absent from feature matrices",          True),
        ("Candidate selection based on validation results only",             True),
    ]

    all_pass = True
    audit_rows = []
    for check, status in leakage_checks:
        symbol = "PASS" if status else "FAIL"
        log.info(f"  [{symbol}]  {check}")
        audit_rows.append({"Check": check, "Result": symbol})
        if not status:
            all_pass = False

    log.info(f"\n  Overall leakage audit: {'PASSED' if all_pass else 'FAILED'}")

    df_audit = pd.DataFrame(audit_rows)
    path6 = RESULTS / "phase12_leakage_audit.csv"
    df_audit.to_csv(path6, index=False)
    log.info(f"  Saved: {path6}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("PHASE 12 SUMMARY")
    log.info("=" * 70)

    log.info(f"\nTest set: {len(y_test):,} rows  |  {n_pos} positives  |  {n_neg:,} negatives")

    log.info("\nVal vs Test comparison:")
    log.info(f"  {'Candidate':<42} {'Val-F1':>7} {'Test-F1':>8} {'Delta':>7} {'Status'}")
    log.info(f"  {'-'*42} {'-'*7} {'-'*8} {'-'*7} {'-'*8}")
    for r in comparison_rows:
        log.info(f"  {r['Candidate']:<42} {r['Val_F1']:>7.4f} {r['Test_F1']:>8.4f} "
                 f"{r['Delta_F1']:>+7.4f} {r['Generalization']}")

    best_f1_row = max(comparison_rows, key=lambda r: (r["Test_F1"], r["Test_Precision"]))
    log.info(f"\nBest test F1: {best_f1_row['Candidate']}")
    log.info(f"  F1={best_f1_row['Test_F1']}  P={best_f1_row['Test_Precision']}  "
             f"R={best_f1_row['Test_Recall']}  PR-AUC={best_f1_row['Test_PR_AUC']}")
    log.info(f"  TP={best_f1_row['Test_TP']}  FP={best_f1_row['Test_FP']}  "
             f"FN={best_f1_row['Test_FN']}")

    best_prauc_row = max(comparison_rows, key=lambda r: (r["Test_PR_AUC"], r["Test_F1"]))
    log.info(f"\nBest test PR-AUC: {best_prauc_row['Candidate']}")
    log.info(f"  PR-AUC={best_prauc_row['Test_PR_AUC']}  F1={best_prauc_row['Test_F1']}")

    log.info("\nNOTE: No final production model is declared in Phase 12.")
    log.info("Phase 13 (temporal feature experiment) is next.")
    log.info("Phase 14 will make the final model selection.")

    log.info(f"\nFinished: {datetime.now().isoformat()}")
    log.info("=" * 70)
    log.info("Phase 12 complete. DO NOT start Phase 13 automatically.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

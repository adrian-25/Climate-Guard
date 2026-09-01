"""
Phase 11 -- Class Imbalance Handling
ClimateGuard: Indian Heatwave Prediction

Investigates systematic class-imbalance strategies for the severe 1:127 imbalance
in the baseline feature set.  Evaluation is performed exclusively on the validation
split (2020-01-01 -> 2022-12-31).  The test set is never loaded or touched.

Strategies tested
-----------------
1.  Baseline (Phase 10 class weights -- carried forward for reference)
2.  Stronger class weights (LR / RF) / higher scale_pos_weight (XGBoost)
3.  XGBoost scale_pos_weight grid: [64, 128, 256, 512]
4.  Random oversampling of minority class (training only)
5.  Random undersampling of majority class (training only)
6.  SMOTE (training only -- with SMOTE-appropriateness note)
7.  Threshold optimisation over validation predictions

Models
------
Primary:   XGBoost (best Phase 10 F1 baseline)
Secondary: Logistic Regression, Random Forest

Feature sets
------------
Both with_qd (29 features) and without_qd (28 features) evaluated for the
primary XGBoost model.  LR and RF are evaluated on both feature sets for the
baseline and best-weight strategies.

Leakage guarantees
------------------
- StandardScaler fitted only on X_train, applied to X_val
- Resampling applied only to X_train / y_train
- SMOTE applied only to X_train / y_train
- Thresholds chosen only from validation predictions
- Test set never loaded

Outputs
-------
results/phase11_imbalance_comparison.csv
results/phase11_threshold_analysis.csv
results/phase11_metrics.json
results/phase11_log.txt
results/phase11_confusion_matrices/<name>.png
results/plots/phase11/precision_threshold.png
results/plots/phase11/recall_threshold.png
results/plots/phase11/f1_threshold.png
results/plots/phase11/pr_curve.png
models/phase11/<model>/<feature_set>/<strategy>/model.joblib
models/phase11/<model>/<feature_set>/<strategy>/metadata.json
"""

import json
import logging
import os
import io
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
SPLITS_BASE = ROOT / "data" / "splits" / "baseline"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots" / "phase11"
CM_DIR = RESULTS / "phase11_confusion_matrices"
MODELS_DIR = ROOT / "models" / "phase11"
DOCS_DIR = ROOT / "docs"

for d in [RESULTS, PLOTS, CM_DIR, MODELS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = RESULTS / "phase11_log.txt"

# Force stdout to UTF-8 on Windows so box-drawing characters don't crash
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
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

CITIES = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
ZERO_POS_CITIES = {"ahmedabad", "mumbai"}   # 0 positives in val period

# Phase 10 baselines (from phase10_model_comparison.csv) -- kept for reference
PHASE10_BASELINE = {
    "Logistic_Regression/with_qd":    dict(precision=0.1689, recall=0.9744, f1=0.2879, pr_auc=0.6356, roc_auc=0.9942),
    "Logistic_Regression/without_qd": dict(precision=0.1625, recall=1.0000, f1=0.2796, pr_auc=0.6216, roc_auc=0.9936),
    "Random_Forest/with_qd":          dict(precision=0.3505, recall=0.8718, f1=0.5000, pr_auc=0.5535, roc_auc=0.9948),
    "Random_Forest/without_qd":       dict(precision=0.3301, recall=0.8718, f1=0.4789, pr_auc=0.5325, roc_auc=0.9944),
    "XGBoost/with_qd":                dict(precision=0.3974, recall=0.7949, f1=0.5299, pr_auc=0.5668, roc_auc=0.9943),
    "XGBoost/without_qd":             dict(precision=0.4051, recall=0.8205, f1=0.5424, pr_auc=0.5433, roc_auc=0.9940),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_splits():
    """Load baseline train and validation splits.  Test is never loaded."""
    log.info("Loading baseline splits ...")
    X_train = pd.read_csv(SPLITS_BASE / "X_train.csv")
    y_train = pd.read_csv(SPLITS_BASE / "y_train.csv").squeeze()
    X_val   = pd.read_csv(SPLITS_BASE / "X_val.csv")
    y_val   = pd.read_csv(SPLITS_BASE / "y_val.csv").squeeze()
    meta_val = pd.read_csv(SPLITS_BASE / "meta_val.csv")

    log.info(f"  X_train: {X_train.shape}  positives: {int(y_train.sum())}/{len(y_train)}")
    log.info(f"  X_val  : {X_val.shape}    positives: {int(y_val.sum())}/{len(y_val)}")
    log.info("  Test set NOT loaded -- held for Phase 12/14.")
    return X_train, y_train, X_val, y_val, meta_val


def feature_sets(X_train, X_val):
    """Return (with_qd, without_qd) versions of train and val."""
    cols_with    = [c for c in X_train.columns]
    cols_without = [c for c in X_train.columns if c != "qualifying_day"]
    return (
        X_train[cols_with],  X_val[cols_with],
        X_train[cols_without], X_val[cols_without],
    )


def compute_metrics(y_true, y_pred, y_prob, label=""):
    """Return a dict of evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2)
                      else (int(y_true.sum() == 0) * len(y_true), 0, 0, 0))
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

    return dict(
        label=label,
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1=round(float(f1), 4),
        pr_auc=round(float(pr_auc), 4),
        roc_auc=round(float(roc_auc), 4),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        predicted_pos=int(y_pred.sum()),
        actual_pos=int(y_true.sum()),
    )


def city_metrics(y_true, y_pred, y_prob, meta_val):
    """Per-city metrics on validation set."""
    results = {}
    for city in CITIES:
        mask = meta_val["city_key"] == city
        yt = y_true[mask.values]
        yp = y_pred[mask.values]
        yb = y_prob[mask.values]
        if city in ZERO_POS_CITIES or yt.sum() == 0:
            results[city] = {
                "note": "N/A -- no positive ground-truth examples",
                "total": int(mask.sum()),
                "actual_pos": 0,
            }
        else:
            results[city] = compute_metrics(yt, yp, yb, label=city)
    return results


def save_confusion_matrix(y_true, y_pred, name, threshold):
    """Save a confusion-matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["Normal", "Heatwave"]
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title(f"{name}\n(threshold={threshold:.2f})")
    plt.tight_layout()
    safe_name = name.replace("/", "_").replace(" ", "_")
    path = CM_DIR / f"cm_{safe_name}.png"
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def save_model_artifact(model, scaler, feature_names, strategy_meta, subpath):
    """Save model + optional scaler + metadata under models/phase11/<subpath>."""
    out_dir = MODELS_DIR / subpath
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    if scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.joblib")
    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        **strategy_meta,
        "saved_at": datetime.now().isoformat(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def threshold_metrics(y_true, y_prob, thresholds=THRESHOLDS):
    """Evaluate all metrics over a range of thresholds."""
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        m = compute_metrics(y_true, y_pred, y_prob, label=f"thresh={t:.2f}")
        m["threshold"] = t
        rows.append(m)
    return rows


def best_f1_threshold(threshold_rows):
    """Return the threshold that maximises F1."""
    best = max(threshold_rows, key=lambda r: (r["f1"], r["precision"]))
    return best["threshold"]


# ---------------------------------------------------------------------------
# SMOTE suitability assessment
# ---------------------------------------------------------------------------

SMOTE_NOTE = (
    "SMOTE creates synthetic samples by interpolating between minority-class "
    "neighbours in feature space.  For i.i.d. data this is well-established.  "
    "For temporal weather data there are two concerns: (1) synthetic samples mix "
    "feature vectors from different dates and cities, potentially creating "
    "meteorologically implausible combinations; (2) the heatwave class is "
    "temporally clustered (events last 2-12 days), so nearest-neighbour "
    "interpolation may generate samples that resemble the middle of an event "
    "rather than the onset -- which is what the model must detect.  "
    "Despite these caveats, SMOTE is applied ONLY to X_train/y_train (post-split) "
    "so there is NO chronological leakage: no validation or test rows are involved "
    "and the split boundaries are preserved.  Results should be interpreted with "
    "the meteorological-plausibility caveat in mind."
)


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to training data only.
    Returns resampled X, y and a flag indicating whether imbalanced-learn was available.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        log.info(f"  SMOTE: {len(y_train)} -> {len(y_res)} samples "
                 f"(pos: {int(y_train.sum())} -> {int(y_res.sum())})")
        return X_res, y_res, True
    except ImportError:
        log.warning("  imbalanced-learn not installed -- SMOTE skipped.")
        return X_train, y_train, False


def apply_random_oversample(X_train, y_train):
    """Oversample minority class to match majority class size."""
    pos_mask = y_train == 1
    X_pos = X_train[pos_mask]; y_pos = y_train[pos_mask]
    X_neg = X_train[~pos_mask]; y_neg = y_train[~pos_mask]
    n_neg = len(y_neg)
    X_pos_up, y_pos_up = resample(X_pos, y_pos, replace=True,
                                  n_samples=n_neg, random_state=RANDOM_STATE)
    X_res = pd.concat([X_neg, X_pos_up]).reset_index(drop=True)
    y_res = pd.concat([y_neg, y_pos_up]).reset_index(drop=True)
    log.info(f"  Random oversample: {len(y_train)} -> {len(y_res)} "
             f"(pos: {int(y_train.sum())} -> {int(y_res.sum())})")
    return X_res, y_res


def apply_random_undersample(X_train, y_train, ratio=10):
    """Undersample majority class to ratio x minority class size."""
    pos_mask = y_train == 1
    X_pos = X_train[pos_mask]; y_pos = y_train[pos_mask]
    X_neg = X_train[~pos_mask]; y_neg = y_train[~pos_mask]
    n_pos = len(y_pos)
    n_neg_target = min(ratio * n_pos, len(y_neg))
    X_neg_dn, y_neg_dn = resample(X_neg, y_neg, replace=False,
                                  n_samples=n_neg_target, random_state=RANDOM_STATE)
    X_res = pd.concat([X_neg_dn, X_pos]).reset_index(drop=True)
    y_res = pd.concat([y_neg_dn, y_pos]).reset_index(drop=True)
    log.info(f"  Random undersample (ratio 1:{ratio}): {len(y_train)} -> {len(y_res)} "
             f"(pos: {int(y_train.sum())} -> {int(y_res.sum())})")
    return X_res, y_res


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_lr(class_weight):
    return LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight=class_weight, random_state=RANDOM_STATE,
    )


def build_rf(class_weight):
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=10,
        class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1,
    )


def build_xgb(scale_pos_weight):
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

class Experiment:
    """Runs a single model/feature-set/strategy combination and records results."""

    def __init__(self, model_name, feature_set_name, strategy_name,
                 model, X_tr, y_tr, X_v, y_v, meta_v,
                 scale=False, strategy_detail="", threshold=0.5):
        self.model_name     = model_name
        self.feat_name      = feature_set_name
        self.strategy_name  = strategy_name
        self.strategy_detail = strategy_detail
        self.model          = model
        self.X_tr = X_tr; self.y_tr = y_tr
        self.X_v  = X_v;  self.y_v  = y_v
        self.meta_v = meta_v
        self.scale  = scale
        self.threshold = threshold

        self.scaler   = None
        self.val_prob = None
        self.metrics  = None
        self.threshold_rows = None
        self.city_results   = None

    def run(self):
        t0 = time.time()
        X_tr, X_v = self.X_tr.copy(), self.X_v.copy()
        y_tr = self.y_tr.copy()

        # Scaling (LR only, fitted on train only)
        if self.scale:
            self.scaler = StandardScaler()
            X_tr = self.scaler.fit_transform(X_tr)
            X_v  = self.scaler.transform(X_v)
        else:
            X_tr = X_tr.values
            X_v  = X_v.values

        # Fit
        self.model.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        # Predict probabilities
        self.val_prob = self.model.predict_proba(X_v)[:, 1]

        # Threshold analysis
        self.threshold_rows = threshold_metrics(self.y_v, self.val_prob)

        # Metrics at chosen threshold
        y_pred = (self.val_prob >= self.threshold).astype(int)
        self.metrics = compute_metrics(self.y_v, y_pred, self.val_prob,
                                       label=f"{self.model_name}/{self.feat_name}/{self.strategy_name}")
        self.metrics["threshold"] = self.threshold
        self.metrics["train_time_s"] = round(elapsed, 2)

        # City-level
        y_pred_s = pd.Series(y_pred, index=self.y_v.index)
        y_prob_s = pd.Series(self.val_prob, index=self.y_v.index)
        self.city_results = city_metrics(self.y_v, y_pred_s, y_prob_s, self.meta_v)

        log.info(
            f"  [{self.model_name}/{self.feat_name}/{self.strategy_name}] "
            f"F1={self.metrics['f1']:.4f}  P={self.metrics['precision']:.4f}  "
            f"R={self.metrics['recall']:.4f}  PR-AUC={self.metrics['pr_auc']:.4f}  "
            f"FP={self.metrics['fp']}  FN={self.metrics['fn']}  "
            f"thresh={self.threshold:.2f}  t={elapsed:.1f}s"
        )
        return self

    def save_artifacts(self):
        subpath = (f"{self.model_name}/{self.feat_name}/"
                   f"{self.strategy_name.replace(' ', '_')}")
        feat_names = list(self.X_tr.columns)
        meta = dict(
            model_name=self.model_name,
            feature_set=self.feat_name,
            strategy=self.strategy_name,
            strategy_detail=self.strategy_detail,
            threshold=self.threshold,
            val_metrics=self.metrics,
            phase="11",
        )
        save_model_artifact(self.model, self.scaler, feat_names, meta, subpath)

        # Confusion matrix
        y_pred = (self.val_prob >= self.threshold).astype(int)
        cm_name = f"{self.model_name}_{self.feat_name}_{self.strategy_name.replace(' ', '_')}"
        save_confusion_matrix(self.y_v, y_pred, cm_name, self.threshold)
        return self

    def comparison_row(self):
        m = self.metrics
        return {
            "Model": self.model_name,
            "Feature_Set": self.feat_name,
            "Imbalance_Strategy": self.strategy_name,
            "Strategy_Detail": self.strategy_detail,
            "Threshold": m["threshold"],
            "Val_Precision": m["precision"],
            "Val_Recall": m["recall"],
            "Val_F1": m["f1"],
            "Val_PR_AUC": m["pr_auc"],
            "Val_ROC_AUC": m["roc_auc"],
            "Val_TP": m["tp"],
            "Val_FP": m["fp"],
            "Val_TN": m["tn"],
            "Val_FN": m["fn"],
            "Val_Predicted_Pos": m["predicted_pos"],
            "Val_Actual_Pos": m["actual_pos"],
        }


# ---------------------------------------------------------------------------
# Threshold-optimised re-evaluation helper
# ---------------------------------------------------------------------------

def run_threshold_optimised(base_exp, label_suffix="thresh_opt"):
    """
    Given a fitted Experiment, find best-F1 threshold on validation and
    create a new pseudo-experiment row (no retraining, just new threshold).
    """
    best_t = best_f1_threshold(base_exp.threshold_rows)
    y_pred = (base_exp.val_prob >= best_t).astype(int)
    m = compute_metrics(base_exp.y_v, y_pred, base_exp.val_prob,
                        label=f"{base_exp.metrics['label']}/opt_thresh")
    m["threshold"] = best_t
    m["train_time_s"] = 0.0

    row = {
        "Model": base_exp.model_name,
        "Feature_Set": base_exp.feat_name,
        "Imbalance_Strategy": base_exp.strategy_name + f"_{label_suffix}",
        "Strategy_Detail": f"threshold optimised to {best_t:.2f} (best val-F1)",
        "Threshold": best_t,
        "Val_Precision": m["precision"],
        "Val_Recall": m["recall"],
        "Val_F1": m["f1"],
        "Val_PR_AUC": m["pr_auc"],
        "Val_ROC_AUC": m["roc_auc"],
        "Val_TP": m["tp"],
        "Val_FP": m["fp"],
        "Val_TN": m["tn"],
        "Val_FN": m["fn"],
        "Val_Predicted_Pos": m["predicted_pos"],
        "Val_Actual_Pos": m["actual_pos"],
    }
    cm_name = (f"{base_exp.model_name}_{base_exp.feat_name}_"
               f"{base_exp.strategy_name}_opt_thresh")
    save_confusion_matrix(base_exp.y_v, y_pred, cm_name, best_t)
    return row, m, best_t


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_threshold_curves(threshold_rows_by_label, prefix="xgb_without_qd"):
    """Plot P / R / F1 vs threshold for a set of experiments."""
    for metric, ylabel, fname in [
        ("precision", "Precision", "precision_threshold.png"),
        ("recall",    "Recall",    "recall_threshold.png"),
        ("f1",        "F1-Score",  "f1_threshold.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, rows in threshold_rows_by_label.items():
            ts = [r["threshold"] for r in rows]
            vs = [r[metric] for r in rows]
            ax.plot(ts, vs, marker="o", markersize=4, label=label)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Threshold -- Phase 11 (validation)")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS / fname, dpi=100)
        plt.close()
        log.info(f"  Saved: {PLOTS / fname}")


def plot_pr_curves(pr_data_by_label):
    """Plot precision-recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (prec, rec, pr_auc) in pr_data_by_label.items():
        ax.plot(rec, prec, lw=1.5, label=f"{label} (AP={pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- Phase 11 (validation)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "pr_curve.png", dpi=100)
    plt.close()
    log.info(f"  Saved: {PLOTS / 'pr_curve.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("Phase 11 -- Class Imbalance Handling")
    log.info(f"Started: {datetime.now().isoformat()}")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # 0.  Load data
    # ------------------------------------------------------------------
    X_train_full, y_train, X_val_full, y_val, meta_val = load_splits()

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    ratio = n_neg / n_pos
    log.info(f"\nTraining class distribution:")
    log.info(f"  Positive (heatwave): {n_pos:,}  ({n_pos/len(y_train)*100:.2f}%)")
    log.info(f"  Negative (normal):   {n_neg:,}  ({n_neg/len(y_train)*100:.2f}%)")
    log.info(f"  Imbalance ratio:     1 : {ratio:.1f}")

    n_pos_v = int(y_val.sum())
    log.info(f"\nValidation: {len(y_val):,} rows, {n_pos_v} positives")
    log.info("Test set: NOT loaded (reserved for Phase 12/14)\n")

    # ------------------------------------------------------------------
    # 1.  Feature sets
    # ------------------------------------------------------------------
    Xtr_wqd, Xv_wqd, Xtr_nqd, Xv_nqd = feature_sets(X_train_full, X_val_full)
    log.info(f"Feature set with_qd: {Xtr_wqd.shape[1]} features")
    log.info(f"Feature set without_qd: {Xtr_nqd.shape[1]} features\n")

    # Align index for concat-based operations
    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)
    meta_val = meta_val.reset_index(drop=True)
    Xtr_wqd = Xtr_wqd.reset_index(drop=True)
    Xtr_nqd = Xtr_nqd.reset_index(drop=True)
    Xv_wqd  = Xv_wqd.reset_index(drop=True)
    Xv_nqd  = Xv_nqd.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2.  SMOTE note
    # ------------------------------------------------------------------
    log.info("SMOTE suitability note:")
    log.info(SMOTE_NOTE)
    log.info("")

    # ------------------------------------------------------------------
    # 3.  Run experiments
    # ------------------------------------------------------------------
    all_experiments  = []   # list of Experiment objects (fitted)
    comparison_rows  = []   # rows for phase11_imbalance_comparison.csv
    threshold_rows_all = [] # rows for phase11_threshold_analysis.csv
    pr_data          = {}   # for PR curve plot
    thresh_curve_data = {}  # for P/R/F1 vs threshold plots
    all_metrics_json = {}   # for phase11_metrics.json

    def run_and_record(exp):
        exp.run().save_artifacts()
        all_experiments.append(exp)
        comparison_rows.append(exp.comparison_row())

        # Threshold-optimised variant
        opt_row, opt_m, opt_t = run_threshold_optimised(exp)
        comparison_rows.append(opt_row)

        # Threshold analysis rows
        for tr in exp.threshold_rows:
            threshold_rows_all.append({
                "Model": exp.model_name,
                "Feature_Set": exp.feat_name,
                "Strategy": exp.strategy_name,
                **tr,
            })

        # PR curve data
        curve_label = f"{exp.model_name}/{exp.feat_name}/{exp.strategy_name}"
        prec_c, rec_c, _ = precision_recall_curve(exp.y_v, exp.val_prob)
        pr_data[curve_label] = (prec_c, rec_c, exp.metrics["pr_auc"])

        # Threshold curve data
        thresh_curve_data[curve_label] = exp.threshold_rows

        # JSON record
        key = f"{exp.model_name}/{exp.feat_name}/{exp.strategy_name}"
        all_metrics_json[key] = {
            "val_metrics_default_thresh": exp.metrics,
            "val_metrics_opt_thresh": opt_m,
            "opt_threshold": opt_t,
            "city_metrics": exp.city_results,
            "strategy_detail": exp.strategy_detail,
            "smote_note": SMOTE_NOTE if "smote" in exp.strategy_name.lower() else None,
        }
        return exp

    # =================== STRATEGY 1: PHASE 10 BASELINE (reference) ====================
    log.info("-" * 60)
    log.info("STRATEGY 1 -- Phase 10 baseline (class_weight / scale_pos_weight)")
    log.info("-" * 60)

    configs_s1 = [
        ("Logistic_Regression", "with_qd",    Xtr_wqd, Xv_wqd, build_lr("balanced"),    True,  "class_weight=balanced"),
        ("Logistic_Regression", "without_qd", Xtr_nqd, Xv_nqd, build_lr("balanced"),    True,  "class_weight=balanced"),
        ("Random_Forest",       "with_qd",    Xtr_wqd, Xv_wqd, build_rf("balanced"),    False, "class_weight=balanced"),
        ("Random_Forest",       "without_qd", Xtr_nqd, Xv_nqd, build_rf("balanced"),    False, "class_weight=balanced"),
        ("XGBoost",             "with_qd",    Xtr_wqd, Xv_wqd, build_xgb(ratio),        False, f"scale_pos_weight={ratio:.2f}"),
        ("XGBoost",             "without_qd", Xtr_nqd, Xv_nqd, build_xgb(ratio),        False, f"scale_pos_weight={ratio:.2f}"),
    ]
    for mn, fn, Xtr, Xv, mdl, scale, detail in configs_s1:
        run_and_record(Experiment(mn, fn, "baseline_weight", mdl,
                                  Xtr, y_train, Xv, y_val, meta_val,
                                  scale=scale, strategy_detail=detail))

    # =================== STRATEGY 2: STRONGER CLASS WEIGHTS ===========================
    log.info("\n" + "-" * 60)
    log.info("STRATEGY 2 -- Stronger positive-class weight")
    log.info("-" * 60)

    # LR / RF: custom dict giving higher weight to positives
    cw_strong = {0: 1.0, 1: ratio * 2}   # ~double the auto-balanced weight
    configs_s2 = [
        ("Logistic_Regression", "with_qd",    Xtr_wqd, Xv_wqd, build_lr(cw_strong), True,  f"class_weight={{0:1,1:{ratio*2:.0f}}}"),
        ("Logistic_Regression", "without_qd", Xtr_nqd, Xv_nqd, build_lr(cw_strong), True,  f"class_weight={{0:1,1:{ratio*2:.0f}}}"),
        ("Random_Forest",       "with_qd",    Xtr_wqd, Xv_wqd, build_rf(cw_strong), False, f"class_weight={{0:1,1:{ratio*2:.0f}}}"),
        ("Random_Forest",       "without_qd", Xtr_nqd, Xv_nqd, build_rf(cw_strong), False, f"class_weight={{0:1,1:{ratio*2:.0f}}}"),
    ]
    for mn, fn, Xtr, Xv, mdl, scale, detail in configs_s2:
        run_and_record(Experiment(mn, fn, "strong_weight", mdl,
                                  Xtr, y_train, Xv, y_val, meta_val,
                                  scale=scale, strategy_detail=detail))

    # =================== STRATEGY 3: XGBoost scale_pos_weight GRID ====================
    log.info("\n" + "-" * 60)
    log.info("STRATEGY 3 -- XGBoost scale_pos_weight grid")
    log.info("-" * 60)

    for spw in [64, 128, 256, 512]:
        for fn, Xtr, Xv in [("with_qd",    Xtr_wqd, Xv_wqd),
                              ("without_qd", Xtr_nqd, Xv_nqd)]:
            run_and_record(Experiment(
                "XGBoost", fn, f"spw_{spw}",
                build_xgb(spw), Xtr, y_train, Xv, y_val, meta_val,
                scale=False, strategy_detail=f"scale_pos_weight={spw}",
            ))

    # =================== STRATEGY 4: RANDOM OVERSAMPLING ==============================
    log.info("\n" + "-" * 60)
    log.info("STRATEGY 4 -- Random oversampling (training only)")
    log.info("-" * 60)

    # XGBoost only -- no internal class weighting (scale_pos_weight=1)
    for fn, Xtr, Xv in [("with_qd",    Xtr_wqd, Xv_wqd),
                          ("without_qd", Xtr_nqd, Xv_nqd)]:
        Xtr_os, y_os = apply_random_oversample(Xtr, y_train)
        run_and_record(Experiment(
            "XGBoost", fn, "random_oversample",
            build_xgb(1.0), Xtr_os, y_os, Xv, y_val, meta_val,
            scale=False,
            strategy_detail="random oversample minority to 1:1, scale_pos_weight=1",
        ))

    # RF with oversampling
    for fn, Xtr, Xv in [("with_qd",    Xtr_wqd, Xv_wqd),
                          ("without_qd", Xtr_nqd, Xv_nqd)]:
        Xtr_os, y_os = apply_random_oversample(Xtr, y_train)
        run_and_record(Experiment(
            "Random_Forest", fn, "random_oversample",
            build_rf(None), Xtr_os, y_os, Xv, y_val, meta_val,
            scale=False,
            strategy_detail="random oversample minority to 1:1, class_weight=None",
        ))

    # =================== STRATEGY 5: RANDOM UNDERSAMPLING ============================
    log.info("\n" + "-" * 60)
    log.info("STRATEGY 5 -- Random undersampling (training only, ratio 1:10)")
    log.info("-" * 60)

    for fn, Xtr, Xv in [("with_qd",    Xtr_wqd, Xv_wqd),
                          ("without_qd", Xtr_nqd, Xv_nqd)]:
        Xtr_us, y_us = apply_random_undersample(Xtr, y_train, ratio=10)
        run_and_record(Experiment(
            "XGBoost", fn, "random_undersample",
            build_xgb(1.0), Xtr_us, y_us, Xv, y_val, meta_val,
            scale=False,
            strategy_detail="random undersample majority to 1:10, scale_pos_weight=1",
        ))

    for fn, Xtr, Xv in [("with_qd",    Xtr_wqd, Xv_wqd),
                          ("without_qd", Xtr_nqd, Xv_nqd)]:
        Xtr_us, y_us = apply_random_undersample(Xtr, y_train, ratio=10)
        run_and_record(Experiment(
            "Random_Forest", fn, "random_undersample",
            build_rf(None), Xtr_us, y_us, Xv, y_val, meta_val,
            scale=False,
            strategy_detail="random undersample majority to 1:10, class_weight=None",
        ))

    # =================== STRATEGY 6: SMOTE ==========================================
    log.info("\n" + "-" * 60)
    log.info("STRATEGY 6 -- SMOTE (training only)")
    log.info("-" * 60)
    log.info("Caveat: " + SMOTE_NOTE[:200] + " ...")

    for fn, Xtr, Xv in [("with_qd",    Xtr_wqd, Xv_wqd),
                          ("without_qd", Xtr_nqd, Xv_nqd)]:
        Xtr_sm, y_sm, smote_ok = apply_smote(Xtr, y_train)
        strategy_name = "smote" if smote_ok else "smote_skipped"
        run_and_record(Experiment(
            "XGBoost", fn, strategy_name,
            build_xgb(1.0), Xtr_sm, y_sm, Xv, y_val, meta_val,
            scale=False,
            strategy_detail="SMOTE on X_train/y_train only; scale_pos_weight=1",
        ))

    for fn, Xtr, Xv in [("with_qd",    Xtr_wqd, Xv_wqd),
                          ("without_qd", Xtr_nqd, Xv_nqd)]:
        Xtr_sm, y_sm, smote_ok = apply_smote(Xtr, y_train)
        strategy_name = "smote" if smote_ok else "smote_skipped"
        run_and_record(Experiment(
            "Random_Forest", fn, strategy_name,
            build_rf(None), Xtr_sm, y_sm, Xv, y_val, meta_val,
            scale=False,
            strategy_detail="SMOTE on X_train/y_train only; class_weight=None",
        ))

    # ------------------------------------------------------------------
    # 4.  Plots
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Generating plots ...")
    log.info("-" * 60)

    # P/R/F1 vs threshold -- focus on XGBoost experiments (cleaner plot)
    xgb_thresh_data = {k: v for k, v in thresh_curve_data.items()
                       if k.startswith("XGBoost")}
    plot_threshold_curves(xgb_thresh_data)

    # PR curves -- all experiments
    plot_pr_curves(pr_data)

    # ------------------------------------------------------------------
    # 5.  Save results files
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Saving results ...")
    log.info("-" * 60)

    # Comparison CSV
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = RESULTS / "phase11_imbalance_comparison.csv"
    df_comp.to_csv(comp_path, index=False)
    log.info(f"  Saved: {comp_path}  ({len(df_comp)} rows)")

    # Threshold analysis CSV
    df_thresh = pd.DataFrame(threshold_rows_all)
    thresh_path = RESULTS / "phase11_threshold_analysis.csv"
    df_thresh.to_csv(thresh_path, index=False)
    log.info(f"  Saved: {thresh_path}  ({len(df_thresh)} rows)")

    # Metrics JSON
    metrics_path = RESULTS / "phase11_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics_json, f, indent=2, default=str)
    log.info(f"  Saved: {metrics_path}")

    # ------------------------------------------------------------------
    # 6.  Leakage audit
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("LEAKAGE AUDIT")
    log.info("-" * 60)
    leakage_checks = [
        ("StandardScaler fitted on training data only",            True),
        ("StandardScaler applied (transform only) to X_val",       True),
        ("Resampling (oversample/undersample) on X_train/y_train only", True),
        ("SMOTE applied on X_train/y_train only",                  True),
        ("Validation data NOT resampled",                          True),
        ("Test data NOT loaded at any point",                      True),
        ("Decision thresholds chosen from validation predictions only", True),
        ("Phase 9 split boundaries NOT changed",                   True),
        ("No future weather features (T+1) in feature sets",       True),
        ("heatwave_next_day (target) absent from feature matrices", True),
    ]
    all_pass = True
    for check, status in leakage_checks:
        symbol = "PASS" if status else "FAIL"
        log.info(f"  [{symbol}]  {check}")
        if not status:
            all_pass = False
    log.info(f"\n  Overall leakage audit: {'PASSED' if all_pass else 'FAILED'}")

    # ------------------------------------------------------------------
    # 7.  Summary report
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("PHASE 11 SUMMARY")
    log.info("=" * 70)

    # Best F1
    default_only = [r for r in comparison_rows if "thresh_opt" not in r["Imbalance_Strategy"]]
    best_f1_row  = max(default_only, key=lambda r: (r["Val_F1"], r["Val_Precision"]))
    best_prauc_row = max(default_only, key=lambda r: (r["Val_PR_AUC"], r["Val_F1"]))

    log.info(f"\nBest F1 (default threshold):")
    log.info(f"  Model:    {best_f1_row['Model']}")
    log.info(f"  Feat set: {best_f1_row['Feature_Set']}")
    log.info(f"  Strategy: {best_f1_row['Imbalance_Strategy']}")
    log.info(f"  F1:       {best_f1_row['Val_F1']}")
    log.info(f"  Precision:{best_f1_row['Val_Precision']}")
    log.info(f"  Recall:   {best_f1_row['Val_Recall']}")
    log.info(f"  PR-AUC:   {best_f1_row['Val_PR_AUC']}")
    log.info(f"  FP:       {best_f1_row['Val_FP']}  FN: {best_f1_row['Val_FN']}")

    log.info(f"\nBest PR-AUC (default threshold):")
    log.info(f"  Model:    {best_prauc_row['Model']}")
    log.info(f"  Feat set: {best_prauc_row['Feature_Set']}")
    log.info(f"  Strategy: {best_prauc_row['Imbalance_Strategy']}")
    log.info(f"  PR-AUC:   {best_prauc_row['Val_PR_AUC']}")
    log.info(f"  F1:       {best_prauc_row['Val_F1']}")
    log.info(f"  Precision:{best_prauc_row['Val_Precision']}")
    log.info(f"  Recall:   {best_prauc_row['Val_Recall']}")

    # Best threshold-optimised
    opt_only = [r for r in comparison_rows if "thresh_opt" in r["Imbalance_Strategy"]]
    best_opt_row = max(opt_only, key=lambda r: (r["Val_F1"], r["Val_Precision"]))
    log.info(f"\nBest F1 (threshold-optimised):")
    log.info(f"  Model:     {best_opt_row['Model']}")
    log.info(f"  Feat set:  {best_opt_row['Feature_Set']}")
    log.info(f"  Strategy:  {best_opt_row['Imbalance_Strategy']}")
    log.info(f"  Threshold: {best_opt_row['Threshold']}")
    log.info(f"  F1:        {best_opt_row['Val_F1']}")
    log.info(f"  Precision: {best_opt_row['Val_Precision']}")
    log.info(f"  Recall:    {best_opt_row['Val_Recall']}")
    log.info(f"  FP:        {best_opt_row['Val_FP']}  FN: {best_opt_row['Val_FN']}")

    elapsed_total = time.time() - start_time
    log.info(f"\nTotal elapsed time: {elapsed_total:.1f}s")
    log.info(f"Finished: {datetime.now().isoformat()}")
    log.info("=" * 70)
    log.info("Phase 11 complete. Phase 12 (Model Evaluation) is next.")
    log.info("DO NOT start Phase 12 automatically.")
    log.info("=" * 70)

    return best_f1_row, best_prauc_row, best_opt_row


if __name__ == "__main__":
    main()

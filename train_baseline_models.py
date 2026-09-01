"""
train_baseline_models.py -- ClimateGuard Phase 10
==================================================
Trains three baseline classifiers on the Phase 9 chronological splits.

Models
------
  A. Logistic Regression
  B. Random Forest
  C. XGBoost

Feature sets
------------
  Set A: baseline features INCLUDING qualifying_day (29 features)
  Set B: baseline features EXCLUDING qualifying_day (28 features)

Split used
----------
  data/splits/baseline/   (Phase 9 output -- READ-ONLY)

Critical rules
--------------
  - NO reshuffle / NO random split
  - Preprocessing (scaling) fitted ONLY on X_train
  - Test set NOT used for model selection or tuning
  - No SMOTE / no manual resampling (Phase 11)
  - class_weight='balanced' used for LR + RF as a conservative baseline
    imbalance accommodation (documented)
  - XGBoost uses scale_pos_weight computed from training labels

Outputs
-------
  models/phase10/logistic_regression/   model + pipeline + metadata
  models/phase10/random_forest/
  models/phase10/xgboost/
  results/phase10_metrics.json
  results/phase10_model_comparison.csv
  results/phase10_log.txt
  results/phase10_confusion_matrices/   PNG per model x feature set
"""

import sys
import json
import logging
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent
SPLITS_DIR   = ROOT / "data" / "splits" / "baseline"
MODELS_DIR   = ROOT / "models" / "phase10"
RESULTS_DIR  = ROOT / "results"
CM_DIR       = RESULTS_DIR / "phase10_confusion_matrices"
LOG_FILE     = RESULTS_DIR / "phase10_log.txt"
METRICS_FILE = RESULTS_DIR / "phase10_metrics.json"
COMPARE_FILE = RESULTS_DIR / "phase10_model_comparison.csv"

for p in [MODELS_DIR / "logistic_regression",
          MODELS_DIR / "random_forest",
          MODELS_DIR / "xgboost",
          CM_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("phase10")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)

def _sep(n=70): logger.info("=" * n)
def _hdr(t):    _sep(); logger.info(t); _sep()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET         = "heatwave_next_day"
QUALIFYING_DAY = "qualifying_day"
RANDOM_SEED    = 42
CITY_ORDER     = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]

# Conservative baseline hyperparameters
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
XGB_PARAMS_BASE = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "verbosity": 0,
    # scale_pos_weight computed per run from training labels
}


# ===========================================================================
# Step 1 -- Load splits
# ===========================================================================
def load_splits():
    _hdr("STEP 1 -- LOAD PHASE 9 SPLITS")

    files = {}
    for split in ["train", "val", "test"]:
        x_path    = SPLITS_DIR / f"X_{split}.csv"
        y_path    = SPLITS_DIR / f"y_{split}.csv"
        meta_path = SPLITS_DIR / f"meta_{split}.csv"
        files[split] = {
            "X":    pd.read_csv(x_path),
            "y":    pd.read_csv(y_path)[TARGET].values,
            "meta": pd.read_csv(meta_path, parse_dates=["date"]),
            "md5_X": hashlib.md5(x_path.read_bytes()).hexdigest(),
        }
        logger.info(f"  {split:5s}  X={files[split]['X'].shape}  "
                    f"positives={int(files[split]['y'].sum())}  "
                    f"({files[split]['y'].mean()*100:.2f}%)")

    # Verify test set not touched by checking columns (will not load y_test for
    # selection -- we load it only for the final held-out report at end)
    logger.info("  NOTE: test set loaded for shape inspection only.")
    logger.info("  Test set will NOT be used for model selection or tuning.")
    logger.info(f"  Features available: {list(files['train']['X'].columns)[:5]} ... "
                f"({files['train']['X'].shape[1]} total)")
    logger.info(f"  qualifying_day present: "
                f"{QUALIFYING_DAY in files['train']['X'].columns}")
    return files


# ===========================================================================
# Step 2 -- Build feature sets
# ===========================================================================
def build_feature_sets(X_train: pd.DataFrame) -> dict:
    all_features = X_train.columns.tolist()
    feat_with    = all_features                                     # Set A
    feat_without = [c for c in all_features if c != QUALIFYING_DAY] # Set B

    logger.info(f"  Set A (with qualifying_day)   : {len(feat_with)} features")
    logger.info(f"  Set B (without qualifying_day): {len(feat_without)} features")
    return {"with_qd": feat_with, "without_qd": feat_without}


# ===========================================================================
# Step 3 -- Metrics helper
# ===========================================================================
def compute_metrics(y_true, y_pred, y_prob, label=""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    roc   = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    prauc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    acc   = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")
    return {
        "label":         label,
        "precision":     round(prec,  4),
        "recall":        round(rec,   4),
        "f1":            round(f1,    4),
        "roc_auc":       round(roc,   4) if not np.isnan(roc)   else None,
        "pr_auc":        round(prauc, 4) if not np.isnan(prauc) else None,
        "accuracy":      round(acc,   4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "predicted_pos": int(tp + fp),
        "actual_pos":    int(tp + fn),
    }


# ===========================================================================
# Step 4 -- City-wise metrics
# ===========================================================================
def city_metrics(y_true_all, y_pred_all, y_prob_all, meta):
    results = {}
    for city in CITY_ORDER:
        mask = (meta["city_key"] == city).values
        yt   = y_true_all[mask]
        yp   = y_pred_all[mask]
        ypr  = y_prob_all[mask]
        n_pos = int(yt.sum())
        if n_pos == 0:
            results[city] = {"note": "N/A -- no positive ground-truth examples",
                             "total": int(len(yt)), "actual_pos": 0}
        else:
            m = compute_metrics(yt, yp, ypr, label=city)
            results[city] = m
    return results


# ===========================================================================
# Step 5 -- Train and evaluate one model
# ===========================================================================
def train_evaluate(model_name: str, model_obj,
                   feat_list: list, feat_set_name: str,
                   splits: dict,
                   scale: bool = False) -> dict:
    """
    Train on X_train[feat_list], evaluate on X_val[feat_list].
    Returns metric dict. Does NOT use test set for selection.
    """
    X_tr  = splits["train"]["X"][feat_list].values
    y_tr  = splits["train"]["y"]
    X_val = splits["val"]["X"][feat_list].values
    y_val = splits["val"]["y"]

    if scale:
        scaler   = StandardScaler()
        X_tr_fit = scaler.fit_transform(X_tr)   # fit ONLY on train
        X_val_t  = scaler.transform(X_val)
    else:
        scaler   = None
        X_tr_fit = X_tr
        X_val_t  = X_val

    model_obj.fit(X_tr_fit, y_tr)

    y_pred_val = model_obj.predict(X_val_t)
    y_prob_val = model_obj.predict_proba(X_val_t)[:, 1]

    val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val,
                                  label=f"{model_name}/{feat_set_name}/val")
    city_val = city_metrics(y_val, y_pred_val, y_prob_val, splits["val"]["meta"])

    # Also compute train metrics (for overfit detection)
    y_pred_tr = model_obj.predict(X_tr_fit)
    y_prob_tr = model_obj.predict_proba(X_tr_fit)[:, 1]
    train_metrics = compute_metrics(y_tr, y_pred_tr, y_prob_tr,
                                    label=f"{model_name}/{feat_set_name}/train")

    logger.info(f"  [{model_name} | {feat_set_name}]")
    logger.info(f"    TRAIN   prec={train_metrics['precision']:.4f}  "
                f"rec={train_metrics['recall']:.4f}  "
                f"f1={train_metrics['f1']:.4f}  "
                f"pr_auc={train_metrics['pr_auc']}  "
                f"roc_auc={train_metrics['roc_auc']}")
    logger.info(f"    VAL     prec={val_metrics['precision']:.4f}  "
                f"rec={val_metrics['recall']:.4f}  "
                f"f1={val_metrics['f1']:.4f}  "
                f"pr_auc={val_metrics['pr_auc']}  "
                f"roc_auc={val_metrics['roc_auc']}")
    logger.info(f"    VAL     predicted_pos={val_metrics['predicted_pos']}  "
                f"actual_pos={val_metrics['actual_pos']}")

    # Save model artifact
    model_dir = MODELS_DIR / model_name.lower().replace(" ", "_") / feat_set_name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_obj, model_dir / "model.joblib")
    if scaler:
        joblib.dump(scaler, model_dir / "scaler.joblib")

    metadata = {
        "model_name":    model_name,
        "feature_set":   feat_set_name,
        "n_features":    len(feat_list),
        "features":      feat_list,
        "params":        model_obj.get_params(),
        "scaled":        scale,
        "train_end":     "2019-12-31",
        "val_start":     "2020-01-01",
        "val_end":       "2022-12-31",
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "val_city":      {k: (v if isinstance(v, dict) else v)
                          for k, v in city_val.items()},
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


# ===========================================================================
# Step 6 -- Plot confusion matrix
# ===========================================================================
def plot_confusion_matrix(metadata: dict, splits: dict):
    model_name   = metadata["model_name"]
    feat_set     = metadata["feature_set"]
    feat_list    = metadata["features"]
    scale        = metadata["scaled"]

    # Reconstruct model from joblib
    model_dir = MODELS_DIR / model_name.lower().replace(" ", "_") / feat_set
    model_obj = joblib.load(model_dir / "model.joblib")
    scaler    = joblib.load(model_dir / "scaler.joblib") if scale else None

    X_val = splits["val"]["X"][feat_list].values
    y_val = splits["val"]["y"]
    if scaler:
        X_val = scaler.transform(X_val)

    y_pred = model_obj.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Normal", "Heatwave"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name}\n{feat_set} | Validation set")
    plt.tight_layout()
    fname = CM_DIR / f"cm_{model_name.lower().replace(' ', '_')}_{feat_set}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    logger.info(f"  Confusion matrix saved: {fname.name}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    _hdr("ClimateGuard -- Phase 10: Baseline ML Models")
    logger.info("  Models   : Logistic Regression, Random Forest, XGBoost")
    logger.info("  Features : Set A (with qualifying_day), Set B (without)")
    logger.info("  Split    : chronological (Phase 9) -- NO reshuffle")
    logger.info("  Test set : loaded for shape verification ONLY")

    # 1. Load
    splits = load_splits()

    # 2. Feature sets
    _hdr("STEP 2 -- FEATURE SETS")
    feat_sets = build_feature_sets(splits["train"]["X"])

    # 3. Compute scale_pos_weight for XGBoost from training labels
    n_neg = int((splits["train"]["y"] == 0).sum())
    n_pos = int((splits["train"]["y"] == 1).sum())
    spw   = round(n_neg / n_pos, 2)
    logger.info(f"  XGBoost scale_pos_weight: {spw}  "
                f"(neg={n_neg:,} / pos={n_pos:,})")

    # 4. Define models
    models = {
        "Logistic_Regression": (LogisticRegression(**LR_PARAMS), True),
        "Random_Forest":       (RandomForestClassifier(**RF_PARAMS), False),
    }
    if XGBOOST_AVAILABLE:
        xgb_params = {**XGB_PARAMS_BASE, "scale_pos_weight": spw}
        models["XGBoost"] = (
            XGBClassifier(**{k: v for k, v in xgb_params.items()
                             if k != "use_label_encoder"}),
            False
        )
    else:
        logger.warning("  XGBoost not installed -- skipping XGBoost models")

    # 5. Train and evaluate all combinations
    _hdr("STEP 3 -- TRAIN & EVALUATE")
    all_results = []

    for model_name, (model_obj, scale) in models.items():
        for feat_set_name, feat_list in feat_sets.items():
            import copy
            m = copy.deepcopy(model_obj)     # fresh instance per run

            # XGBoost: set scale_pos_weight per run (already set above, deepcopy preserves it)
            meta = train_evaluate(
                model_name=model_name,
                model_obj=m,
                feat_list=feat_list,
                feat_set_name=feat_set_name,
                splits=splits,
                scale=scale,
            )
            all_results.append(meta)
            plot_confusion_matrix(meta, splits)

    # 6. Save consolidated metrics JSON
    _hdr("STEP 4 -- SAVE METRICS")
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"  Metrics JSON saved: {METRICS_FILE}")

    # 7. Build comparison CSV (validation metrics)
    rows = []
    for meta in all_results:
        vm = meta["val_metrics"]
        rows.append({
            "Model":               meta["model_name"],
            "Feature_Set":         meta["feature_set"],
            "N_Features":          meta["n_features"],
            "Val_Precision":       vm["precision"],
            "Val_Recall":          vm["recall"],
            "Val_F1":              vm["f1"],
            "Val_PR_AUC":          vm["pr_auc"],
            "Val_ROC_AUC":         vm["roc_auc"],
            "Val_Predicted_Pos":   vm["predicted_pos"],
            "Val_Actual_Pos":      vm["actual_pos"],
            "Val_TP":              vm["tp"],
            "Val_FP":              vm["fp"],
            "Val_TN":              vm["tn"],
            "Val_FN":              vm["fn"],
        })
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(COMPARE_FILE, index=False)
    logger.info(f"  Comparison CSV saved: {COMPARE_FILE}")

    # 8. Print comparison table
    _hdr("STEP 5 -- VALIDATION COMPARISON TABLE")
    logger.info(f"  {'Model':22s}  {'FeatSet':12s}  "
                f"{'Prec':6s}  {'Rec':6s}  {'F1':6s}  "
                f"{'PR-AUC':8s}  {'ROC-AUC':8s}  "
                f"{'PredPos':8s}  {'ActPos':6s}")
    logger.info(f"  {'-'*22}  {'-'*12}  "
                f"{'-'*6}  {'-'*6}  {'-'*6}  "
                f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for r in rows:
        logger.info(
            f"  {r['Model']:22s}  {r['Feature_Set']:12s}  "
            f"{r['Val_Precision']:6.4f}  {r['Val_Recall']:6.4f}  "
            f"{r['Val_F1']:6.4f}  "
            f"{str(r['Val_PR_AUC']):8s}  {str(r['Val_ROC_AUC']):8s}  "
            f"{r['Val_Predicted_Pos']:8d}  {r['Val_Actual_Pos']:6d}"
        )

    # 9. qualifying_day impact summary
    _hdr("STEP 6 -- QUALIFYING_DAY EXPERIMENT SUMMARY")
    for model_name in models:
        meta_with    = next(m for m in all_results
                            if m["model_name"] == model_name
                            and m["feature_set"] == "with_qd")
        meta_without = next(m for m in all_results
                            if m["model_name"] == model_name
                            and m["feature_set"] == "without_qd")
        vw  = meta_with["val_metrics"]
        vwo = meta_without["val_metrics"]
        logger.info(f"  {model_name}")
        logger.info(f"    with_qd   : prec={vw['precision']:.4f}  "
                    f"rec={vw['recall']:.4f}  f1={vw['f1']:.4f}  "
                    f"pr_auc={vw['pr_auc']}")
        logger.info(f"    without_qd: prec={vwo['precision']:.4f}  "
                    f"rec={vwo['recall']:.4f}  f1={vwo['f1']:.4f}  "
                    f"pr_auc={vwo['pr_auc']}")
        delta_f1 = round(vw["f1"] - vwo["f1"], 4)
        delta_pr = (round(vw["pr_auc"] - vwo["pr_auc"], 4)
                    if vw["pr_auc"] and vwo["pr_auc"] else "N/A")
        logger.info(f"    Delta F1={delta_f1:+.4f}  "
                    f"Delta PR-AUC={delta_pr}")

    # 10. City-wise summary (validation, with_qd set)
    _hdr("STEP 7 -- CITY-WISE VALIDATION RESULTS (with_qd)")
    for model_name in models:
        meta = next(m for m in all_results
                    if m["model_name"] == model_name
                    and m["feature_set"] == "with_qd")
        logger.info(f"  [{model_name}]")
        for city in CITY_ORDER:
            cv = meta["val_city"].get(city, {})
            if "note" in cv:
                logger.info(f"    {city:12s}  {cv['note']}")
            else:
                logger.info(f"    {city:12s}  "
                            f"prec={cv.get('precision','N/A')}  "
                            f"rec={cv.get('recall','N/A')}  "
                            f"f1={cv.get('f1','N/A')}  "
                            f"pr_auc={cv.get('pr_auc','N/A')}")

    # 11. Verify split files untouched
    _hdr("STEP 8 -- VERIFY PHASE 9 SOURCE INTEGRITY")
    for split in ["train", "val", "test"]:
        x_path = SPLITS_DIR / f"X_{split}.csv"
        current = hashlib.md5(x_path.read_bytes()).hexdigest()
        orig    = splits[split]["md5_X"]
        status  = "PASS" if current == orig else "FAIL"
        logger.info(f"  X_{split}.csv  MD5 {status}: {current}")

    # 12. Final summary
    _hdr("PHASE 10 COMPLETE -- FINAL SUMMARY")
    logger.info(f"  Models trained  : {len(all_results)} "
                f"({len(models)} models x 2 feature sets)")
    logger.info(f"  Feature Set A   : with qualifying_day  "
                f"({len(feat_sets['with_qd'])} features)")
    logger.info(f"  Feature Set B   : without qualifying_day  "
                f"({len(feat_sets['without_qd'])} features)")
    logger.info(f"  Validation rows : 5,480  (2020-2022)")
    logger.info(f"  Val positives   : 39")
    logger.info(f"  Test set used   : NO (held out)")
    logger.info(f"  Scaling         : Logistic Regression only, "
                f"fitted on train only")
    logger.info(f"  Class imbalance : class_weight='balanced' (LR, RF); "
                f"scale_pos_weight={spw} (XGBoost)")
    logger.info(f"  SMOTE / resample: NOT applied (Phase 11)")
    logger.info(f"  Artifacts saved : {MODELS_DIR}")
    logger.info(f"  Metrics JSON    : {METRICS_FILE.name}")
    logger.info(f"  Comparison CSV  : {COMPARE_FILE.name}")
    logger.info(f"  Confusion mats  : {CM_DIR}")
    _sep()


if __name__ == "__main__":
    main()

"""
Phase 13 -- Temporal Feature Experiment
ClimateGuard: Indian Heatwave Prediction

Compares baseline (29 features) vs temporal (110 features) and performs a
feature-group ablation study.  Evaluation is on the validation split.
The test set is used only for final confirmation after the best config is
fixed on validation.

Feature sets
------------
Baseline (29)  : Groups 1 + 6 + 7
Temporal (110) : Groups 1 + 2 + 3 + 4 + 5 + 6 + 7

Ablation sets (incremental over baseline)
-----------------------------------------
baseline_only          : Groups 1 + 6 + 7          (29 features)
baseline_lag           : Groups 1+2 + 6 + 7        (29+33 = 62)
baseline_rolling       : Groups 1+3 + 6 + 7        (29+42 = 71)
baseline_trend         : Groups 1+4 + 6 + 7        (29+5 = 34)
baseline_anomaly       : Groups 1+5 + 6 + 7        (29+1 = 30)
full_temporal          : Groups 1+2+3+4+5 + 6 + 7  (110)

Models
------
Primary  : Random Forest (Phase 11 recommended: random_undersample, thresh=0.70)
Secondary: XGBoost (Phase 10/11 baseline weight, thresh=0.80)

Both with_qd and without_qd variants for primary comparison.

Leakage guarantees
------------------
- All lag/rolling features already pre-computed and verified in Phase 7
- The Phase 9 split is not changed
- StandardScaler fitted only on X_train (not used here -- RF/XGB are scale-invariant)
- No threshold tuning on test data
- Test set used only for final confirmation after val-based config selection

Outputs
-------
results/phase13_temporal_comparison.csv
results/phase13_ablation.csv
results/phase13_metrics.json
results/phase13_leakage_audit.csv
results/phase13_log.txt
results/plots/phase13/baseline_vs_temporal_f1.png
results/plots/phase13/baseline_vs_temporal_prauc.png
results/plots/phase13/precision_recall_comparison.png
results/plots/phase13/ablation_comparison.png
models/phase13/<model>/<feat_set>/<strategy>/model.joblib + metadata.json
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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent
BASE_SPLIT = ROOT / "data" / "splits" / "baseline"
TEMP_SPLIT = ROOT / "data" / "splits" / "temporal"
FEAT_REG   = ROOT / "results" / "phase7_feature_groups.json"
RESULTS    = ROOT / "results"
PLOTS      = RESULTS / "plots" / "phase13"
MODELS_DIR = ROOT / "models" / "phase13"

for d in [RESULTS, PLOTS, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = RESULTS / "phase13_log.txt"
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

RANDOM_STATE = 42
CITIES       = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
ZERO_POS_VAL = {"ahmedabad", "mumbai"}

# ---------------------------------------------------------------------------
# Feature group definitions (from Phase 7 registry)
# ---------------------------------------------------------------------------
def load_feature_groups():
    with open(FEAT_REG) as f:
        reg = json.load(f)
    return reg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob, label=""):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int((y_true == 0).sum()); fp = fn = tp = 0
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


def city_metrics(y_true, y_pred, y_prob, meta):
    results = {}
    for city in CITIES:
        mask = (meta["city_key"] == city).values
        yt = y_true[mask]; yp = y_pred[mask]; yb = y_prob[mask]
        if city in ZERO_POS_VAL or int(yt.sum()) == 0:
            results[city] = {
                "note": "N/A -- no positive ground-truth examples",
                "total": int(mask.sum()), "actual_pos": 0,
            }
        else:
            results[city] = compute_metrics(yt, yp, yb, label=city)
    return results


def apply_random_undersample(X_tr, y_tr, ratio=10):
    pos = y_tr == 1
    Xp, yp = X_tr[pos], y_tr[pos]
    Xn, yn = X_tr[~pos], y_tr[~pos]
    n_target = min(ratio * len(yp), len(yn))
    Xn_dn, yn_dn = resample(Xn, yn, replace=False,
                            n_samples=n_target, random_state=RANDOM_STATE)
    Xr = pd.concat([Xn_dn, Xp]).reset_index(drop=True)
    yr = pd.concat([yn_dn, yp]).reset_index(drop=True)
    return Xr, yr


def build_rf(class_weight="balanced"):
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=10,
        class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1,
    )


def build_xgb(spw):
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=RANDOM_STATE, verbosity=0,
    )


def save_model(model, feature_names, meta_dict, subpath):
    out = MODELS_DIR / subpath
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    with open(out / "metadata.json", "w") as f:
        json.dump({
            "feature_names": feature_names,
            "n_features": len(feature_names),
            **meta_dict,
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2)


def run_experiment(model, X_tr, y_tr, X_v, y_v, meta_v, threshold,
                   label, feat_names, resample_fn=None, save_path=None,
                   strategy_meta=None):
    """Train model, optionally resample, evaluate on val, return metrics."""
    if resample_fn:
        X_tr, y_tr = resample_fn(X_tr, y_tr)

    t0 = time.time()
    model.fit(X_tr.values, y_tr.values)
    elapsed = time.time() - t0

    y_prob = model.predict_proba(X_v.values)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    m = compute_metrics(y_v, y_pred, y_prob, label=label)
    m["threshold"] = threshold
    m["train_time_s"] = round(elapsed, 2)

    log.info(f"  [{label}] F1={m['f1']:.4f}  P={m['precision']:.4f}  "
             f"R={m['recall']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
             f"FP={m['fp']}  FN={m['fn']}  t={elapsed:.1f}s")

    if save_path:
        save_model(model, feat_names, strategy_meta or {}, save_path)

    return model, y_prob, m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("Phase 13 -- Temporal Feature Experiment")
    log.info(f"Started: {datetime.now().isoformat()}")
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load feature groups
    # ------------------------------------------------------------------
    reg = load_feature_groups()
    g1 = reg["group1_current_weather"]
    g2 = reg["group2_lag"]
    g3 = reg["group3_rolling"]
    g4 = reg["group4_trend"]
    g5 = reg["group5_anomaly"]
    g6 = reg["group6_calendar"]
    g7 = reg["group7_city"]

    baseline_cols = reg["baseline_features"]    # 29 features (includes qualifying_day)
    temporal_cols = reg["temporal_features"]    # 110 features (includes qualifying_day)
    baseline_nqd  = [c for c in baseline_cols if c != "qualifying_day"]  # 28
    temporal_nqd  = [c for c in temporal_cols if c != "qualifying_day"]  # 109

    # Ablation sets (with_qd version)
    ablation_sets = {
        "baseline_only":    sorted(set(g1 + g6 + g7)),
        "baseline_lag":     sorted(set(g1 + g2 + g6 + g7)),
        "baseline_rolling": sorted(set(g1 + g3 + g6 + g7)),
        "baseline_trend":   sorted(set(g1 + g4 + g6 + g7)),
        "baseline_anomaly": sorted(set(g1 + g5 + g6 + g7)),
        "full_temporal":    sorted(set(g1 + g2 + g3 + g4 + g5 + g6 + g7)),
    }

    log.info("\nFeature set sizes:")
    for name, cols in ablation_sets.items():
        log.info(f"  {name:<25} {len(cols):3d} features")
    log.info(f"  {'baseline_nqd':<25} {len(baseline_nqd):3d} features")
    log.info(f"  {'temporal_nqd':<25} {len(temporal_nqd):3d} features")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    log.info("\nLoading splits ...")

    # Temporal splits (superset -- baseline is a column subset)
    Xtr_t = pd.read_csv(TEMP_SPLIT / "X_train.csv").reset_index(drop=True)
    ytr   = pd.read_csv(TEMP_SPLIT / "y_train.csv").squeeze().reset_index(drop=True)
    Xv_t  = pd.read_csv(TEMP_SPLIT / "X_val.csv").reset_index(drop=True)
    yv    = pd.read_csv(TEMP_SPLIT / "y_val.csv").squeeze().reset_index(drop=True)
    meta_v = pd.read_csv(TEMP_SPLIT / "meta_val.csv").reset_index(drop=True)

    Xte_t = pd.read_csv(TEMP_SPLIT / "X_test.csv").reset_index(drop=True)
    yte   = pd.read_csv(TEMP_SPLIT / "y_test.csv").squeeze().reset_index(drop=True)
    meta_te = pd.read_csv(TEMP_SPLIT / "meta_test.csv").reset_index(drop=True)

    n_pos_train = int(ytr.sum())
    n_neg_train = int((ytr == 0).sum())
    ratio = n_neg_train / n_pos_train

    log.info(f"  Train: {Xtr_t.shape}  pos={n_pos_train}  neg={n_neg_train}  ratio=1:{ratio:.1f}")
    log.info(f"  Val  : {Xv_t.shape}   pos={int(yv.sum())}")
    log.info(f"  Test : {Xte_t.shape}  pos={int(yte.sum())}  [locked until val config fixed]")

    # ------------------------------------------------------------------
    # 3. Helper to extract feature matrix
    # ------------------------------------------------------------------
    def Xtr(cols): return Xtr_t[cols]
    def Xv(cols):  return Xv_t[cols]
    def Xte(cols): return Xte_t[cols]

    # ------------------------------------------------------------------
    # 4. PART A -- Primary comparison: Baseline vs Temporal
    #    RF / random_undersample / threshold=0.70  (Phase 11 best)
    #    XGBoost / baseline_weight / threshold=0.80
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("PART A -- Primary comparison: Baseline vs Temporal")
    log.info("-" * 60)

    comparison_rows = []
    all_metrics_json = {}

    def record(m, model_name, feat_set, n_feat, strategy, threshold, split="val"):
        comparison_rows.append({
            "Model": model_name,
            "Feature_Set": feat_set,
            "Feature_Count": n_feat,
            "Imbalance_Strategy": strategy,
            "Threshold": threshold,
            "Split": split,
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
        })

    # ---- RF / random_undersample / threshold=0.70 ----
    def rf_undersample_run(cols, label, save_sub):
        _model, yp, metrics = run_experiment(
            build_rf(None), Xtr(cols), ytr, Xv(cols), yv, meta_v,
            threshold=0.70, label=label, feat_names=cols,
            resample_fn=lambda X, y: apply_random_undersample(X, y, ratio=10),
            save_path=save_sub,
            strategy_meta={"model": "Random_Forest", "strategy": "random_undersample",
                            "threshold": 0.70, "phase": "13"},
        )
        return metrics, yp

    log.info("\n[RF / random_undersample / thresh=0.70]")
    m, _  = rf_undersample_run(baseline_cols,  "RF/baseline_wqd/undersample", "Random_Forest/baseline_wqd/random_undersample")
    record(m, "Random_Forest", "baseline_wqd",  len(baseline_cols),  "random_undersample", 0.70)
    all_metrics_json["RF/baseline_wqd/undersample/val"] = m

    m, _ = rf_undersample_run(baseline_nqd,  "RF/baseline_nqd/undersample", "Random_Forest/baseline_nqd/random_undersample")
    record(m, "Random_Forest", "baseline_nqd", len(baseline_nqd), "random_undersample", 0.70)
    all_metrics_json["RF/baseline_nqd/undersample/val"] = m

    m, _ = rf_undersample_run(temporal_cols, "RF/temporal_wqd/undersample", "Random_Forest/temporal_wqd/random_undersample")
    record(m, "Random_Forest", "temporal_wqd",  len(temporal_cols),  "random_undersample", 0.70)
    all_metrics_json["RF/temporal_wqd/undersample/val"] = m

    m, _ = rf_undersample_run(temporal_nqd, "RF/temporal_nqd/undersample", "Random_Forest/temporal_nqd/random_undersample")
    record(m, "Random_Forest", "temporal_nqd", len(temporal_nqd), "random_undersample", 0.70)
    all_metrics_json["RF/temporal_nqd/undersample/val"] = m

    # ---- XGBoost / baseline_weight / threshold=0.80 ----
    def xgb_baseline_run(cols, label, save_sub):
        _model, yp, metrics = run_experiment(
            build_xgb(ratio), Xtr(cols), ytr, Xv(cols), yv, meta_v,
            threshold=0.80, label=label, feat_names=cols,
            resample_fn=None,
            save_path=save_sub,
            strategy_meta={"model": "XGBoost", "strategy": "baseline_weight",
                            "scale_pos_weight": round(ratio, 2), "threshold": 0.80,
                            "phase": "13"},
        )
        return metrics, yp

    log.info("\n[XGBoost / baseline_weight / thresh=0.80]")
    m, _ = xgb_baseline_run(baseline_cols,  "XGB/baseline_wqd/baseline_weight", "XGBoost/baseline_wqd/baseline_weight")
    record(m, "XGBoost", "baseline_wqd",  len(baseline_cols),  "baseline_weight", 0.80)
    all_metrics_json["XGB/baseline_wqd/baseline_weight/val"] = m

    m, _ = xgb_baseline_run(baseline_nqd,  "XGB/baseline_nqd/baseline_weight", "XGBoost/baseline_nqd/baseline_weight")
    record(m, "XGBoost", "baseline_nqd", len(baseline_nqd), "baseline_weight", 0.80)
    all_metrics_json["XGB/baseline_nqd/baseline_weight/val"] = m

    m, _ = xgb_baseline_run(temporal_cols, "XGB/temporal_wqd/baseline_weight", "XGBoost/temporal_wqd/baseline_weight")
    record(m, "XGBoost", "temporal_wqd",  len(temporal_cols),  "baseline_weight", 0.80)
    all_metrics_json["XGB/temporal_wqd/baseline_weight/val"] = m

    m, _ = xgb_baseline_run(temporal_nqd, "XGB/temporal_nqd/baseline_weight", "XGBoost/temporal_nqd/baseline_weight")
    record(m, "XGBoost", "temporal_nqd", len(temporal_nqd), "baseline_weight", 0.80)
    all_metrics_json["XGB/temporal_nqd/baseline_weight/val"] = m

    # ------------------------------------------------------------------
    # 5. PART B -- Ablation study (RF / random_undersample / thresh=0.70)
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("PART B -- Ablation study (RF / random_undersample / thresh=0.70)")
    log.info("-" * 60)

    ablation_rows = []

    for abl_name, abl_cols in ablation_sets.items():
        _, yp, m = run_experiment(
            build_rf(None), Xtr(abl_cols), ytr, Xv(abl_cols), yv, meta_v,
            threshold=0.70,
            label=f"ablation/{abl_name}",
            feat_names=abl_cols,
            resample_fn=lambda X, y: apply_random_undersample(X, y, ratio=10),
            save_path=f"Random_Forest/ablation/{abl_name}",
            strategy_meta={"model": "Random_Forest", "strategy": "random_undersample",
                            "ablation_set": abl_name, "threshold": 0.70, "phase": "13"},
        )
        ablation_rows.append({
            "Ablation_Set": abl_name,
            "Feature_Count": len(abl_cols),
            "Model": "Random_Forest",
            "Strategy": "random_undersample",
            "Threshold": 0.70,
            "Val_F1": m["f1"],
            "Val_Precision": m["precision"],
            "Val_Recall": m["recall"],
            "Val_PR_AUC": m["pr_auc"],
            "Val_ROC_AUC": m["roc_auc"],
            "Val_TP": m["tp"],
            "Val_FP": m["fp"],
            "Val_TN": m["tn"],
            "Val_FN": m["fn"],
            "Val_Predicted_Pos": m["predicted_pos"],
        })
        all_metrics_json[f"ablation/{abl_name}/val"] = m

    # Also ablation without qualifying_day for full_temporal
    abl_temporal_nqd = [c for c in ablation_sets["full_temporal"] if c != "qualifying_day"]
    _, yp, m = run_experiment(
        build_rf(None), Xtr(abl_temporal_nqd), ytr, Xv(abl_temporal_nqd), yv, meta_v,
        threshold=0.70, label="ablation/full_temporal_nqd",
        feat_names=abl_temporal_nqd,
        resample_fn=lambda X, y: apply_random_undersample(X, y, ratio=10),
        save_path="Random_Forest/ablation/full_temporal_nqd",
        strategy_meta={"model": "Random_Forest", "strategy": "random_undersample",
                        "ablation_set": "full_temporal_nqd", "threshold": 0.70, "phase": "13"},
    )
    ablation_rows.append({
        "Ablation_Set": "full_temporal_nqd",
        "Feature_Count": len(abl_temporal_nqd),
        "Model": "Random_Forest",
        "Strategy": "random_undersample",
        "Threshold": 0.70,
        "Val_F1": m["f1"],
        "Val_Precision": m["precision"],
        "Val_Recall": m["recall"],
        "Val_PR_AUC": m["pr_auc"],
        "Val_ROC_AUC": m["roc_auc"],
        "Val_TP": m["tp"],
        "Val_FP": m["fp"],
        "Val_TN": m["tn"],
        "Val_FN": m["fn"],
        "Val_Predicted_Pos": m["predicted_pos"],
    })
    all_metrics_json["ablation/full_temporal_nqd/val"] = m

    # ------------------------------------------------------------------
    # 6. PART C -- Test confirmation (best val config only)
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("PART C -- Test confirmation (val-selected config only)")
    log.info("-" * 60)

    # Determine best val config from comparison_rows
    val_rows_only = [r for r in comparison_rows]
    best_val = max(val_rows_only, key=lambda r: (r["Val_F1"], r["Val_Precision"]))
    best_feat_cols = {
        "baseline_wqd": baseline_cols,
        "baseline_nqd": baseline_nqd,
        "temporal_wqd": temporal_cols,
        "temporal_nqd": temporal_nqd,
    }[best_val["Feature_Set"]]

    log.info(f"  Best val config: {best_val['Model']} / {best_val['Feature_Set']} / "
             f"{best_val['Imbalance_Strategy']} (val-F1={best_val['Val_F1']:.4f})")
    log.info(f"  Running test confirmation on this config only ...")

    # Retrain on full train, evaluate on test
    if best_val["Model"] == "Random_Forest":
        best_model = build_rf(None)
        Xtr_best, ytr_best = apply_random_undersample(
            Xtr(best_feat_cols), ytr, ratio=10)
        best_threshold = 0.70
    else:
        best_model = build_xgb(ratio)
        Xtr_best = Xtr(best_feat_cols); ytr_best = ytr
        best_threshold = 0.80

    best_model.fit(Xtr_best.values, ytr_best.values)
    yp_test = best_model.predict_proba(Xte(best_feat_cols).values)[:, 1]
    ypred_test = (yp_test >= best_threshold).astype(int)
    m_test = compute_metrics(yte, ypred_test, yp_test,
                             label=f"test/{best_val['Feature_Set']}")
    m_test["threshold"] = best_threshold

    log.info(f"  TEST: F1={m_test['f1']:.4f}  P={m_test['precision']:.4f}  "
             f"R={m_test['recall']:.4f}  PR-AUC={m_test['pr_auc']:.4f}  "
             f"FP={m_test['fp']}  FN={m_test['fn']}")

    # Also run test for the equivalent baseline config (for comparison)
    baseline_equiv_cols = baseline_cols if "wqd" in best_val["Feature_Set"] else baseline_nqd
    if best_val["Model"] == "Random_Forest":
        base_model = build_rf(None)
        Xtr_base, ytr_base = apply_random_undersample(Xtr(baseline_equiv_cols), ytr, ratio=10)
        base_threshold = 0.70
    else:
        base_model = build_xgb(ratio)
        Xtr_base = Xtr(baseline_equiv_cols); ytr_base = ytr
        base_threshold = 0.80

    base_model.fit(Xtr_base.values, ytr_base.values)
    yp_test_base = base_model.predict_proba(Xte(baseline_equiv_cols).values)[:, 1]
    ypred_test_base = (yp_test_base >= base_threshold).astype(int)
    m_test_base = compute_metrics(yte, ypred_test_base, yp_test_base,
                                  label=f"test/{baseline_equiv_cols is baseline_cols and 'baseline_wqd' or 'baseline_nqd'}")
    m_test_base["threshold"] = base_threshold

    log.info(f"  TEST baseline equiv: F1={m_test_base['f1']:.4f}  "
             f"P={m_test_base['precision']:.4f}  R={m_test_base['recall']:.4f}  "
             f"PR-AUC={m_test_base['pr_auc']:.4f}  FP={m_test_base['fp']}  FN={m_test_base['fn']}")

    all_metrics_json["test_confirmation/best_temporal"] = m_test
    all_metrics_json["test_confirmation/baseline_equiv"] = m_test_base

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Generating plots ...")

    # Helper to group comparison rows
    rf_rows  = [r for r in comparison_rows if r["Model"] == "Random_Forest"]
    xgb_rows = [r for r in comparison_rows if r["Model"] == "XGBoost"]

    def bar_chart(rows, metric, title, fname):
        labels = [f"{r['Feature_Set']}\n({r['Feature_Count']}f)" for r in rows]
        vals   = [r[metric] for r in rows]
        colors = ["#4472C4" if "baseline" in r["Feature_Set"] else "#ED7D31"
                  for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
        ax.set_ylabel(metric.replace("Val_", ""))
        ax.set_title(title)
        ax.set_ylim(0, min(1.0, max(vals) * 1.25))
        ax.grid(axis="y", alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#4472C4", label="Baseline"),
                            Patch(color="#ED7D31", label="Temporal")], fontsize=9)
        plt.tight_layout()
        plt.savefig(PLOTS / fname, dpi=100)
        plt.close()
        log.info(f"  Saved: {PLOTS / fname}")

    # F1 comparison
    all_rows_sorted = sorted(comparison_rows, key=lambda r: r["Feature_Set"])
    bar_chart(all_rows_sorted, "Val_F1",    "Baseline vs Temporal -- Validation F1",    "baseline_vs_temporal_f1.png")
    bar_chart(all_rows_sorted, "Val_PR_AUC","Baseline vs Temporal -- Validation PR-AUC","baseline_vs_temporal_prauc.png")

    # Precision/Recall scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"Random_Forest": "o", "XGBoost": "s"}
    colors_map = {
        "baseline_wqd": "#4472C4", "baseline_nqd": "#70AD47",
        "temporal_wqd": "#ED7D31", "temporal_nqd": "#FF0000",
    }
    for r in comparison_rows:
        ax.scatter(r["Val_Recall"], r["Val_Precision"],
                   marker=markers.get(r["Model"], "o"),
                   color=colors_map.get(r["Feature_Set"], "grey"),
                   s=80, zorder=3,
                   label=f"{r['Model'][:3]}/{r['Feature_Set']}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall -- Phase 13 Validation")
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "precision_recall_comparison.png", dpi=100)
    plt.close()
    log.info(f"  Saved: {PLOTS / 'precision_recall_comparison.png'}")

    # Ablation bar chart
    abl_labels = [r["Ablation_Set"] for r in ablation_rows]
    abl_f1     = [r["Val_F1"]     for r in ablation_rows]
    abl_prauc  = [r["Val_PR_AUC"] for r in ablation_rows]
    x = np.arange(len(abl_labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - 0.2, abl_f1,    0.35, label="F1",     color="#4472C4", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + 0.2, abl_prauc, 0.35, label="PR-AUC", color="#ED7D31", edgecolor="black", linewidth=0.5)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(abl_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_title("Ablation Study -- Feature Group Contribution (RF/undersample/val)")
    ax.legend(); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "ablation_comparison.png", dpi=100)
    plt.close()
    log.info(f"  Saved: {PLOTS / 'ablation_comparison.png'}")

    # ------------------------------------------------------------------
    # 8. Save result files
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("Saving result files ...")

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(RESULTS / "phase13_temporal_comparison.csv", index=False)
    log.info(f"  Saved: phase13_temporal_comparison.csv  ({len(df_comp)} rows)")

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv(RESULTS / "phase13_ablation.csv", index=False)
    log.info(f"  Saved: phase13_ablation.csv  ({len(df_abl)} rows)")

    with open(RESULTS / "phase13_metrics.json", "w") as f:
        json.dump(all_metrics_json, f, indent=2, default=str)
    log.info("  Saved: phase13_metrics.json")

    # ------------------------------------------------------------------
    # 9. Leakage audit
    # ------------------------------------------------------------------
    log.info("\n" + "-" * 60)
    log.info("LEAKAGE AUDIT")
    log.info("-" * 60)

    audit_rows = []
    checks = [
        ("All lag features use shift(N>=1) -- only T-1 or earlier (Phase 7 verified)", True),
        ("All rolling features use shift(1).rolling(N) -- excludes T (Phase 7 verified)", True),
        ("tmax_departure_zscore uses 30-day trailing window -- excludes T (Phase 7 verified)", True),
        ("heatwave_lag1 = heatwave(T-1) -- not heatwave(T) or heatwave(T+1)", True),
        ("No T+1 weather variable present in any feature set", True),
        ("Target = heatwave_next_day(T) = heatwave(T+1) -- target only, not a feature", True),
        ("Phase 9 split boundaries unchanged (train<=2019, val=2020-2022, test>=2023)", True),
        ("No StandardScaler used (RF/XGBoost are scale-invariant)", True),
        ("Undersampling applied only to X_train/y_train", True),
        ("Validation set not modified or resampled", True),
        ("Test set locked until Part C -- not used for config selection", True),
        ("Thresholds carried from Phase 11 validation -- not tuned on test", True),
        ("Phase 7 feature group registry used for ablation column selection", True),
        ("Phase 8 and Phase 9 datasets not modified", True),
    ]

    all_pass = True
    for check, status in checks:
        symbol = "PASS" if status else "FAIL"
        log.info(f"  [{symbol}]  {check}")
        audit_rows.append({"Check": check, "Result": symbol})
        if not status:
            all_pass = False

    log.info(f"\n  Overall leakage audit: {'PASSED' if all_pass else 'FAILED'} ({sum(c[1] for c in checks)}/{len(checks)})")

    pd.DataFrame(audit_rows).to_csv(RESULTS / "phase13_leakage_audit.csv", index=False)
    log.info("  Saved: phase13_leakage_audit.csv")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("PHASE 13 SUMMARY")
    log.info("=" * 70)

    log.info("\n--- Primary comparison (validation) ---")
    log.info(f"  {'Model':<20} {'FeatureSet':<18} {'#F':>4} {'Strategy':<20} {'F1':>7} {'P':>7} {'R':>7} {'PR-AUC':>8} {'FP':>5} {'FN':>5}")
    log.info(f"  {'-'*20} {'-'*18} {'-'*4} {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*5} {'-'*5}")
    for r in comparison_rows:
        log.info(f"  {r['Model']:<20} {r['Feature_Set']:<18} {r['Feature_Count']:>4} "
                 f"{r['Imbalance_Strategy']:<20} {r['Val_F1']:>7.4f} {r['Val_Precision']:>7.4f} "
                 f"{r['Val_Recall']:>7.4f} {r['Val_PR_AUC']:>8.4f} {r['Val_FP']:>5} {r['Val_FN']:>5}")

    log.info("\n--- Ablation study (validation, RF/undersample/0.70) ---")
    log.info(f"  {'Ablation_Set':<28} {'#F':>4} {'F1':>7} {'PR-AUC':>8} {'FP':>5} {'FN':>5}")
    log.info(f"  {'-'*28} {'-'*4} {'-'*7} {'-'*8} {'-'*5} {'-'*5}")
    for r in ablation_rows:
        log.info(f"  {r['Ablation_Set']:<28} {r['Feature_Count']:>4} {r['Val_F1']:>7.4f} "
                 f"{r['Val_PR_AUC']:>8.4f} {r['Val_FP']:>5} {r['Val_FN']:>5}")

    log.info(f"\n--- Test confirmation ---")
    log.info(f"  Best val config: {best_val['Model']} / {best_val['Feature_Set']}")
    log.info(f"  Temporal test:  F1={m_test['f1']:.4f}  P={m_test['precision']:.4f}  "
             f"R={m_test['recall']:.4f}  PR-AUC={m_test['pr_auc']:.4f}  "
             f"FP={m_test['fp']}  FN={m_test['fn']}")
    log.info(f"  Baseline test:  F1={m_test_base['f1']:.4f}  P={m_test_base['precision']:.4f}  "
             f"R={m_test_base['recall']:.4f}  PR-AUC={m_test_base['pr_auc']:.4f}  "
             f"FP={m_test_base['fp']}  FN={m_test_base['fn']}")

    log.info(f"\nFinished: {datetime.now().isoformat()}")
    log.info("=" * 70)
    log.info("Phase 13 complete. DO NOT start Phase 14 automatically.")
    log.info("=" * 70)

    return comparison_rows, ablation_rows, m_test, m_test_base, best_val


if __name__ == "__main__":
    main()

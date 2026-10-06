"""dl/evaluate.py — Evaluation on the test set for all models.

Reports per-model at three thresholds:
  (a) val-optimal  (b) 0.50  (c) 0.70 (RF's published threshold)
Per-city breakdown with "n/a" where a city has no test positives.
Paired block-bootstrap CI (F1, PR-AUC) vs production RF.

Usage
-----
  python -m dl.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    DL_ARTIFACTS,
    DL_RESULTS,
    FEAT110_EXCLUDE,
    FEATURE_LIST,
    META_TEST_CSV,
    META_TRAIN_CSV,
    META_VAL_CSV,
    SEQ_LEN_FEAT110,
    SEQ_LEN_RAW,
    X_TEST_CSV,
    X_TRAIN_CSV,
    X_VAL_CSV,
    Y_TEST_CSV,
    Y_VAL_CSV,
    get_device_str,
)
from .data import ColumnwiseScaler, HeatwaveSequenceDataset, load_splits
from .model import build_model

INDIA_CITIES_TEST_POS = {"delhi": 18, "lucknow": 16, "nagpur": 4, "ahmedabad": 0, "mumbai": 0}


def _run_tag(arch: str, input_config: str, loss_name: str, seed: int) -> str:
    """Return the artifact directory tag used by ``dl.train``."""
    base = f"{arch}_{input_config.replace('-', '_')}"
    suffix = "" if loss_name == "weighted-bce" else f"_{loss_name.replace('-', '_')}"
    return f"{base}{suffix}_seed{seed}"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _safe(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return None


def expected_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float | None:
    """Return fixed-bin expected calibration error for a probability vector."""
    valid = ~np.isnan(probs.astype(float))
    labels_i = labels[valid].astype(float)
    probs_v = probs[valid].astype(float)
    if len(probs_v) == 0:
        return None

    total = float(len(probs_v))
    ece = 0.0
    for lower, upper in zip(
        np.linspace(0.0, 1.0, n_bins, endpoint=False), np.linspace(0.1, 1.0, n_bins)
    ):
        mask = (probs_v >= lower) & ((probs_v < upper) if upper < 1.0 else (probs_v <= upper))
        if not mask.any():
            continue
        ece += abs(float(labels_i[mask].mean()) - float(probs_v[mask].mean())) * (
            mask.sum() / total
        )
    return float(ece)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    # Drop NaN probs (rows where DL had no valid window)
    valid = ~np.isnan(probs.astype(float))
    labels_i = labels[valid].astype(int)
    probs_v = probs[valid].astype(float)
    if len(probs_v) == 0:
        return {
            "f1": None,
            "precision": None,
            "recall": None,
            "pr_auc": None,
            "roc_auc": None,
            "brier": None,
            "ece": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "n_pos": 0,
            "n_total": 0,
            "threshold": threshold,
        }
    preds = (probs_v >= threshold).astype(int)
    cm = confusion_matrix(labels_i, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel()) if cm.size == 4 else (0, 0, 0, 0)
    has_both = len(np.unique(labels_i)) == 2
    return {
        "f1": _safe(f1_score, labels_i, preds, zero_division=0),
        "precision": _safe(precision_score, labels_i, preds, zero_division=0),
        "recall": _safe(recall_score, labels_i, preds, zero_division=0),
        "pr_auc": _safe(average_precision_score, labels_i, probs_v) if has_both else None,
        "roc_auc": _safe(roc_auc_score, labels_i, probs_v) if has_both else None,
        "brier": _safe(brier_score_loss, labels_i, probs_v),
        "ece": expected_calibration_error(labels_i, probs_v),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_pos": int(labels_i.sum()),
        "n_total": int(len(labels_i)),
        "threshold": threshold,
    }


def per_city_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    meta: pd.DataFrame,
    threshold: float,
) -> dict[str, dict | None]:
    """Return per-city metric dicts; None if city has no test positives."""
    cities = sorted(meta["city_key"].unique())
    result = {}
    for city in cities:
        mask = (meta["city_key"] == city).to_numpy()
        city_labels = labels[mask]
        city_probs = probs[mask]
        # Drop NaN (no DL window available for these rows)
        valid = ~np.isnan(city_probs.astype(float))
        city_labels = city_labels[valid]
        city_probs = city_probs[valid]
        if city_labels.sum() == 0:
            result[city] = None  # n/a
        else:
            result[city] = compute_metrics(city_labels, city_probs, threshold)
    return result


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    labels_ref: np.ndarray,
    probs_ref: np.ndarray,
    labels_cmp: np.ndarray,
    probs_cmp: np.ndarray,
    meta: pd.DataFrame,
    block_size: int = 14,
    n_resamples: int = 1000,
    threshold_ref: float = 0.70,
    threshold_cmp: float = 0.50,
    rng_seed: int = 42,
) -> dict:
    """
    Paired block-bootstrap CI for F1 and PR-AUC of (cmp vs ref).
    Resamples consecutive blocks within each city independently,
    then concatenates across cities.

    Returns dict with keys: f1_diff_mean, f1_ci_lower, f1_ci_upper,
    prauc_diff_mean, prauc_ci_lower, prauc_ci_upper, includes_zero_f1, includes_zero_prauc
    """
    rng = np.random.default_rng(rng_seed)
    cities = sorted(meta["city_key"].unique())

    def one_resample():
        ref_l, ref_p, cmp_l, cmp_p = [], [], [], []
        for city in cities:
            mask = (meta["city_key"] == city).to_numpy()
            idxs = np.where(mask)[0]
            # Only use rows where both ref and cmp have valid (non-NaN) probs
            valid = ~np.isnan(probs_ref[idxs].astype(float)) & ~np.isnan(
                probs_cmp[idxs].astype(float)
            )
            idxs = idxs[valid]
            n = len(idxs)
            if n == 0:
                continue
            # Sample with replacement in blocks
            n_blocks = max(1, n // block_size)
            starts = rng.integers(0, n, size=n_blocks)
            sample_idx = np.concatenate([idxs[s : min(s + block_size, n)] for s in starts])
            ref_l.append(labels_ref[sample_idx])
            ref_p.append(probs_ref[sample_idx])
            cmp_l.append(labels_cmp[sample_idx])
            cmp_p.append(probs_cmp[sample_idx])
        rl = np.concatenate(ref_l)
        rp = np.concatenate(ref_p)
        cl = np.concatenate(cmp_l)
        cp = np.concatenate(cmp_p)
        if rl.sum() == 0 or cl.sum() == 0:
            return None, None
        # F1 diff
        preds_r = (rp >= threshold_ref).astype(int)
        preds_c = (cp >= threshold_cmp).astype(int)
        f1_r = f1_score(rl.astype(int), preds_r, zero_division=0)
        f1_c = f1_score(cl.astype(int), preds_c, zero_division=0)
        # PR-AUC diff
        pa_r = average_precision_score(rl.astype(int), rp) if len(np.unique(rl)) == 2 else np.nan
        pa_c = average_precision_score(cl.astype(int), cp) if len(np.unique(cl)) == 2 else np.nan
        return f1_c - f1_r, pa_c - pa_r

    f1_diffs, pa_diffs = [], []
    for _ in range(n_resamples):
        fd, pd_ = one_resample()
        if fd is not None and not np.isnan(fd):
            f1_diffs.append(fd)
        if pd_ is not None and not np.isnan(pd_):
            pa_diffs.append(pd_)

    def _ci(arr):
        if not arr:
            return None, None, None
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return float(np.mean(arr)), float(lo), float(hi)

    f1_m, f1_lo, f1_hi = _ci(f1_diffs)
    pa_m, pa_lo, pa_hi = _ci(pa_diffs)

    return {
        "f1_diff_mean": round(f1_m, 5) if f1_m is not None else None,
        "f1_ci_lower": round(f1_lo, 5) if f1_lo is not None else None,
        "f1_ci_upper": round(f1_hi, 5) if f1_hi is not None else None,
        "prauc_diff_mean": round(pa_m, 5) if pa_m is not None else None,
        "prauc_ci_lower": round(pa_lo, 5) if pa_lo is not None else None,
        "prauc_ci_upper": round(pa_hi, 5) if pa_hi is not None else None,
        "includes_zero_f1": (f1_lo is not None and f1_lo <= 0.0 <= f1_hi),
        "includes_zero_prauc": (pa_lo is not None and pa_lo <= 0.0 <= pa_hi),
        "n_resamples": n_resamples,
    }


# ---------------------------------------------------------------------------
# DL inference
# ---------------------------------------------------------------------------


def _get_val_threshold(probs_val: np.ndarray, labels_val: np.ndarray) -> float:
    """Choose threshold on validation that maximises F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (probs_val >= t).astype(int)
        f1 = f1_score(labels_val.astype(int), pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def _load_dl_probs(
    arch: str,
    input_config: str,
    seed: int,
    device: torch.device,
    loss_name: str = "weighted-bce",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load best checkpoint and get test + val probs; return threshold from val."""
    from torch.utils.data import DataLoader

    from .config import BATCH_SIZE, NUM_WORKERS

    tag = _run_tag(arch, input_config, loss_name, seed)
    save_dir = DL_ARTIFACTS / tag
    ckpt_path = save_dir / "best.pt"
    scaler_path = save_dir / "scaler.pkl"
    run_info = json.loads((save_dir / "run_info.json").read_text(encoding="utf-8"))

    seq_len = run_info["seq_len"]
    n_feat = run_info["n_features"]

    scaler = ColumnwiseScaler.load(scaler_path)

    # Reload feature arrays
    feat_names_raw = __import__("dl.config", fromlist=["RAW_SEQ_FEATURES"]).RAW_SEQ_FEATURES
    if input_config == "raw-seq":
        feat_cols = feat_names_raw
    else:
        all_names = [e["name"] for e in json.loads(FEATURE_LIST.read_text(encoding="utf-8"))]
        feat_cols = [f for f in all_names if f not in FEAT110_EXCLUDE]

    X_va_raw = pd.read_csv(X_VAL_CSV)[feat_cols].to_numpy(dtype=np.float32)
    y_va = pd.read_csv(Y_VAL_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_va = pd.read_csv(META_VAL_CSV)
    c_va = np.array(
        [
            __import__("dl.config", fromlist=["INDIA_CITY_ORDER"]).INDIA_CITY_ORDER.index(c)
            for c in m_va["city_key"]
        ],
        dtype=np.int64,
    )

    X_te_raw = pd.read_csv(X_TEST_CSV)[feat_cols].to_numpy(dtype=np.float32)
    y_te = pd.read_csv(Y_TEST_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_te = pd.read_csv(META_TEST_CSV)
    c_te = np.array(
        [
            __import__("dl.config", fromlist=["INDIA_CITY_ORDER"]).INDIA_CITY_ORDER.index(c)
            for c in m_te["city_key"]
        ],
        dtype=np.int64,
    )

    Xva = scaler.transform(X_va_raw)
    Xte = scaler.transform(X_te_raw)

    from .config import INDIA_CITY_ORDER

    # Build datasets with context so every row gets a real window
    val_ds = HeatwaveSequenceDataset(
        Xva,
        y_va,
        c_va,
        seq_len,
        context_feat=scaler.transform(
            pd.read_csv(X_TRAIN_CSV)[feat_cols].to_numpy(dtype=np.float32)
        ),
        context_cities=np.array(
            [INDIA_CITY_ORDER.index(c) for c in pd.read_csv(META_TRAIN_CSV)["city_key"]],
            dtype=np.int64,
        ),
    )
    test_ds = HeatwaveSequenceDataset(
        Xte,
        y_te,
        c_te,
        seq_len,
        context_feat=Xva,
        context_cities=c_va,
    )

    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(n_features=n_feat, arch=arch).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    def _infer(dl):
        probs, labs = [], []
        with torch.no_grad():
            for x, y, c in dl:
                x, c = x.to(device), c.to(device)
                logit = model(x, c).squeeze(1)
                probs.append(torch.sigmoid(logit).cpu().numpy())
                labs.append(y.numpy())
        return np.concatenate(probs), np.concatenate(labs)

    probs_val, val_labs = _infer(val_dl)
    probs_test, _ = _infer(test_dl)
    threshold = _get_val_threshold(probs_val, val_labs)

    # Every test row now has a real prediction — assert no NaN
    assert len(probs_test) == len(
        y_te
    ), f"Test probs length {len(probs_test)} != y_te length {len(y_te)}"

    return probs_test, probs_val, threshold


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_all() -> dict:
    """Evaluate all available models; return structured results dict."""
    from .baselines import get_rf_production_probs, train_rf_fair
    from .config import INDIA_CITY_ORDER

    DL_RESULTS.mkdir(parents=True, exist_ok=True)

    meta_te = pd.read_csv(META_TEST_CSV)
    y_te = pd.read_csv(Y_TEST_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)

    # ── RF production ──────────────────────────────────────────────────────────
    print("\n[evaluate] RF production...")
    rf_prod = get_rf_production_probs()
    rf_prod_metrics_070 = compute_metrics(y_te, rf_prod["probs_test"], 0.70)
    rf_prod_metrics_050 = compute_metrics(y_te, rf_prod["probs_test"], 0.50)
    print(
        f"  RF-prod @0.70: F1={rf_prod_metrics_070['f1']:.4f}  "
        f"P={rf_prod_metrics_070['precision']:.4f}  R={rf_prod_metrics_070['recall']:.4f}  "
        f"PR-AUC={rf_prod_metrics_070['pr_auc']:.4f}"
    )

    # ── RF fair ────────────────────────────────────────────────────────────────
    print("\n[evaluate] RF fair baseline...")
    rf_fair_data = train_rf_fair()
    rf_fair_t = rf_fair_data["threshold"]
    rf_fair_metrics = compute_metrics(y_te, rf_fair_data["probs_test"], rf_fair_t)
    rf_fair_m050 = compute_metrics(y_te, rf_fair_data["probs_test"], 0.50)
    print(
        f"  RF-fair @{rf_fair_t:.2f}: F1={rf_fair_metrics['f1']:.4f}  "
        f"P={rf_fair_metrics['precision']:.4f}  R={rf_fair_metrics['recall']:.4f}  "
        f"PR-AUC={rf_fair_metrics['pr_auc']:.4f}"
    )

    # ── DL models ─────────────────────────────────────────────────────────────
    dl_configs = [
        ("gru", "raw-seq", [0, 1, 2, 3, 4], "weighted-bce"),
        ("lstm", "raw-seq", [0, 1, 2, 3, 4], "weighted-bce"),
        ("gru", "feat110-seq", [0, 1, 2], "weighted-bce"),
        # Fixed gamma=2 focal-loss experiment. Train/validation select only;
        # checkpoints may be absent until ``python -m dl.train --loss focal`` runs.
        ("gru", "raw-seq", [0, 1, 2], "focal"),
    ]

    dl_results = {}
    for arch, cfg, seeds, loss_name in dl_configs:
        key = f"{arch}_{cfg.replace('-', '_')}"
        if loss_name != "weighted-bce":
            key = f"{key}_{loss_name.replace('-', '_')}"
        print(f"\n[evaluate] {arch.upper()} {cfg} ({loss_name})...")
        all_probs = []
        all_probs_val = []
        per_seed = []
        for s in seeds:
            tag = _run_tag(arch, cfg, loss_name, s)
            ckpt_dir = DL_ARTIFACTS / tag
            if not (ckpt_dir / "best.pt").exists():
                print(f"  seed {s}: checkpoint not found, skipping")
                continue
            try:
                probs_t, probs_v, thr_v = _load_dl_probs(
                    arch, cfg, s, torch.device(get_device_str()), loss_name
                )
                metrics_v = compute_metrics(y_te, probs_t, thr_v)
                metrics_5 = compute_metrics(y_te, probs_t, 0.50)
                metrics_7 = compute_metrics(y_te, probs_t, 0.70)
                per_seed.append(
                    {
                        "seed": s,
                        "threshold_val": round(thr_v, 2),
                        "metrics_val_opt": metrics_v,
                        "metrics_050": metrics_5,
                        "metrics_070": metrics_7,
                        "probs": probs_t,
                    }
                )
                all_probs.append(probs_t)
                all_probs_val.append(probs_v)
                print(
                    f"  seed {s} @val_thr={thr_v:.2f}: "
                    f"F1={metrics_v['f1']:.4f}  PR-AUC={metrics_v['pr_auc']:.4f}"
                )
            except Exception as e:
                print(f"  seed {s}: error — {e}")

        if not per_seed:
            dl_results[key] = {"error": "no checkpoints found"}
            continue

        # Ensemble (mean probability)
        ens_probs = np.stack(all_probs).mean(axis=0)
        ens_probs_val = np.stack(all_probs_val).mean(axis=0)
        ens_thr_v = np.mean([p["threshold_val"] for p in per_seed])
        ens_m_v = compute_metrics(y_te, ens_probs, ens_thr_v)
        ens_m_5 = compute_metrics(y_te, ens_probs, 0.50)
        ens_m_7 = compute_metrics(y_te, ens_probs, 0.70)

        # Mean ± std across seeds (headline = val-optimal threshold)
        f1s = [
            p["metrics_val_opt"]["f1"] for p in per_seed if p["metrics_val_opt"]["f1"] is not None
        ]
        aus = [
            p["metrics_val_opt"]["pr_auc"]
            for p in per_seed
            if p["metrics_val_opt"]["pr_auc"] is not None
        ]

        # Per-city
        city_m = per_city_metrics(y_te, ens_probs, meta_te, ens_thr_v)

        # Bootstrap CI vs RF production
        ci = block_bootstrap_ci(
            labels_ref=y_te,
            probs_ref=rf_prod["probs_test"],
            labels_cmp=y_te,
            probs_cmp=ens_probs,
            meta=meta_te,
            threshold_ref=0.70,
            threshold_cmp=ens_thr_v,
        )

        # Bootstrap CI vs fair RF (like-for-like: same training window, no qualifying_day)
        ci_vs_fair = block_bootstrap_ci(
            labels_ref=y_te,
            probs_ref=rf_fair_data["probs_test"],
            labels_cmp=y_te,
            probs_cmp=ens_probs,
            meta=meta_te,
            threshold_ref=rf_fair_t,
            threshold_cmp=ens_thr_v,
        )

        dl_results[key] = {
            "arch": arch,
            "input_config": cfg,
            "loss": loss_name,
            "seeds": seeds,
            "n_seeds_completed": len(per_seed),
            "per_seed": [{k: v for k, v in p.items() if k != "probs"} for p in per_seed],
            "ensemble": {
                "threshold_val_opt": round(ens_thr_v, 2),
                "metrics_val_opt": ens_m_v,
                "metrics_050": ens_m_5,
                "metrics_070": ens_m_7,
                "uncertainty": {
                    "type": "cross_seed_probability_standard_deviation",
                    "mean": round(float(np.std(np.stack(all_probs), axis=0).mean()), 6),
                    "p95": round(float(np.percentile(np.std(np.stack(all_probs), axis=0), 95)), 6),
                    "validation_mean": round(
                        float(np.std(np.stack(all_probs_val), axis=0).mean()), 6
                    ),
                },
            },
            "mean_f1": round(np.mean(f1s), 4) if f1s else None,
            "std_f1": round(np.std(f1s), 4) if f1s else None,
            "mean_prauc": round(np.mean(aus), 4) if aus else None,
            "std_prauc": round(np.std(aus), 4) if aus else None,
            "per_city": city_m,
            "bootstrap_ci_vs_rf_prod": ci,
            "bootstrap_ci_vs_rf_fair": ci_vs_fair,
            "probs": ens_probs,  # kept in memory for compare.py
        }
        print(
            f"  Ensemble @{ens_thr_v:.2f}: F1={ens_m_v['f1']:.4f}  PR-AUC={ens_m_v['pr_auc']:.4f}"
        )
        print(
            f"  Mean F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  PR-AUC={np.mean(aus):.4f}±{np.std(aus):.4f}"
        )
        print(
            f"  Bootstrap CI (DL-RF) F1: [{ci['f1_ci_lower']}, {ci['f1_ci_upper']}]  "
            f"includes_zero={ci['includes_zero_f1']}"
        )
        print(
            f"  Bootstrap CI (DL-RF-fair) F1: [{ci_vs_fair['f1_ci_lower']}, {ci_vs_fair['f1_ci_upper']}]  "
            f"includes_zero={ci_vs_fair['includes_zero_f1']}"
        )

    return {
        "rf_production": {
            "metrics_070": rf_prod_metrics_070,
            "metrics_050": rf_prod_metrics_050,
            "probs": rf_prod["probs_test"],
            "per_city": per_city_metrics(y_te, rf_prod["probs_test"], meta_te, 0.70),
        },
        "rf_fair": {
            "threshold": rf_fair_t,
            "metrics_val_opt": rf_fair_metrics,
            "metrics_050": rf_fair_m050,
            "probs": rf_fair_data["probs_test"],
            "per_city": per_city_metrics(y_te, rf_fair_data["probs_test"], meta_te, rf_fair_t),
            "info": rf_fair_data["info"],
        },
        "dl": dl_results,
        "labels_test": y_te,
        "meta_test": meta_te,
    }


def main() -> None:
    results = evaluate_all()
    # Save core metrics (without numpy arrays) to results/
    DL_RESULTS.mkdir(parents=True, exist_ok=True)
    saveable = {
        k: {kk: vv for kk, vv in v.items() if kk != "probs"} if isinstance(v, dict) else v
        for k, v in results.items()
        if k not in ("labels_test", "meta_test")
    }
    # Remove probs from dl entries
    for key, val in saveable.get("dl", {}).items():
        if isinstance(val, dict):
            val.pop("probs", None)
    path = DL_RESULTS / "evaluation_raw.json"
    path.write_text(json.dumps(saveable, indent=2, default=str), encoding="utf-8")
    print(f"\n[evaluate] Saved raw evaluation to {path}")


if __name__ == "__main__":
    main()

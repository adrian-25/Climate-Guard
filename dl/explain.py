"""dl/explain.py — Feature attribution for the best DL model.

Uses Integrated Gradients (Captum). Known issue: cuDNN RNN backward fails
outside training mode, so we run on CPU with cudnn disabled.

Falls back to permutation importance if Captum fails.

Usage
-----
  python -m dl.explain
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import (
    DL_ARTIFACTS,
    DL_RESULTS,
    FEAT110_EXCLUDE,
    FEATURE_LIST,
    META_TEST_CSV,
    RAW_SEQ_FEATURES,
    ROOT,
    SEQ_LEN_RAW,
    X_TEST_CSV,
    Y_TEST_CSV,
)
from .data import ColumnwiseScaler, HeatwaveSequenceDataset
from .model import build_model

# Best model to explain: GRU raw-seq seed 0 (first seed of headline config)
EXPLAIN_ARCH = "gru"
EXPLAIN_CONFIG = "raw-seq"
EXPLAIN_SEED = 0
FEAT_NAMES = RAW_SEQ_FEATURES
N_EXAMPLES = 10  # high-risk test examples to visualise


def _load_model_and_data():
    tag = f"{EXPLAIN_ARCH}_{EXPLAIN_CONFIG.replace('-','_')}_seed{EXPLAIN_SEED}"
    save_dir = DL_ARTIFACTS / tag
    ckpt_path = save_dir / "best.pt"
    scaler_path = save_dir / "scaler.pkl"
    run_info = json.loads((save_dir / "run_info.json").read_text(encoding="utf-8"))
    n_feat = run_info["n_features"]
    seq_len = run_info["seq_len"]

    device = torch.device("cpu")  # attribution always on CPU

    scaler = ColumnwiseScaler.load(scaler_path)
    X_te_raw = pd.read_csv(X_TEST_CSV)[FEAT_NAMES].to_numpy(dtype=np.float32)
    y_te = pd.read_csv(Y_TEST_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_te = pd.read_csv(META_TEST_CSV)

    from .config import INDIA_CITY_ORDER

    c_te = np.array([INDIA_CITY_ORDER.index(c) for c in m_te["city_key"]], dtype=np.int64)
    Xte = scaler.transform(X_te_raw)

    ds = HeatwaveSequenceDataset(Xte, y_te, c_te, seq_len)
    model = build_model(n_features=n_feat, arch=EXPLAIN_ARCH).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    return model, ds, y_te, m_te, device, seq_len


# ---------------------------------------------------------------------------
# Integrated Gradients wrapper
# ---------------------------------------------------------------------------


class _WrapForIG(nn.Module):
    """Wrapper that takes (x, city) as separate args; IG needs single tensor."""

    def __init__(self, model, city_tensor):
        super().__init__()
        self.model = model
        self.city = city_tensor

    def forward(self, x):
        logit = self.model(x, self.city)
        return torch.sigmoid(logit)


def _integrated_gradients(model, x_batch, city_batch, device):
    """
    Compute IG for each example in x_batch.
    x_batch: (B, seq_len, F)  city_batch: (B,)
    Returns attributions: (B, seq_len, F)
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return None

    x = x_batch.to(device).float().requires_grad_(True)
    c = city_batch.to(device)
    baseline = torch.zeros_like(x)

    attributions_list = []
    for i in range(len(x)):
        xi = x[i : i + 1]
        ci = c[i : i + 1]
        bl = baseline[i : i + 1]
        wrapped = _WrapForIG(model, ci)
        wrapped.eval()
        ig = IntegratedGradients(wrapped)
        # cuDNN RNN backward is broken in eval mode; force CPU+no-cuDNN
        with torch.backends.cudnn.flags(enabled=False):
            attr, _ = ig.attribute(
                xi, bl, target=0, return_convergence_delta=True, n_steps=50, internal_batch_size=1
            )
        attributions_list.append(attr.detach().cpu().numpy())  # (1, seq, F)

    return np.concatenate(attributions_list, axis=0)  # (B, seq, F)


# ---------------------------------------------------------------------------
# Permutation importance fallback
# ---------------------------------------------------------------------------


def _permutation_importance(model, ds, device, n_perm=5):
    """Permute each feature across all windows and measure PR-AUC drop."""
    from sklearn.metrics import average_precision_score
    from torch.utils.data import DataLoader

    from .config import BATCH_SIZE, NUM_WORKERS

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    def _score(dl_inner):
        probs, labs = [], []
        with torch.no_grad():
            for xx, yy, cc in dl_inner:
                xx, cc = xx.to(device), cc.to(device)
                logit = model(xx, cc).squeeze(1)
                probs.append(torch.sigmoid(logit).cpu().numpy())
                labs.append(yy.numpy())
        return average_precision_score(np.concatenate(labs).astype(int), np.concatenate(probs))

    baseline_score = _score(dl)
    # HeatwaveSequenceDataset stores features/labels/cities as private tensors:
    # _target_feat (N, F), _target_labels (N,), _target_cities (N,).
    # seq_len is a public attribute. These are stable internal fields; if the
    # dataset API changes, update here accordingly.
    n_feat = ds._target_feat.shape[1]
    drops = []
    rng = np.random.default_rng(42)
    for fi in range(n_feat):
        fi_drops = []
        for _ in range(n_perm):
            feat_copy = ds._target_feat.clone()
            perm_idx = rng.permutation(len(feat_copy))
            feat_copy[:, fi] = feat_copy[perm_idx, fi]
            perm_ds = HeatwaveSequenceDataset(
                feat_copy.numpy(),
                ds._target_labels.numpy(),
                ds._target_cities.numpy(),
                ds.seq_len,
            )
            perm_dl = DataLoader(
                perm_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
            )
            fi_drops.append(baseline_score - _score(perm_dl))
        drops.append(float(np.mean(fi_drops)))
    return np.array(drops)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    DL_RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"[explain] Loading {EXPLAIN_ARCH} {EXPLAIN_CONFIG} seed {EXPLAIN_SEED}...")
    try:
        model, ds, y_te, m_te, device, seq_len = _load_model_and_data()
    except FileNotFoundError as e:
        print(f"[explain] Checkpoint not found: {e}")
        print("[explain] Run dl.train first.")
        return

    n_feat = ds._target_feat.shape[1]  # see _permutation_importance for field docs
    feat_names = FEAT_NAMES[:n_feat]

    # ── Find N_EXAMPLES high-risk test windows ────────────────────────────────
    from torch.utils.data import DataLoader

    from .config import BATCH_SIZE, NUM_WORKERS

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    all_probs = []
    with torch.no_grad():
        for x, y, c in dl:
            logit = model(x.to(device), c.to(device)).squeeze(1)
            all_probs.append(torch.sigmoid(logit).cpu().numpy())
    all_probs = np.concatenate(all_probs)
    high_idx = np.argsort(all_probs)[::-1][:N_EXAMPLES]

    # ── Try Integrated Gradients ──────────────────────────────────────────────
    ig_success = False
    try:
        from captum.attr import IntegratedGradients

        print("[explain] Running Integrated Gradients on high-risk examples...")
        x_batch = torch.stack([ds[i][0] for i in high_idx])
        c_batch = torch.stack([ds[i][2] for i in high_idx])
        attrs = _integrated_gradients(model, x_batch, c_batch, device)
        if attrs is not None:
            ig_success = True
            # Aggregate over time → (B, F)
            mean_abs_attr = np.abs(attrs).mean(axis=1)  # (B, F)
            global_attr = mean_abs_attr.mean(axis=0)  # (F,)

            # Top-15 feature importance chart
            top15 = np.argsort(global_attr)[::-1][:15]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(
                range(15),
                global_attr[top15][::-1],
                color=["#C8421B" if i < 3 else "#8C8580" for i in range(15)],
            )
            ax.set_yticks(range(15))
            ax.set_yticklabels([feat_names[i] for i in top15][::-1], fontsize=9)
            ax.set_xlabel("Mean |IG attribution| over high-risk examples")
            ax.set_title(
                f"Top 15 features — {EXPLAIN_ARCH.upper()} {EXPLAIN_CONFIG} "
                f"(Integrated Gradients, {N_EXAMPLES} high-risk examples)"
            )
            fig.tight_layout()
            fig.savefig(DL_RESULTS / "dl_feature_importance_ig.png", dpi=150)
            plt.close(fig)
            print(f"[explain] Saved dl_feature_importance_ig.png")

            # Time × feature heatmap for first example
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(
                attrs[0].T,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-np.abs(attrs[0]).max(),
                vmax=np.abs(attrs[0]).max(),
            )
            ax.set_yticks(range(n_feat))
            ax.set_yticklabels(feat_names, fontsize=7)
            ax.set_xlabel("Time step (0 = oldest)")
            ax.set_ylabel("Feature")
            ax.set_title("IG attributions — one high-risk example")
            plt.colorbar(im, ax=ax, label="Attribution")
            fig.tight_layout()
            fig.savefig(DL_RESULTS / "dl_ig_heatmap.png", dpi=150)
            plt.close(fig)
            print(f"[explain] Saved dl_ig_heatmap.png")

            # Save ranked list
            ranked = sorted(zip(feat_names, global_attr.tolist()), key=lambda x: -x[1])
            print("[explain] Top 10 features (IG):")
            for fname, imp in ranked[:10]:
                print(f"  {fname:<40s} {imp:.6f}")

    except Exception as e:
        print(f"[explain] Captum/IG failed: {e}")
        print("[explain] Falling back to permutation importance...")

    if not ig_success:
        print("[explain] Computing permutation importance (fallback)...")
        drops = _permutation_importance(model, ds, device)
        top15 = np.argsort(drops)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            range(15),
            drops[top15][::-1],
            color=["#C8421B" if i < 3 else "#8C8580" for i in range(15)],
        )
        ax.set_yticks(range(15))
        ax.set_yticklabels([feat_names[i] for i in top15][::-1], fontsize=9)
        ax.set_xlabel("PR-AUC drop when feature permuted")
        ax.set_title(
            f"Top 15 features — {EXPLAIN_ARCH.upper()} " f"(Permutation importance fallback)"
        )
        fig.tight_layout()
        fig.savefig(DL_RESULTS / "dl_feature_importance_perm.png", dpi=150)
        plt.close(fig)
        print(f"[explain] Saved dl_feature_importance_perm.png")
        ranked = sorted(zip(feat_names, drops.tolist()), key=lambda x: -x[1])
        print("[explain] Top 10 features (permutation):")
        for fname, imp in ranked[:10]:
            print(f"  {fname:<40s} {imp:.6f}")

    # ── Compare with RF top features (computed, not hardcoded) ──────────────
    print("\n[explain] Computing RF vs DL feature importance comparison...")
    try:
        import joblib
        from scipy.stats import spearmanr

        # Load RF global importances
        rf_model = joblib.load(ROOT / "models" / "final" / "climateguard_final_model.joblib")
        rf_feat_names_raw = [
            e["name"]
            for e in json.loads((ROOT / "models" / "final" / "feature_list.json").read_text())
        ]
        rf_importances = dict(zip(rf_feat_names_raw, rf_model.feature_importances_.tolist()))

        # DL attribution is only for the raw-seq features (a subset of the RF's 110)
        # qualifying_day is excluded from DL but present in RF — can't compare directly.
        # Restrict to DL features that also appear in the RF.
        common = [f for f in feat_names if f in rf_importances]
        if ig_success and len(common) > 0:
            # Build ranked lists on the common subset
            dl_vals = np.array([global_attr[feat_names.index(f)] for f in common])
            rf_vals = np.array([rf_importances[f] for f in common])
            dl_ranks = np.argsort(-dl_vals)  # descending
            rf_ranks = np.argsort(-rf_vals)
            # Spearman rank correlation
            rho, pval = spearmanr(
                [list(dl_ranks).index(i) for i in range(len(common))],
                [list(rf_ranks).index(i) for i in range(len(common))],
            )
            # Top-5 overlap
            dl_top5 = set([common[i] for i in np.argsort(-dl_vals)[:5]])
            rf_top5 = set([common[i] for i in np.argsort(-rf_vals)[:5]])
            overlap = dl_top5 & rf_top5
            print(
                f"  Common features compared: {len(common)} (out of {len(feat_names)} DL / {len(rf_feat_names_raw)} RF)"
            )
            print(f"  Note: qualifying_day is in RF but excluded from DL by design.")
            print(f"  Spearman rank correlation on common features: rho={rho:.3f}  p={pval:.3f}")
            print(f"  DL top-5: {sorted(dl_top5)}")
            print(f"  RF top-5 (restricted to DL features): {sorted(rf_top5)}")
            print(f"  Overlap in top-5: {sorted(overlap)}")
            note = (
                f"Computed on {len(common)} features common to both DL (raw-seq) and RF. "
                f"Spearman rho={rho:.3f} (p={pval:.3f}). "
                f"Top-5 overlap: {sorted(overlap)}. "
                f"Note: qualifying_day excluded from DL; RF relies on it heavily."
            )
        else:
            note = (
                "IG attributions not available or no common features. " "Direct comparison skipped."
            )
        # Save the note to a file for the website/docs
        (DL_RESULTS / "explain_comparison_note.txt").write_text(note, encoding="utf-8")
        print(f"[explain] Saved explain_comparison_note.txt")
    except Exception as ex:
        note = f"Comparison computation failed: {ex}"
        print(f"[explain] WARNING: {note}")


if __name__ == "__main__":
    main()

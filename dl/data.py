"""dl/data.py — dataset and DataLoader construction for the DL comparison module.

Key design decisions
--------------------
* A window ending on test row t uses the previous seq_len-1 rows as INPUT
  context from the preceding split (same city), so that EVERY target row gets
  a real DL prediction. No NaN padding. The (city, date) test key set and
  per-city positive counts exactly match the RF's test set.

* Specifically:
  - Training windows: built within the train split only. The first seq_len-1
    rows of each city do not form windows (fine for training diversity).
  - Val windows: context prefix = last (seq_len-1) rows of each city in train.
    Every val row gets a window.
  - Test windows: context prefix = last (seq_len-1) rows of each city in val.
    Every test row gets a window.
  Input context is INPUTS ONLY — its labels are never exposed.

* Windows never cross city boundaries.
* Scaler fitted on train rows only.
* Label = heatwave_next_day at the LAST timestep of the window (index t).
* qualifying_day and heatwave_lag1 excluded; see dl/config.py for rationale.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    FEAT110_EXCLUDE,
    FEATURE_LIST,
    INDIA_CITY_ORDER,
    META_TEST_CSV,
    META_TRAIN_CSV,
    META_VAL_CSV,
    NUM_WORKERS,
    RAW_SEQ_FEATURES,
    SEQ_LEN_FEAT110,
    SEQ_LEN_RAW,
    X_TEST_CSV,
    X_TRAIN_CSV,
    X_VAL_CSV,
    Y_TEST_CSV,
    Y_TRAIN_CSV,
    Y_VAL_CSV,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_110_feature_names() -> list[str]:
    raw = json.loads(FEATURE_LIST.read_text(encoding="utf-8"))
    return [e["name"] for e in raw]


def _city_to_idx(city_key: str) -> int:
    return INDIA_CITY_ORDER.index(city_key)


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------


class HeatwaveSequenceDataset(Dataset):
    """
    Sliding-window dataset that gives EVERY target row a real prediction.

    For rows near the start of a city's block in the target split that
    don't have enough target-side history, the window is completed with
    context rows from the preceding split (inputs only, no labels).

    Parameters
    ----------
    target_feat    : (N, F) float32 — feature matrix for the target split
    target_labels  : (N,)   float32 — labels for the target split
    target_cities  : (N,)   int64   — city indices for the target split
    seq_len        : window length
    context_feat   : (M, F) float32 — context rows (input only); may be None
    context_cities : (M,)   int64   — city indices for context rows; may be None
    """

    def __init__(
        self,
        target_feat: np.ndarray,
        target_labels: np.ndarray,
        target_cities: np.ndarray,
        seq_len: int,
        context_feat: np.ndarray | None = None,
        context_cities: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # ── Build per-city context prefix (last seq_len-1 context rows) ────────
        # ctx_prefix[city_idx] = np.ndarray of shape (<=seq_len-1, F)
        ctx_prefix: dict[int, np.ndarray] = {}
        if context_feat is not None and context_cities is not None:
            for city_idx in np.unique(target_cities).tolist():
                mask = context_cities == city_idx
                rows = context_feat[mask]
                keep = min(seq_len - 1, len(rows))
                if keep > 0:
                    ctx_prefix[int(city_idx)] = rows[-keep:]
        self._ctx_prefix = ctx_prefix

        # ── Build windows per city ──────────────────────────────────────────────
        # Work city by city so we never cross boundaries.
        # For each city, find its contiguous block(s) in the target split.
        # Within each block, generate one window per row.
        N = len(target_feat)
        # (t_row, n_target_rows, t_start_in_block, city_idx)
        indices: list[tuple[int, int, int, int]] = []

        i = 0
        while i < N:
            city_idx = int(target_cities[i])
            # Find the end of this city's contiguous block
            j = i
            while j < N and target_cities[j] == city_idx:
                j += 1
            # Block is target_feat[i:j] for city city_idx
            prefix = ctx_prefix.get(city_idx, np.empty((0, target_feat.shape[1])))
            n_ctx = len(prefix)

            for local_t in range(j - i):
                global_t = i + local_t  # index into full target array
                # target rows available: target_feat[i : global_t+1] → local_t+1 rows
                n_target_rows = local_t + 1
                n_needed_ctx = seq_len - n_target_rows
                if n_needed_ctx <= 0:
                    # Full window from target only
                    t_start = global_t - seq_len + 1
                    indices.append((global_t, n_target_rows, t_start, city_idx))
                elif n_ctx >= n_needed_ctx:
                    # Need some context, we have enough
                    t_start = i  # target portion starts at block start
                    indices.append((global_t, n_target_rows, t_start, city_idx))
                # else: skip (not enough context — only affects first rows of
                #       train cities where no earlier split exists)

            i = j

        self._indices = indices  # list of (global_t_row, n_target, t_start, city)

        # ── Store tensors ────────────────────────────────────────────────────────
        self._target_feat = torch.from_numpy(target_feat).float()
        self._target_labels = torch.from_numpy(target_labels).float()
        self._target_cities = torch.from_numpy(target_cities).long()
        self._ctx_tensors: dict[int, torch.Tensor] = {
            k: torch.from_numpy(v).float() for k, v in ctx_prefix.items()
        }

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_t, n_target, t_start, city_idx = self._indices[idx]
        n_needed_ctx = self.seq_len - n_target

        # Target portion: t_start..global_t (inclusive)
        t_feat = self._target_feat[t_start : global_t + 1]

        if n_needed_ctx > 0:
            ctx = self._ctx_tensors[city_idx]
            ctx_slice = ctx[-n_needed_ctx:]  # (n_needed_ctx, F)
            x = torch.cat([ctx_slice, t_feat], dim=0)
        else:
            x = t_feat

        y = self._target_labels[global_t]
        c = self._target_cities[global_t]
        return x, y, c

    @property
    def target_row_indices(self) -> list[int]:
        """Global target-array index for each window (for alignment checks)."""
        return [entry[0] for entry in self._indices]


# ---------------------------------------------------------------------------
# Scaler (fitted on train, applied to val/test)
# ---------------------------------------------------------------------------


class ColumnwiseScaler:
    """Standard scaler per feature column (ignores time structure)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ColumnwiseScaler":
        self.mean_ = X.mean(axis=0, keepdims=True).astype(np.float32)
        std = X.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-8] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Scaler not fitted"
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)

    @classmethod
    def load(cls, path: Path) -> "ColumnwiseScaler":
        with open(path, "rb") as f:
            d = pickle.load(f)
        sc = cls()
        sc.mean_ = d["mean"]
        sc.std_ = d["std"]
        return sc


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _get_feat110_cols() -> list[str]:
    all_names = _load_110_feature_names()
    return [n for n in all_names if n not in FEAT110_EXCLUDE]


def _extract_features(X_df: pd.DataFrame, input_config: str) -> np.ndarray:
    if input_config == "raw-seq":
        cols = RAW_SEQ_FEATURES
        missing = [c for c in cols if c not in X_df.columns]
        if missing:
            raise ValueError(f"raw-seq: columns missing: {missing}")
        return X_df[cols].to_numpy(dtype=np.float32)
    elif input_config == "feat110-seq":
        cols = _get_feat110_cols()
        missing = [c for c in cols if c not in X_df.columns]
        if missing:
            raise ValueError(f"feat110-seq: columns missing: {missing}")
        return X_df[cols].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unknown input_config: {input_config!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_splits(
    input_config: str = "raw-seq",
    seq_len: int | None = None,
    batch_size: int = 256,
    refit_scaler: bool = True,
    scaler_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load train/val/test splits, fit a scaler on train, build Datasets.

    Val and test datasets use context rows from the preceding split so
    EVERY row gets a real prediction — no NaN padding, no mismatch with the
    RF's 4,865-row test set.
    """
    if seq_len is None:
        seq_len = SEQ_LEN_RAW if input_config == "raw-seq" else SEQ_LEN_FEAT110

    print(f"[data] Loading splits for config={input_config!r}, seq_len={seq_len}")
    X_tr = pd.read_csv(X_TRAIN_CSV)
    y_tr = pd.read_csv(Y_TRAIN_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_tr = pd.read_csv(META_TRAIN_CSV)

    X_va = pd.read_csv(X_VAL_CSV)
    y_va = pd.read_csv(Y_VAL_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_va = pd.read_csv(META_VAL_CSV)

    X_te = pd.read_csv(X_TEST_CSV)
    y_te = pd.read_csv(Y_TEST_CSV)["heatwave_next_day"].to_numpy(dtype=np.float32)
    m_te = pd.read_csv(META_TEST_CSV)

    # ── Verify test set matches RF ─────────────────────────────────────────────
    assert len(m_te) == 4865, f"Expected 4865 test rows, got {len(m_te)}"
    assert int(y_te.sum()) == 38, f"Expected 38 test positives, got {int(y_te.sum())}"
    _exp = {"delhi": 18, "lucknow": 16, "nagpur": 4, "ahmedabad": 0, "mumbai": 0}
    for city, exp_n in _exp.items():
        got = int(y_te[m_te["city_key"].to_numpy() == city].sum())
        assert got == exp_n, f"{city}: expected {exp_n} test positives, got {got}"

    assert m_tr["date"].max() < m_va["date"].min(), "Train/val overlap"
    assert m_va["date"].max() < m_te["date"].min(), "Val/test overlap"

    print(
        f"[data] Test set: {len(m_te)} rows, "
        f"date range {m_te['date'].min()} to {m_te['date'].max()}, "
        f"positives {int(y_te.sum())}"
    )

    # ── Extract features ────────────────────────────────────────────────────────
    Xtr = _extract_features(X_tr, input_config)
    Xva = _extract_features(X_va, input_config)
    Xte = _extract_features(X_te, input_config)

    feature_names: list[str] = (
        RAW_SEQ_FEATURES if input_config == "raw-seq" else _get_feat110_cols()
    )
    n_features = len(feature_names)
    print(f"[data] Feature count: {n_features}")

    # ── Scale: fit on train only ────────────────────────────────────────────────
    if refit_scaler:
        scaler = ColumnwiseScaler().fit(Xtr)
        if scaler_path is not None:
            scaler.save(scaler_path)
    else:
        assert scaler_path is not None and scaler_path.exists()
        scaler = ColumnwiseScaler.load(scaler_path)

    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # ── City arrays ────────────────────────────────────────────────────────────
    c_tr = np.array([_city_to_idx(c) for c in m_tr["city_key"]], dtype=np.int64)
    c_va = np.array([_city_to_idx(c) for c in m_va["city_key"]], dtype=np.int64)
    c_te = np.array([_city_to_idx(c) for c in m_te["city_key"]], dtype=np.int64)

    # ── Datasets ─────────────────────────────────────────────────────────────
    # Train: no context — first seq_len-1 rows per city get no window (fine).
    train_ds = HeatwaveSequenceDataset(Xtr_s, y_tr, c_tr, seq_len)

    # Val: context = last (seq_len-1) train rows per city (inputs only).
    val_ds = HeatwaveSequenceDataset(
        Xva_s,
        y_va,
        c_va,
        seq_len,
        context_feat=Xtr_s,
        context_cities=c_tr,
    )

    # Test: context = last (seq_len-1) val rows per city (inputs only).
    test_ds = HeatwaveSequenceDataset(
        Xte_s,
        y_te,
        c_te,
        seq_len,
        context_feat=Xva_s,
        context_cities=c_va,
    )

    # Every val and test row must have a window (context ensures this).
    assert len(val_ds) == len(m_va), f"Val: expected {len(m_va)} windows, got {len(val_ds)}"
    assert len(test_ds) == len(m_te), f"Test: expected {len(m_te)} windows, got {len(test_ds)}"

    print(
        f"[data] Windows — train: {len(train_ds):,}  "
        f"val: {len(val_ds):,}  test: {len(test_ds):,}"
    )

    # ── pos_weight (train only, capped) ────────────────────────────────────────
    from .config import POS_WEIGHT_CAP

    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    raw_pw = n_neg / max(n_pos, 1.0)
    pos_weight = min(raw_pw, POS_WEIGHT_CAP)
    print(f"[data] pos_weight (raw={raw_pw:.1f}, capped={pos_weight:.1f})")

    # ── DataLoaders ─────────────────────────────────────────────────────────────
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "test_dl": test_dl,
        "scaler": scaler,
        "feature_names": feature_names,
        "pos_weight": pos_weight,
        "test_meta": m_te[["city_key", "date"]].reset_index(drop=True),
        "val_meta": m_va[["city_key", "date"]].reset_index(drop=True),
        "n_features": n_features,
        "seq_len": seq_len,
    }

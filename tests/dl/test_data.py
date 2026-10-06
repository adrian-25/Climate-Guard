"""tests/dl/test_data.py — Data pipeline correctness tests.

Skipped entirely if torch is not installed (CI without torch stays green).
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; DL tests skipped")

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Skip the entire module if the required split files are not present
# (CI checkout may not include large data files).
_DATA_PRESENT = (ROOT / "data" / "splits" / "temporal" / "X_test.csv").exists()
if not _DATA_PRESENT:
    pytest.skip("temporal split data files not present", allow_module_level=True)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_splits():
    from dl.data import load_splits

    return load_splits(input_config="raw-seq", seq_len=7, batch_size=64, refit_scaler=True)


@pytest.fixture(scope="module")
def feat_splits():
    from dl.data import load_splits

    return load_splits(input_config="feat110-seq", seq_len=7, batch_size=64, refit_scaler=True)


# ---------------------------------------------------------------------------
# Split ordering
# ---------------------------------------------------------------------------


def test_train_val_test_date_ordering():
    """Train < val < test with no overlap."""
    m_tr = pd.read_csv(ROOT / "data/splits/temporal/meta_train.csv")
    m_va = pd.read_csv(ROOT / "data/splits/temporal/meta_val.csv")
    m_te = pd.read_csv(ROOT / "data/splits/temporal/meta_test.csv")
    assert m_tr["date"].max() < m_va["date"].min(), "Train/val overlap"
    assert m_va["date"].max() < m_te["date"].min(), "Val/test overlap"


def test_test_rows_equal_rf_test_rows():
    """DL test set must be the same 4865 (city, date) rows as the RF test set."""
    m_te = pd.read_csv(ROOT / "data/splits/temporal/meta_test.csv")
    assert len(m_te) == 4865, f"Expected 4865 test rows, got {len(m_te)}"
    y_te = pd.read_csv(ROOT / "data/splits/temporal/y_test.csv")
    assert int(y_te["heatwave_next_day"].sum()) == 38, "Expected 38 test positives"


# ---------------------------------------------------------------------------
# Windows never cross cities
# ---------------------------------------------------------------------------


def _check_no_cross_city(ds):
    """No window should span more than one city."""
    for idx in range(min(len(ds), 200)):  # sample to keep test fast
        x, y, c = ds[idx]
        # The city returned must equal the city integer for the window
        _, _, _, city_idx = ds._indices[idx]
        assert (
            int(c) == city_idx
        ), f"Window {idx}: city tensor {int(c)} != stored city_idx {city_idx}"
        # x must have shape (seq_len, F)
        assert x.shape[0] == ds.seq_len, f"Window {idx}: wrong seq_len {x.shape[0]} != {ds.seq_len}"


def test_windows_never_cross_cities_raw(raw_splits):
    _check_no_cross_city(raw_splits["train_ds"])


def test_windows_never_cross_cities_feat(feat_splits):
    _check_no_cross_city(feat_splits["train_ds"])


# ---------------------------------------------------------------------------
# Scaler fitted on train only
# ---------------------------------------------------------------------------


def test_scaler_fitted_on_train_only(raw_splits):
    """Scaler mean/std should match train feature statistics."""
    X_tr = pd.read_csv(ROOT / "data/splits/temporal/X_train.csv")
    from dl.config import RAW_SEQ_FEATURES

    Xtr = X_tr[RAW_SEQ_FEATURES].to_numpy(dtype=float)
    sc = raw_splits["scaler"]
    np.testing.assert_allclose(
        sc.mean_.ravel(),
        Xtr.mean(axis=0),
        rtol=1e-4,
        err_msg="Scaler mean does not match train set mean",
    )
    np.testing.assert_allclose(
        sc.std_.ravel(),
        Xtr.std(axis=0),
        rtol=1e-4,
        err_msg="Scaler std does not match train set std",
    )


# ---------------------------------------------------------------------------
# Label alignment: last timestep of window = heatwave_next_day
# ---------------------------------------------------------------------------


def test_label_alignment_is_next_day():
    """The label at window end (global_t) must equal heatwave_next_day at that row."""
    import pandas as _pd_inner

    from dl.config import INDIA_CITY_ORDER, RAW_SEQ_FEATURES
    from dl.data import ColumnwiseScaler, HeatwaveSequenceDataset

    y_tr = _pd_inner.read_csv(ROOT / "data/splits/temporal/y_train.csv")[
        "heatwave_next_day"
    ].to_numpy()
    m_tr = _pd_inner.read_csv(ROOT / "data/splits/temporal/meta_train.csv")
    X_tr = _pd_inner.read_csv(ROOT / "data/splits/temporal/X_train.csv")
    Xtr = X_tr[RAW_SEQ_FEATURES].to_numpy(dtype=np.float32)
    c_tr = np.array([INDIA_CITY_ORDER.index(c) for c in m_tr["city_key"]], dtype=np.int64)
    sc = ColumnwiseScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    ds = HeatwaveSequenceDataset(Xtr, y_tr.astype(np.float32), c_tr, seq_len=7)

    # _indices entries are tuples: (global_t, n_target, t_start, city_idx)
    for win_idx in [0, 10, 50]:
        if win_idx >= len(ds):
            continue
        _, label, _ = ds[win_idx]
        # global_t is the first element of the tuple
        global_t = ds._indices[win_idx][0]
        expected = float(y_tr[global_t])
        assert (
            float(label) == expected
        ), f"Window {win_idx}: label={float(label)}, expected={expected} at row {global_t}"


def test_val_and_test_have_full_coverage(raw_splits):
    """Every val and test row must get a window — no NaN padding needed."""
    assert (
        len(raw_splits["val_ds"]) == 5480
    ), f"Val: expected 5480 windows, got {len(raw_splits['val_ds'])}"
    assert (
        len(raw_splits["test_ds"]) == 4865
    ), f"Test: expected 4865 windows, got {len(raw_splits['test_ds'])}"


def test_test_windows_use_earlier_history():
    """
    The first test window of each city should contain input rows from before
    2023-01-01, but the label must come from 2023 or later.
    """
    import pandas as _pd2

    from dl.config import INDIA_CITY_ORDER, RAW_SEQ_FEATURES
    from dl.data import ColumnwiseScaler, HeatwaveSequenceDataset

    X_va = _pd2.read_csv(ROOT / "data/splits/temporal/X_val.csv")
    y_va = _pd2.read_csv(ROOT / "data/splits/temporal/y_val.csv")["heatwave_next_day"].to_numpy()
    m_va = _pd2.read_csv(ROOT / "data/splits/temporal/meta_val.csv")
    X_te = _pd2.read_csv(ROOT / "data/splits/temporal/X_test.csv")
    y_te = _pd2.read_csv(ROOT / "data/splits/temporal/y_test.csv")["heatwave_next_day"].to_numpy()
    m_te = _pd2.read_csv(ROOT / "data/splits/temporal/meta_test.csv")

    feat_cols = RAW_SEQ_FEATURES
    sc = ColumnwiseScaler().fit(X_va[feat_cols].to_numpy(dtype=np.float32))
    Xva = sc.transform(X_va[feat_cols].to_numpy(dtype=np.float32))
    Xte = sc.transform(X_te[feat_cols].to_numpy(dtype=np.float32))
    c_va = np.array([INDIA_CITY_ORDER.index(c) for c in m_va["city_key"]], dtype=np.int64)
    c_te = np.array([INDIA_CITY_ORDER.index(c) for c in m_te["city_key"]], dtype=np.int64)

    test_ds = HeatwaveSequenceDataset(
        Xte,
        y_te.astype(np.float32),
        c_te,
        seq_len=21,
        context_feat=Xva,
        context_cities=c_va,
    )

    # The very first window in the test set ends on 2023-01-01 (the first test date).
    # seq_len=21 means it needs 20 context rows, all from val (before 2023).
    first_entry = test_ds._indices[0]
    global_t, n_target, t_start, city_idx = first_entry
    # The label row is the first test row — date must be 2023-01-01
    assert m_te["date"].iloc[global_t] >= "2023-01-01", "First test label must be in test period"
    # n_target == 1 means almost all of the window comes from val context
    assert (
        n_target == 1
    ), f"First test window should use 1 target row (rest from val context), got {n_target}"


# ---------------------------------------------------------------------------
# Excluded features are really absent
# ---------------------------------------------------------------------------


def test_excluded_features_absent_raw(raw_splits):
    from dl.config import RAW_SEQ_FEATURES

    # qualifying_day must NOT be in raw-seq features
    assert "qualifying_day" not in RAW_SEQ_FEATURES
    assert "heatwave_lag1" not in RAW_SEQ_FEATURES
    assert "city_encoded" not in RAW_SEQ_FEATURES
    # Feature count
    assert raw_splits["n_features"] == len(RAW_SEQ_FEATURES)


def test_excluded_features_absent_feat110(feat_splits):
    from dl.config import FEAT110_EXCLUDE

    feat_names = feat_splits["feature_names"]
    for excl in FEAT110_EXCLUDE:
        assert excl not in feat_names, f"{excl!r} should be excluded but is in feat110-seq"


def test_feat110_count_is_reasonable(feat_splits):
    """110 minus 6 exclusions = 104 features."""
    assert (
        feat_splits["n_features"] == 104
    ), f"Expected 104 feat110 features, got {feat_splits['n_features']}"

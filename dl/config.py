"""dl/config.py — centralised configuration for the DL comparison module."""

from __future__ import annotations

import os
from pathlib import Path

# NOTE: torch is NOT imported at the module level here — it would be imported
# during pytest collection even for tests that skip on no-torch, and the
# cuda.is_available() call can hang on Windows without admin CUDA drivers.
# Instead, torch is imported lazily inside get_device() and in the modules
# that actually need it (model.py, train.py, etc.).

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# Input data (never written to)
LABELLED_CSV = ROOT / "data" / "processed" / "weather_labelled.csv"
FEATURE_CSV = ROOT / "data" / "features" / "ml_temporal.csv"  # 110-feature ML dataset
X_TRAIN_CSV = ROOT / "data" / "splits" / "temporal" / "X_train.csv"
Y_TRAIN_CSV = ROOT / "data" / "splits" / "temporal" / "y_train.csv"
META_TRAIN_CSV = ROOT / "data" / "splits" / "temporal" / "meta_train.csv"
X_VAL_CSV = ROOT / "data" / "splits" / "temporal" / "X_val.csv"
Y_VAL_CSV = ROOT / "data" / "splits" / "temporal" / "y_val.csv"
META_VAL_CSV = ROOT / "data" / "splits" / "temporal" / "meta_val.csv"
X_TEST_CSV = ROOT / "data" / "splits" / "temporal" / "X_test.csv"
Y_TEST_CSV = ROOT / "data" / "splits" / "temporal" / "y_test.csv"
META_TEST_CSV = ROOT / "data" / "splits" / "temporal" / "meta_test.csv"
FEATURE_LIST = ROOT / "models" / "final" / "feature_list.json"
PROD_MODEL = ROOT / "models" / "final" / "climateguard_final_model.joblib"

# Output (written only by dl/ scripts)
DL_ARTIFACTS = ROOT / "dl" / "artifacts"
DL_RESULTS = ROOT / "dl" / "results"


# ── Device ───────────────────────────────────────────────────────────────────
# Stored as a string so config.py never imports torch at module level.
# Each module that needs a torch.device does: torch.device(DEVICE_STR)
DEVICE_STR: str = "cpu"  # overridden at runtime by get_device()


def get_device_str() -> str:
    """Return 'cuda' if CUDA is available, else 'cpu'. Lazy-imports torch."""
    import torch  # noqa: PLC0415

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ── Random seeds ──────────────────────────────────────────────────────────────
SEEDS: list[int] = [0, 1, 2, 3, 4]

# ── Sequence window ──────────────────────────────────────────────────────────
SEQ_LEN_RAW = 21  # raw-seq input config
SEQ_LEN_FEAT110 = 7  # feat110-seq input config

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 60
LR = 3e-4
PATIENCE = 8  # early stopping on val PR-AUC
POS_WEIGHT_CAP = 50.0  # cap for BCEWithLogitsLoss pos_weight (avoid extreme values)
# One pre-registered rare-event ablation. Its gamma is fixed before looking at
# test results; selection continues to use validation PR-AUC only.
FOCAL_GAMMA = 2.0
NUM_WORKERS = 0  # Windows-safe (no forking)

# ── Architecture ─────────────────────────────────────────────────────────────
GRU_HIDDEN = 64
LSTM_HIDDEN = 64
NUM_LAYERS = 1
DROPOUT = 0.2
CITY_EMBED_DIM = 4  # learned city embedding size
MLP_HIDDEN = 32

# ── City encoding ─────────────────────────────────────────────────────────────
# Ordered list used for the city embedding (0-indexed integer → city)
INDIA_CITY_ORDER: list[str] = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
NUM_INDIA_CITIES = len(INDIA_CITY_ORDER)

# ── Input feature configs ─────────────────────────────────────────────────────

# Config A — raw-seq
# 15 ERA5 daily vars + tmax_normal + tmax_departure + month_sin/cos + doy_sin/cos
# City identity handled by the learned embedding, NOT as a column.
# Excluded: qualifying_day (structural correlation with target label — documented
#           in docs/heatwave_labeling_methodology.md and MODEL_CARD.md),
#           heatwave_lag1 (derived from the heatwave label itself; would leak
#           label-correlated info across the sequence boundary),
#           city_encoded / latitude / longitude / is_coastal (replaced by embedding).
RAW_SEQ_FEATURES: list[str] = [
    # 15 ERA5 weather variables
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    # Derived from ERA5 (not from the label)
    "tmax_normal",
    "tmax_departure",
    # Calendar cyclical encodings
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

# Config B — feat110-seq
# The 110 engineered features minus the exclusions listed above.
# Load dynamically from FEATURE_LIST in data.py so the set stays in sync
# with the frozen contract.
FEAT110_EXCLUDE: set[str] = {
    "qualifying_day",  # structural correlation with heatwave label
    "heatwave_lag1",  # derived from the label; leaks label information
    "city_encoded",  # replaced by learned embedding
    "latitude",  # replaced by embedding
    "longitude",  # replaced by embedding
    "is_coastal",  # replaced by embedding
}

# ── Baseline RF hyperparameters (must match production exactly) ───────────────
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES = 10
RF_SEED = 42
# The "fair" RF is trained on train rows only (≤ 2019-12-31), without
# qualifying_day and heatwave_lag1, with threshold chosen on validation.
# This makes it comparable to the DL models.
RF_FAIR_EXCLUDE: set[str] = {
    "qualifying_day",
    "heatwave_lag1",
    "city_encoded",
    "latitude",
    "longitude",
    "is_coastal",
}

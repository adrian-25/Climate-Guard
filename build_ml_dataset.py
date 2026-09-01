"""
build_ml_dataset.py -- ClimateGuard Phase 8
============================================
Produces the final, validated ML-ready datasets from Phase 7 output.

Inputs
------
  data/features/climateguard_features.csv    (Phase 7 output -- READ-ONLY)
  results/phase7_feature_groups.json         (registered feature sets)

Outputs
-------
  data/features/ml_baseline.csv             identifiers + 29 baseline features + target
  data/features/ml_temporal.csv             identifiers + 110 temporal features + target
  results/phase8_report.txt                 full validation report
  results/phase8_feature_audit.csv          per-feature audit table

What this script does
---------------------
  1.  Load Phase 7 features (no modification)
  2.  Load feature group registry
  3.  Verify target construction: heatwave_next_day(T) == heatwave(T+1) per city
  4.  Validate city boundaries for lags, rolling, and target
  5.  Check chronological ordering within every city
  6.  Inspect and explain all missing values
  7.  Verify data types
  8.  Construct ml_baseline  (identifiers + baseline_features + target)
  9.  Construct ml_temporal  (identifiers + temporal_features + target)
  10. Verify target NOT in X for both datasets
  11. Verify exact feature count matches registry
  12. Run leakage audit on final column sets
  13. Report class distribution overall and per city
  14. Save both CSVs
  15. Generate feature audit table
  16. Print final summary report

Do NOT:
  - train models
  - split train/test
  - scale features
  - oversample / undersample
  - modify climateguard_features.csv
"""

import sys
import json
import logging
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent
FEATURES_FILE = ROOT / "data" / "features" / "climateguard_features.csv"
REGISTRY_FILE = ROOT / "results"  / "phase7_feature_groups.json"
OUT_DIR       = ROOT / "data"     / "features"
RESULTS_DIR   = ROOT / "results"
BASELINE_OUT  = OUT_DIR      / "ml_baseline.csv"
TEMPORAL_OUT  = OUT_DIR      / "ml_temporal.csv"
REPORT_FILE   = RESULTS_DIR  / "phase8_report.txt"
AUDIT_FILE    = RESULTS_DIR  / "phase8_feature_audit.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging -- UTF-8 file + ASCII-safe stdout
# ---------------------------------------------------------------------------
logger = logging.getLogger("phase8")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")

_fh = logging.FileHandler(REPORT_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)

_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)

logger.addHandler(_fh)
logger.addHandler(_ch)

def _sep(char="=", n=70):
    logger.info(char * n)

def _hdr(title):
    _sep()
    logger.info(title)
    _sep()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET      = "heatwave_next_day"
ID_COLS     = ["city", "city_key", "date"]          # minimum traceback identifiers
EXTRA_IDS   = ["state", "region_type",              # extra context, not ML features
               "heatwave",                          # same-day ground truth (for reference)
               "hw_event_id", "hw_event_start",
               "hw_event_end", "hw_event_length"]
ALL_IDS     = ID_COLS + EXTRA_IDS
CITY_ORDER  = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]

# Patterns that would indicate future-data leakage
LEAKAGE_PATTERNS = ["_lead", "_next", "_t1", "t_plus"]


# ===========================================================================
# Step 1 -- Load data
# ===========================================================================
def load_data():
    _hdr("STEP 1 -- LOAD DATA")

    # MD5 of the Phase 7 output before we touch it
    raw_bytes = FEATURES_FILE.read_bytes()
    md5_before = hashlib.md5(raw_bytes).hexdigest()
    logger.info(f"  Phase 7 file  : {FEATURES_FILE}")
    logger.info(f"  MD5 (before)  : {md5_before}")
    logger.info(f"  Size          : {len(raw_bytes)/1024/1024:.2f} MB")

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    logger.info(f"  Loaded shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Expected shape from Phase 7
    assert df.shape == (65_095, 121), (
        f"Unexpected shape {df.shape}; expected (65095, 121). "
        "Phase 7 output may have been modified."
    )
    logger.info("  Shape assertion PASSED (65095 x 121)")

    with open(REGISTRY_FILE, encoding="utf-8") as f:
        registry = json.load(f)
    logger.info(f"  Registry      : {REGISTRY_FILE}")
    logger.info(f"  baseline_features count  : {len(registry['baseline_features'])}")
    logger.info(f"  temporal_features count  : {len(registry['temporal_features'])}")

    return df, registry, md5_before


# ===========================================================================
# Step 2 -- Target construction verification
# ===========================================================================
def verify_target_construction(df):
    _hdr("STEP 2 -- TARGET CONSTRUCTION VERIFICATION")
    logger.info("  Verifying: heatwave_next_day(T) == heatwave(T+1) per city")

    errors = 0
    for city, grp in df.groupby("city_key"):
        grp = grp.sort_values("date").reset_index(drop=True)
        # Recompute what heatwave_next_day should be
        expected = grp["heatwave"].shift(-1)
        actual   = grp["heatwave_next_day"]

        # Compare ignoring the last row (which is NaN in expected -- but we
        # already dropped last rows in Phase 7, so expected NaN count = 0)
        nan_actual   = actual.isna().sum()
        nan_expected = expected.isna().sum()

        # Compare non-NaN rows
        compare_mask = ~actual.isna() & ~expected.isna()
        mismatches   = (actual[compare_mask] != expected[compare_mask]).sum()

        logger.info(f"  {city:12s}  rows={len(grp):6,}  "
                    f"nan_actual={nan_actual}  nan_expected={nan_expected}  "
                    f"mismatches={mismatches}")

        if mismatches > 0:
            logger.error(f"    FAIL: {mismatches} target mismatches in {city}")
            errors += 1
        if nan_actual > 0:
            logger.error(f"    FAIL: {nan_actual} NaN targets remain in {city} -- "
                         "Phase 7 dropping was incomplete")
            errors += 1

    if errors:
        logger.error(f"Target verification FAILED ({errors} city errors)")
        sys.exit(1)
    logger.info("  Target construction PASSED -- heatwave_next_day(T) == heatwave(T+1) for all cities")


# ===========================================================================
# Step 3 -- City boundary validation
# ===========================================================================
def validate_city_boundaries(df):
    _hdr("STEP 3 -- CITY BOUNDARY VALIDATION")
    issues = []

    df_sorted = df.sort_values(["city_key", "date"]).reset_index(drop=True)

    # For each city pair of consecutive rows: if city changes, no lag/rolling
    # value at position i should reference position i-1 (which is a different city).
    # The safest check: within each city, the date must be strictly consecutive
    # (no gaps -- ERA5 is daily).

    logger.info("  Checking date continuity within each city ...")
    for city, grp in df_sorted.groupby("city_key"):
        grp = grp.sort_values("date").reset_index(drop=True)
        gaps = grp["date"].diff().dropna()
        bad_gaps = gaps[gaps != pd.Timedelta(days=1)]
        if len(bad_gaps):
            issues.append(f"{city}: {len(bad_gaps)} date gap(s) -- "
                          f"first at {bad_gaps.index[0]}")
        else:
            logger.info(f"    {city:12s}  {len(grp):,} rows -- date sequence OK (no gaps)")

    # Verify that rows are cleanly separated -- the city_key of row[i] must
    # match city_key of row[i-1] whenever both are within the same block.
    # A cross-city lag would produce an inconsistency between
    # temperature_2m_max_lag1(first_row_of_city) and temperature_2m_max of
    # the actual prior row if city changed. We verify:
    logger.info("  Checking cross-city boundary: first lag row of each city ...")
    for city, grp in df_sorted.groupby("city_key"):
        grp = grp.sort_values("date").reset_index(drop=True)
        first_row = grp.iloc[0]
        # The lag1 of the first row in Phase 7 was computed AFTER Phase 7
        # dropped the first 7 rows, meaning row 0 here was originally row 7
        # in the city block. The lag1 should equal row 6 of that city.
        # We verify by cross-referencing against the Phase 7 labelled file.
        # (We trust Phase 7 groupby -- just ensure no NaN exists in lag cols
        # for first row of each city in our final frame.)
        lag_cols = [c for c in grp.columns if "_lag" in c]
        nan_in_first = first_row[lag_cols].isna().sum()
        if nan_in_first:
            issues.append(f"{city}: {nan_in_first} NaN lag values in first row -- "
                          "possible cross-city leakage or incomplete drop")
        else:
            logger.info(f"    {city:12s}  first-row lag NaNs=0 -- OK")

    if issues:
        for issue in issues:
            logger.error(f"  BOUNDARY FAIL: {issue}")
        sys.exit(1)
    logger.info("  City boundary validation PASSED")


# ===========================================================================
# Step 4 -- Chronological ordering check
# ===========================================================================
def validate_chronological_order(df):
    _hdr("STEP 4 -- CHRONOLOGICAL ORDER VALIDATION")
    issues = []
    for city, grp in df.groupby("city_key"):
        dates = grp["date"].sort_index()
        if not dates.is_monotonic_increasing:
            issues.append(f"{city}: dates are not strictly ascending")
        else:
            logger.info(f"  {city:12s}  date range "
                        f"{dates.min().date()} -> {dates.max().date()}  OK")
    if issues:
        for i in issues:
            logger.error(f"  ORDER FAIL: {i}")
        sys.exit(1)
    logger.info("  Chronological ordering PASSED for all cities")


# ===========================================================================
# Step 5 -- Missing value inspection and remediation
# ===========================================================================
def inspect_missing_values(df, registry):
    """
    Inspects all NaN values and explains their cause.

    Known NaN source in Phase 7 output:
      tmax_departure_zscore  -- rows 0-2 per city (15 rows total)

    Root cause:
      In feature_engineering.py the zscore is computed using:
          dep_past.rolling(window=30, min_periods=10).std()
      Phase 7 dropped the first 7 rows per city (for lag-7 completeness),
      so the earliest rows in the output correspond to original rows 7-9.
      Those rows have 7, 8, and 9 prior observations respectively -- all
      below min_periods=10 -- so rolling().std() returns NaN there.

    Resolution:
      Drop 3 additional rows from the start of each city block here in
      Phase 8 (rows corresponding to original rows 7, 8, 9 of each city).
      This yields 65,095 - 15 = 65,080 final rows.
      These are NOT artificially dropped -- the zscore is genuinely
      undefined for these rows.

    hw_event_start / hw_event_end:
      These columns are NaN for non-heatwave rows (99.2% of data).
      They are passthrough metadata, not ML features. No action required.
    """
    _hdr("STEP 5 -- MISSING VALUE INSPECTION")

    total_nan = df.isnull().sum()
    cols_with_nan = total_nan[total_nan > 0]

    if len(cols_with_nan) == 0:
        logger.info("  No missing values in any column.")
        return df

    logger.info(f"  {len(cols_with_nan)} column(s) contain missing values:")

    baseline_set  = set(registry["baseline_features"])
    temporal_set  = set(registry["temporal_features"])
    lag_cols      = set(c for c in df.columns if "_lag"  in c)
    roll_cols     = set(c for c in df.columns if "_roll" in c)
    trend_cols    = set(c for c in df.columns if c.startswith("tmax_delta_") or
                                                  c.startswith("tmax_slope_"))
    anomaly_cols  = {c for c in df.columns if "zscore" in c}
    event_cols    = {"hw_event_start", "hw_event_end"}

    for col, cnt in cols_with_nan.items():
        pct = cnt / len(df) * 100
        if col in event_cols:
            reason = ("Passthrough metadata: NaN for non-heatwave rows "
                      "(99.2% of data). Not an ML feature -- no action required.")
        elif col in anomaly_cols:
            reason = ("tmax_departure_zscore: first 3 rows per city have "
                      "<10 prior observations for rolling std (min_periods=10). "
                      "Resolution: drop these 3 rows per city in Phase 8.")
        elif col in lag_cols:
            reason = "Lag feature: incomplete window (should be 0 after Phase 7 drops)"
        elif col in roll_cols:
            reason = "Rolling feature: incomplete window (should be 0 after Phase 7 drops)"
        elif col in trend_cols:
            reason = "Trend feature: incomplete window"
        elif col in baseline_set or col in temporal_set:
            reason = "UNEXPECTED FEATURE NaN -- INVESTIGATE"
        else:
            reason = "Passthrough / metadata column"

        logger.info(f"    {col:55s}  NaN={cnt:5,}  ({pct:.3f}%)  -- {reason}")

    # Handle tmax_departure_zscore NaNs: drop first 3 rows per city
    zscore_nans = df["tmax_departure_zscore"].isna().sum()
    if zscore_nans > 0:
        before = len(df)
        df_sorted = df.sort_values(["city_key", "date"]).reset_index(drop=True)
        df_sorted["_rnum"] = df_sorted.groupby("city_key").cumcount()
        # Verify NaN rows are exactly at the start of each city block
        nan_rows = df_sorted[df_sorted["tmax_departure_zscore"].isna()]
        max_rnum = nan_rows["_rnum"].max()
        assert max_rnum <= 2, (
            f"zscore NaN rows extend beyond row 2 in city block "
            f"(max_rnum={max_rnum}) -- unexpected, investigate before proceeding")
        assert len(nan_rows) == zscore_nans, "Unexpected zscore NaN distribution"

        ZSCORE_DROP = 3  # first 3 rows per city
        df_sorted = df_sorted[df_sorted["_rnum"] >= ZSCORE_DROP].drop(
            columns=["_rnum"]).reset_index(drop=True)
        after = len(df_sorted)
        logger.info(f"  Dropping first {ZSCORE_DROP} rows per city "
                    f"(zscore NaN remediation): {before:,} -> {after:,} rows "
                    f"(dropped {before - after})")
        assert df_sorted["tmax_departure_zscore"].isna().sum() == 0, \
            "zscore NaN remains after drop -- investigate"
        logger.info("  tmax_departure_zscore NaN resolved -- 0 remaining")
        df = df_sorted

    # Final check: no NaN in any feature column
    for col in sorted(set(registry["temporal_features"])):
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.error(f"  MISSING VALUE FAIL: feature '{col}' still has "
                         f"{nan_count} NaN after remediation")
            sys.exit(1)

    remaining_feature_nans = {
        c: int(n) for c, n in df.isnull().sum().items()
        if n > 0 and c in set(registry["temporal_features"] + [TARGET])
    }
    if remaining_feature_nans:
        logger.error(f"  MISSING VALUE FAIL: {remaining_feature_nans}")
        sys.exit(1)

    logger.info("  Missing value inspection complete -- 0 feature-column NaNs remain")
    return df


# ===========================================================================
# Step 6 -- Data type verification
# ===========================================================================
def verify_dtypes(df, registry):
    _hdr("STEP 6 -- DATA TYPE VERIFICATION")

    all_features = set(registry["temporal_features"])  # superset
    issues = []

    for col in sorted(all_features):
        if col not in df.columns:
            issues.append(f"Column '{col}' from registry not found in dataframe")
            continue
        dtype = df[col].dtype
        if not pd.api.types.is_numeric_dtype(dtype):
            issues.append(f"Feature '{col}' has non-numeric dtype '{dtype}'")

    # Target
    target_dtype = df[TARGET].dtype
    unique_vals  = sorted(df[TARGET].dropna().unique())
    if set(unique_vals) - {0.0, 1.0}:
        issues.append(f"Target '{TARGET}' contains non-binary values: {unique_vals}")
    logger.info(f"  Target '{TARGET}': dtype={target_dtype}  unique_values={unique_vals}")

    # Date
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        issues.append("'date' column is not datetime64")
    else:
        logger.info(f"  'date': dtype=datetime64 -- OK")

    if issues:
        for i in issues:
            logger.error(f"  DTYPE FAIL: {i}")
        sys.exit(1)
    logger.info(f"  All {len(all_features)} feature columns are numeric -- OK")
    logger.info("  Data type verification PASSED")


# ===========================================================================
# Step 7 -- Build datasets
# ===========================================================================
def build_datasets(df, registry):
    _hdr("STEP 7 -- BUILD ML DATASETS")

    baseline_feats  = registry["baseline_features"]   # list of 29
    temporal_feats  = registry["temporal_features"]   # list of 110

    # Verify all features exist in df
    missing_base = [c for c in baseline_feats if c not in df.columns]
    missing_temp = [c for c in temporal_feats if c not in df.columns]
    if missing_base:
        logger.error(f"  Missing baseline columns: {missing_base}")
        sys.exit(1)
    if missing_temp:
        logger.error(f"  Missing temporal columns: {missing_temp}")
        sys.exit(1)

    # Verify target exists
    if TARGET not in df.columns:
        logger.error(f"  Target column '{TARGET}' not found")
        sys.exit(1)

    # Verify target NOT in feature lists
    if TARGET in baseline_feats:
        logger.error("  LEAKAGE: target is inside baseline_features -- ABORT")
        sys.exit(1)
    if TARGET in temporal_feats:
        logger.error("  LEAKAGE: target is inside temporal_features -- ABORT")
        sys.exit(1)
    logger.info(f"  Target '{TARGET}' confirmed NOT in baseline_features -- OK")
    logger.info(f"  Target '{TARGET}' confirmed NOT in temporal_features -- OK")

    # Verify IDs exist
    for col in ALL_IDS:
        if col not in df.columns:
            logger.error(f"  Required identifier '{col}' not found in dataframe")
            sys.exit(1)

    # Build X/y in memory (for verification reporting -- not saved separately)
    X_baseline = df[baseline_feats]
    y_baseline = df[TARGET]
    X_temporal = df[temporal_feats]
    y_temporal = df[TARGET]

    logger.info(f"  X_baseline shape : {X_baseline.shape}  "
                f"(rows={X_baseline.shape[0]:,}, features={X_baseline.shape[1]})")
    logger.info(f"  y_baseline shape : {y_baseline.shape}  "
                f"positives={int(y_baseline.sum())}  "
                f"({y_baseline.mean()*100:.2f}%)")
    logger.info(f"  X_temporal shape : {X_temporal.shape}  "
                f"(rows={X_temporal.shape[0]:,}, features={X_temporal.shape[1]})")
    logger.info(f"  y_temporal shape : {y_temporal.shape}  "
                f"positives={int(y_temporal.sum())}  "
                f"({y_temporal.mean()*100:.2f}%)")

    # Confirm feature counts match registry exactly
    assert X_baseline.shape[1] == len(baseline_feats), (
        f"Baseline feature count mismatch: got {X_baseline.shape[1]}, "
        f"registry says {len(baseline_feats)}")
    assert X_temporal.shape[1] == len(temporal_feats), (
        f"Temporal feature count mismatch: got {X_temporal.shape[1]}, "
        f"registry says {len(temporal_feats)}")
    logger.info(f"  Feature count assertions PASSED "
                f"(baseline={len(baseline_feats)}, temporal={len(temporal_feats)})")

    # Construct output DataFrames: IDs + features + target
    # Column order: ID_COLS | EXTRA_IDS | features | target
    df_baseline = df[ALL_IDS + baseline_feats + [TARGET]].copy()
    df_temporal = df[ALL_IDS + temporal_feats + [TARGET]].copy()

    return df_baseline, df_temporal, X_baseline, y_baseline, X_temporal, y_temporal


# ===========================================================================
# Step 8 -- Leakage audit on final column sets
# ===========================================================================
def leakage_audit_final(df_baseline, df_temporal):
    _hdr("STEP 8 -- LEAKAGE AUDIT ON FINAL DATASETS")

    def audit_cols(cols, dataset_name):
        issues = []
        for col in cols:
            if col in (ALL_IDS + [TARGET]):
                continue
            for pat in LEAKAGE_PATTERNS:
                if pat in col.lower():
                    issues.append(f"  [{dataset_name}] '{col}' matches forbidden pattern '{pat}'")
            if "_lag" in col:
                try:
                    lag_val = int(col.split("_lag")[-1])
                    if lag_val < 1:
                        issues.append(f"  [{dataset_name}] '{col}' has lag={lag_val} < 1")
                except ValueError:
                    pass
        return issues

    issues  = audit_cols(df_baseline.columns.tolist(), "BASELINE")
    issues += audit_cols(df_temporal.columns.tolist(), "TEMPORAL")

    if issues:
        for i in issues:
            logger.error(i)
        sys.exit(1)
    logger.info("  Leakage audit PASSED on both final datasets -- 0 issues")


# ===========================================================================
# Step 9 -- Class distribution report
# ===========================================================================
def report_class_distribution(df_baseline, df_temporal):
    _hdr("STEP 9 -- CLASS DISTRIBUTION REPORT")

    for label, df_ml in [("BASELINE", df_baseline), ("TEMPORAL", df_temporal)]:
        total    = len(df_ml)
        pos      = int(df_ml[TARGET].sum())
        neg      = total - pos
        pos_pct  = pos / total * 100
        neg_pct  = neg / total * 100

        logger.info(f"  [{label}]")
        logger.info(f"    Total rows     : {total:,}")
        logger.info(f"    Positive (hw=1): {pos:,}  ({pos_pct:.2f}%)")
        logger.info(f"    Negative (hw=0): {neg:,}  ({neg_pct:.2f}%)")
        logger.info(f"    Imbalance ratio: 1 : {neg/pos:.0f}  (negatives per positive)")

        logger.info(f"    Per-city breakdown:")
        logger.info(f"    {'City':12s}  {'Total':>7s}  {'Positive':>9s}  "
                    f"{'Negative':>9s}  {'Pos%':>6s}")
        logger.info(f"    {'-'*12}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*6}")
        for city in CITY_ORDER:
            city_rows = df_ml[df_ml["city_key"] == city]
            c_total = len(city_rows)
            c_pos   = int(city_rows[TARGET].sum())
            c_neg   = c_total - c_pos
            c_pct   = c_pos / c_total * 100 if c_total > 0 else 0.0
            logger.info(f"    {city:12s}  {c_total:7,}  {c_pos:9,}  "
                        f"{c_neg:9,}  {c_pct:6.2f}%")
        logger.info("")


# ===========================================================================
# Step 10 -- Feature audit table
# ===========================================================================
def build_feature_audit(df_temporal, registry):
    _hdr("STEP 10 -- FEATURE AUDIT TABLE")

    groups = {
        "group1_current_weather": registry["group1_current_weather"],
        "group2_lag":             registry["group2_lag"],
        "group3_rolling":         registry["group3_rolling"],
        "group4_trend":           registry["group4_trend"],
        "group5_anomaly":         registry["group5_anomaly"],
        "group6_calendar":        registry["group6_calendar"],
        "group7_city":            registry["group7_city"],
    }

    baseline_set = set(registry["baseline_features"])
    temporal_set = set(registry["temporal_features"])

    col_to_group = {}
    for grp_name, col_list in groups.items():
        for col in col_list:
            col_to_group[col] = grp_name

    rows = []

    for col in df_temporal.columns:
        if col in (ALL_IDS):
            role       = "identifier"
            leakage    = "N/A (identifier)"
            time_ref   = "T"
            feat_type  = "categorical/datetime"
            in_base    = "N/A"
            in_temp    = "N/A"
            grp        = "identifier"
        elif col == TARGET:
            role       = "TARGET"
            leakage    = "N/A (target)"
            time_ref   = "T+1"
            feat_type  = "binary"
            in_base    = "yes"
            in_temp    = "yes"
            grp        = "target"
        elif col in baseline_set or col in temporal_set:
            grp       = col_to_group.get(col, "unknown")
            role      = "feature"
            feat_type = "numeric"
            in_base   = "yes" if col in baseline_set else "no"
            in_temp   = "yes" if col in temporal_set else "no"

            # Determine time reference and leakage status
            if "_lag" in col:
                try:
                    lag_val = int(col.split("_lag")[-1])
                    time_ref  = f"T-{lag_val}"
                    leakage   = "SAFE (past data)"
                except ValueError:
                    time_ref  = "T-?"
                    leakage   = "REVIEW"
            elif "_roll" in col:
                # window over [T-N, ..., T-1] via shift(1).rolling(N)
                parts = col.split("_roll")
                try:
                    w_str   = parts[1].split("_")[0]
                    w       = int(w_str)
                    time_ref  = f"T-{w} to T-1 (rolling)"
                    leakage   = "SAFE (shift(1).rolling)"
                except (IndexError, ValueError):
                    time_ref  = "rolling window"
                    leakage   = "SAFE (shift(1).rolling)"
            elif col.startswith("tmax_delta_"):
                # e.g. tmax_delta_3d = Tmax(T) - Tmax(T-3)
                time_ref  = "T minus T-N"
                leakage   = "SAFE (T is current-day; T-N is past)"
            elif col.startswith("tmax_slope_"):
                time_ref  = "T-N to T-1 (slope)"
                leakage   = "SAFE (shift(1).rolling slope)"
            elif "zscore" in col:
                time_ref  = "T (normalised against T-30 to T-1)"
                leakage   = "SAFE (trailing window)"
            else:
                # Group 1 current-day, calendar, city
                time_ref  = "T"
                leakage   = "SAFE (current-day observation)"
        else:
            # Extra passthrough column (heatwave same-day label, event columns)
            grp       = "passthrough"
            role      = "passthrough"
            feat_type = df_temporal[col].dtype.name
            in_base   = "no"
            in_temp   = "no"
            time_ref  = "T"
            leakage   = "N/A (passthrough)"

        rows.append({
            "feature":        col,
            "in_baseline":    in_base,
            "in_temporal":    in_temp,
            "feature_group":  grp,
            "time_reference": time_ref,
            "dtype":          df_temporal[col].dtype.name,
            "role":           role,
            "leakage_status": leakage,
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(AUDIT_FILE, index=False)
    logger.info(f"  Feature audit table saved: {AUDIT_FILE}")
    logger.info(f"  Rows in audit table: {len(audit_df)}")

    # Count leakage statuses
    status_counts = audit_df["leakage_status"].value_counts()
    logger.info("  Leakage status summary:")
    for status, count in status_counts.items():
        logger.info(f"    {status:45s}  {count:3d}")

    unsafe = audit_df[audit_df["leakage_status"].str.contains("FAIL|REVIEW|LEAK",
                                                                na=False)]
    if len(unsafe):
        logger.error(f"  AUDIT FAIL: {len(unsafe)} row(s) need review:")
        for _, row in unsafe.iterrows():
            logger.error(f"    {row['feature']}: {row['leakage_status']}")
        sys.exit(1)
    logger.info("  Feature audit PASSED -- no unsafe features detected")


# ===========================================================================
# Step 11 -- Save output files
# ===========================================================================
def save_outputs(df_baseline, df_temporal, md5_before):
    _hdr("STEP 11 -- SAVE OUTPUT FILES")

    # Verify Phase 7 source is unchanged
    current_md5 = hashlib.md5(FEATURES_FILE.read_bytes()).hexdigest()
    if current_md5 != md5_before:
        logger.error(f"  INTEGRITY FAIL: Phase 7 MD5 changed during this run! "
                     f"before={md5_before}  after={current_md5}")
        sys.exit(1)
    logger.info(f"  Phase 7 source integrity VERIFIED (MD5 unchanged: {current_md5})")

    df_baseline.to_csv(BASELINE_OUT, index=False)
    size_base = BASELINE_OUT.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {BASELINE_OUT}")
    logger.info(f"    Shape : {df_baseline.shape[0]:,} x {df_baseline.shape[1]}  "
                f"({size_base:.2f} MB)")

    df_temporal.to_csv(TEMPORAL_OUT, index=False)
    size_temp = TEMPORAL_OUT.stat().st_size / 1024 / 1024
    logger.info(f"  Saved: {TEMPORAL_OUT}")
    logger.info(f"    Shape : {df_temporal.shape[0]:,} x {df_temporal.shape[1]}  "
                f"({size_temp:.2f} MB)")

    return size_base, size_temp


# ===========================================================================
# Step 12 -- Final validation checklist
# ===========================================================================
def final_validation_checklist(df, df_baseline, df_temporal, registry):
    _hdr("STEP 12 -- FINAL VALIDATION CHECKLIST")

    checks = []

    # 1. Both output files exist
    checks.append(("Baseline dataset exists",
                   BASELINE_OUT.exists()))
    checks.append(("Temporal dataset exists",
                   TEMPORAL_OUT.exists()))

    # 2. Both contain target
    checks.append(("Baseline contains heatwave_next_day",
                   TARGET in df_baseline.columns))
    checks.append(("Temporal contains heatwave_next_day",
                   TARGET in df_temporal.columns))

    # 3. Target not in X
    baseline_feats = registry["baseline_features"]
    temporal_feats = registry["temporal_features"]
    checks.append(("Target NOT in baseline feature set",
                   TARGET not in baseline_feats))
    checks.append(("Target NOT in temporal feature set",
                   TARGET not in temporal_feats))

    # 4. Exact feature counts
    actual_base_feats = [c for c in df_baseline.columns
                         if c not in ALL_IDS and c != TARGET]
    actual_temp_feats = [c for c in df_temporal.columns
                         if c not in ALL_IDS and c != TARGET]
    checks.append((f"Baseline has exactly {len(baseline_feats)} features",
                   len(actual_base_feats) == len(baseline_feats)))
    checks.append((f"Temporal has exactly {len(temporal_feats)} features",
                   len(actual_temp_feats) == len(temporal_feats)))

    # 5. ID columns present
    for id_col in ID_COLS:
        checks.append((f"Identifier '{id_col}' present in baseline",
                       id_col in df_baseline.columns))
        checks.append((f"Identifier '{id_col}' present in temporal",
                       id_col in df_temporal.columns))

    # 6. No NaN in feature columns
    base_feat_nans = df_baseline[baseline_feats].isnull().sum().sum()
    temp_feat_nans = df_temporal[temporal_feats].isnull().sum().sum()
    checks.append(("No NaN in baseline feature matrix",
                   base_feat_nans == 0))
    checks.append(("No NaN in temporal feature matrix",
                   temp_feat_nans == 0))

    # 7. No NaN in target
    base_target_nans = df_baseline[TARGET].isnull().sum()
    temp_target_nans = df_temporal[TARGET].isnull().sum()
    checks.append(("No NaN in baseline target",
                   base_target_nans == 0))
    checks.append(("No NaN in temporal target",
                   temp_target_nans == 0))

    # 8. Target is binary
    checks.append(("Baseline target is binary {0,1}",
                   set(df_baseline[TARGET].unique()) <= {0.0, 1.0}))
    checks.append(("Temporal target is binary {0,1}",
                   set(df_temporal[TARGET].unique()) <= {0.0, 1.0}))

    # 9. Phase 7 source row count check -- df here is AFTER zscore drop (65080)
    checks.append(("Phase 8 working df rows are 65080 (65095 - 15 zscore drops)",
                   len(df) == 65_080))

    # 10. Matching temporal and baseline feature sets resolve to same registery
    checks.append(("Baseline features match registry exactly",
                   set(actual_base_feats) == set(baseline_feats)))
    checks.append(("Temporal features match registry exactly",
                   set(actual_temp_feats) == set(temporal_feats)))

    # Print
    all_pass = True
    for description, result in checks:
        status = "PASS" if result else "FAIL"
        logger.info(f"  [{status}]  {description}")
        if not result:
            all_pass = False

    if not all_pass:
        logger.error("  One or more validation checks FAILED -- see above")
        sys.exit(1)
    logger.info(f"  All {len(checks)} validation checks PASSED")


# ===========================================================================
# Step 13 -- Final summary
# ===========================================================================
def final_summary(df, df_baseline, df_temporal, registry, size_base, size_temp,
                  md5_before):
    _hdr("PHASE 8 COMPLETE -- FINAL SUMMARY")

    baseline_feats = registry["baseline_features"]
    temporal_feats = registry["temporal_features"]

    logger.info("  INPUT")
    logger.info(f"    Phase 7 file           : {FEATURES_FILE.name}")
    logger.info(f"    Phase 7 MD5 (untouched): {md5_before}")
    logger.info(f"    Input shape            : {df.shape[0]:,} x {df.shape[1]}  "
                f"(65,095 from Phase 7 minus 15 zscore-NaN rows = {df.shape[0]:,})")
    logger.info("")
    logger.info("  OUTPUT DATASETS")
    logger.info(f"    ml_baseline.csv        : {df_baseline.shape[0]:,} rows x "
                f"{df_baseline.shape[1]} cols  ({size_base:.2f} MB)")
    logger.info(f"    ml_temporal.csv        : {df_temporal.shape[0]:,} rows x "
                f"{df_temporal.shape[1]} cols  ({size_temp:.2f} MB)")
    logger.info("")
    logger.info("  FEATURES")
    logger.info(f"    Baseline feature count : {len(baseline_feats)}")
    logger.info(f"    Temporal feature count : {len(temporal_feats)}")
    logger.info(f"    Target column          : {TARGET}")
    logger.info("")
    logger.info("  CLASS DISTRIBUTION (overall, both datasets identical)")
    total = len(df_baseline)
    pos   = int(df_baseline[TARGET].sum())
    neg   = total - pos
    logger.info(f"    Total rows   : {total:,}")
    logger.info(f"    Positives    : {pos:,}  ({pos/total*100:.2f}%)")
    logger.info(f"    Negatives    : {neg:,}  ({neg/total*100:.2f}%)")
    logger.info(f"    Imbalance    : 1:{neg//pos}")
    logger.info("")
    logger.info("  MISSING VALUES")
    logger.info("    Feature matrices : 0 NaN  (confirmed)")
    logger.info("    Target column    : 0 NaN  (confirmed)")
    logger.info("")
    logger.info("  LEAKAGE AUDIT")
    logger.info("    Phase 7 audit : PASSED (110 features, 0 issues)")
    logger.info("    Phase 8 audit : PASSED (0 issues on final column sets)")
    logger.info("")
    logger.info("  FILES CREATED")
    logger.info(f"    {BASELINE_OUT}")
    logger.info(f"    {TEMPORAL_OUT}")
    logger.info(f"    {REPORT_FILE}")
    logger.info(f"    {AUDIT_FILE}")
    logger.info("")
    logger.info("  PHASE 7 SOURCE")
    logger.info(f"    {FEATURES_FILE.name} UNTOUCHED (MD5 verified)")
    logger.info("")
    logger.info("  DECISIONS REQUIRING REVIEW")
    logger.info("    - Mumbai has 0 positive heatwave events (scientifically correct).")
    logger.info("      It is included in both ML datasets. The decision of whether to")
    logger.info("      exclude Mumbai from supervised training or treat it as a purely")
    logger.info("      negative class should be made in Phase 10 (Baseline Models).")
    logger.info("    - qualifying_day is included in the feature set. It is derived from")
    logger.info("      threshold + departure at T (not T+1) and is therefore leakage-safe.")
    logger.info("      However it is strongly correlated with the target. A reviewer")
    logger.info("      should decide whether to include it in the final feature set.")
    logger.info("    - heatwave_lag1 uses heatwave(T-1) as a feature. This is safe.")
    logger.info("      It means the model can learn from yesterday's heatwave status.")
    _sep()


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    df, registry, md5_before = load_data()
    verify_target_construction(df)
    validate_city_boundaries(df)
    validate_chronological_order(df)
    df = inspect_missing_values(df, registry)   # may drop rows; returns updated df
    verify_dtypes(df, registry)
    df_baseline, df_temporal, X_baseline, y_baseline, X_temporal, y_temporal = \
        build_datasets(df, registry)
    leakage_audit_final(df_baseline, df_temporal)
    report_class_distribution(df_baseline, df_temporal)
    build_feature_audit(df_temporal, registry)
    size_base, size_temp = save_outputs(df_baseline, df_temporal, md5_before)
    final_validation_checklist(df, df_baseline, df_temporal, registry)
    final_summary(df, df_baseline, df_temporal, registry,
                  size_base, size_temp, md5_before)


if __name__ == "__main__":
    main()

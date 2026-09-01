"""
time_series_split.py -- ClimateGuard Phase 9
=============================================
Produces chronological train / validation / test splits from the Phase 8
validated ML datasets.

Inputs  (read-only)
-------------------
  data/features/ml_baseline.csv
  data/features/ml_temporal.csv

Outputs
-------
  data/splits/baseline/X_train.csv    X_val.csv    X_test.csv
  data/splits/baseline/y_train.csv    y_val.csv    y_test.csv
  data/splits/baseline/meta_train.csv meta_val.csv meta_test.csv

  data/splits/temporal/X_train.csv    X_val.csv    X_test.csv
  data/splits/temporal/y_train.csv    y_val.csv    y_test.csv
  data/splits/temporal/meta_train.csv meta_val.csv meta_test.csv

  results/phase9_split_report.json
  results/phase9_split_log.txt
  results/phase9_leakage_audit.csv

Split boundaries (year-based, inclusive)
-----------------------------------------
  TRAIN      : 1990-01-11  ->  2019-12-31
  VALIDATION : 2020-01-01  ->  2022-12-31
  TEST       : 2023-01-01  ->  2025-08-30

Critical rules
--------------
  - NO random shuffling
  - NO scaling / normalisation
  - NO SMOTE / resampling
  - NO model training
  - Target (heatwave_next_day) kept out of X matrices
  - city, city_key, date saved in meta files (not in X)
  - Split applied identically to both datasets
"""

import sys
import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent
BASELINE_IN   = ROOT / "data" / "features" / "ml_baseline.csv"
TEMPORAL_IN   = ROOT / "data" / "features" / "ml_temporal.csv"
SPLITS_BASE   = ROOT / "data"    / "splits"
RESULTS_DIR   = ROOT / "results"
LOG_FILE      = RESULTS_DIR / "phase9_split_log.txt"
REPORT_FILE   = RESULTS_DIR / "phase9_split_report.json"
AUDIT_FILE    = RESULTS_DIR / "phase9_leakage_audit.csv"

for p in [SPLITS_BASE / "baseline", SPLITS_BASE / "temporal", RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("phase9")
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
TARGET     = "heatwave_next_day"
META_COLS  = ["city", "city_key", "date", "state", "region_type",
              "heatwave", "hw_event_id", "hw_event_start",
              "hw_event_end", "hw_event_length"]
ID_COLS    = ["city", "city_key", "date"]     # minimum traceback set

# Chronological split boundaries (year-inclusive)
TRAIN_END_YEAR = 2019
VAL_START_YEAR = 2020
VAL_END_YEAR   = 2022
TEST_START_YEAR = 2023

# Exact date strings (derived from dataset inspection)
TRAIN_END_DATE  = "2019-12-31"
VAL_START_DATE  = "2020-01-01"
VAL_END_DATE    = "2022-12-31"
TEST_START_DATE = "2023-01-01"

CITY_ORDER = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]


# ===========================================================================
# Step 1 -- Load and verify inputs
# ===========================================================================
def load_inputs():
    _hdr("STEP 1 -- LOAD PHASE 8 DATASETS")

    results = {}
    for label, path in [("baseline", BASELINE_IN), ("temporal", TEMPORAL_IN)]:
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        df  = pd.read_csv(path, parse_dates=["date"], low_memory=False)
        logger.info(f"  [{label}]  {path.name}  shape={df.shape}  md5={md5}")
        assert len(df) == 65_080, (
            f"Expected 65080 rows for {label}, got {len(df)}")
        assert TARGET in df.columns, f"Target '{TARGET}' missing from {label}"
        results[label] = {"df": df, "md5": md5}

    # Both datasets must have identical row sets (same city+date index)
    base_idx = (results["baseline"]["df"][["city_key", "date"]]
                .sort_values(["city_key", "date"]).reset_index(drop=True))
    temp_idx = (results["temporal"]["df"][["city_key", "date"]]
                .sort_values(["city_key", "date"]).reset_index(drop=True))
    assert base_idx.equals(temp_idx), (
        "baseline and temporal city+date indices do not match -- cannot apply same split")
    logger.info("  city+date index match: PASS (both datasets have identical rows)")

    # Date range
    df_ref = results["baseline"]["df"]
    logger.info(f"  Overall date range: "
                f"{df_ref['date'].min().date()} -> {df_ref['date'].max().date()}")
    logger.info(f"  Target binary check: "
                f"{sorted(df_ref[TARGET].unique())} "
                f"-- positives={int(df_ref[TARGET].sum())}")
    return results


# ===========================================================================
# Step 2 -- Define and apply split
# ===========================================================================
def apply_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns a dict with keys 'train', 'val', 'test'.
    Split is year-based and applied uniformly to all cities.
    """
    year = df["date"].dt.year
    masks = {
        "train": year <= TRAIN_END_YEAR,
        "val":   (year >= VAL_START_YEAR) & (year <= VAL_END_YEAR),
        "test":  year >= TEST_START_YEAR,
    }
    splits = {}
    for name, mask in masks.items():
        splits[name] = df[mask].sort_values(["city_key", "date"]).reset_index(drop=True)

    # Verify all rows are accounted for (no gaps, no overlap)
    total = sum(len(s) for s in splits.values())
    assert total == len(df), (
        f"Split total {total} != input total {len(df)} -- rows lost or duplicated")
    return splits


# ===========================================================================
# Step 3 -- Extract X, y, meta
# ===========================================================================
def extract_Xy_meta(splits: dict, dataset_name: str) -> dict:
    """
    For each split: separate into X (features), y (target), meta (identifiers).
    Features = all columns that are neither meta nor target.
    """
    result = {}
    for split_name, df in splits.items():
        # Meta columns that exist in this dataframe
        meta_present = [c for c in META_COLS if c in df.columns]
        # Feature columns: everything except meta and target
        feature_cols = [c for c in df.columns
                        if c not in meta_present and c != TARGET]
        # Double-check: target must not be in features
        assert TARGET not in feature_cols, (
            f"[{dataset_name}/{split_name}] Target found in feature columns -- ABORT")
        # Double-check: no identifier accidentally in features
        for id_col in ID_COLS:
            assert id_col not in feature_cols, (
                f"[{dataset_name}/{split_name}] Identifier '{id_col}' in feature cols")

        X    = df[feature_cols].reset_index(drop=True)
        y    = df[[TARGET]].reset_index(drop=True)
        meta = df[meta_present].reset_index(drop=True)

        result[split_name] = {"X": X, "y": y, "meta": meta,
                               "feature_cols": feature_cols}

    logger.info(f"  [{dataset_name}] Feature columns: {len(result['train']['feature_cols'])}")
    logger.info(f"  [{dataset_name}] Target: {TARGET} -- confirmed absent from X")
    return result


# ===========================================================================
# Step 4 -- Leakage audit
# ===========================================================================
def leakage_audit(splits_base: dict, splits_temp: dict) -> list[dict]:
    _hdr("STEP 4 -- SPLIT LEAKAGE AUDIT")
    audit_rows = []
    all_pass   = True

    for dataset_name, splits in [("baseline", splits_base),
                                   ("temporal", splits_temp)]:
        train_df = splits["train"]["meta"]
        val_df   = splits["val"]["meta"]
        test_df  = splits["test"]["meta"]

        # --- 1. Global date ordering ---
        train_max = train_df["date"].max()
        val_min   = val_df["date"].min()
        val_max   = val_df["date"].max()
        test_min  = test_df["date"].min()
        test_max  = test_df["date"].max()

        check1 = train_max < val_min
        check2 = val_max   < test_min
        logger.info(f"  [{dataset_name}]  max(train)={train_max.date()}  "
                    f"min(val)={val_min.date()}  "
                    f"max(val)={val_max.date()}  "
                    f"min(test)={test_min.date()}  "
                    f"max(test)={test_max.date()}")
        logger.info(f"  [{dataset_name}]  max(train) < min(val) : "
                    f"{'PASS' if check1 else 'FAIL'}  "
                    f"({train_max.date()} < {val_min.date()})")
        logger.info(f"  [{dataset_name}]  max(val)   < min(test): "
                    f"{'PASS' if check2 else 'FAIL'}  "
                    f"({val_max.date()} < {test_min.date()})")
        if not (check1 and check2):
            all_pass = False

        # --- 2. Per-city date ordering ---
        for city in CITY_ORDER:
            c_train = train_df[train_df["city_key"] == city]["date"]
            c_val   = val_df  [val_df  ["city_key"] == city]["date"]
            c_test  = test_df [test_df ["city_key"] == city]["date"]

            c1 = c_train.max() < c_val.min()
            c2 = c_val.max()   < c_test.min()
            status = "PASS" if (c1 and c2) else "FAIL"
            if not (c1 and c2):
                all_pass = False
            logger.info(f"    {dataset_name}/{city}  "
                        f"train_end={c_train.max().date()}  "
                        f"val_start={c_val.min().date()}  "
                        f"val_end={c_val.max().date()}  "
                        f"test_start={c_test.min().date()}  "
                        f"[{status}]")
            audit_rows.append({
                "dataset":    dataset_name,
                "city":       city,
                "train_end":  c_train.max().date().isoformat(),
                "val_start":  c_val.min().date().isoformat(),
                "val_end":    c_val.max().date().isoformat(),
                "test_start": c_test.min().date().isoformat(),
                "test_end":   c_test.max().date().isoformat(),
                "train_max_lt_val_min": c1,
                "val_max_lt_test_min":  c2,
                "status":     status,
            })

        # --- 3. No duplicate city+date across splits ---
        all_pairs = pd.concat([
            train_df[["city_key", "date"]].assign(split="train"),
            val_df  [["city_key", "date"]].assign(split="val"),
            test_df [["city_key", "date"]].assign(split="test"),
        ])
        dups = all_pairs.duplicated(subset=["city_key", "date"], keep=False)
        dup_count = dups.sum()
        logger.info(f"  [{dataset_name}]  city+date duplicates across splits: "
                    f"{dup_count}  ({'PASS' if dup_count == 0 else 'FAIL'})")
        if dup_count > 0:
            all_pass = False

        # --- 4. Target not in X ---
        for split_name in ["train", "val", "test"]:
            X_cols = splits[split_name]["feature_cols"]
            in_X = TARGET in X_cols
            logger.info(f"  [{dataset_name}/{split_name}]  "
                        f"target NOT in X: {'PASS' if not in_X else 'FAIL'}")
            if in_X:
                all_pass = False

    if not all_pass:
        logger.error("  LEAKAGE AUDIT: one or more checks FAILED -- see above")
        sys.exit(1)
    logger.info("  LEAKAGE AUDIT: ALL CHECKS PASSED")
    return audit_rows


# ===========================================================================
# Step 5 -- Class distribution report
# ===========================================================================
def class_distribution_report(extracted: dict, dataset_name: str) -> dict:
    report = {"dataset": dataset_name, "splits": {}}
    logger.info(f"  [{dataset_name}]")
    logger.info(f"  {'Split':8s}  {'Total':>7s}  {'Pos':>5s}  {'Neg':>7s}  "
                f"{'Pos%':>6s}")
    logger.info(f"  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*6}")

    for split_name in ["train", "val", "test"]:
        y = extracted[split_name]["y"][TARGET]
        total = len(y)
        pos   = int(y.sum())
        neg   = total - pos
        pct   = pos / total * 100

        logger.info(f"  {split_name:8s}  {total:7,}  {pos:5,}  "
                    f"{neg:7,}  {pct:6.2f}%")

        # Per-city
        meta = extracted[split_name]["meta"]
        city_dist = {}
        for city in CITY_ORDER:
            mask   = meta["city_key"] == city
            c_y    = y[mask]
            c_tot  = len(c_y)
            c_pos  = int(c_y.sum())
            c_neg  = c_tot - c_pos
            c_pct  = c_pos / c_tot * 100 if c_tot > 0 else 0.0
            city_dist[city] = {"total": c_tot, "pos": c_pos,
                               "neg": c_neg, "pos_pct": round(c_pct, 4)}
            logger.info(f"    {city:12s}  total={c_tot:5,}  pos={c_pos:4,}  "
                        f"neg={c_neg:5,}  pos%={c_pct:.2f}%")

        report["splits"][split_name] = {
            "total": total, "pos": pos, "neg": neg,
            "pos_pct": round(pct, 4), "by_city": city_dist,
        }
    logger.info("")
    return report


# ===========================================================================
# Step 6 -- Save split files
# ===========================================================================
def save_splits(extracted: dict, out_dir: Path, dataset_name: str,
                md5_before: str):
    _hdr(f"STEP 6 -- SAVE SPLITS: {dataset_name.upper()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        X    = extracted[split_name]["X"]
        y    = extracted[split_name]["y"]
        meta = extracted[split_name]["meta"]

        X.to_csv   (out_dir / f"X_{split_name}.csv",    index=False)
        y.to_csv   (out_dir / f"y_{split_name}.csv",    index=False)
        meta.to_csv(out_dir / f"meta_{split_name}.csv", index=False)

        logger.info(f"  {split_name:5s}  X={X.shape}  y={y.shape}  "
                    f"meta={meta.shape}")
        logger.info(f"    X_{split_name}.csv   "
                    f"y_{split_name}.csv   meta_{split_name}.csv")

    logger.info(f"  Output directory: {out_dir}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    _hdr("ClimateGuard -- Phase 9: Train / Validation / Test Split")
    logger.info(f"  Split boundaries:")
    logger.info(f"    TRAIN      : 1990-01-11  ->  {TRAIN_END_DATE}")
    logger.info(f"    VALIDATION : {VAL_START_DATE}  ->  {VAL_END_DATE}")
    logger.info(f"    TEST       : {TEST_START_DATE}  ->  2025-08-30")
    logger.info(f"  Split method : chronological / year-based (NO random shuffle)")

    # 1. Load
    inputs = load_inputs()

    # 2. Apply split to each dataset
    _hdr("STEP 2 -- APPLY CHRONOLOGICAL SPLIT")
    split_data = {}
    for dataset_name, info in inputs.items():
        df     = info["df"]
        splits = apply_split(df)
        for sname, sdf in splits.items():
            n   = len(sdf)
            pos = int(sdf[TARGET].sum())
            logger.info(f"  [{dataset_name}/{sname}]  rows={n:,}  "
                        f"positives={pos}  ({pos/n*100:.2f}%)")
        split_data[dataset_name] = splits

    # 3. Extract X / y / meta
    _hdr("STEP 3 -- EXTRACT X / y / meta")
    extracted = {}
    for dataset_name, splits in split_data.items():
        extracted[dataset_name] = extract_Xy_meta(splits, dataset_name)

    # 4. Leakage audit (operates on both datasets together)
    audit_rows = leakage_audit(extracted["baseline"], extracted["temporal"])

    # 5. Class distribution reports
    _hdr("STEP 5 -- CLASS DISTRIBUTION REPORT")
    reports = {}
    for dataset_name in ["baseline", "temporal"]:
        rep = class_distribution_report(extracted[dataset_name], dataset_name)
        reports[dataset_name] = rep

    # 6. Save split files
    save_splits(extracted["baseline"],
                SPLITS_BASE / "baseline",
                "baseline",
                inputs["baseline"]["md5"])
    save_splits(extracted["temporal"],
                SPLITS_BASE / "temporal",
                "temporal",
                inputs["temporal"]["md5"])

    # 7. Save leakage audit CSV
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT_FILE, index=False)
    logger.info(f"  Leakage audit CSV saved: {AUDIT_FILE}")

    # 8. Verify Phase 8 sources untouched
    _hdr("STEP 7 -- VERIFY PHASE 8 SOURCE INTEGRITY")
    for dataset_name, info in inputs.items():
        path = BASELINE_IN if dataset_name == "baseline" else TEMPORAL_IN
        current_md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if current_md5 != info["md5"]:
            logger.error(f"  INTEGRITY FAIL: {path.name} was modified! "
                         f"before={info['md5']}  after={current_md5}")
            sys.exit(1)
        logger.info(f"  {path.name}  MD5 UNCHANGED: {current_md5}  PASS")

    # 9. Build full JSON report
    report = {
        "phase": 9,
        "split_method": "chronological / year-based",
        "random_shuffle": False,
        "boundaries": {
            "train_start": "1990-01-11",
            "train_end":   TRAIN_END_DATE,
            "val_start":   VAL_START_DATE,
            "val_end":     VAL_END_DATE,
            "test_start":  TEST_START_DATE,
            "test_end":    "2025-08-30",
        },
        "leakage_audit_passed": all(r["status"] == "PASS" for r in audit_rows),
        "datasets": {},
    }
    for dataset_name in ["baseline", "temporal"]:
        extr = extracted[dataset_name]
        report["datasets"][dataset_name] = {
            "n_features":     len(extr["train"]["feature_cols"]),
            "phase8_md5":     inputs[dataset_name]["md5"],
            "splits": {
                sname: {
                    "X_shape":   list(extr[sname]["X"].shape),
                    "y_shape":   list(extr[sname]["y"].shape),
                    "meta_shape": list(extr[sname]["meta"].shape),
                    "positives": int(extr[sname]["y"][TARGET].sum()),
                    "pos_pct":   round(
                        extr[sname]["y"][TARGET].mean() * 100, 4),
                }
                for sname in ["train", "val", "test"]
            },
            "class_distribution": reports[dataset_name],
        }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Split report saved: {REPORT_FILE}")

    # 10. Final summary
    _hdr("PHASE 9 COMPLETE -- FINAL SUMMARY")
    logger.info("  SPLIT BOUNDARIES")
    logger.info(f"    TRAIN      : 1990-01-11 -> {TRAIN_END_DATE}")
    logger.info(f"    VALIDATION : {VAL_START_DATE} -> {VAL_END_DATE}")
    logger.info(f"    TEST       : {TEST_START_DATE} -> 2025-08-30")
    logger.info("")
    for dataset_name in ["baseline", "temporal"]:
        extr = extracted[dataset_name]
        logger.info(f"  [{dataset_name.upper()}]  "
                    f"features={len(extr['train']['feature_cols'])}")
        for sname in ["train", "val", "test"]:
            s = extr[sname]
            pos = int(s["y"][TARGET].sum())
            tot = len(s["y"])
            logger.info(f"    {sname:5s}  X={s['X'].shape}  "
                        f"y={s['y'].shape}  pos={pos}  ({pos/tot*100:.2f}%)")
    logger.info("")
    logger.info("  LEAKAGE AUDIT: PASSED")
    logger.info("  PHASE 8 SOURCE FILES: UNTOUCHED (MD5 verified)")
    logger.info("  NO scaling, NO SMOTE, NO model training performed")
    logger.info("")
    logger.info("  FILES CREATED:")
    for sub in ["baseline", "temporal"]:
        for fn in ["X_train", "X_val", "X_test",
                   "y_train", "y_val", "y_test",
                   "meta_train", "meta_val", "meta_test"]:
            logger.info(f"    data/splits/{sub}/{fn}.csv")
    logger.info(f"    results/phase9_split_report.json")
    logger.info(f"    results/phase9_split_log.txt")
    logger.info(f"    results/phase9_leakage_audit.csv")
    logger.info("")
    logger.info("  CONCERNS / DECISIONS REQUIRING REVIEW:")
    logger.info("    1. qualifying_day is included in X -- it is leakage-safe (same-day T)")
    logger.info("       but is strongly correlated with the target. Review in Phase 10/12.")
    logger.info("    2. Mumbai has 0 positives in ALL splits (train/val/test).")
    logger.info("       Training-inclusion decision deferred to Phase 10.")
    logger.info("    3. Class imbalance is severe (approx 1:128 overall).")
    logger.info("       This will be addressed with class weights / SMOTE in Phase 11.")
    logger.info("    4. Validation period (2020-2022) contains only 3 years.")
    logger.info("       This is intentional -- test set must use the most recent data.")
    _sep()


if __name__ == "__main__":
    main()

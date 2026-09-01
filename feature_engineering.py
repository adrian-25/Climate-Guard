"""
feature_engineering.py — ClimateGuard Phase 7
==============================================
Input  : data/processed/weather_labelled.csv   (read-only, validated Phase 6 artifact)
Output : data/features/climateguard_features.csv

Feature groups built
────────────────────
1  Current-day weather   — 15 weather variables at T + tmax_departure
2  Lag features          — T-1, T-2, T-3, T-7 for 7 key variables
3  Rolling features      — 3-day and 7-day rolling mean/max/min for key variables
4  Trend features        — Tmax delta T→T-1, T→T-3, T→T-7; slope over 3d and 7d
5  Anomaly features      — tmax_departure already in group 1; + z-score normalised departure
6  Calendar features     — month, day_of_year, season; sin/cos cyclical encoding for month
7  City features         — label-encoded city_key + lat/lon (kept for geographic signal)

Target (leakage-safe)
─────────────────────
heatwave_next_day = heatwave.shift(-1) applied PER CITY  (predicting T+1 from T)
Last row of each city (T+1 unknown) is dropped.
First ~7 rows per city (incomplete lag windows) are dropped.

Leakage audit
─────────────
The script prints a leakage audit table before saving, verifying that no
feature is derived from weather data at T+1 or later.

Baseline vs temporal tagging
─────────────────────────────
The output CSV contains ALL features.  A separate list of column names for
each experimental set is written to results/phase7_feature_groups.json so
Phase 13 can reconstruct each set without duplicating the data.

Usage
─────
    python feature_engineering.py
"""

import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
INPUT_FILE    = ROOT / "data" / "processed" / "weather_labelled.csv"
OUTPUT_DIR    = ROOT / "data" / "features"
OUTPUT_FILE   = OUTPUT_DIR / "climateguard_features.csv"
RESULTS_DIR   = ROOT / "results"
LOG_FILE      = RESULTS_DIR / "phase7_feature_engineering_log.txt"
GROUPS_FILE   = RESULTS_DIR / "phase7_feature_groups.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("phase7")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

# ── constants ──────────────────────────────────────────────────────────────────
# Key variables used for lag / rolling / trend features
KEY_WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "surface_pressure_mean",
]

# All 15 weather variables available in weather_labelled.csv
ALL_WEATHER_VARS = [
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
]

LAG_OFFSETS   = [1, 2, 3, 7]
ROLL_WINDOWS  = [3, 7]
ROLL_STATS    = ["mean", "max", "min"]

CITY_ORDER = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "autumn", 11: "autumn",
}
SEASON_ENCODE = {"winter": 0, "spring": 1, "monsoon": 2, "autumn": 3}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: rolling slope (linear trend)
# ══════════════════════════════════════════════════════════════════════════════
def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling linear slope over `window` observations.
    Uses ONLY past values (shift(1) before rolling) to prevent leakage.
    Returns NaN for windows with < 2 valid data points.
    """
    slopes = [np.nan] * len(series)
    values = series.values
    for i in range(window, len(values) + 1):
        chunk = values[i - window:i]
        valid = ~np.isnan(chunk)
        if valid.sum() >= 2:
            x = np.arange(window)[valid]
            y = chunk[valid]
            slope, *_ = linregress(x, y)
            slopes[i - 1] = slope
    return pd.Series(slopes, index=series.index)


# ══════════════════════════════════════════════════════════════════════════════
# Main feature builder — operates on a single city's DataFrame
# ══════════════════════════════════════════════════════════════════════════════
def build_features_for_city(city_df: pd.DataFrame) -> pd.DataFrame:
    """
    Receives a single-city DataFrame sorted by date ascending.
    Returns the same DataFrame with all feature columns appended.
    No cross-city contamination is possible since this function is
    called exclusively inside groupby.apply.
    """
    df = city_df.copy().sort_values("date").reset_index(drop=True)
    city_key = df["city_key"].iloc[0]
    n = len(df)
    logger.debug(f"  Building features for {city_key:12s}  ({n} rows)")

    tmax = df["temperature_2m_max"]

    # ── GROUP 1: current-day weather (passthrough) ────────────────────────────
    # ALL_WEATHER_VARS and tmax_departure already in df — no new columns needed.
    # They are included in the final column selection later.

    # ── GROUP 2: lag features ─────────────────────────────────────────────────
    for var in KEY_WEATHER_VARS:
        for lag in LAG_OFFSETS:
            col = f"{var}_lag{lag}"
            df[col] = df[var].shift(lag)

    # Also lag tmax_departure (useful for anomaly persistence)
    for lag in LAG_OFFSETS:
        df[f"tmax_departure_lag{lag}"] = df["tmax_departure"].shift(lag)

    # Also lag heatwave status (yesterday's heatwave state — safe, it is T-1)
    # NOTE: heatwave(T) itself is NOT used as a feature — only heatwave(T-1)
    df["heatwave_lag1"] = df["heatwave"].shift(1)

    # ── GROUP 3: rolling features ─────────────────────────────────────────────
    # Pattern: shift(1) first, then rolling(N).
    # This means the rolling window is over [T-N, ..., T-1], excluding T itself.
    # Result assigned back to index T → strictly uses only past data.
    for var in KEY_WEATHER_VARS:
        past = df[var].shift(1)           # T-1 is oldest value in window
        for w in ROLL_WINDOWS:
            roll = past.rolling(window=w, min_periods=w)
            df[f"{var}_roll{w}_mean"] = roll.mean()
            df[f"{var}_roll{w}_max"]  = roll.max()
            df[f"{var}_roll{w}_min"]  = roll.min()

    # ── GROUP 4: trend features ───────────────────────────────────────────────
    # Point-in-time deltas  (all use only T or earlier via shift)
    df["tmax_delta_1d"] = tmax - tmax.shift(1)   # T minus T-1
    df["tmax_delta_3d"] = tmax - tmax.shift(3)   # T minus T-3
    df["tmax_delta_7d"] = tmax - tmax.shift(7)   # T minus T-7

    # Rolling linear slopes over [T-2, T-1] (3d window ending at T-1)
    # and [T-6, ..., T-1] (7d window ending at T-1)
    tmax_past = tmax.shift(1)  # shift so window ends at T-1, not T
    df["tmax_slope_3d"] = rolling_slope(tmax_past, window=3)
    df["tmax_slope_7d"] = rolling_slope(tmax_past, window=7)

    # ── GROUP 5: anomaly features ─────────────────────────────────────────────
    # tmax_departure already included from Group 1 (passthrough).
    # Add z-score of departure relative to its 30-day trailing std:
    dep_past = df["tmax_departure"].shift(1)
    dep_roll30_std  = dep_past.rolling(window=30, min_periods=10).std()
    dep_roll30_mean = dep_past.rolling(window=30, min_periods=10).mean()
    df["tmax_departure_zscore"] = (
        (df["tmax_departure"] - dep_roll30_mean) /
        dep_roll30_std.replace(0, np.nan)
    )

    # ── GROUP 6: calendar features ────────────────────────────────────────────
    dates = pd.to_datetime(df["date"])
    df["month"]        = dates.dt.month
    df["day_of_year"]  = dates.dt.dayofyear
    df["season"]       = dates.dt.month.map(SEASON_MAP)
    df["season_code"]  = df["season"].map(SEASON_ENCODE)

    # Cyclical encoding (preserves Jan–Dec continuity for month)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]      = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"]      = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # ── GROUP 7: city features ────────────────────────────────────────────────
    # city_key label encoding
    df["city_encoded"] = df["city_key"].map(
        {c: i for i, c in enumerate(CITY_ORDER)}
    )
    # latitude and longitude kept as numeric geographic signal
    # region_type binary flag
    df["is_coastal"] = (df["region_type"] == "coastal").astype(int)

    # ── TARGET: heatwave_next_day ─────────────────────────────────────────────
    # shift(-1): tomorrow's heatwave label.  Last row per city → NaN (dropped later).
    df["heatwave_next_day"] = df["heatwave"].shift(-1)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Leakage audit
# ══════════════════════════════════════════════════════════════════════════════
KNOWN_LEAKAGE_PATTERNS = [
    # T+1 weather would look like e.g. "temperature_2m_max_lead1"
    "_lead", "_next",
    # Any weather variable that hasn't been lagged or rolled — raw names at T
    # are ALLOWED as Group 1 current features.
]

# Columns that are explicitly allowed even though they look "current":
ALLOWED_CURRENT = set(ALL_WEATHER_VARS + ["tmax_normal", "tmax_departure",
                                           "qualifying_day"])


def run_leakage_audit(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Checks that no feature column encodes T+1 or later weather data.

    Rules:
    - Lag columns must have lag >= 1  (i.e., named _lag1, _lag2, …)
    - Rolling columns use shift(1) pattern — confirmed by construction
    - Delta columns compare T vs T-N where N>=1 — the T component is the
      current-day reading, which IS allowed in Group 1
    - Calendar features: purely calendar math, no weather
    - City features: static metadata
    - Target (heatwave_next_day): excluded from feature_cols
    """
    logger.info("")
    logger.info("═" * 70)
    logger.info("LEAKAGE AUDIT")
    logger.info("═" * 70)

    issues = []

    for col in feature_cols:
        # Explicitly forbidden patterns
        if any(pat in col for pat in KNOWN_LEAKAGE_PATTERNS):
            issues.append((col, "FORBIDDEN pattern detected"))
            continue

        # Check lag columns have positive lag (lag0 would be same-day duplicate)
        if "_lag" in col:
            try:
                lag_val = int(col.split("_lag")[-1])
                if lag_val < 1:
                    issues.append((col, f"lag={lag_val} is not >= 1"))
            except ValueError:
                pass  # non-numeric suffix, inspect manually
            continue

        # Rolling: constructed via shift(1).rolling(N) — safe by design
        if "_roll" in col:
            continue

        # Trend/delta: tmax_delta_Nd compares T vs T-N; T is current-day (allowed)
        if col.startswith("tmax_delta_") or col.startswith("tmax_slope_"):
            continue

        # Anomaly features
        if "departure" in col or "zscore" in col:
            continue

        # Calendar
        if col in {"month", "day_of_year", "season", "season_code",
                   "month_sin", "month_cos", "doy_sin", "doy_cos"}:
            continue

        # City metadata
        if col in {"city_encoded", "is_coastal", "latitude", "longitude",
                   "city_key", "region_type", "state", "city"}:
            continue

        # Current-day weather (Group 1) — allowed
        if col in ALLOWED_CURRENT:
            continue

        # heatwave_lag1 — safe (yesterday's status)
        if col == "heatwave_lag1":
            continue

        # If we reach here for an unknown column, flag for manual review
        issues.append((col, "UNKNOWN — manual review required"))

    if issues:
        logger.error(f"LEAKAGE AUDIT: {len(issues)} potential issue(s) found!")
        for col, reason in issues:
            logger.error(f"  ❌ {col:55s}  {reason}")
        sys.exit(1)
    else:
        logger.info(f"LEAKAGE AUDIT PASSED — {len(feature_cols)} features checked, 0 issues.")
        logger.info("  All lag features have lag >= 1 (past data only)")
        logger.info("  All rolling features use shift(1).rolling(N) pattern")
        logger.info("  Delta features compare T vs T-N (T itself is current-day weather, allowed)")
        logger.info("  No _lead / _next suffixes detected")
        logger.info("  Target (heatwave_next_day) is excluded from feature list")
    logger.info("═" * 70)
    logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# Column sets for Phase 13 experiment
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_group_registry(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary mapping experiment name → sorted list of feature columns.

    baseline_features: Group 1 (current weather) + Group 6 (calendar) + Group 7 (city)
    temporal_features: All groups (baseline + lags + rolling + trends + anomaly)
    """
    all_cols = df.columns.tolist()

    # Group 1 — current day weather + tmax_departure
    group1 = ALL_WEATHER_VARS + ["tmax_normal", "tmax_departure", "qualifying_day"]
    group1 = [c for c in group1 if c in all_cols]

    # Group 2 — lags
    group2 = [c for c in all_cols if "_lag" in c]

    # Group 3 — rolling
    group3 = [c for c in all_cols if "_roll" in c]

    # Group 4 — trend
    group4 = [c for c in all_cols if c.startswith("tmax_delta_") or c.startswith("tmax_slope_")]

    # Group 5 — anomaly (zscore only; tmax_departure is already in group1)
    group5 = [c for c in all_cols if "zscore" in c]

    # Group 6 — calendar
    group6 = ["month", "day_of_year", "season_code",
               "month_sin", "month_cos", "doy_sin", "doy_cos"]
    group6 = [c for c in group6 if c in all_cols]

    # Group 7 — city
    group7 = ["city_encoded", "is_coastal", "latitude", "longitude"]
    group7 = [c for c in group7 if c in all_cols]

    baseline = sorted(set(group1 + group6 + group7))
    temporal = sorted(set(group1 + group2 + group3 + group4 + group5 + group6 + group7))

    return {
        "group1_current_weather": sorted(group1),
        "group2_lag":             sorted(group2),
        "group3_rolling":         sorted(group3),
        "group4_trend":           sorted(group4),
        "group5_anomaly":         sorted(group5),
        "group6_calendar":        sorted(group6),
        "group7_city":            sorted(group7),
        "baseline_features":      baseline,
        "temporal_features":      temporal,
        "target":                 "heatwave_next_day",
        "passthrough_ids":        ["city", "city_key", "date", "state",
                                   "region_type", "heatwave",
                                   "hw_event_id", "hw_event_start",
                                   "hw_event_end", "hw_event_length"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("ClimateGuard — Phase 7: Feature Engineering")
    logger.info(f"Input : {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info("")

    # ── 1. Load labelled dataset ───────────────────────────────────────────────
    logger.info("Loading weather_labelled.csv …")
    df_raw = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    logger.info(f"  Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")
    assert df_raw.shape == (65_135, 30), (
        f"Unexpected shape {df_raw.shape}; expected (65135, 30). "
        "weather_labelled.csv may have been modified."
    )
    logger.info("  Shape assertion passed (65135 × 30)")

    # ── 2. Sort to guarantee temporal order ───────────────────────────────────
    df_raw = df_raw.sort_values(["city_key", "date"]).reset_index(drop=True)

    # ── 3. Build features per city ────────────────────────────────────────────
    logger.info("")
    logger.info("Building features (grouped by city_key) …")
    parts = []
    for city, city_df in df_raw.groupby("city_key", sort=False):
        enriched = build_features_for_city(city_df)
        parts.append(enriched)
    df = pd.concat(parts, ignore_index=True)
    logger.info(f"  After feature construction: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 4. Drop incomplete rows ────────────────────────────────────────────────
    # (a) Last row per city — target is NaN (T+1 does not exist)
    # (b) First 7 rows per city — lag-7 window is incomplete
    MIN_LAG_ROWS = 7   # lag_7 needs 7 prior rows; rolling_7 also needs 7

    logger.info("")
    logger.info("Dropping incomplete rows …")
    before = len(df)

    # Mark last row per city
    df = df.sort_values(["city_key", "date"]).reset_index(drop=True)
    df["_row_num"] = df.groupby("city_key").cumcount()
    df["_city_total"] = df.groupby("city_key")["city_key"].transform("count")

    # Drop last row (target NaN) and first MIN_LAG_ROWS rows (incomplete lags)
    mask_keep = (df["_row_num"] >= MIN_LAG_ROWS) & (df["_row_num"] < df["_city_total"] - 1)
    df = df[mask_keep].drop(columns=["_row_num", "_city_total"]).reset_index(drop=True)

    after = len(df)
    dropped = before - after
    logger.info(f"  Dropped {dropped:,} rows ({MIN_LAG_ROWS} head + 1 tail per city × 5 cities = {(MIN_LAG_ROWS + 1) * 5})")
    logger.info(f"  Remaining: {after:,} rows")

    # ── 5. Sanity check — no NaN in core feature columns ──────────────────────
    logger.info("")
    logger.info("Checking for unexpected NaNs in core lag/rolling columns …")
    lag_cols   = [c for c in df.columns if "_lag"  in c]
    roll_cols  = [c for c in df.columns if "_roll" in c]
    check_cols = lag_cols + roll_cols + ["heatwave_next_day"]
    nan_report = df[check_cols].isnull().sum()
    nan_issues = nan_report[nan_report > 0]
    if len(nan_issues):
        logger.warning(f"  NaN counts in core columns (may be OK for very sparse rolling):")
        for col, cnt in nan_issues.items():
            logger.warning(f"    {col:55s}  {cnt:,}")
    else:
        logger.info("  No unexpected NaNs found in lag/rolling/target columns. ✓")

    # ── 6. Leakage audit ──────────────────────────────────────────────────────
    registry = build_feature_group_registry(df)
    all_feature_cols = registry["temporal_features"]   # superset
    run_leakage_audit(df, all_feature_cols)

    # ── 7. Print feature summary ───────────────────────────────────────────────
    logger.info("Feature group summary")
    logger.info("─" * 60)
    for group_key in ["group1_current_weather", "group2_lag", "group3_rolling",
                       "group4_trend", "group5_anomaly", "group6_calendar",
                       "group7_city"]:
        cols = registry[group_key]
        logger.info(f"  {group_key:30s}  {len(cols):3d} features")
    logger.info("─" * 60)
    logger.info(f"  {'baseline_features':30s}  {len(registry['baseline_features']):3d} features")
    logger.info(f"  {'temporal_features':30s}  {len(registry['temporal_features']):3d} features")
    logger.info("")

    # ── 8. Class balance check ────────────────────────────────────────────────
    logger.info("Class balance — heatwave_next_day")
    logger.info("─" * 60)
    overall = df["heatwave_next_day"].value_counts(dropna=True)
    total = overall.sum()
    for val, cnt in sorted(overall.items()):
        label = "heatwave" if val == 1.0 else "normal"
        logger.info(f"  {label:10s} ({int(val)})  {cnt:6,}  ({cnt/total*100:.2f}%)")
    logger.info("")
    logger.info("Per-city class balance:")
    for city, grp in df.groupby("city_key"):
        hw = int(grp["heatwave_next_day"].sum())
        tot = int(grp["heatwave_next_day"].count())
        pct = hw / tot * 100 if tot > 0 else 0.0
        logger.info(f"  {city:12s}  hw={hw:4d}  total={tot:6,}  hw%={pct:.2f}%")
    logger.info("")

    # ── 9. Save feature groups JSON ───────────────────────────────────────────
    with open(GROUPS_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Feature group registry saved → {GROUPS_FILE}")

    # ── 10. Save output CSV ────────────────────────────────────────────────────
    # Column ordering: IDs first, then all features, target last
    id_cols = ["city", "city_key", "state", "region_type",
               "latitude", "longitude", "date"]
    event_cols = ["heatwave", "hw_event_id", "hw_event_start",
                  "hw_event_end", "hw_event_length",
                  "tmax_normal", "qualifying_day"]
    # All other feature cols (exclude id/event/season string/target)
    exclude = set(id_cols + event_cols +
                  ["heatwave_next_day", "season"])  # keep season_code, drop string version
    feature_cols_ordered = [c for c in df.columns
                             if c not in exclude and c != "heatwave_next_day"]

    final_cols = (id_cols + event_cols + feature_cols_ordered + ["heatwave_next_day"])
    # Ensure no duplicates
    seen = set()
    final_cols_dedup = []
    for c in final_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            final_cols_dedup.append(c)

    df_out = df[final_cols_dedup]

    df_out.to_csv(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    logger.info(f"Output saved  → {OUTPUT_FILE}")
    logger.info(f"  Shape  : {df_out.shape[0]:,} rows × {df_out.shape[1]} columns")
    logger.info(f"  Size   : {size_mb:.2f} MB")
    logger.info("")

    # ── 11. Final summary ──────────────────────────────────────────────────────
    logger.info("═" * 70)
    logger.info("PHASE 7 COMPLETE")
    logger.info("═" * 70)
    logger.info(f"  Input rows       : 65,135  (5 cities × 13,027 days)")
    logger.info(f"  Output rows      : {df_out.shape[0]:,}  (dropped {65135 - df_out.shape[0]:,} incomplete rows)")
    logger.info(f"  Total columns    : {df_out.shape[1]}")
    logger.info(f"  Feature cols     : {len(feature_cols_ordered)}")
    logger.info(f"  Baseline set     : {len(registry['baseline_features'])} features")
    logger.info(f"  Temporal set     : {len(registry['temporal_features'])} features")
    logger.info(f"  Target column    : heatwave_next_day")
    logger.info(f"  Leakage audit    : PASSED")
    logger.info(f"  Output file      : {OUTPUT_FILE}")
    logger.info(f"  Feature registry : {GROUPS_FILE}")
    logger.info("═" * 70)


if __name__ == "__main__":
    main()

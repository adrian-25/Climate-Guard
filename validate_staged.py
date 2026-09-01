"""
validate_staged.py
ClimateGuard — Validates all 4 newly staged city files before promotion.

Checks ALL 11 required criteria:
  1.  All 4 target cities present in staging
  2.  ~13,027 rows per city
  3.  Date range 1990-01-01 to 2025-08-31
  4.  No missing dates
  5.  No duplicate city+date combinations
  6.  All 16 weather variables present
  7.  Previously empty 10 variables now populated (>0 non-null)
  8.  Missing-value percentages per city per variable
  9.  Data types correct
  10. Physically impossible values check
  11. Latitude/longitude and metadata consistency

Usage:
  python validate_staged.py

Exits 0 if all checks pass, 1 if any check fails.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

STAGING_DIR = Path("data/raw_staging")
RAW_DIR     = Path("data/raw")

START_DATE = "1990-01-01"
END_DATE   = "2025-08-31"

EXPECTED_ROWS_MIN = 13020
EXPECTED_ROWS_MAX = 13035

WEATHER_COLS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

# The 10 columns that were 100% empty for lucknow/nagpur/ahmedabad
PREVIOUSLY_EMPTY = [
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

TARGET_CITIES = {
    "lucknow":   {"lat": 26.8467, "lon": 80.9462, "region": "plains",  "state": "Uttar Pradesh"},
    "nagpur":    {"lat": 21.1458, "lon": 79.0882, "region": "plains",  "state": "Maharashtra"},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714, "region": "plains",  "state": "Gujarat"},
    "mumbai":    {"lat": 19.0760, "lon": 72.8777, "region": "coastal", "state": "Maharashtra"},
}

PHYSICAL_RANGES = {
    "temperature_2m_max":         (-20, 60),
    "temperature_2m_min":         (-20, 50),
    "temperature_2m_mean":        (-20, 55),
    "apparent_temperature_max":   (-30, 70),
    "apparent_temperature_min":   (-30, 60),
    "apparent_temperature_mean":  (-30, 65),
    "precipitation_sum":          (0,   600),
    "rain_sum":                   (0,   600),
    "wind_speed_10m_max":         (0,   200),
    "wind_gusts_10m_max":         (0,   300),
    "relative_humidity_2m_max":   (0,   100),
    "relative_humidity_2m_min":   (0,   100),
    "relative_humidity_2m_mean":  (0,   100),
    "surface_pressure_mean":      (800, 1100),
    "shortwave_radiation_sum":    (0,    50),
    "et0_fao_evapotranspiration": (0,    20),
}

fail_count = 0
warn_count = 0


def PASS(msg):  print(f"    [PASS] {msg}")
def FAIL(msg):
    global fail_count
    fail_count += 1
    print(f"    [FAIL] {msg}")
def WARN(msg):
    global warn_count
    warn_count += 1
    print(f"    [WARN] {msg}")


print("=" * 68)
print("ClimateGuard — Staged Data Validation")
print("=" * 68)

# ── CHECK 1: Files exist ───────────────────────────────────────────────────────
print("\n[CHECK 1] Staged files exist")
for ck in TARGET_CITIES:
    fp = STAGING_DIR / f"{ck}_era5_raw.csv"
    if fp.exists():
        size_mb = fp.stat().st_size / 1e6
        PASS(f"{ck}_era5_raw.csv  ({size_mb:.2f} MB)")
    else:
        FAIL(f"{ck}_era5_raw.csv  NOT FOUND in {STAGING_DIR}")

# ── Load all staged files ─────────────────────────────────────────────────────
staged = {}
for ck in TARGET_CITIES:
    fp = STAGING_DIR / f"{ck}_era5_raw.csv"
    if fp.exists():
        staged[ck] = pd.read_csv(fp, parse_dates=["date"])

if not staged:
    print("\nFATAL: No staged files loaded. Cannot continue validation.")
    sys.exit(1)

# ── Per-city validation ────────────────────────────────────────────────────────
for ck, meta in TARGET_CITIES.items():
    if ck not in staged:
        continue
    df = staged[ck]
    print(f"\n{'='*68}")
    print(f"  CITY: {ck.upper()} ({len(df)} rows)")
    print(f"{'='*68}")

    # CHECK 2: Row count
    print("\n  [CHECK 2] Row count")
    if EXPECTED_ROWS_MIN <= len(df) <= EXPECTED_ROWS_MAX:
        PASS(f"{len(df)} rows (expected {EXPECTED_ROWS_MIN}–{EXPECTED_ROWS_MAX})")
    else:
        FAIL(f"{len(df)} rows — outside expected range [{EXPECTED_ROWS_MIN}, {EXPECTED_ROWS_MAX}]")

    # CHECK 3: Date range
    print("\n  [CHECK 3] Date range")
    actual_start = df["date"].min().date()
    actual_end   = df["date"].max().date()
    if str(actual_start) == START_DATE and str(actual_end) == END_DATE:
        PASS(f"{actual_start} to {actual_end}")
    else:
        FAIL(f"Got {actual_start} to {actual_end}, expected {START_DATE} to {END_DATE}")

    # CHECK 4: No missing dates
    print("\n  [CHECK 4] No missing dates")
    full_range    = pd.date_range(START_DATE, END_DATE, freq="D")
    missing_dates = full_range.difference(df["date"])
    if len(missing_dates) == 0:
        PASS("No missing dates")
    else:
        FAIL(f"{len(missing_dates)} missing dates. First few: {[str(d.date()) for d in missing_dates[:5]]}")

    # CHECK 5: No duplicate dates
    print("\n  [CHECK 5] No duplicate dates")
    dup_dates = df["date"].duplicated().sum()
    if dup_dates == 0:
        PASS("No duplicate dates")
    else:
        FAIL(f"{dup_dates} duplicate dates found")

    # CHECK 6: All 16 columns present
    print("\n  [CHECK 6] All 16 weather variables present")
    all_present = True
    for col in WEATHER_COLS:
        if col not in df.columns:
            FAIL(f"Column missing: {col}")
            all_present = False
    if all_present:
        PASS("All 16 weather columns present")

    # CHECK 7: Previously empty columns now populated
    print("\n  [CHECK 7] Previously empty columns now populated")
    for col in PREVIOUSLY_EMPTY:
        if col not in df.columns:
            FAIL(f"Column still missing: {col}")
            continue
        null_pct = df[col].isnull().mean() * 100
        if df[col].isnull().all():
            FAIL(f"{col} — still 100% empty!")
        elif null_pct > 1.0:
            WARN(f"{col} — {null_pct:.2f}% missing (>1% threshold)")
        else:
            PASS(f"{col} — populated ({null_pct:.4f}% missing)")

    # CHECK 8: Missing-value percentages (all columns)
    print("\n  [CHECK 8] Missing-value % — all columns")
    print(f"    {'Column':<40} {'Missing':>8} {'Pct':>8}")
    print("    " + "-" * 60)
    for col in WEATHER_COLS:
        n   = df[col].isnull().sum()
        pct = n / len(df) * 100
        flag = "  <-- ATTENTION" if pct > 1.0 else ""
        print(f"    {col:<40} {n:>8} {pct:>7.4f}%{flag}")

    # CHECK 9: Data types
    print("\n  [CHECK 9] Data types")
    for col in WEATHER_COLS:
        if col in df.columns:
            dtype = str(df[col].dtype)
            ok = "int" in dtype or "float" in dtype
            if ok:
                PASS(f"{col}: {dtype}")
            else:
                FAIL(f"{col}: unexpected dtype {dtype}")

    # CHECK 10: Physical ranges
    print("\n  [CHECK 10] Physical range validation")
    range_ok = True
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        below = int((vals < lo).sum())
        above = int((vals > hi).sum())
        if below == 0 and above == 0:
            PASS(f"{col}: [{vals.min():.2f}, {vals.max():.2f}] within [{lo}, {hi}]")
        else:
            FAIL(f"{col}: {below} below {lo}, {above} above {hi}. Range: [{vals.min():.2f}, {vals.max():.2f}]")
            range_ok = False

    # CHECK 11: Metadata consistency
    print("\n  [CHECK 11] Metadata consistency")
    for field, expected_val in [
        ("latitude",    meta["lat"]),
        ("longitude",   meta["lon"]),
        ("region_type", meta["region"]),
        ("state",       meta["state"]),
    ]:
        if field not in df.columns:
            FAIL(f"Metadata column missing: {field}")
            continue
        uniq = df[field].unique().tolist()
        if len(uniq) == 1 and str(uniq[0]) == str(expected_val):
            PASS(f"{field}: {uniq[0]}")
        elif len(uniq) == 1:
            FAIL(f"{field}: got {uniq[0]}, expected {expected_val}")
        else:
            FAIL(f"{field}: inconsistent values: {uniq}")

# ── CROSS-CITY SUMMARY TABLE ───────────────────────────────────────────────────
print()
print("=" * 68)
print("CROSS-CITY MISSING-VALUE SUMMARY TABLE")
print("=" * 68)
header = f"{'Variable':<40}" + "".join(f"{ck:>12}" for ck in TARGET_CITIES)
print(header)
print("-" * (40 + 12 * len(TARGET_CITIES)))
for col in WEATHER_COLS:
    row_str = f"{col:<40}"
    for ck in TARGET_CITIES:
        if ck in staged and col in staged[ck].columns:
            pct = staged[ck][col].isnull().mean() * 100
            row_str += f"{pct:>11.2f}%"
        else:
            row_str += f"{'N/A':>12}"
    print(row_str)

# ── FINAL VERDICT ──────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("VALIDATION RESULT")
print("=" * 68)
print(f"  Failures : {fail_count}")
print(f"  Warnings : {warn_count}")
print()
if fail_count == 0:
    print("  STAGED DATA INTEGRITY STATUS: PASS")
    print("  Safe to run promote_staged.py")
    sys.exit(0)
else:
    print("  STAGED DATA INTEGRITY STATUS: FAIL")
    print("  DO NOT promote. Fix issues and re-run download_safe.py")
    sys.exit(1)

"""
inspect_raw.py
ClimateGuard - Step 1: Raw Data Inspection
Run: python inspect_raw.py
Outputs a full data-quality report to the console and saves it as
results/data_quality_report_raw.txt
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR     = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "data_quality_report_raw.txt"

EXPECTED_CITIES = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
ACTUAL_CITIES_FOUND = []  # will be populated after loading

WEATHER_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

META_COLS = ["city", "city_key", "latitude", "longitude", "region_type", "state", "date"]

lines = []

def p(s=""):
    print(s)
    lines.append(s)

def section(title):
    p()
    p("=" * 70)
    p(f"  {title}")
    p("=" * 70)

def subsection(title):
    p()
    p(f"--- {title} ---")

# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
section("1. FILE INVENTORY")
csv_files = sorted(RAW_DIR.glob("*.csv"))
p(f"Directory : {RAW_DIR.resolve()}")
p(f"CSV files found: {len(csv_files)}")
p()
for f in csv_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    p(f"  {f.name:<40} {size_mb:.2f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD COMBINED FILE
# ─────────────────────────────────────────────────────────────────────────────
section("2. COMBINED FILE — all_cities_era5_raw.csv")
combined_path = RAW_DIR / "all_cities_era5_raw.csv"

if not combined_path.exists():
    p("FATAL: combined file not found. Aborting.")
    sys.exit(1)

df = pd.read_csv(combined_path)
p(f"Shape        : {df.shape[0]} rows  x  {df.shape[1]} columns")

# Column list
p()
p("Columns:")
for i, col in enumerate(df.columns, 1):
    p(f"  {i:2d}. {col}")

# Data types
p()
p("Data types:")
for col, dtype in df.dtypes.items():
    p(f"  {col:<40} {dtype}")

# Parse date properly
df["date"] = pd.to_datetime(df["date"])

# ─────────────────────────────────────────────────────────────────────────────
# 3. CITY COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
section("3. CITY COVERAGE")
city_counts = df.groupby("city_key").size().sort_index()
p(f"{'City key':<15} {'City name':<20} {'Region':<10} {'State':<20} {'Rows':>7}")
p("-" * 80)
for ck, count in city_counts.items():
    sub = df[df["city_key"] == ck].iloc[0]
    p(f"  {ck:<13} {sub['city']:<20} {sub['region_type']:<10} {sub['state']:<20} {count:>7}")
p()
missing_cities = [c for c in EXPECTED_CITIES if c not in city_counts.index]
if missing_cities:
    p(f"MISSING expected cities: {missing_cities}")
else:
    p("All 5 expected cities present.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATE RANGE AND COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
section("4. DATE RANGE AND COVERAGE PER CITY")
p(f"{'City':<15} {'Start':<12} {'End':<12} {'Rows':>7} {'Expected':>9} {'Missing dates':>14} {'Dup dates':>10}")
p("-" * 85)
for ck in sorted(df["city_key"].unique()):
    sub = df[df["city_key"] == ck].copy()
    sub_dates = sub["date"].sort_values().reset_index(drop=True)
    start     = sub_dates.min()
    end       = sub_dates.max()
    full_range = pd.date_range(start=start, end=end, freq="D")
    expected  = len(full_range)
    actual    = len(sub)
    missing_n = len(full_range.difference(sub_dates))
    dup_n     = sub_dates.duplicated().sum()
    p(f"  {ck:<13} {str(start.date()):<12} {str(end.date()):<12} {actual:>7} {expected:>9} {missing_n:>14} {dup_n:>10}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DUPLICATE ROW CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("5. DUPLICATE ROW CHECK")
full_dups = df.duplicated().sum()
city_date_dups = df.duplicated(subset=["city_key", "date"]).sum()
p(f"Fully duplicate rows    : {full_dups}")
p(f"city_key + date duplicates : {city_date_dups}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MISSING VALUE SUMMARY — OVERALL
# ─────────────────────────────────────────────────────────────────────────────
section("6. MISSING VALUES — OVERALL (ALL CITIES COMBINED)")
null_counts = df[WEATHER_COLS].isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(4)
p(f"{'Variable':<40} {'Missing':>8} {'Pct (%)':>10}")
p("-" * 62)
for col in WEATHER_COLS:
    p(f"  {col:<38} {null_counts[col]:>8} {null_pct[col]:>10.4f}")
total_cells = len(df) * len(WEATHER_COLS)
total_null  = null_counts.sum()
p()
p(f"Total cells (weather cols) : {total_cells}")
p(f"Total missing              : {total_null}")
p(f"Overall missing pct        : {total_null/total_cells*100:.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MISSING VALUES — PER CITY PER VARIABLE
# ─────────────────────────────────────────────────────────────────────────────
section("7. MISSING VALUES — PER CITY PER VARIABLE")
for ck in sorted(df["city_key"].unique()):
    sub = df[df["city_key"] == ck]
    subsection(f"{ck.upper()} ({len(sub)} rows)")
    p(f"  {'Variable':<40} {'Missing':>8} {'Pct (%)':>10}")
    p("  " + "-" * 60)
    any_missing = False
    for col in WEATHER_COLS:
        n = sub[col].isnull().sum()
        pct = n / len(sub) * 100
        flag = "  <-- ATTENTION" if pct > 1 else ""
        if n > 0:
            any_missing = True
        p(f"    {col:<38} {n:>8} {pct:>10.4f}{flag}")
    if not any_missing:
        p("    No missing values.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CONSTANT COLUMNS CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("8. CONSTANT COLUMNS CHECK")
p("Checking for columns with zero variance (constant values)...")
for col in WEATHER_COLS:
    if df[col].nunique(dropna=True) <= 1:
        p(f"  CONSTANT: {col}  (unique value: {df[col].unique()})")
    else:
        p(f"  OK      : {col}  ({df[col].nunique()} unique values)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. BASIC STATISTICS PER VARIABLE (ALL CITIES COMBINED)
# ─────────────────────────────────────────────────────────────────────────────
section("9. DESCRIPTIVE STATISTICS — ALL CITIES COMBINED")
desc = df[WEATHER_COLS].describe().T
p(f"{'Variable':<40} {'min':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'max':>8} {'mean':>8}")
p("-" * 96)
for col in WEATHER_COLS:
    row = desc.loc[col]
    p(f"  {col:<38} {row['min']:>8.2f} {row['25%']:>8.2f} {row['50%']:>8.2f} {row['75%']:>8.2f} {row['max']:>8.2f} {row['mean']:>8.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PHYSICAL RANGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
section("10. PHYSICAL RANGE VALIDATION")

checks = {
    "temperature_2m_max":          (-20, 60,  "°C — plausible for India: -10 to 55"),
    "temperature_2m_min":          (-20, 50,  "°C — plausible for India: -5 to 40"),
    "temperature_2m_mean":         (-20, 55,  "°C"),
    "apparent_temperature_max":    (-30, 70,  "°C — feels-like can exceed actual"),
    "apparent_temperature_min":    (-30, 60,  "°C"),
    "apparent_temperature_mean":   (-30, 65,  "°C"),
    "precipitation_sum":           (0,  500,  "mm/day — 500mm extreme but possible in monsoon"),
    "rain_sum":                    (0,  500,  "mm/day"),
    "wind_speed_10m_max":          (0,  200,  "km/h"),
    "wind_gusts_10m_max":          (0,  300,  "km/h"),
    "relative_humidity_2m_max":    (0,  100,  "%"),
    "relative_humidity_2m_min":    (0,  100,  "%"),
    "relative_humidity_2m_mean":   (0,  100,  "%"),
    "surface_pressure_mean":       (800, 1100,"hPa"),
    "shortwave_radiation_sum":     (0,  50,   "MJ/m² — daily total"),
    "et0_fao_evapotranspiration":  (0,  20,   "mm/day"),
}

for col, (lo, hi, note) in checks.items():
    if col not in df.columns:
        p(f"  MISSING COL: {col}")
        continue
    below = (df[col] < lo).sum()
    above = (df[col] > hi).sum()
    neg   = (df[col] < 0).sum() if lo >= 0 else 0
    status = "OK" if (below == 0 and above == 0) else "FLAG"
    p(f"  [{status}] {col:<38}  range [{lo}, {hi}]  |  below: {below}  above: {above}  negative: {neg}  ({note})")
    if below > 0:
        vals = df.loc[df[col] < lo, col]
        p(f"         Below-range examples: {vals.head(5).tolist()}")
    if above > 0:
        vals = df.loc[df[col] > hi, col]
        p(f"         Above-range examples: {vals.head(5).tolist()}")

# Special: humidity can't exceed 100 or be negative
for hcol in ["relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean"]:
    over100 = (df[hcol] > 100).sum()
    neg     = (df[hcol] < 0).sum()
    if over100 > 0 or neg > 0:
        p(f"  HUMIDITY ALERT {hcol}: >100 => {over100}, <0 => {neg}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. META-COLUMN CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────
section("11. META-COLUMN CONSISTENCY")
for ck in sorted(df["city_key"].unique()):
    sub = df[df["city_key"] == ck]
    p(f"\n  {ck}:")
    for col in ["city", "latitude", "longitude", "region_type", "state"]:
        unique_vals = sub[col].unique()
        status = "OK" if len(unique_vals) == 1 else "INCONSISTENT"
        p(f"    [{status}] {col}: {unique_vals.tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. INDIVIDUAL CITY FILES CHECK
# ─────────────────────────────────────────────────────────────────────────────
section("12. INDIVIDUAL CITY FILES — ROW COUNTS vs COMBINED")
p(f"{'File':<40} {'File rows':>10} {'In combined':>12} {'Match?':>8}")
p("-" * 74)
for ck in EXPECTED_CITIES:
    fp = RAW_DIR / f"{ck}_era5_raw.csv"
    if fp.exists():
        dfc = pd.read_csv(fp)
        file_rows = len(dfc)
    else:
        file_rows = "MISSING"
    combined_rows = int((df["city_key"] == ck).sum())
    match = "YES" if str(file_rows) == str(combined_rows) else "NO"
    p(f"  {ck+'_era5_raw.csv':<38} {str(file_rows):>10} {combined_rows:>12} {match:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("13. SUMMARY")
p(f"Total rows in combined file : {len(df)}")
p(f"Total columns               : {len(df.columns)}")
p(f"Cities present              : {sorted(df['city_key'].unique().tolist())}")
p(f"Date range (combined)       : {df['date'].min().date()} to {df['date'].max().date()}")
p(f"Duplicate rows              : {full_dups}")
p(f"city+date duplicates        : {city_date_dups}")
p(f"Weather variables           : {len(WEATHER_COLS)}")
total_null = df[WEATHER_COLS].isnull().sum().sum()
p(f"Total missing weather vals  : {total_null}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────────────────────────────────────────
report_text = "\n".join(lines)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nReport saved to: {OUTPUT_FILE}")

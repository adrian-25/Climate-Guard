"""
promote_staged.py
ClimateGuard — Promotes validated staged files to data/raw/ and rebuilds combined CSV.

Only run this AFTER validate_staged.py exits 0 (PASS).

What this script does:
  1. Copies staged city files (lucknow, nagpur, ahmedabad, mumbai) into data/raw/
  2. Rebuilds all_cities_era5_raw.csv from all 5 final city files
  3. Prints a final row-count and completeness summary

What it does NOT do:
  - Modify data/raw/delhi_era5_raw.csv (already complete)
  - Delete staging files (kept for reference)
  - Delete backup files in data/raw_backup/

Usage:
  python promote_staged.py
"""

import shutil
import pandas as pd
from pathlib import Path

STAGING_DIR = Path("data/raw_staging")
RAW_DIR     = Path("data/raw")
BACKUP_DIR  = Path("data/raw_backup")

ALL_CITIES  = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]

WEATHER_COLS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

print("=" * 65)
print("ClimateGuard — Promote Staged Files to data/raw/")
print("=" * 65)

# ── STEP 1: Copy staged files ─────────────────────────────────────────────────
PROMOTE_CITIES = ["lucknow", "nagpur", "ahmedabad", "mumbai"]

print("\nCopying staged files to data/raw/ ...")
for ck in PROMOTE_CITIES:
    src = STAGING_DIR / f"{ck}_era5_raw.csv"
    dst = RAW_DIR     / f"{ck}_era5_raw.csv"
    if not src.exists():
        print(f"  [FAIL] Staged file not found: {src}")
        raise FileNotFoundError(f"Missing staged file: {src}")
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / 1e6
    print(f"  Promoted: {ck}_era5_raw.csv  ({size_mb:.2f} MB)")

# ── STEP 2: Rebuild combined CSV ──────────────────────────────────────────────
print("\nRebuilding all_cities_era5_raw.csv ...")
dfs = []
for ck in ALL_CITIES:
    fp = RAW_DIR / f"{ck}_era5_raw.csv"
    if not fp.exists():
        print(f"  [MISSING] {ck}_era5_raw.csv — skipping")
        continue
    df = pd.read_csv(fp, parse_dates=["date"])
    empty_cols = [c for c in WEATHER_COLS if c in df.columns and df[c].isnull().all()]
    status = "COMPLETE" if not empty_cols else f"INCOMPLETE: {empty_cols}"
    print(f"  {ck:<14} {len(df)} rows  [{status}]")
    dfs.append(df)

if not dfs:
    print("ERROR: No city files loaded. Cannot build combined CSV.")
    raise RuntimeError("No city files found.")

combined = pd.concat(dfs, ignore_index=True)
combined_path = RAW_DIR / "all_cities_era5_raw.csv"
combined.to_csv(combined_path, index=False)

print(f"\nCombined file : {combined_path}")
print(f"Total rows    : {len(combined)}")
print(f"Total columns : {len(combined.columns)}")
print(f"Cities        : {sorted(combined['city_key'].unique().tolist())}")

# ── STEP 3: Final summary ─────────────────────────────────────────────────────
print()
print("=" * 65)
print("PROMOTION COMPLETE — data/raw/ contents:")
print("=" * 65)
for fp in sorted(RAW_DIR.glob("*.csv")):
    size_mb = fp.stat().st_size / 1e6
    print(f"  {fp.name:<40} {size_mb:.2f} MB")

print()
print("Backup preserved in:", BACKUP_DIR.resolve())
print("Staging files still at:", STAGING_DIR.resolve())
print()
print("Next step: run final_validation.py to confirm data/raw/ integrity.")

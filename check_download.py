"""
check_download.py
Verify that the ERA5-Land download completed correctly.
Run: python check_download.py
"""
import pandas as pd
import os

RAW_DIR = os.path.join("data", "raw")

EXPECTED_CITIES = {
    "delhi":       {"region": "plains",  "tmax_threshold": 40},
    "lucknow":     {"region": "plains",  "tmax_threshold": 40},
    "nagpur":      {"region": "plains",  "tmax_threshold": 40},
    "ahmedabad":   {"region": "plains",  "tmax_threshold": 40},
    "bhubaneswar": {"region": "coastal", "tmax_threshold": 37},
}

EXPECTED_ROWS_MIN = 12900
EXPECTED_ROWS_MAX = 13100

print("=" * 60)
print("ClimateGuard -- Download Verification")
print("=" * 60)

all_ok = True

for city_key, meta in EXPECTED_CITIES.items():
    filepath = os.path.join(RAW_DIR, f"{city_key}_era5_raw.csv")
    print(f"\n--- {city_key.upper()} ({meta['region']}, threshold {meta['tmax_threshold']}C) ---")

    if not os.path.exists(filepath):
        print(f"  [FAIL] File not found: {filepath}")
        all_ok = False
        continue

    df = pd.read_csv(filepath, parse_dates=["date"])

    n = len(df)
    row_ok = EXPECTED_ROWS_MIN <= n <= EXPECTED_ROWS_MAX
    print(f"  Rows       : {n}  [{'OK' if row_ok else 'WARNING'}]  (expected {EXPECTED_ROWS_MIN}-{EXPECTED_ROWS_MAX})")
    if not row_ok:
        all_ok = False

    print(f"  Date range : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Columns    : {len(df.columns)}")

    tmax_col = "temperature_2m_max"
    if tmax_col in df.columns:
        tmax_min = df[tmax_col].min()
        tmax_max = df[tmax_col].max()
        temp_ok  = (0 <= tmax_min) and (tmax_max <= 55)
        print(f"  Tmax range : {tmax_min:.1f}C to {tmax_max:.1f}C  [{'OK' if temp_ok else 'WARNING'}]")
        if not temp_ok:
            all_ok = False

        threshold = meta["tmax_threshold"]
        hot_days  = (df[tmax_col] >= threshold).sum()
        hot_pct   = hot_days / len(df) * 100
        print(f"  Days >={threshold}C  : {hot_days} ({hot_pct:.1f}%) -- candidate heatwave days")
    else:
        print(f"  [FAIL] Column '{tmax_col}' not found")
        all_ok = False

    null_pct     = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = null_pct[null_pct > 5]
    if len(high_missing) > 0:
        print(f"  Missing    : WARNING -- {high_missing.to_dict()}")
        all_ok = False
    else:
        print(f"  Missing    : OK (max {null_pct.max():.2f}% in any column)")

print(f"\n--- COMBINED FILE ---")
combined_path = os.path.join(RAW_DIR, "all_cities_era5_raw.csv")

if not os.path.exists(combined_path):
    print(f"  [FAIL] Not found: {combined_path}")
    all_ok = False
else:
    df_all = pd.read_csv(combined_path, parse_dates=["date"])
    print(f"  Total rows : {len(df_all)}")
    print(f"  Columns    : {len(df_all.columns)}")
    print(f"  Cities     : {sorted(df_all['city_key'].unique().tolist())}")
    print(f"  Rows/city  :")
    for city, count in df_all.groupby("city_key").size().items():
        print(f"    {city}: {count}")

print(f"\n{'=' * 60}")
if all_ok:
    print("RESULT: ALL CHECKS PASSED -- ready for Phase 4 (EDA)")
else:
    print("RESULT: SOME CHECKS FAILED -- review warnings above")
print("=" * 60)

"""
run_inspection.py
ClimateGuard - Complete raw data inspection.
Run: python run_inspection.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")

WEATHER_COLS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_DIR / "all_cities_era5_raw.csv")
df["date"] = pd.to_datetime(df["date"])

print("=" * 70)
print("SECTION 1: SHAPE & SCHEMA")
print("=" * 70)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print()
print("Column names & dtypes:")
for col, dtype in df.dtypes.items():
    print(f"  {col:<40} {str(dtype)}")

# ── CITY COVERAGE ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 2: CITY COVERAGE")
print("=" * 70)
city_counts = df.groupby("city_key").size()
for ck, cnt in city_counts.items():
    row = df[df["city_key"] == ck].iloc[0]
    print(f"  {ck:<14} | {row['city']:<20} | {row['region_type']:<8} | {row['state']:<20} | rows={cnt}")

EXPECTED = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
ACTUAL   = sorted(df["city_key"].unique().tolist())
print()
print(f"Expected cities : {EXPECTED}")
print(f"Actual cities   : {ACTUAL}")
missing = [c for c in EXPECTED if c not in ACTUAL]
extra   = [c for c in ACTUAL if c not in EXPECTED]
print(f"MISSING from expected : {missing}")
print(f"EXTRA (not expected)  : {extra}")

# ── DATE COVERAGE ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 3: DATE COVERAGE PER CITY")
print("=" * 70)
print(f"  {'City':<14} {'Start':<12} {'End':<12} {'Rows':>7} {'Expected':>9} {'MissDates':>10} {'DupDates':>9}")
print("  " + "-" * 80)
for ck in sorted(df["city_key"].unique()):
    sub   = df[df["city_key"] == ck]
    dates = sub["date"].sort_values()
    full  = pd.date_range(dates.min(), dates.max(), freq="D")
    miss  = len(full.difference(dates))
    dups  = int(dates.duplicated().sum())
    print(f"  {ck:<14} {str(dates.min().date()):<12} {str(dates.max().date()):<12} {len(sub):>7} {len(full):>9} {miss:>10} {dups:>9}")

# ── DUPLICATES ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 4: DUPLICATE ROWS")
print("=" * 70)
full_dups = df.duplicated().sum()
key_dups  = df.duplicated(subset=["city_key", "date"]).sum()
print(f"  Full duplicate rows          : {full_dups}")
print(f"  city_key + date duplicates   : {key_dups}")

# ── MISSING VALUES (OVERALL) ──────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 5: MISSING VALUES — ALL CITIES COMBINED")
print("=" * 70)
nc  = df[WEATHER_COLS].isnull().sum()
pct = (nc / len(df) * 100).round(4)
print(f"  {'Variable':<40} {'Count':>8} {'Pct':>8}")
print("  " + "-" * 60)
for col in WEATHER_COLS:
    flag = "  <-- ATTENTION" if pct[col] > 0.5 else ""
    print(f"  {col:<40} {nc[col]:>8} {pct[col]:>8.4f}%{flag}")
total_miss = nc.sum()
total_cell = len(df) * len(WEATHER_COLS)
print(f"\n  Total weather cells  : {total_cell}")
print(f"  Total missing        : {total_miss}")
print(f"  Overall missing pct  : {total_miss / total_cell * 100:.4f}%")

# ── MISSING VALUES PER CITY ───────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 6: MISSING VALUES — PER CITY")
print("=" * 70)
for ck in sorted(df["city_key"].unique()):
    sub = df[df["city_key"] == ck]
    print(f"\n  [{ck.upper()}] ({len(sub)} rows)")
    any_miss = False
    for col in WEATHER_COLS:
        n = sub[col].isnull().sum()
        p = n / len(sub) * 100
        if n > 0:
            any_miss = True
            print(f"    {col:<40} missing={n}  ({p:.4f}%)")
    if not any_miss:
        print("    No missing values.")

# ── DESCRIPTIVE STATS ─────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 7: DESCRIPTIVE STATISTICS (ALL CITIES)")
print("=" * 70)
desc = df[WEATHER_COLS].describe().T
print(f"  {'Variable':<40} {'min':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'max':>8} {'mean':>8}")
print("  " + "-" * 96)
for col in WEATHER_COLS:
    r = desc.loc[col]
    print(f"  {col:<40} {r['min']:>8.2f} {r['25%']:>8.2f} {r['50%']:>8.2f} {r['75%']:>8.2f} {r['max']:>8.2f} {r['mean']:>8.2f}")

# ── PHYSICAL VALIDITY ─────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 8: PHYSICAL RANGE CHECKS")
print("=" * 70)

RANGES = {
    "temperature_2m_max":         (-20, 60),
    "temperature_2m_min":         (-20, 50),
    "temperature_2m_mean":        (-20, 55),
    "apparent_temperature_max":   (-30, 70),
    "apparent_temperature_min":   (-30, 60),
    "apparent_temperature_mean":  (-30, 65),
    "precipitation_sum":          (0,  600),
    "rain_sum":                   (0,  600),
    "wind_speed_10m_max":         (0,  200),
    "wind_gusts_10m_max":         (0,  300),
    "relative_humidity_2m_max":   (0,  100),
    "relative_humidity_2m_min":   (0,  100),
    "relative_humidity_2m_mean":  (0,  100),
    "surface_pressure_mean":      (800, 1100),
    "shortwave_radiation_sum":    (0,   50),
    "et0_fao_evapotranspiration": (0,   20),
}

all_ok = True
for col, (lo, hi) in RANGES.items():
    sub_col = df[col].dropna()
    below = int((sub_col < lo).sum())
    above = int((sub_col > hi).sum())
    actual_min = sub_col.min()
    actual_max = sub_col.max()
    status = "OK  " if (below == 0 and above == 0) else "FLAG"
    if status == "FLAG":
        all_ok = False
    print(f"  [{status}] {col:<40} actual=[{actual_min:.2f}, {actual_max:.2f}]  below_range={below}  above_range={above}")
    if below > 0:
        examples = df.loc[df[col] < lo, ["city_key", "date", col]].head(3)
        for _, row in examples.iterrows():
            print(f"         BELOW: {row['city_key']} {row['date'].date()} = {row[col]}")
    if above > 0:
        examples = df.loc[df[col] > hi, ["city_key", "date", col]].head(3)
        for _, row in examples.iterrows():
            print(f"         ABOVE: {row['city_key']} {row['date'].date()} = {row[col]}")

if all_ok:
    print("\n  All columns within physical bounds.")

# ── CONSTANT COLUMNS ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 9: CONSTANT / LOW-VARIANCE COLUMNS")
print("=" * 70)
for col in WEATHER_COLS:
    nuniq = df[col].nunique(dropna=True)
    std   = df[col].std()
    if nuniq <= 1:
        print(f"  CONSTANT : {col}  (unique={nuniq})")
    elif std < 0.01:
        print(f"  NEAR-ZERO VARIANCE : {col}  std={std:.6f}")
    else:
        print(f"  OK       : {col}  unique={nuniq}  std={std:.4f}")

# ── META CONSISTENCY ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 10: META-COLUMN CONSISTENCY PER CITY")
print("=" * 70)
for ck in sorted(df["city_key"].unique()):
    sub = df[df["city_key"] == ck]
    print(f"\n  [{ck}]")
    for col in ["city", "latitude", "longitude", "region_type", "state"]:
        uniq = sub[col].unique().tolist()
        status = "OK" if len(uniq) == 1 else "INCONSISTENT"
        print(f"    [{status}] {col}: {uniq}")

# ── INDIVIDUAL FILES vs COMBINED ──────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 11: INDIVIDUAL FILE ROWS vs COMBINED")
print("=" * 70)
for ck in EXPECTED:
    fp = RAW_DIR / f"{ck}_era5_raw.csv"
    if fp.exists():
        n_file = len(pd.read_csv(fp))
        n_comb = int((df["city_key"] == ck).sum())
        match  = "YES" if n_file == n_comb else "NO"
        print(f"  {ck:<14} file={n_file}  combined={n_comb}  match={match}")
    else:
        n_comb = int((df["city_key"] == ck).sum())
        print(f"  {ck:<14} file=MISSING  combined={n_comb}")

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

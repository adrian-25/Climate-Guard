"""
data_cleaning.py
ClimateGuard Phase 5 — Data Cleaning
=====================================
Input  : data/raw/all_cities_era5_raw.csv   (READ-ONLY — never modified)
Output : data/processed/weather_cleaned.csv

Cleaning actions:
  1. Load and validate raw dataset
  2. Verify date dtype and chronological ordering
  3. Verify numerical columns are numeric
  4. Verify city metadata consistency
  5. Check duplicates
  6. Check missing values
  7. Check physical ranges
  8. Remove rain_sum (identical to precipitation_sum, r=1.000)
  9. Sort by city_key, date
 10. Save to data/processed/weather_cleaned.csv

What is NOT changed:
  - No rows removed (0 invalid observations exist)
  - No imputation (0 missing values)
  - No outlier removal (extremes are the signal)
  - No lag/rolling features
  - No heatwave labels

Run:
    python data_cleaning.py
"""

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ── PATHS ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
RAW_CSV     = ROOT / "data" / "raw" / "all_cities_era5_raw.csv"
PROCESSED   = ROOT / "data" / "processed"
OUT_CSV     = PROCESSED / "weather_cleaned.csv"
DOCS_DIR    = ROOT / "docs"

PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
CITY_ORDER      = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
EXPECTED_ROWS   = 65135
EXPECTED_CITIES = 5
ROWS_PER_CITY   = 13027
DATE_START      = "1990-01-01"
DATE_END        = "2025-08-31"

WEATHER_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",                          # rain_sum will be removed
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

PHYSICAL_RANGES = {
    "temperature_2m_max":           (-20,  60),
    "temperature_2m_min":           (-20,  50),
    "temperature_2m_mean":          (-20,  55),
    "apparent_temperature_max":     (-30,  70),
    "apparent_temperature_min":     (-30,  60),
    "apparent_temperature_mean":    (-30,  65),
    "precipitation_sum":            (  0, 600),
    "wind_speed_10m_max":           (  0, 200),
    "wind_gusts_10m_max":           (  0, 300),
    "relative_humidity_2m_max":     (  0, 100),
    "relative_humidity_2m_min":     (  0, 100),
    "relative_humidity_2m_mean":    (  0, 100),
    "surface_pressure_mean":        (800,1100),
    "shortwave_radiation_sum":      (  0,  50),
    "et0_fao_evapotranspiration":   (  0,  20),
}

# ── REPORT BUFFER ──────────────────────────────────────────────────────────────
LOG = []
PASS_COUNT = 0
FAIL_COUNT = 0

def log(line=""):
    LOG.append(line)
    print(line)

def PASS(msg):
    global PASS_COUNT
    PASS_COUNT += 1
    log(f"    [PASS] {msg}")

def FAIL(msg):
    global FAIL_COUNT
    FAIL_COUNT += 1
    log(f"    [FAIL] {msg}")
    # Hard stop on any validation failure — do not produce a bad output file
    print("\n  FATAL: Validation failure detected. Halting. Do not promote output.")
    sys.exit(1)

def section(title):
    bar = "=" * 68
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)

# ── MD5 HASH OF RAW FILE (to verify it hasn't changed later) ──────────────────
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

raw_hash_before = md5(RAW_CSV)

# ══════════════════════════════════════════════════════════════════════════════
section("PHASE 5 — DATA CLEANING")
log(f"  Input  : {RAW_CSV.relative_to(ROOT)}")
log(f"  Output : {OUT_CSV.relative_to(ROOT)}")
log(f"  Raw MD5: {raw_hash_before}  (will re-check after save)")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 1 — LOAD RAW DATASET")

df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
log(f"  Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")

if len(df_raw) == EXPECTED_ROWS:
    PASS(f"Row count = {len(df_raw):,}")
else:
    FAIL(f"Row count {len(df_raw):,} ≠ expected {EXPECTED_ROWS:,}")

if df_raw.shape[1] == 23:
    PASS(f"Column count = 23")
else:
    FAIL(f"Column count {df_raw.shape[1]} ≠ expected 23")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 2 — DATE DTYPE AND CHRONOLOGICAL ORDER")

if pd.api.types.is_datetime64_any_dtype(df_raw["date"]):
    PASS("date column is datetime64")
else:
    FAIL(f"date column dtype = {df_raw['date'].dtype}  (expected datetime64)")

for ck in CITY_ORDER:
    sub   = df_raw[df_raw["city_key"] == ck]["date"]
    diffs = sub.sort_values().diff().dropna()
    if (diffs == pd.Timedelta(days=1)).all():
        PASS(f"{ck}: dates are consecutive daily (no gaps, no jumps)")
    else:
        non_one = diffs[diffs != pd.Timedelta(days=1)]
        FAIL(f"{ck}: non-1-day gaps found: {non_one.head(3).tolist()}")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 3 — DATE RANGE")

actual_start = str(df_raw["date"].min().date())
actual_end   = str(df_raw["date"].max().date())
if actual_start == DATE_START and actual_end == DATE_END:
    PASS(f"Date range {actual_start} → {actual_end}")
else:
    FAIL(f"Date range {actual_start} → {actual_end}  (expected {DATE_START} → {DATE_END})")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 4 — NUMERICAL COLUMN DTYPES")

for col in WEATHER_VARS:
    dtype = str(df_raw[col].dtype)
    if "int" in dtype or "float" in dtype:
        PASS(f"{col}: {dtype}")
    else:
        FAIL(f"{col}: dtype={dtype}  (expected numeric)")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 5 — CITY METADATA CONSISTENCY")

CITY_META = {
    "delhi":     {"city": "New Delhi",  "state": "Delhi",          "region_type": "plains",  "latitude": 28.6139, "longitude": 77.2090},
    "lucknow":   {"city": "Lucknow",    "state": "Uttar Pradesh",  "region_type": "plains",  "latitude": 26.8467, "longitude": 80.9462},
    "nagpur":    {"city": "Nagpur",     "state": "Maharashtra",    "region_type": "plains",  "latitude": 21.1458, "longitude": 79.0882},
    "ahmedabad": {"city": "Ahmedabad",  "state": "Gujarat",        "region_type": "plains",  "latitude": 23.0225, "longitude": 72.5714},
    "mumbai":    {"city": "Mumbai",     "state": "Maharashtra",    "region_type": "coastal", "latitude": 19.0760, "longitude": 72.8777},
}

for ck, expected in CITY_META.items():
    sub = df_raw[df_raw["city_key"] == ck]
    if len(sub) == 0:
        FAIL(f"{ck}: city not found in dataset")
        continue
    n = len(sub)
    if n == ROWS_PER_CITY:
        PASS(f"{ck}: {n:,} rows")
    else:
        FAIL(f"{ck}: {n:,} rows  (expected {ROWS_PER_CITY:,})")
    for field, exp_val in expected.items():
        uniq = sub[field].unique()
        if len(uniq) == 1:
            actual = uniq[0]
            if field in ("latitude", "longitude"):
                # float comparison with tolerance
                if abs(float(actual) - float(exp_val)) < 0.001:
                    PASS(f"  {ck}.{field} = {actual}")
                else:
                    FAIL(f"  {ck}.{field} = {actual}  (expected {exp_val})")
            else:
                if str(actual) == str(exp_val):
                    PASS(f"  {ck}.{field} = {actual}")
                else:
                    FAIL(f"  {ck}.{field} = {actual}  (expected {exp_val})")
        else:
            FAIL(f"  {ck}.{field}: inconsistent values {uniq}")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 6 — DUPLICATES")

full_dups = df_raw.duplicated().sum()
key_dups  = df_raw.duplicated(subset=["city_key", "date"]).sum()

if full_dups == 0:
    PASS(f"Full duplicate rows = {full_dups}")
else:
    FAIL(f"Full duplicate rows = {full_dups}  (expected 0)")

if key_dups == 0:
    PASS(f"city_key + date duplicates = {key_dups}")
else:
    FAIL(f"city_key + date duplicates = {key_dups}  (expected 0)")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 7 — MISSING VALUES")

total_miss = df_raw[WEATHER_VARS].isnull().sum().sum()
if total_miss == 0:
    PASS(f"Missing values across all weather variables = {total_miss}")
else:
    for col in WEATHER_VARS:
        n = df_raw[col].isnull().sum()
        if n > 0:
            log(f"    {col}: {n} missing")
    FAIL(f"Total missing = {total_miss}  (expected 0)")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 8 — PHYSICAL RANGES")

range_ok = True
for col, (lo, hi) in PHYSICAL_RANGES.items():
    vals   = df_raw[col].dropna()
    below  = int((vals < lo).sum())
    above  = int((vals > hi).sum())
    if below == 0 and above == 0:
        PASS(f"{col}: [{vals.min():.2f}, {vals.max():.2f}]  within [{lo}, {hi}]")
    else:
        FAIL(f"{col}: {below} below {lo}, {above} above {hi}")

# ══════════════════════════════════════════════════════════════════════════════
section("CHECK 9 — rain_sum IDENTITY VERIFICATION")

# Confirm rain_sum == precipitation_sum before removal
diff = (df_raw["rain_sum"] - df_raw["precipitation_sum"]).abs().max()
if diff == 0.0:
    PASS(f"rain_sum == precipitation_sum everywhere (max abs diff = {diff})")
else:
    FAIL(f"rain_sum != precipitation_sum in some rows (max diff = {diff})")

log()
log("  Decision: rain_sum is removed.")
log("  Reason  : It is identical to precipitation_sum (r=1.000, max diff=0.0).")
log("            Retaining it would introduce perfect multicollinearity with no")
log("            additional information for any downstream model.")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 10 — BUILD CLEANED DATASET")

# Columns to keep (ordered)
META_COLS    = ["city", "city_key", "latitude", "longitude", "region_type", "state"]
DATE_COL     = ["date"]
WEATHER_KEEP = [c for c in WEATHER_VARS if c != "rain_sum"]   # 15 variables

FINAL_COLS = META_COLS + DATE_COL + WEATHER_KEEP

log(f"  Columns in raw    : {df_raw.shape[1]}")
log(f"  Columns removed   : 1  (rain_sum)")
log(f"  Columns retained  : {len(FINAL_COLS)}")
log(f"  Rows unchanged    : {len(df_raw):,}")

# Build the cleaned dataframe — sort by city_key then date
df_clean = (
    df_raw[FINAL_COLS]
    .sort_values(["city_key", "date"])
    .reset_index(drop=True)
)

# Verify sort order
for ck in CITY_ORDER:
    sub   = df_clean[df_clean["city_key"] == ck]["date"]
    if sub.is_monotonic_increasing:
        PASS(f"{ck}: dates are monotonically increasing after sort")
    else:
        FAIL(f"{ck}: dates not monotonically increasing after sort")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 11 — SAVE weather_cleaned.csv")

df_clean.to_csv(OUT_CSV, index=False)
out_size_mb = OUT_CSV.stat().st_size / 1e6
log(f"  Saved : {OUT_CSV.relative_to(ROOT)}")
log(f"  Size  : {out_size_mb:.2f} MB")
log(f"  Rows  : {len(df_clean):,}")
log(f"  Cols  : {df_clean.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
section("POST-SAVE VALIDATION — CLEANED FILE")

df_check = pd.read_csv(OUT_CSV, parse_dates=["date"])
log(f"  Re-loaded: {len(df_check):,} rows × {df_check.shape[1]} columns")

if len(df_check) == EXPECTED_ROWS:
    PASS(f"Row count preserved = {len(df_check):,}")
else:
    FAIL(f"Row count {len(df_check):,} ≠ {EXPECTED_ROWS:,}")

if df_check.shape[1] == len(FINAL_COLS):
    PASS(f"Column count = {df_check.shape[1]}")
else:
    FAIL(f"Column count {df_check.shape[1]} ≠ expected {len(FINAL_COLS)}")

if "rain_sum" not in df_check.columns:
    PASS("rain_sum is absent from cleaned file")
else:
    FAIL("rain_sum still present in cleaned file")

if "precipitation_sum" in df_check.columns:
    PASS("precipitation_sum is present in cleaned file")
else:
    FAIL("precipitation_sum missing from cleaned file")

miss = df_check[WEATHER_KEEP].isnull().sum().sum()
if miss == 0:
    PASS(f"Missing values in cleaned file = {miss}")
else:
    FAIL(f"Missing values in cleaned file = {miss}  (expected 0)")

key_dups2 = df_check.duplicated(subset=["city_key", "date"]).sum()
if key_dups2 == 0:
    PASS(f"city_key+date duplicates in cleaned file = {key_dups2}")
else:
    FAIL(f"city_key+date duplicates = {key_dups2}  (expected 0)")

cities_in_clean = sorted(df_check["city_key"].unique().tolist())
if cities_in_clean == sorted(CITY_ORDER):
    PASS(f"All 5 cities present: {cities_in_clean}")
else:
    FAIL(f"Cities mismatch: {cities_in_clean}")

start_c = str(df_check["date"].min().date())
end_c   = str(df_check["date"].max().date())
if start_c == DATE_START and end_c == DATE_END:
    PASS(f"Date range {start_c} → {end_c}")
else:
    FAIL(f"Date range {start_c} → {end_c}  (expected {DATE_START} → {DATE_END})")

# Physical range check on cleaned file
for col, (lo, hi) in PHYSICAL_RANGES.items():
    if col == "rain_sum":
        continue
    vals  = df_check[col].dropna()
    below = int((vals < lo).sum())
    above = int((vals > hi).sum())
    if below == 0 and above == 0:
        PASS(f"{col}: [{vals.min():.2f}, {vals.max():.2f}] in [{lo}, {hi}]")
    else:
        FAIL(f"{col}: {below} below {lo}, {above} above {hi}")

# ══════════════════════════════════════════════════════════════════════════════
section("FINAL CHECK — RAW DATASET UNTOUCHED")

raw_hash_after = md5(RAW_CSV)
if raw_hash_before == raw_hash_after:
    PASS(f"Raw dataset MD5 unchanged: {raw_hash_after}")
else:
    FAIL(f"RAW DATASET MODIFIED!  before={raw_hash_before}  after={raw_hash_after}")

raw_reload = pd.read_csv(RAW_CSV, parse_dates=["date"])
if len(raw_reload) == EXPECTED_ROWS and raw_reload.shape[1] == 23:
    PASS(f"Raw file re-loads as {len(raw_reload):,} rows × {raw_reload.shape[1]} cols  (unchanged)")
else:
    FAIL(f"Raw file shape changed: {raw_reload.shape}")

if "rain_sum" in raw_reload.columns:
    PASS("rain_sum still present in raw file (raw is untouched)")
else:
    FAIL("rain_sum missing from raw file — raw was modified!")

# ══════════════════════════════════════════════════════════════════════════════
section("SUMMARY")

log(f"\n  Original shape  : {EXPECTED_ROWS:,} rows × 23 columns")
log(f"  Cleaned shape   : {len(df_clean):,} rows × {df_clean.shape[1]} columns")
log(f"  Columns removed : 1  (rain_sum)")
log(f"  Rows removed    : 0")
log(f"  Missing values  : 0")
log(f"  Outliers removed: 0  (none — extremes retained)")
log()
log(f"  Checks passed   : {PASS_COUNT}")
log(f"  Checks failed   : {FAIL_COUNT}")
log()
if FAIL_COUNT == 0:
    log("  DATA CLEANING STATUS: PASS")
    log(f"  Output: {OUT_CSV.relative_to(ROOT)}")
else:
    log("  DATA CLEANING STATUS: FAIL")
    log("  Do NOT use output file.")

log()
log("  Final columns in weather_cleaned.csv:")
for i, col in enumerate(FINAL_COLS, 1):
    log(f"    {i:>2}. {col}")

# Save log
log_path = ROOT / "results" / "phase5_cleaning_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
log(f"\n  Log saved: {log_path.relative_to(ROOT)}")

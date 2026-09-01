"""
download_safe.py
ClimateGuard — Safe re-download for cities with missing/incomplete data.

SAFETY CONTRACT:
  - Downloads go to data/raw_staging/ ONLY.
  - data/raw/ is NEVER touched by this script.
  - Promotion to data/raw/ is a separate explicit step (promote_staged.py).

TARGET CITIES:
  lucknow    — 10 variables 100% empty (ERA5-Land failure)
  nagpur     — same
  ahmedabad  — same
  mumbai     — never downloaded

MODEL:
  ERA5 (0.25 degree) — confirmed to return all 16 daily variables.

DATE RANGE:
  1990-01-01 to 2025-08-31  (~13,027 expected rows)

RATE LIMIT HANDLING:
  Open-Meteo free tier has an hourly request cap.
  On 429, this script waits up to 70 minutes and retries.
  Total wait budget: 8 attempts × (retry_wait) per city.

USAGE:
  python download_safe.py
"""

import os
import time
import sys
import requests
import pandas as pd
from datetime import datetime

# ── PATHS ─────────────────────────────────────────────────────────────────────
STAGING_DIR = os.path.join("data", "raw_staging")
RAW_DIR     = os.path.join("data", "raw")

os.makedirs(STAGING_DIR, exist_ok=True)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
API_URL    = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "1990-01-01"
END_DATE   = "2025-08-31"
MODEL      = "era5"
EXPECTED_ROWS_MIN = 13020
EXPECTED_ROWS_MAX = 13035

# Wait time (seconds) after a 429 before retrying.
# Open-Meteo resets hourly — we poll every 5 minutes.
RATE_LIMIT_WAIT = 300   # 5 minutes per retry
MAX_RATE_RETRIES = 15   # up to 15 × 5 min = 75 minutes total wait

DAILY_VARIABLES = [
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

CITIES_TO_DOWNLOAD = {
    "lucknow": {
        "name":        "Lucknow",
        "latitude":    26.8467,
        "longitude":   80.9462,
        "region_type": "plains",
        "state":       "Uttar Pradesh",
    },
    "nagpur": {
        "name":        "Nagpur",
        "latitude":    21.1458,
        "longitude":   79.0882,
        "region_type": "plains",
        "state":       "Maharashtra",
    },
    "ahmedabad": {
        "name":        "Ahmedabad",
        "latitude":    23.0225,
        "longitude":   72.5714,
        "region_type": "plains",
        "state":       "Gujarat",
    },
    "mumbai": {
        "name":        "Mumbai",
        "latitude":    19.0760,
        "longitude":   72.8777,
        "region_type": "coastal",
        "state":       "Maharashtra",
    },
}


def ts():
    return datetime.now().strftime("%H:%M:%S")


def api_get(params, label="request"):
    """
    Make a GET request to Open-Meteo.
    Handles 429 with polling. Returns parsed JSON or None on failure.
    """
    for rate_attempt in range(1, MAX_RATE_RETRIES + 1):
        try:
            print(f"  [{ts()}] Sending {label} ...", flush=True)
            r = requests.get(API_URL, params=params, timeout=180)

            if r.status_code == 429:
                print(f"  [{ts()}] 429 Rate limited. "
                      f"Waiting {RATE_LIMIT_WAIT}s "
                      f"(rate retry {rate_attempt}/{MAX_RATE_RETRIES}) ...", flush=True)
                time.sleep(RATE_LIMIT_WAIT)
                continue

            r.raise_for_status()
            data = r.json()

            if data.get("error"):
                reason = data.get("reason", "unknown error")
                print(f"  [{ts()}] API error: {reason}", flush=True)
                # If it is a rate-limit message buried in the JSON
                if "hourly" in reason.lower() or "limit" in reason.lower():
                    print(f"  [{ts()}] Rate limit in JSON body. "
                          f"Waiting {RATE_LIMIT_WAIT}s ...", flush=True)
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                return None

            return data

        except requests.exceptions.Timeout:
            print(f"  [{ts()}] Timeout. Waiting 30s ...", flush=True)
            time.sleep(30)
        except requests.exceptions.ConnectionError as e:
            print(f"  [{ts()}] Connection error: {e}. Waiting 60s ...", flush=True)
            time.sleep(60)
        except Exception as e:
            print(f"  [{ts()}] Unexpected error: {e}. Waiting 60s ...", flush=True)
            time.sleep(60)

    print(f"  [{ts()}] Exhausted rate-limit retries for {label}.", flush=True)
    return None


def quick_validate(df, city_key):
    """Returns (passed: bool, issues: list[str])."""
    issues = []
    if not (EXPECTED_ROWS_MIN <= len(df) <= EXPECTED_ROWS_MAX):
        issues.append(f"Row count {len(df)} outside [{EXPECTED_ROWS_MIN}, {EXPECTED_ROWS_MAX}]")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    full_range    = pd.date_range(START_DATE, END_DATE, freq="D")
    missing_dates = full_range.difference(df["date"])
    if len(missing_dates) > 0:
        issues.append(f"{len(missing_dates)} missing dates")
    dup_dates = df["date"].duplicated().sum()
    if dup_dates > 0:
        issues.append(f"{dup_dates} duplicate dates")
    for col in DAILY_VARIABLES:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif df[col].isnull().all():
            issues.append(f"Column entirely empty: {col}")
    return (len(issues) == 0, issues)


def download_city(city_key, info):
    """Download one city to staging. Returns DataFrame or None."""
    out_path = os.path.join(STAGING_DIR, f"{city_key}_era5_raw.csv")

    print(f"\n{'='*60}", flush=True)
    print(f"  Downloading: {info['name']} ({info['state']}, {info['region_type']})", flush=True)
    print(f"  Lat: {info['latitude']}  Lon: {info['longitude']}  Model: {MODEL}", flush=True)
    print(f"  [{ts()}] Starting ...", flush=True)

    params = {
        "latitude":   info["latitude"],
        "longitude":  info["longitude"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   "Asia/Kolkata",
        "models":     MODEL,
    }

    data = api_get(params, label=f"full download {city_key}")
    if data is None:
        print(f"  [{ts()}] FAILED: no data returned for {city_key}", flush=True)
        return None

    daily = data.get("daily", {})
    if not daily:
        print(f"  [{ts()}] FAILED: empty daily block for {city_key}", flush=True)
        return None

    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)

    df.insert(0, "city",        info["name"])
    df.insert(1, "city_key",    city_key)
    df.insert(2, "latitude",    info["latitude"])
    df.insert(3, "longitude",   info["longitude"])
    df.insert(4, "region_type", info["region_type"])
    df.insert(5, "state",       info["state"])
    df["date"] = pd.to_datetime(df["date"])

    # Report
    empty_cols = [c for c in DAILY_VARIABLES if c in df.columns and df[c].isnull().all()]
    print(f"  Rows   : {len(df)}", flush=True)
    print(f"  Period : {df['date'].min().date()} to {df['date'].max().date()}", flush=True)
    print(f"  Cols   : {len(df.columns)}", flush=True)
    if empty_cols:
        print(f"  EMPTY COLUMNS ({len(empty_cols)}): {empty_cols}", flush=True)
    else:
        print(f"  All 16 weather variables populated.", flush=True)

    ok, issues = quick_validate(df, city_key)
    if not ok:
        print(f"  VALIDATION ISSUES: {issues}", flush=True)

    df.to_csv(out_path, index=False)
    status = "PASS" if ok else "FAIL"
    print(f"  [{ts()}] Saved to staging [{status}] -> {out_path}", flush=True)
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("ClimateGuard — Safe Staged Download", flush=True)
    print(f"Model      : {MODEL}", flush=True)
    print(f"Period     : {START_DATE} to {END_DATE}", flush=True)
    print(f"Cities     : {list(CITIES_TO_DOWNLOAD.keys())}", flush=True)
    print(f"Target dir : {STAGING_DIR}  (data/raw/ untouched)", flush=True)
    print(f"Start time : {ts()}", flush=True)
    print("=" * 60, flush=True)

    # ── API TEST ───────────────────────────────────────────────────────────
    print(f"\n[{ts()}] Testing API (3-day sample, Lucknow) ...", flush=True)
    test_data = api_get({
        "latitude":   26.8467,
        "longitude":  80.9462,
        "start_date": "2020-06-01",
        "end_date":   "2020-06-03",
        "daily":      "temperature_2m_max,precipitation_sum,wind_speed_10m_max,surface_pressure_mean",
        "timezone":   "Asia/Kolkata",
        "models":     MODEL,
    }, label="API test")

    if test_data is None:
        print("FATAL: API test failed. Cannot proceed.", flush=True)
        sys.exit(1)

    td = test_data.get("daily", {})
    precip = td.get("precipitation_sum", [None])[0]
    wind   = td.get("wind_speed_10m_max",  [None])[0]
    press  = td.get("surface_pressure_mean", [None])[0]
    print(f"[{ts()}] API OK — precip={precip}  wind={wind}  pressure={press}", flush=True)

    if precip is None and wind is None and press is None:
        print("WARNING: ERA5 model returned all nulls on test. "
              "Will attempt full download but results may be incomplete.", flush=True)

    # ── DOWNLOAD LOOP ──────────────────────────────────────────────────────
    results = {}
    city_list = list(CITIES_TO_DOWNLOAD.items())

    for i, (city_key, info) in enumerate(city_list):
        staged_path = os.path.join(STAGING_DIR, f"{city_key}_era5_raw.csv")

        # Skip if already staged and valid
        if os.path.exists(staged_path):
            existing = pd.read_csv(staged_path, parse_dates=["date"])
            ok, issues = quick_validate(existing, city_key)
            if ok:
                print(f"\n  [{city_key}] Already staged and valid — skipping.", flush=True)
                results[city_key] = (True, [])
                continue
            else:
                print(f"\n  [{city_key}] Staged file invalid ({issues}) — re-downloading.", flush=True)

        df = download_city(city_key, info)

        if df is not None:
            ok, issues = quick_validate(df, city_key)
            results[city_key] = (ok, issues)
        else:
            results[city_key] = (False, ["Download failed"])

        # Polite pause between cities (skip after last)
        if i < len(city_list) - 1:
            print(f"\n  [{ts()}] Pausing 30s before next city ...", flush=True)
            time.sleep(30)

    # ── SUMMARY ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("STAGING DOWNLOAD SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    all_ok = True
    for city_key, (ok, issues) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {city_key:<14} [{status}]", end="", flush=True)
        if issues:
            print(f"  -> {issues}", flush=True)
            all_ok = False
        else:
            print(flush=True)

    print(flush=True)
    if all_ok:
        print("All 4 staged downloads PASSED.", flush=True)
        print("Run: python validate_staged.py", flush=True)
        print("Then: python promote_staged.py", flush=True)
    else:
        print("One or more downloads FAILED. Check logs above.", flush=True)
        print("Do NOT run promote_staged.py until all pass.", flush=True)
    print(f"Done at: {ts()}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

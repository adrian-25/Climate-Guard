"""
download_missing.py
Downloads nagpur and ahmedabad (rate-limited on first attempt).
Has retry logic with longer waits for 429 errors.
"""
import os, time, requests, pandas as pd

API_URL    = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "1990-01-01"
END_DATE   = "2025-08-31"
RAW_DIR    = os.path.join("data", "raw")

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

MISSING = {
    "nagpur": {
        "name": "Nagpur", "latitude": 21.1458, "longitude": 79.0882,
        "region_type": "plains", "state": "Maharashtra"
    },
    "ahmedabad": {
        "name": "Ahmedabad", "latitude": 23.0225, "longitude": 72.5714,
        "region_type": "plains", "state": "Gujarat"
    },
}

def download_one(city_key, info):
    out_path = os.path.join(RAW_DIR, f"{city_key}_era5_raw.csv")
    if os.path.exists(out_path):
        print(f"Already exists: {out_path} - skipping")
        return True

    print(f"\nDownloading {info['name']}...")
    params = {
        "latitude": info["latitude"], "longitude": info["longitude"],
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata", "models": "era5_land",
    }

    for attempt in range(1, 6):
        try:
            print(f"  Attempt {attempt}/5...")
            r = requests.get(API_URL, params=params, timeout=120)

            if r.status_code == 429:
                wait_time = 60 * attempt   # 60s, 120s, 180s, 240s, 300s
                print(f"  Rate limited (429). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            r.raise_for_status()
            data = r.json()

            if "error" in data:
                print(f"  API error: {data.get('reason')}")
                return False

            df = pd.DataFrame(data["daily"])
            df.rename(columns={"time": "date"}, inplace=True)
            df.insert(0, "city",        info["name"])
            df.insert(1, "city_key",    city_key)
            df.insert(2, "latitude",    info["latitude"])
            df.insert(3, "longitude",   info["longitude"])
            df.insert(4, "region_type", info["region_type"])
            df.insert(5, "state",       info["state"])
            df["date"] = pd.to_datetime(df["date"])
            df.to_csv(out_path, index=False)
            print(f"  OK - {len(df)} rows saved to {out_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
            time.sleep(30)

    print(f"  FAILED after 5 attempts.")
    return False


# Wait before starting to let rate limit reset
print("Waiting 120 seconds for rate limit to reset before downloading...")
time.sleep(120)

for city_key, info in MISSING.items():
    success = download_one(city_key, info)
    if success and list(MISSING.keys()).index(city_key) < len(MISSING) - 1:
        print("  Waiting 90 seconds before next city...")
        time.sleep(90)

# Rebuild combined file
print("\nRebuilding combined file from all 5 city files...")
all_cities = ["delhi", "lucknow", "nagpur", "ahmedabad", "bhubaneswar"]
dfs = []
for city_key in all_cities:
    fp = os.path.join(RAW_DIR, f"{city_key}_era5_raw.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp, parse_dates=["date"])
        dfs.append(df)
        print(f"  Loaded {city_key}: {len(df)} rows")
    else:
        print(f"  MISSING: {fp}")

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(RAW_DIR, "all_cities_era5_raw.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined: {combined_path}")
    print(f"Rows    : {len(combined)}")
    print(f"Cities  : {sorted(combined['city_key'].unique().tolist())}")

print("\nDone.")

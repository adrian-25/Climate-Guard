"""
download_missing2.py — downloads nagpur and ahmedabad only
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

for city_key, info in MISSING.items():
    out_path = os.path.join(RAW_DIR, f"{city_key}_era5_raw.csv")
    if os.path.exists(out_path):
        print(f"Already exists: {city_key} — skipping")
        continue

    print(f"Downloading {info['name']}...")
    params = {
        "latitude": info["latitude"], "longitude": info["longitude"],
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata", "models": "era5_land",
    }

    for attempt in range(1, 4):
        try:
            r = requests.get(API_URL, params=params, timeout=180)
            if r.status_code == 429:
                print(f"  Rate limited. Waiting 90s (attempt {attempt})...")
                time.sleep(90)
                continue
            r.raise_for_status()
            data = r.json()
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
            print(f"  Saved {len(df)} rows -> {out_path}")
            break
        except Exception as e:
            print(f"  Error (attempt {attempt}): {e}")
            time.sleep(30)

    # 30 second gap between cities
    print("  Pausing 30s before next city...")
    time.sleep(30)

# Rebuild combined file
print("\nRebuilding combined CSV...")
all_cities = ["delhi", "lucknow", "nagpur", "ahmedabad", "bhubaneswar"]
dfs = []
for ck in all_cities:
    fp = os.path.join(RAW_DIR, f"{ck}_era5_raw.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp, parse_dates=["date"])
        dfs.append(df)
        print(f"  {ck}: {len(df)} rows")
    else:
        print(f"  MISSING: {ck}")

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    cp = os.path.join(RAW_DIR, "all_cities_era5_raw.csv")
    combined.to_csv(cp, index=False)
    print(f"\nCombined: {cp}")
    print(f"Rows    : {len(combined)}")
    print(f"Cities  : {sorted(combined['city_key'].unique().tolist())}")

print("\nDone.")

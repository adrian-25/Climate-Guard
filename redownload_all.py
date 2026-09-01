"""
redownload_all.py
Re-downloads all 5 cities using model=era5 which provides all required variables.
ERA5-Land only has temperature/humidity. ERA5 has everything including
wind, pressure, precipitation, radiation, apparent temperature.
"""
import os, time, requests, pandas as pd

API_URL    = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "1990-01-01"
END_DATE   = "2025-08-31"
RAW_DIR    = os.path.join("data", "raw")

# ERA5 (0.25 degree) has all required variables.
# ERA5-Land only has temperature/humidity — missing wind, pressure, precip.
MODEL = "era5"

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

CITIES = {
    "delhi": {
        "name": "New Delhi", "latitude": 28.6139, "longitude": 77.2090,
        "region_type": "plains", "state": "Delhi"
    },
    "lucknow": {
        "name": "Lucknow", "latitude": 26.8467, "longitude": 80.9462,
        "region_type": "plains", "state": "Uttar Pradesh"
    },
    "nagpur": {
        "name": "Nagpur", "latitude": 21.1458, "longitude": 79.0882,
        "region_type": "plains", "state": "Maharashtra"
    },
    "ahmedabad": {
        "name": "Ahmedabad", "latitude": 23.0225, "longitude": 72.5714,
        "region_type": "plains", "state": "Gujarat"
    },
    "bhubaneswar": {
        "name": "Bhubaneswar", "latitude": 20.2961, "longitude": 85.8245,
        "region_type": "coastal", "state": "Odisha"
    },
}

os.makedirs(RAW_DIR, exist_ok=True)

def download_city(city_key, info):
    out_path = os.path.join(RAW_DIR, f"{city_key}_era5_raw.csv")
    print(f"\nDownloading {info['name']} (ERA5 model)...")

    params = {
        "latitude":   info["latitude"],
        "longitude":  info["longitude"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   "Asia/Kolkata",
        "models":     MODEL,
    }

    for attempt in range(1, 5):
        try:
            r = requests.get(API_URL, params=params, timeout=180)
            if r.status_code == 429:
                wait = 90 * attempt
                print(f"  Rate limited. Waiting {wait}s (attempt {attempt})...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                print(f"  API error: {data.get('reason')}")
                return None

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

            # Report which columns have data
            non_null_cols = [c for c in df.columns if df[c].notna().any() and c not in
                             ["city","city_key","latitude","longitude","region_type","state","date"]]
            null_cols = [c for c in DAILY_VARIABLES if c not in non_null_cols]

            print(f"  Saved {len(df)} rows to {out_path}")
            print(f"  Columns with data : {len(non_null_cols)}")
            if null_cols:
                print(f"  Still empty       : {null_cols}")
            else:
                print(f"  All weather columns have data -- OK")
            return df

        except Exception as e:
            print(f"  Error (attempt {attempt}): {e}")
            time.sleep(30)

    print(f"  FAILED for {info['name']}")
    return None


dfs = []
city_list = list(CITIES.items())
for i, (city_key, info) in enumerate(city_list):
    df = download_city(city_key, info)
    if df is not None:
        dfs.append(df)
    if i < len(city_list) - 1:
        print(f"  Waiting 30s before next city...")
        time.sleep(30)

# Rebuild combined
print("\nRebuilding combined CSV...")
all_keys = ["delhi", "lucknow", "nagpur", "ahmedabad", "bhubaneswar"]
final_dfs = []
for ck in all_keys:
    fp = os.path.join(RAW_DIR, f"{ck}_era5_raw.csv")
    if os.path.exists(fp):
        d = pd.read_csv(fp, parse_dates=["date"])
        final_dfs.append(d)
        print(f"  {ck}: {len(d)} rows")
    else:
        print(f"  MISSING: {ck}")

if final_dfs:
    combined = pd.concat(final_dfs, ignore_index=True)
    cp = os.path.join(RAW_DIR, "all_cities_era5_raw.csv")
    combined.to_csv(cp, index=False)
    print(f"\nCombined file : {cp}")
    print(f"Total rows    : {len(combined)}")
    print(f"Total columns : {len(combined.columns)}")
    print(f"Cities        : {sorted(combined['city_key'].unique().tolist())}")

print("\nDone.")

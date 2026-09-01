"""
download_final.py
Downloads Mumbai (replacing Bhubaneswar) + fixes Lucknow, Nagpur, Ahmedabad
which still have incomplete data (10 empty columns from earlier era5_land attempt).
Uses ERA5 model (0.25 degree) which has all required variables.
"""
import os, time, requests, pandas as pd

API_URL    = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "1990-01-01"
END_DATE   = "2025-08-31"
RAW_DIR    = os.path.join("data", "raw")
MODEL      = "era5"

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

# Cities to download / re-download
# Delhi is already complete — skipped
# Bhubaneswar replaced by Mumbai
CITIES_TO_DOWNLOAD = {
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
    "mumbai": {
        "name": "Mumbai", "latitude": 19.0760, "longitude": 72.8777,
        "region_type": "coastal", "state": "Maharashtra"
    },
}

# Final 5 cities for the combined file
ALL_CITIES = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]

os.makedirs(RAW_DIR, exist_ok=True)


def needs_download(city_key):
    fp = os.path.join(RAW_DIR, f"{city_key}_era5_raw.csv")
    if not os.path.exists(fp):
        return True
    df = pd.read_csv(fp)
    empty = [c for c in DAILY_VARIABLES if c in df.columns and df[c].isnull().all()]
    return len(empty) > 0


def download_city(city_key, info):
    out_path = os.path.join(RAW_DIR, f"{city_key}_era5_raw.csv")
    print(f"\nDownloading {info['name']}...")

    params = {
        "latitude":   info["latitude"],
        "longitude":  info["longitude"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   "Asia/Kolkata",
        "models":     MODEL,
    }

    for attempt in range(1, 8):
        try:
            r = requests.get(API_URL, params=params, timeout=180)
            if r.status_code == 429:
                wait = 120
                print(f"  Rate limited (429). Waiting {wait}s (attempt {attempt}/7)...")
                time.sleep(wait)
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

            empty = [c for c in DAILY_VARIABLES if df[c].isnull().all()]
            print(f"  Rows   : {len(df)}")
            print(f"  Period : {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"  Status : {'ALL COLUMNS OK' if not empty else f'EMPTY: {empty}'}")
            return True

        except Exception as e:
            print(f"  Error (attempt {attempt}): {e}")
            time.sleep(60)

    print(f"  FAILED for {info['name']}")
    return False


# Test API first
print("Testing API availability...")
test = requests.get(API_URL, params={
    "latitude": 26.85, "longitude": 80.95,
    "start_date": "2020-01-01", "end_date": "2020-01-03",
    "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
    "timezone": "Asia/Kolkata", "models": MODEL
}, timeout=30)

if test.status_code == 429:
    print("Rate limited. Waiting 3 minutes before starting...")
    time.sleep(180)
elif test.status_code == 200:
    data = test.json()
    precip = data.get("daily", {}).get("precipitation_sum", [None])[0]
    wind   = data.get("daily", {}).get("wind_speed_10m_max", [None])[0]
    print(f"API OK. Test values — precip: {precip}, wind: {wind}")
    if precip is None and wind is None:
        print("WARNING: ERA5 model still not returning precip/wind. Will try anyway.")
else:
    print(f"Unexpected status: {test.status_code}")

# Also delete old bhubaneswar file since we are replacing it
bbs_path = os.path.join(RAW_DIR, "bhubaneswar_era5_raw.csv")
if os.path.exists(bbs_path):
    os.remove(bbs_path)
    print("Removed old bhubaneswar file.")

print(f"\nStarting downloads for {len(CITIES_TO_DOWNLOAD)} cities...")
print("=" * 50)

city_list = list(CITIES_TO_DOWNLOAD.items())
for i, (city_key, info) in enumerate(city_list):
    if not needs_download(city_key):
        print(f"\n{city_key} already complete -- skipping")
        continue
    download_city(city_key, info)
    if i < len(city_list) - 1:
        print(f"  Waiting 45s before next city...")
        time.sleep(45)

# Rebuild combined file with final 5 cities
print("\n" + "=" * 50)
print("Rebuilding combined CSV with final 5 cities...")
final_dfs = []
for ck in ALL_CITIES:
    fp = os.path.join(RAW_DIR, f"{ck}_era5_raw.csv")
    if os.path.exists(fp):
        d = pd.read_csv(fp, parse_dates=["date"])
        empty_cols = [c for c in DAILY_VARIABLES if c in d.columns and d[c].isnull().all()]
        status = "COMPLETE" if not empty_cols else f"INCOMPLETE ({len(empty_cols)} empty)"
        print(f"  {ck:12} : {len(d)} rows  [{status}]")
        final_dfs.append(d)
    else:
        print(f"  {ck:12} : MISSING")

if final_dfs:
    combined = pd.concat(final_dfs, ignore_index=True)
    cp = os.path.join(RAW_DIR, "all_cities_era5_raw.csv")
    combined.to_csv(cp, index=False)
    print(f"\nCombined file : {cp}")
    print(f"Total rows    : {len(combined)}")
    print(f"Total columns : {len(combined.columns)}")
    print(f"Cities        : {sorted(combined['city_key'].unique().tolist())}")

print("\nDone.")

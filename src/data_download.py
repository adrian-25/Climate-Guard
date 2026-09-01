"""
src/data_download.py

PURPOSE:
    Download ERA5-Land daily weather data from the Open-Meteo Historical
    Weather API for five heatwave-prone Indian cities.

SOURCE:
    Open-Meteo Historical Weather API (https://open-meteo.com)
    Underlying model: ERA5-Land (ECMWF), 0.1 degree resolution (~11 km)

COVERAGE:
    1990-01-01 to 2025-08-31
    Five cities: New Delhi, Lucknow, Nagpur, Ahmedabad, Bhubaneswar

OUTPUT:
    One CSV file per city in data/raw/
    Combined CSV with all cities in data/raw/all_cities_era5_raw.csv

CITATION:
    Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software].
    Zenodo. https://doi.org/10.5281/ZENODO.7970649
    ERA5-Land: Munoz-Sabater et al. (2021), ECMWF.
"""

import os
import time
import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Five cities covering major heatwave-prone regions of India.
# region_type determines the IMD Tmax threshold for heatwave labeling:
#   plains  → Tmax >= 40°C required
#   coastal → Tmax >= 37°C required
CITIES = {
    "delhi": {
        "name": "New Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "region_type": "plains",
        "state": "Delhi"
    },
    "lucknow": {
        "name": "Lucknow",
        "latitude": 26.8467,
        "longitude": 80.9462,
        "region_type": "plains",
        "state": "Uttar Pradesh"
    },
    "nagpur": {
        "name": "Nagpur",
        "latitude": 21.1458,
        "longitude": 79.0882,
        "region_type": "plains",
        "state": "Maharashtra"
    },
    "ahmedabad": {
        "name": "Ahmedabad",
        "latitude": 23.0225,
        "longitude": 72.5714,
        "region_type": "plains",
        "state": "Gujarat"
    },
    "bhubaneswar": {
        "name": "Bhubaneswar",
        "latitude": 20.2961,
        "longitude": 85.8245,
        "region_type": "coastal",
        "state": "Odisha"
    }
}

# Date range.
# 1990-01-01 gives 30+ years for computing 1991-2020 climatological normals.
# 2025-08-31 is the most recent complete period.
START_DATE = "1990-01-01"
END_DATE   = "2025-08-31"

# Open-Meteo archive endpoint
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily variables to request from ERA5-Land.
# These cover all variables needed for IMD heatwave labeling and feature engineering.
DAILY_VARIABLES = [
    "temperature_2m_max",          # Daily Tmax (°C) — primary heatwave criterion
    "temperature_2m_min",          # Daily Tmin (°C) — warm night detection
    "temperature_2m_mean",         # Daily Tmean (°C)
    "apparent_temperature_max",    # Feels-like max (°C) — heat stress
    "apparent_temperature_min",    # Feels-like min (°C)
    "apparent_temperature_mean",   # Feels-like mean (°C)
    "precipitation_sum",           # Total daily precipitation (mm)
    "rain_sum",                    # Rain component (mm)
    "wind_speed_10m_max",          # Max wind speed at 10m (km/h)
    "wind_gusts_10m_max",          # Max wind gust (km/h)
    "relative_humidity_2m_max",    # Max relative humidity (%)
    "relative_humidity_2m_min",    # Min relative humidity (%)
    "relative_humidity_2m_mean",   # Mean relative humidity (%)
    "surface_pressure_mean",       # Mean surface pressure (hPa)
    "shortwave_radiation_sum",     # Total solar radiation (MJ/m²)
    "et0_fao_evapotranspiration",  # Reference evapotranspiration (mm)
]

# Output folder — two levels up from src/, into data/raw/
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def download_city_data(city_key: str, city_info: dict) -> pd.DataFrame:
    """
    Download ERA5-Land daily data for a single city from Open-Meteo.

    Parameters
    ----------
    city_key  : str  — Short key used for filenames (e.g., 'delhi')
    city_info : dict — City metadata (name, lat, lon, region_type, state)

    Returns
    -------
    pd.DataFrame with daily weather data, or None if download failed.
    """
    params = {
        "latitude":   city_info["latitude"],
        "longitude":  city_info["longitude"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   "Asia/Kolkata",  # IST — ensures correct daily aggregation
        "models":     "era5_land",     # Explicitly request ERA5-Land (0.1 degree)
    }

    print(f"\n  Downloading: {city_info['name']} ({city_info['state']})")
    print(f"  Lat: {city_info['latitude']}  Lon: {city_info['longitude']}  Type: {city_info['region_type']}")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=120)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"  All {max_retries} attempts failed. Skipping {city_info['name']}.")
                return None

    data = response.json()

    if "error" in data:
        print(f"  API Error: {data.get('reason', 'Unknown error')}")
        return None

    daily_data = data.get("daily", {})
    if not daily_data:
        print(f"  No daily data returned.")
        return None

    df = pd.DataFrame(daily_data)
    df.rename(columns={"time": "date"}, inplace=True)

    # Add metadata columns
    df.insert(0, "city",        city_info["name"])
    df.insert(1, "city_key",    city_key)
    df.insert(2, "latitude",    city_info["latitude"])
    df.insert(3, "longitude",   city_info["longitude"])
    df.insert(4, "region_type", city_info["region_type"])
    df.insert(5, "state",       city_info["state"])

    df["date"] = pd.to_datetime(df["date"])

    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dataframes = []
    failed_cities  = []

    print("=" * 60)
    print("ClimateGuard — ERA5-Land Data Download")
    print(f"API: Open-Meteo Historical (ERA5-Land model)")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Cities: {len(CITIES)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    for city_key, city_info in CITIES.items():
        df = download_city_data(city_key, city_info)

        if df is not None:
            output_path = os.path.join(OUTPUT_DIR, f"{city_key}_era5_raw.csv")
            df.to_csv(output_path, index=False)
            print(f"  Saved -> {output_path}")
            all_dataframes.append(df)
        else:
            failed_cities.append(city_key)

        # Polite pause — Open-Meteo is a free service
        time.sleep(2)

    if all_dataframes:
        combined_df   = pd.concat(all_dataframes, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "all_cities_era5_raw.csv")
        combined_df.to_csv(combined_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"COMBINED FILE: {combined_path}")
        print(f"Total rows   : {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")
        print(f"Cities done  : {len(all_dataframes)}/{len(CITIES)}")
        if failed_cities:
            print(f"FAILED       : {failed_cities}")
        print("=" * 60)
        print("Download complete.")
        return combined_df

    print("No data downloaded.")
    return None


if __name__ == "__main__":
    main()

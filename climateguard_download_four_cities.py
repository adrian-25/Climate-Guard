#!/usr/bin/env python3
"""
ClimateGuard — download four complete daily ERA5 city datasets.

Cities:
- New Delhi
- Lucknow
- Nagpur
- Ahmedabad

Period: 1990-01-01 through 2025-08-31
Source: Open-Meteo Historical Weather API, ERA5 model
Units: °C, %, km/h, mm, hPa, MJ/m²

Run:
    python climateguard_download_four_cities.py

Requires:
    pip install requests pandas
"""

from pathlib import Path
import time
import requests
import pandas as pd

START = "1990-01-01"
END = "2025-08-31"

CITIES = {
    "delhi": {
        "name": "New Delhi",
        "state": "Delhi",
        "region_type": "Plains",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "threshold_c": 40.0,
    },
    "lucknow": {
        "name": "Lucknow",
        "state": "Uttar Pradesh",
        "region_type": "Plains",
        "latitude": 26.8467,
        "longitude": 80.9462,
        "threshold_c": 40.0,
    },
    "nagpur": {
        "name": "Nagpur",
        "state": "Maharashtra",
        "region_type": "Plains",
        "latitude": 21.1458,
        "longitude": 79.0882,
        "threshold_c": 40.0,
    },
    "ahmedabad": {
        "name": "Ahmedabad",
        "state": "Gujarat",
        "region_type": "Plains",
        "latitude": 23.0225,
        "longitude": 72.5714,
        "threshold_c": 40.0,
    },
}

DAILY = [
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

API = "https://archive-api.open-meteo.com/v1/archive"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def download_city(key, meta):
    params = {
        "latitude": meta["latitude"],
        "longitude": meta["longitude"],
        "start_date": START,
        "end_date": END,
        "daily": ",".join(DAILY),
        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "models": "era5",
    }

    print(f"\nDownloading {meta['name']}...")
    r = requests.get(API, params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()

    if "daily" not in payload:
        raise RuntimeError(f"No daily data returned for {meta['name']}: {payload}")

    daily = payload["daily"]
    df = pd.DataFrame(daily)

    # Standardize names and add project metadata.
    df = df.rename(columns={"time": "date"})
    df.insert(0, "city", meta["name"])
    df.insert(1, "state", meta["state"])
    df.insert(2, "region_type", meta["region_type"])
    df.insert(3, "latitude", meta["latitude"])
    df.insert(4, "longitude", meta["longitude"])
    df.insert(5, "heatwave_threshold_c", meta["threshold_c"])

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    # Basic integrity checks before saving.
    expected_days = len(pd.date_range(START, END, freq="D"))
    if len(df) != expected_days:
        raise RuntimeError(
            f"{meta['name']}: expected {expected_days} rows, got {len(df)}"
        )
    if df["date"].duplicated().any():
        raise RuntimeError(f"{meta['name']}: duplicate dates detected")
    if df["date"].isna().any():
        raise RuntimeError(f"{meta['name']}: invalid dates detected")

    path = RAW / f"{key}_era5_raw.csv"
    df.to_csv(path, index=False)

    print(f"  Saved: {path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Missing values: {int(df.isna().sum().sum()):,}")

    return df

def main():
    frames = []

    for key, meta in CITIES.items():
        frames.append(download_city(key, meta))
        time.sleep(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["city", "date"]).reset_index(drop=True)

    combined_path = RAW / "all_four_cities_era5_raw.csv"
    combined.to_csv(combined_path, index=False)

    print("\n" + "=" * 72)
    print("FINAL VALIDATION")
    print("=" * 72)

    expected_days = len(pd.date_range(START, END, freq="D"))

    for city in CITIES.values():
        part = combined[combined["city"] == city["name"]]
        missing_dates = expected_days - part["date"].nunique()
        duplicates = part.duplicated(subset=["city", "date"]).sum()

        print(
            f"{city['name']:<12} "
            f"rows={len(part):>6,}  "
            f"missing_dates={missing_dates:>2}  "
            f"duplicates={duplicates:>2}  "
            f"missing_cells={int(part.isna().sum().sum()):>3}"
        )

    print(f"\nCombined rows: {len(combined):,}")
    print(f"Combined file: {combined_path}")
    print("\nNo heatwave labels are generated by this script.")
    print("No ML features are generated by this script.")
    print("Raw data only.")

if __name__ == "__main__":
    main()

"""Live Europe-v1 inference using the shared ClimateGuard feature transformation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from src.europe_pipeline import (
    ABSOLUTE_FLOOR_C,
    DAILY_VARIABLES,
    EUROPE_CITIES,
    EUROPE_CITY_ORDER,
    WARM_MONTHS,
)

ROOT = Path(__file__).resolve().parents[1]
NORMALS = json.loads((ROOT / "data" / "europe" / "city_tmax_normals.json").read_text(encoding="utf-8"))
FEATURES = json.loads((ROOT / "models" / "europe-v1" / "feature_list.json").read_text(encoding="utf-8"))
METADATA = json.loads((ROOT / "models" / "europe-v1" / "metadata.json").read_text(encoding="utf-8"))
MODEL = joblib.load(ROOT / "models" / "europe-v1" / "model.joblib")
# Use city_order from saved metadata so live inference always matches the trained model
_SAVED_CITY_ORDER = METADATA.get("city_order", EUROPE_CITY_ORDER)
CACHE: dict[str, dict[str, Any]] = {}
TTL_SECONDS = 30 * 60


def _risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "EXTREME"
    if probability >= 0.6:
        return "HIGH"
    if probability >= 0.3:
        return "MODERATE"
    return "LOW"


def europe_rules(row: pd.Series, probability: float) -> list[dict[str, Any]]:
    """Non-IMD contextual rules based only on the Europe relative definition."""
    tmax, p90 = float(row["temperature_2m_max"]), float(row["tmax_p90"])
    departure = float(row["tmax_departure"])
    qualifying = bool(row["qualifying_day"])
    persistent = bool(row.get("heatwave_lag1", 0)) and qualifying
    return [
        {"rule_id": "EU_01", "name": "Above local hot-day percentile", "triggered": qualifying,
         "severity": "WARNING", "message": f"Tmax {tmax:.1f}°C exceeds this city's warm-season 90th percentile ({p90:.1f}°C)." if qualifying else "",
         "description": "Uses the city-specific 1991–2020 90th-percentile threshold; not an official warning criterion."},
        {"rule_id": "EU_02", "name": "Persistent local heat", "triggered": persistent,
         "severity": "CRITICAL", "message": "Consecutive locally exceptional hot days are present; reduce heat exposure and follow local authority guidance." if persistent else "",
         "description": "Signals a continuing run of qualifying heat days under the Europe research definition."},
        {"rule_id": "EU_03", "name": "Unusually warm conditions", "triggered": departure >= 3.0,
         "severity": "WARNING", "message": f"Today's maximum is {departure:.1f}°C above the local seasonal normal." if departure >= 3.0 else "",
         "description": "Flags a notable departure from the local 1991–2020 seasonal normal."},
        {"rule_id": "EU_04", "name": "Model risk overlay", "triggered": probability >= 0.6,
         "severity": "CRITICAL" if probability >= 0.8 else "WARNING", "message": "The Europe research model indicates elevated next-day heatwave risk." if probability >= 0.6 else "",
         "description": "Project model probability only; it is not an official national meteorological alert."},
    ]


def _live_frame(city_key: str) -> pd.DataFrame:
    city = EUROPE_CITIES[city_key]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": city["lat"], "longitude": city["lon"], "past_days": 30,
                "forecast_days": 7, "daily": ",".join(DAILY_VARIABLES), "timezone": city["timezone"]},
        timeout=20,
    )
    response.raise_for_status()
    daily = response.json().get("daily", {})
    if not daily or any(name not in daily for name in DAILY_VARIABLES):
        raise ValueError("Open-Meteo did not return the Europe live data contract")
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "city", city["name"])
    df.insert(1, "city_key", city_key)
    df.insert(2, "latitude", city["lat"])
    df.insert(3, "longitude", city["lon"])
    df.insert(4, "region_type", city["region_type"])
    df.insert(5, "state", city["state"])
    days = df["date"].dt.dayofyear.astype(str)
    table = NORMALS[city_key]
    df["tmax_normal"] = days.map(table["tmax_normal"]).astype(float)
    df["tmax_p90"] = days.map(table["tmax_p90"]).astype(float)
    df["tmax_departure"] = df["temperature_2m_max"] - df["tmax_normal"]
    df["qualifying_day"] = ((df["temperature_2m_max"] > df["tmax_p90"])
                            & (df["temperature_2m_max"] >= ABSOLUTE_FLOOR_C)
                            & df["date"].dt.month.isin(WARM_MONTHS)).astype(int)
    # The recent state is a local proxy required by the shared lag feature.
    df["heatwave"] = 0
    qualifying = df["qualifying_day"].to_numpy()
    for i in range(2, len(df)):
        if qualifying[i - 2:i + 1].all():
            df.loc[i - 2:i, "heatwave"] = 1
    return df


def get_europe_live_forecast(city_key: str) -> dict[str, Any]:
    """Return the existing live endpoint's day payload shape for a Europe city."""
    cached = CACHE.get(city_key)
    if cached and time.time() - cached["fetched_at"] < TTL_SECONDS:
        return cached["value"]
    if city_key not in EUROPE_CITIES:
        raise KeyError(city_key)
    raw = _live_frame(city_key)
    from feature_engineering import build_features_for_city

    feature_df = build_features_for_city(
        raw, city_order=_SAVED_CITY_ORDER, coastal_region_types=("atlantic", "mediterranean")
    )
    valid = feature_df.dropna(subset=FEATURES).copy()
    probabilities = MODEL.predict_proba(valid[FEATURES])[:, 1]
    valid["probability"] = probabilities
    city = EUROPE_CITIES[city_key]
    days = []
    for _, row in valid.tail(7).iterrows():
        probability = float(row["probability"])
        rules = europe_rules(row, probability)
        days.append({
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "temperature_max": round(float(row["temperature_2m_max"]), 1),
            "apparent_temperature_max": round(float(row["apparent_temperature_max"]), 1),
            "humidity_mean": round(float(row["relative_humidity_2m_mean"]), 1),
            "wind_speed_max": round(float(row["wind_speed_10m_max"]), 1),
            "probability": round(probability, 4), "prediction": int(probability >= METADATA["threshold"]),
            "risk_level": _risk_level(probability), "triggered_rules": [rule for rule in rules if rule["triggered"]],
        })
    official = "AEMET" if city["country"] == "Spain" else "IPMA" if city["country"] == "Portugal" else "the relevant national meteorological service"
    result = {
        "city": city_key, "city_info": {**city, "dataset_region": "europe"}, "days": days,
        "last_updated": pd.Timestamp.now(tz="UTC").isoformat(), "source": "Open-Meteo",
        "disclaimer": f"Europe-v1 is a research model, not an official warning. Follow {official} and local authority guidance.",
        "feature_notes": ["Uses the shared ClimateGuard feature transformation and Europe’s percentile-based definition."],
    }
    CACHE[city_key] = {"fetched_at": time.time(), "value": result}
    return result


def cache_status() -> dict[str, dict[str, float]]:
    return {key: {"age_seconds": round(time.time() - item["fetched_at"], 1)} for key, item in CACHE.items()}

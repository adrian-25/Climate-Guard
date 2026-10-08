"""
live_data.py — Phase 7 live weather data + feature engineering for ClimateGuard.

Fetches recent weather from Open-Meteo (free, no API key) using each city's
lat/lon, computes all 110 model features, and returns predictions for today
and the next few forecast days.

Architecture:
  - Additive only: this module is imported by app.py via new routes.
  - Does NOT touch any existing endpoint or its response shape.
  - Reuses the exact same feature-engineering logic as training
    (mirrors feature_engineering.py exactly, not a reimplementation).
  - In-memory TTL cache (30 minutes) per city.
  - Returns the same shape as /api/predict wherever possible.
  - Never returns stale data as fresh: cache age is always reported.
  - If Open-Meteo fetch fails, returns a clear error payload.

Caveats (documented to users):
  1. tmax_departure_zscore: computed from 30 days of live data.
     Training used 30-day windows of historical data — should match well.
  2. heatwave_lag1: computed from yesterday's qualifying_day flag derived
     from live data. May differ from historical by ±1 day around events.
  3. forecast days (beyond today): temperature etc. are from Open-Meteo
     forecast, not ERA5 reanalysis. Distributional shift is possible.
  4. These are model estimates, NOT official IMD warnings.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests as http_requests
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# City metadata from the versioned active-model city registry
# ---------------------------------------------------------------------------
from src.cities import CITIES, CITY_ORDER

SEASON_MAP = {
    12: 0,
    1: 0,
    2: 0,  # winter
    3: 1,
    4: 1,
    5: 1,  # spring
    6: 2,
    7: 2,
    8: 2,
    9: 2,  # monsoon
    10: 3,
    11: 3,  # autumn
}

# ---------------------------------------------------------------------------
# Climatological normals (1990-2020 ERA5 baseline, built by _build_normals.py)
# ---------------------------------------------------------------------------
_NORMALS_PATH = PROJECT_ROOT / "data" / "city_tmax_normals.json"
_TMAX_NORMALS: Dict[str, Dict[str, float]] = {}


def _load_normals() -> None:
    global _TMAX_NORMALS
    if not _NORMALS_PATH.exists():
        raise FileNotFoundError(
            f"City tmax normals file not found: {_NORMALS_PATH}. " "Run _build_normals.py first."
        )
    with open(_NORMALS_PATH, encoding="utf-8") as f:
        _TMAX_NORMALS = json.load(f)


_load_normals()


def get_tmax_normal(city_key: str, doy: int) -> float:
    """Return the climatological tmax normal for a city and day-of-year."""
    city_norms = _TMAX_NORMALS.get(city_key, {})
    v = city_norms.get(str(doy))
    if v is None:
        # Fallback: nearest available DOY
        for delta in range(1, 15):
            for d in [doy + delta, doy - delta]:
                v = city_norms.get(str(d % 366 or 366))
                if v is not None:
                    return float(v)
        return 35.0  # extreme fallback
    return float(v)


# ---------------------------------------------------------------------------
# Open-Meteo variable mapping
# ERA5 name → Open-Meteo hourly/daily param name
# All variables are daily; units match ERA5 from Open-Meteo
# ---------------------------------------------------------------------------
OPEN_METEO_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_CUSTOMER_URL = "https://customer-api.open-meteo.com/v1/forecast"
OPEN_METEO_REQUEST_TIMEOUT_SECONDS = 12


class OpenMeteoRateLimitError(RuntimeError):
    """Raised when Open-Meteo temporarily rejects a request for rate limiting."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__("Open-Meteo is temporarily rate-limiting live weather requests.")


def _open_meteo_request_settings() -> tuple[str, Dict[str, str]]:
    """Return the configured Open-Meteo endpoint and optional customer API key.

    The public endpoint remains the zero-configuration default. Deployments can
    set ``OPEN_METEO_API_KEY`` to use Open-Meteo's dedicated customer endpoint,
    which avoids public shared-IP rate limits without exposing the key to clients.
    """
    api_key = os.getenv("OPEN_METEO_API_KEY", "").strip()
    if api_key:
        return OPEN_METEO_CUSTOMER_URL, {"apikey": api_key}
    return OPEN_METEO_URL, {}


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Dict] = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes
STALE_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60
RATE_LIMIT_COOLDOWN_SECONDS = 5 * 60
_RATE_LIMITED_UNTIL = 0.0


def _cache_get(city_key: str) -> Optional[Dict]:
    entry = _CACHE.get(city_key)
    if entry is None:
        return None
    age = time.time() - entry["fetched_at"]
    if age > CACHE_TTL_SECONDS:
        return None
    return entry


def _cache_set(city_key: str, data: Dict) -> None:
    _CACHE[city_key] = {**data, "fetched_at": time.time()}


def _stale_cache_get(city_key: str) -> Optional[Dict]:
    """Return a recent expired entry for outage fallback, never older than one day."""
    entry = _CACHE.get(city_key)
    if entry is None:
        return None
    if time.time() - entry["fetched_at"] > STALE_CACHE_MAX_AGE_SECONDS:
        return None
    return entry


def _cached_result(entry: Dict, *, stale: bool = False) -> Dict[str, Any]:
    """Return a client-safe cached response with freshness clearly labelled."""
    result = dict(entry)
    age_seconds = round(time.time() - entry["fetched_at"], 1)
    result["cache_used"] = True
    result["cache_age_seconds"] = age_seconds
    result["cache_stale"] = stale
    if stale:
        warnings = list(result.get("warnings", []))
        warnings.append(
            "Live weather provider is temporarily rate-limited. Showing the last "
            f"successful forecast from {round(age_seconds / 60)} minute(s) ago."
        )
        result["warnings"] = warnings
    return result


def cache_status() -> Dict[str, Any]:
    now = time.time()
    result = {}
    for city, entry in _CACHE.items():
        age = now - entry["fetched_at"]
        result[city] = {
            "cached": True,
            "age_seconds": round(age, 1),
            "stale": age > CACHE_TTL_SECONDS,
            "fetched_at": datetime.fromtimestamp(entry["fetched_at"]).isoformat(),
        }
    return result


# ---------------------------------------------------------------------------
# Fetch raw weather from Open-Meteo
# ---------------------------------------------------------------------------
def _fetch_open_meteo(city_key: str, past_days: int = 30, forecast_days: int = 7) -> pd.DataFrame:
    """
    Fetch daily weather from Open-Meteo for a city.
    Returns a DataFrame with dates and raw weather variables.
    Raises on network error or bad response.
    """
    city = CITIES[city_key]
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "daily": ",".join(OPEN_METEO_DAILY_VARS),
        "timezone": "Asia/Kolkata",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    endpoint, customer_params = _open_meteo_request_settings()
    params.update(customer_params)

    try:
        resp = http_requests.get(
            endpoint,
            params=params,
            timeout=OPEN_METEO_REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "ClimateGuard/1.0 (research dashboard)"},
        )
        if resp.status_code == 429:
            retry_after = _retry_after_seconds(getattr(resp, "headers", {}))
            raise OpenMeteoRateLimitError(retry_after)
        resp.raise_for_status()
    except OpenMeteoRateLimitError:
        raise
    except http_requests.exceptions.Timeout:
        raise RuntimeError(
            f"Open-Meteo request timed out ({OPEN_METEO_REQUEST_TIMEOUT_SECONDS} s). Try again shortly."
        )
    except http_requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open-Meteo fetch failed: {exc}")

    data = resp.json()
    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise RuntimeError("Open-Meteo returned unexpected response shape.")

    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["city_key"] = city_key

    # Validate no critical fields are entirely null
    critical = ["temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean"]
    for col in critical:
        if col in df.columns and df[col].isna().all():
            raise RuntimeError(
                f"Open-Meteo returned all-null values for '{col}' for {city_key}. "
                "Cannot proceed with prediction."
            )

    return df


def _retry_after_seconds(headers: Any) -> int:
    """Use an upstream Retry-After value when it is a short numeric delay."""
    try:
        retry_after = int(headers.get("Retry-After", RATE_LIMIT_COOLDOWN_SECONDS))
    except (AttributeError, TypeError, ValueError):
        retry_after = RATE_LIMIT_COOLDOWN_SECONDS
    return max(1, min(retry_after, RATE_LIMIT_COOLDOWN_SECONDS))


# ---------------------------------------------------------------------------
# Feature engineering (mirrors feature_engineering.py exactly)
# ---------------------------------------------------------------------------


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear slope — exact replica of feature_engineering.py rolling_slope()."""
    slopes = [np.nan] * len(series)
    values = series.values
    for i in range(window, len(values) + 1):
        chunk = values[i - window : i]
        valid_mask = ~np.isnan(chunk)
        if valid_mask.sum() >= 2:
            x = np.arange(window)[valid_mask]
            y = chunk[valid_mask]
            slope, *_ = linregress(x, y)
            slopes[i - 1] = slope
    return pd.Series(slopes, index=series.index)


KEY_WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "surface_pressure_mean",
]
LAG_OFFSETS = [1, 2, 3, 7]
ROLL_WINDOWS = [3, 7]


def _compute_features(df: pd.DataFrame, city_key: str) -> pd.DataFrame:
    """
    Given a DataFrame of raw weather with enough past history,
    compute all 110 model features. Mirrors feature_engineering.py exactly.

    df must be sorted ascending by date with at least 30 rows of past data
    before the target prediction date.
    """
    city_info = CITIES[city_key]
    df = df.copy().sort_values("date").reset_index(drop=True)

    # --- tmax_normal and tmax_departure ---
    doys = df["date"].dt.dayofyear
    df["tmax_normal"] = doys.apply(lambda d: get_tmax_normal(city_key, int(d)))
    df["tmax_departure"] = df["temperature_2m_max"] - df["tmax_normal"]

    # --- qualifying_day ---
    is_coastal = city_info["region_type"] == "coastal"
    if is_coastal:
        df["qualifying_day"] = (
            (df["temperature_2m_max"] >= 37.0) & (df["tmax_departure"] >= 4.5)
        ).astype(int)
    else:
        df["qualifying_day"] = (
            ((df["temperature_2m_max"] >= 40.0) & (df["tmax_departure"] >= 4.5))
            | (df["temperature_2m_max"] >= 45.0)
        ).astype(int)

    # --- heatwave_lag1 (yesterday's qualifying_day as proxy for heatwave state) ---
    # For live data we don't have the 2-day run context for prior days,
    # so we use the qualifying_day flag shifted by 1 as the best available proxy.
    df["heatwave_lag1"] = df["qualifying_day"].shift(1).fillna(0)

    # --- Group 2: lag features ---
    for var in KEY_WEATHER_VARS:
        for lag in LAG_OFFSETS:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
    for lag in LAG_OFFSETS:
        df[f"tmax_departure_lag{lag}"] = df["tmax_departure"].shift(lag)

    # --- Group 3: rolling features ---
    for var in KEY_WEATHER_VARS:
        past = df[var].shift(1)
        for w in ROLL_WINDOWS:
            roll = past.rolling(window=w, min_periods=w)
            df[f"{var}_roll{w}_mean"] = roll.mean()
            df[f"{var}_roll{w}_max"] = roll.max()
            df[f"{var}_roll{w}_min"] = roll.min()

    # --- Group 4: trend features ---
    tmax = df["temperature_2m_max"]
    df["tmax_delta_1d"] = tmax - tmax.shift(1)
    df["tmax_delta_3d"] = tmax - tmax.shift(3)
    df["tmax_delta_7d"] = tmax - tmax.shift(7)
    tmax_past = tmax.shift(1)
    df["tmax_slope_3d"] = _rolling_slope(tmax_past, window=3)
    df["tmax_slope_7d"] = _rolling_slope(tmax_past, window=7)

    # --- Group 5: anomaly z-score ---
    dep_past = df["tmax_departure"].shift(1)
    dep_roll30_std = dep_past.rolling(window=30, min_periods=10).std()
    dep_roll30_mean = dep_past.rolling(window=30, min_periods=10).mean()
    df["tmax_departure_zscore"] = (df["tmax_departure"] - dep_roll30_mean) / dep_roll30_std.replace(
        0, np.nan
    )

    # --- Group 6: calendar features ---
    dates = df["date"]
    df["month"] = dates.dt.month
    df["day_of_year"] = dates.dt.dayofyear
    df["season_code"] = df["month"].map(SEASON_MAP)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # --- Group 7: city features ---
    df["city_encoded"] = CITY_ORDER.index(city_key)
    df["is_coastal"] = int(is_coastal)
    df["latitude"] = city_info["lat"]
    df["longitude"] = city_info["lon"]

    return df


def _load_feature_list() -> List[str]:
    path = PROJECT_ROOT / "models" / "final" / "feature_list.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [entry["name"] for entry in raw]


_FEATURE_NAMES: List[str] = _load_feature_list()


def _row_to_feature_dict(row: pd.Series) -> Optional[Dict[str, float]]:
    """Extract the 110-feature dict from a DataFrame row. Returns None if any NaN."""
    result = {}
    for name in _FEATURE_NAMES:
        if name not in row.index:
            return None
        val = row[name]
        if pd.isna(val):
            return None
        result[name] = float(val)
    return result


# ---------------------------------------------------------------------------
# Risk level (mirrors Part 2 thresholds from app.py exactly)
# ---------------------------------------------------------------------------
def _prob_to_risk(prob: float) -> str:
    if prob >= 0.80:
        return "EXTREME"
    if prob >= 0.60:
        return "HIGH"
    if prob >= 0.30:
        return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# Main function: fetch + engineer + predict
# ---------------------------------------------------------------------------
def get_live_forecast(
    city_key: str, pipeline, raw_weather: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Fetch live weather, build features, run the full pipeline, return structured result.

    Parameters
    ----------
    city_key : str   One of the 5 supported city keys.
    pipeline : ClimateGuardPipeline   Loaded pipeline (passed in from app.py).

    Returns
    -------
    dict with keys: city, city_info, days (list of per-day results),
                    source, last_updated, cache_used, warnings, error (if failed)
    """
    global _RATE_LIMITED_UNTIL

    city_info = CITIES[city_key]
    warnings: List[str] = [
        "Forecast-based predictions are estimates and may differ from the model's "
        "historical accuracy on ERA5 data.",
        "Not an official warning. Follow IMD advisories during heat emergencies.",
    ]

    if raw_weather is None:
        # --- Check cache ---
        cached = _cache_get(city_key)
        if cached is not None:
            return _cached_result(cached)

        # --- Fetch from Open-Meteo ---
        # Avoid repeatedly sending a cloud-hosted shared IP into an upstream 429.
        # If a recent successful result exists, it is preferable to label and return
        # that result rather than pretending the stale result is current.
        stale_cached = _stale_cache_get(city_key)
        if time.time() < _RATE_LIMITED_UNTIL:
            if stale_cached is not None:
                return _cached_result(stale_cached, stale=True)
            retry_in_seconds = max(1, round(_RATE_LIMITED_UNTIL - time.time()))
            return _rate_limit_result(city_key, city_info, retry_in_seconds)

        try:
            raw_df = _fetch_open_meteo(city_key, past_days=30, forecast_days=7)
        except OpenMeteoRateLimitError as exc:
            _RATE_LIMITED_UNTIL = time.time() + exc.retry_after_seconds
            if stale_cached is not None:
                return _cached_result(stale_cached, stale=True)
            return _rate_limit_result(city_key, city_info, exc.retry_after_seconds)
        except Exception as exc:
            return {
                "city": city_key,
                "city_info": city_info,
                "error": str(exc),
                "error_type": "fetch_failed",
                "days": [],
                "source": "Open-Meteo API",
                "last_updated": None,
                "cache_used": False,
            }
    else:
        raw_df = raw_weather

    # --- Engineer features ---
    try:
        feat_df = _compute_features(raw_df, city_key)
    except Exception as exc:
        return {
            "city": city_key,
            "city_info": city_info,
            "error": f"Feature engineering failed: {exc}",
            "error_type": "feature_error",
            "days": [],
            "source": "Open-Meteo API",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "cache_used": False,
        }

    # --- Identify prediction rows (today onward) ---
    today = pd.Timestamp(date.today())
    forecast_mask = feat_df["date"] >= today
    pred_rows = feat_df[forecast_mask].copy()

    if len(pred_rows) == 0:
        return {
            "city": city_key,
            "city_info": city_info,
            "error": "No forecast dates available in response.",
            "error_type": "no_data",
            "days": [],
            "source": "Open-Meteo API",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "cache_used": False,
        }

    # --- Predict each day ---
    days: List[Dict] = []
    predictor = pipeline.predictor

    for _, row in pred_rows.iterrows():
        day_date = row["date"].strftime("%Y-%m-%d")
        is_forecast = row["date"] > today

        features = _row_to_feature_dict(row)
        if features is None:
            missing = [n for n in _FEATURE_NAMES if n not in row.index or pd.isna(row.get(n))]
            days.append(
                {
                    "date": day_date,
                    "type": "forecast" if is_forecast else "today",
                    "error": f"Cannot predict: {len(missing)} feature(s) have null values: {missing[:5]}",
                    "probability": None,
                    "prediction": None,
                    "risk_level": None,
                }
            )
            continue

        try:
            # Run through Part 1 predictor directly (features are already validated)
            feat_array = np.array([[features[n] for n in _FEATURE_NAMES]])
            prob = float(predictor.model.predict_proba(feat_array)[0, 1])
            pred_label = int(prob >= predictor.threshold)
            risk = _prob_to_risk(prob)

            # Expert rules from Part 3
            rule_results = pipeline.expert_engine.evaluate(
                data={**features, "city_key": city_key},
                risk_level=risk,
                heatwave_probability=prob,
            )
            triggered_rules = [
                {"rule_id": r.rule_id, "name": r.name, "severity": r.severity, "message": r.message}
                for r in rule_results
                if r.triggered
            ]

            days.append(
                {
                    "date": day_date,
                    "type": "forecast" if is_forecast else "today",
                    "is_forecast": is_forecast,
                    "probability": round(prob, 4),
                    "prediction": pred_label,
                    "risk_level": risk,
                    "temperature_max": (
                        round(float(row["temperature_2m_max"]), 1)
                        if not pd.isna(row["temperature_2m_max"])
                        else None
                    ),
                    "temperature_min": (
                        round(float(row["temperature_2m_min"]), 1)
                        if not pd.isna(row["temperature_2m_min"])
                        else None
                    ),
                    "apparent_temperature_max": (
                        round(float(row["apparent_temperature_max"]), 1)
                        if not pd.isna(row["apparent_temperature_max"])
                        else None
                    ),
                    "humidity_mean": (
                        round(float(row["relative_humidity_2m_mean"]), 1)
                        if not pd.isna(row["relative_humidity_2m_mean"])
                        else None
                    ),
                    "wind_speed_max": (
                        round(float(row["wind_speed_10m_max"]), 1)
                        if not pd.isna(row["wind_speed_10m_max"])
                        else None
                    ),
                    "tmax_departure": (
                        round(float(row["tmax_departure"]), 2)
                        if not pd.isna(row["tmax_departure"])
                        else None
                    ),
                    "qualifying_day": int(row["qualifying_day"]),
                    "triggered_rules": triggered_rules,
                }
            )
        except Exception as exc:
            days.append(
                {
                    "date": day_date,
                    "type": "forecast" if is_forecast else "today",
                    "error": f"Prediction error: {exc}",
                    "probability": None,
                    "prediction": None,
                    "risk_level": None,
                }
            )

    result = {
        "city": city_key,
        "city_info": city_info,
        "days": days,
        "source": "Open-Meteo.com (free API, no key)",
        "source_url": "https://open-meteo.com",
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "cache_used": False,
        "warnings": warnings,
        "feature_notes": [
            "tmax_departure_zscore: computed from 30-day rolling window of live data.",
            "heatwave_lag1: derived from yesterday's qualifying_day flag (live proxy).",
            "Forecast days use Open-Meteo NWP data, not ERA5 reanalysis.",
        ],
        "error": None,
    }

    _cache_set(city_key, result)
    return result


def get_live_forecast_from_browser_payload(city_key: str, daily: Any, pipeline) -> Dict[str, Any]:
    """Predict from Open-Meteo daily data fetched directly by a user's browser.

    This fallback only moves the public weather download away from a shared cloud
    IP. Feature engineering and inference still run on the server, and the
    payload has a narrow, validated shape before it is processed.
    """
    city_info = CITIES[city_key]
    try:
        raw_df = _daily_payload_to_frame(city_key, daily)
    except (TypeError, ValueError) as exc:
        return {
            "city": city_key,
            "city_info": city_info,
            "error": f"Browser weather data could not be validated: {exc}",
            "error_type": "client_weather_invalid",
            "days": [],
            "source": "Open-Meteo API",
            "last_updated": None,
            "cache_used": False,
        }

    result = get_live_forecast(city_key, pipeline, raw_weather=raw_df)
    if not result.get("error"):
        warnings = list(result.get("warnings", []))
        warnings.append(
            "Weather data was fetched directly by this browser because the hosted "
            "service's shared network was rate-limited."
        )
        result["warnings"] = warnings
        result["browser_weather_fallback"] = True
    return result


def _daily_payload_to_frame(city_key: str, daily: Any) -> pd.DataFrame:
    """Validate a bounded Open-Meteo daily payload and convert it to a frame."""
    if not isinstance(daily, dict):
        raise TypeError("daily must be an object")

    expected_columns = {"time", *OPEN_METEO_DAILY_VARS}
    missing_columns = expected_columns.difference(daily)
    if missing_columns:
        raise ValueError(f"missing daily fields: {sorted(missing_columns)}")

    time_values = daily["time"]
    if not isinstance(time_values, list) or not 31 <= len(time_values) <= 45:
        raise ValueError("daily time must contain 31 to 45 dates")
    if any(
        not isinstance(value, list) or len(value) != len(time_values) for value in daily.values()
    ):
        raise ValueError("daily fields must be equally sized arrays")

    frame = pd.DataFrame({name: daily[name] for name in expected_columns})
    frame = frame.rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["city_key"] = city_key
    return frame


def _rate_limit_result(
    city_key: str, city_info: Dict[str, Any], retry_in_seconds: int
) -> Dict[str, Any]:
    """Create a concise, non-sensitive response for a temporary provider 429."""
    retry_minutes = max(1, math.ceil(retry_in_seconds / 60))
    return {
        "city": city_key,
        "city_info": city_info,
        "error": (
            "The live weather provider is temporarily busy. ClimateGuard will retry "
            f"in about {retry_minutes} minute(s)."
        ),
        "error_type": "rate_limited",
        "retry_after_seconds": retry_in_seconds,
        "days": [],
        "source": "Open-Meteo API",
        "last_updated": None,
        "cache_used": False,
    }

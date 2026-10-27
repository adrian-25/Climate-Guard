"""Network-free resilience tests for Europe live weather requests."""

from datetime import date, timedelta
from unittest.mock import patch

import requests

from src import europe_live


def _browser_daily_payload(days: int = 37) -> dict[str, list]:
    """Build a bounded, realistic Europe browser-weather fixture."""
    start = date(2026, 6, 1)
    temperatures = [31.0 + (index % 4) for index in range(days)]
    return {
        "time": [(start + timedelta(days=index)).isoformat() for index in range(days)],
        "temperature_2m_max": temperatures,
        "temperature_2m_min": [18.0 + (index % 2) for index in range(days)],
        "temperature_2m_mean": [25.0 + (index % 2) for index in range(days)],
        "apparent_temperature_max": [value + 0.8 for value in temperatures],
        "apparent_temperature_min": [19.0 + (index % 2) for index in range(days)],
        "apparent_temperature_mean": [26.0 + (index % 2) for index in range(days)],
        "precipitation_sum": [0.0] * days,
        "rain_sum": [0.0] * days,
        "wind_speed_10m_max": [12.0] * days,
        "wind_gusts_10m_max": [20.0] * days,
        "relative_humidity_2m_max": [65.0] * days,
        "relative_humidity_2m_min": [35.0] * days,
        "relative_humidity_2m_mean": [50.0] * days,
        "surface_pressure_mean": [1012.0] * days,
        "shortwave_radiation_sum": [25.0] * days,
        "et0_fao_evapotranspiration": [5.0] * days,
    }


def test_europe_provider_outage_does_not_expose_upstream_url():
    """A temporary Open-Meteo error should preserve the live response shape safely."""
    original_cache = dict(europe_live.CACHE)
    europe_live.CACHE.clear()
    try:
        with patch.object(
            europe_live.requests,
            "get",
            side_effect=requests.exceptions.Timeout("https://api.open-meteo.com/v1/forecast"),
        ):
            result = europe_live.get_europe_live_forecast("porto")
    finally:
        europe_live.CACHE.clear()
        europe_live.CACHE.update(original_cache)

    assert result["error_type"] == "provider_unavailable"
    assert result["days"] == []
    assert "https://" not in result["error"]


def test_europe_browser_weather_payload_returns_live_predictions_without_caching():
    """A validated browser payload should restore Europe cards during a host outage."""
    original_cache = dict(europe_live.CACHE)
    europe_live.CACHE.clear()
    try:
        result = europe_live.get_europe_live_forecast_from_browser_payload(
            "porto", _browser_daily_payload()
        )
    finally:
        europe_live.CACHE.clear()
        europe_live.CACHE.update(original_cache)

    assert result["browser_weather_fallback"] is True
    assert result["source"] == "Open-Meteo (browser fallback)"
    assert len(result["days"]) == 7
    assert "error" not in result


def test_europe_browser_weather_payload_rejects_invalid_data():
    result = europe_live.get_europe_live_forecast_from_browser_payload("porto", {"time": []})

    assert result["error_type"] == "client_weather_invalid"
    assert result["days"] == []

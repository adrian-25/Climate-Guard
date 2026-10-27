"""Network-free resilience tests for Europe live weather requests."""

from unittest.mock import patch

import requests

from src import europe_live


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

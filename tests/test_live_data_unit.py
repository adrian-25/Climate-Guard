"""Network-free unit tests for live Open-Meteo integration."""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

import live_data


class LiveDataTests(unittest.TestCase):
    def test_fetch_uses_expected_city_and_returns_dataframe(self):
        fake = Mock()
        fake.json.return_value = {
            "daily": {
                "time": ["2026-01-01"],
                "temperature_2m_max": [40],
                "temperature_2m_min": [25],
                "relative_humidity_2m_mean": [30],
            }
        }
        with patch.object(live_data.http_requests, "get", return_value=fake) as get:
            frame = live_data._fetch_open_meteo("delhi", past_days=1, forecast_days=1)
        self.assertEqual(frame.loc[0, "city_key"], "delhi")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["date"]))
        self.assertEqual(
            get.call_args.kwargs["params"]["latitude"], live_data.CITIES["delhi"]["lat"]
        )

    def test_fetch_translates_upstream_failure(self):
        with patch.object(live_data.http_requests, "get", side_effect=requests.exceptions.Timeout):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                live_data._fetch_open_meteo("delhi")

    def test_fetch_translates_rate_limit_without_exposing_request_url(self):
        fake = Mock()
        fake.status_code = 429
        fake.headers = {"Retry-After": "120"}
        with patch.object(live_data.http_requests, "get", return_value=fake):
            with self.assertRaises(live_data.OpenMeteoRateLimitError) as error:
                live_data._fetch_open_meteo("delhi")
        self.assertEqual(error.exception.retry_after_seconds, 120)
        self.assertNotIn("https://", str(error.exception))

    def test_rate_limit_response_is_concise_and_has_no_network_call(self):
        original_until = live_data._RATE_LIMITED_UNTIL
        live_data._RATE_LIMITED_UNTIL = live_data.time.time() + 120
        try:
            with patch.object(live_data.http_requests, "get") as get:
                result = live_data.get_live_forecast("delhi", pipeline=Mock())
            get.assert_not_called()
        finally:
            live_data._RATE_LIMITED_UNTIL = original_until

        self.assertEqual(result["error_type"], "rate_limited")
        self.assertEqual(result["days"], [])
        self.assertNotIn("https://", result["error"])

    def test_rate_limited_request_uses_recent_stale_cache(self):
        original_until = live_data._RATE_LIMITED_UNTIL
        original_cache = dict(live_data._CACHE)
        live_data._CACHE["delhi"] = {
            "city": "delhi",
            "days": [],
            "warnings": ["Original forecast notice."],
            "fetched_at": live_data.time.time() - live_data.CACHE_TTL_SECONDS - 1,
        }
        live_data._RATE_LIMITED_UNTIL = live_data.time.time() + 120
        try:
            result = live_data.get_live_forecast("delhi", pipeline=Mock())
        finally:
            live_data._CACHE.clear()
            live_data._CACHE.update(original_cache)
            live_data._RATE_LIMITED_UNTIL = original_until

        self.assertTrue(result["cache_used"])
        self.assertTrue(result["cache_stale"])
        self.assertTrue(any("rate-limited" in warning for warning in result["warnings"]))

    def test_risk_boundaries(self):
        self.assertEqual(live_data._prob_to_risk(0.0), "LOW")
        self.assertEqual(live_data._prob_to_risk(0.3), "MODERATE")
        self.assertEqual(live_data._prob_to_risk(0.6), "HIGH")
        self.assertEqual(live_data._prob_to_risk(0.8), "EXTREME")

    def test_live_features_match_training_contract(self):
        feature_path = (
            Path(__file__).resolve().parents[1] / "models" / "final" / "feature_list.json"
        )
        with feature_path.open(encoding="utf-8") as handle:
            training_features = [item["name"] for item in json.load(handle)]
        self.assertEqual(list(live_data._FEATURE_NAMES), training_features)
        self.assertEqual(len(live_data._FEATURE_NAMES), 110)


if __name__ == "__main__":
    unittest.main()

"""Network-free unit tests for live Open-Meteo integration."""

import json
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
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

    def test_temporary_provider_outage_is_sanitised_for_clients(self):
        fake = Mock()
        fake.status_code = 503
        fake.headers = {"Retry-After": "60"}
        original_until = live_data._PROVIDER_UNAVAILABLE_UNTIL
        original_cache = dict(live_data._CACHE)
        live_data._CACHE.clear()
        try:
            with patch.object(live_data.http_requests, "get", return_value=fake):
                result = live_data.get_live_forecast("lucknow", pipeline=Mock())
        finally:
            live_data._PROVIDER_UNAVAILABLE_UNTIL = original_until
            live_data._CACHE.clear()
            live_data._CACHE.update(original_cache)

        self.assertEqual(result["error_type"], "provider_unavailable")
        self.assertEqual(result["retry_after_seconds"], 60)
        self.assertNotIn("https://", result["error"])
        self.assertNotIn("503", result["error"])

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

    def test_browser_daily_payload_is_bounded_and_converted_to_a_city_frame(self):
        dates = pd.date_range("2026-01-01", periods=31).strftime("%Y-%m-%d").tolist()
        daily = {"time": dates}
        daily.update({field: [1.0] * len(dates) for field in live_data.OPEN_METEO_DAILY_VARS})

        frame = live_data._daily_payload_to_frame("delhi", daily)

        self.assertEqual(len(frame), 31)
        self.assertEqual(frame.loc[0, "city_key"], "delhi")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["date"]))

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

    def test_live_forecast_includes_drift_after_sufficient_history(self):
        """A network-free live run includes PSI based on observed rows only."""
        dates = pd.date_range(date.today() - timedelta(days=41), periods=48)
        raw_weather = pd.DataFrame({"date": dates, "city_key": "delhi"})
        for index, field in enumerate(live_data.OPEN_METEO_DAILY_VARS):
            raw_weather[field] = [20.0 + index + (day * 0.2) for day in range(len(dates))]

        pipeline = Mock()
        pipeline.predictor.threshold = 0.70
        pipeline.predictor.model.predict_proba.return_value = np.array([[0.8, 0.2]])
        pipeline.expert_engine.evaluate.return_value = []
        original_cache = dict(live_data._CACHE)
        original_drift = dict(live_data._LATEST_DRIFT_BY_CITY)
        try:
            result = live_data.get_live_forecast(
                "delhi", pipeline=pipeline, raw_weather=raw_weather
            )
        finally:
            live_data._CACHE.clear()
            live_data._CACHE.update(original_cache)
            live_data._LATEST_DRIFT_BY_CITY.clear()
            live_data._LATEST_DRIFT_BY_CITY.update(original_drift)

        self.assertIsNone(result["error"])
        self.assertEqual(result["drift"]["status"], "available")
        self.assertGreaterEqual(result["drift"]["sample_size"], 30)
        self.assertTrue(all("risk_score" in day for day in result["days"] if not day.get("error")))


if __name__ == "__main__":
    unittest.main()

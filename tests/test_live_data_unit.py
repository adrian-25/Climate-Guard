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

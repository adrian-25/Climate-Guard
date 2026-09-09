"""
ClimateGuard Part 3 — Test Suite
tests/test_part3.py

Uses unittest (matching the project's test_part2.py pattern) so it runs
directly with:
    python tests/test_part3.py
    python -m unittest tests.test_part3 -v

Test groups
-----------
A. InputValidator
B. FeatureContractValidator
C. FeatureTransformer
D. ETLPipeline (single record)
E. ETLPipeline (batch)
F. Expert Rules — individual rule functions
G. ExpertRuleEngine
H. ClimateGuardPipeline (full pipeline)
I. Error handling
J. JSON serialisation
K. Real-data smoke test (X_test.csv)
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_LIST_PATH = PROJECT_ROOT / "models" / "final" / "feature_list.json"
X_TEST_PATH       = PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv"
META_TEST_PATH    = PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv"

# ---------------------------------------------------------------------------
# Load the canonical 110 feature names once
# ---------------------------------------------------------------------------
with open(FEATURE_LIST_PATH, encoding="utf-8") as _f:
    _FEATURE_LIST_RAW = json.load(_f)
FEATURE_NAMES: List[str] = [e["name"] for e in _FEATURE_LIST_RAW]
assert len(FEATURE_NAMES) == 110, f"Expected 110 features, got {len(FEATURE_NAMES)}"

# ---------------------------------------------------------------------------
# Module-level shared objects (loaded once for efficiency)
# ---------------------------------------------------------------------------
_PIPELINE = None
_X_TEST   = None
_META_TEST = None

def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from src.integration import ClimateGuardPipeline
        _PIPELINE = ClimateGuardPipeline()
    return _PIPELINE

def _get_test_data():
    global _X_TEST, _META_TEST
    if _X_TEST is None:
        _X_TEST   = pd.read_csv(X_TEST_PATH)
        _META_TEST = pd.read_csv(META_TEST_PATH)
    return _X_TEST, _META_TEST


# ---------------------------------------------------------------------------
# Helper: build a synthetic valid feature record
# ---------------------------------------------------------------------------

def _make_normal_record(
    city_key: str = "delhi",
    date: str = "2024-01-15",
    tmax: float = 22.0,
    tmin: float = 8.0,
    tmean: float = 15.0,
    tmax_departure: float = -2.0,
    tmax_departure_zscore: float = -1.2,
    tmax_normal: float = 24.0,
    roll7_mean: float = 21.0,
) -> Dict:
    """Build a minimal valid record with all 110 features at sensible defaults."""
    record: Dict = {}

    scalar_overrides = {
        "temperature_2m_max":  tmax,
        "temperature_2m_min":  tmin,
        "temperature_2m_mean": tmean,
        "tmax_departure":      tmax_departure,
        "tmax_departure_zscore": tmax_departure_zscore,
        "tmax_normal":         tmax_normal,
        "temperature_2m_max_roll7_mean": roll7_mean,
        "apparent_temperature_max":  tmax + 1.0,
        "apparent_temperature_mean": tmean + 0.5,
        "apparent_temperature_min":  tmin - 0.5,
        "relative_humidity_2m_max":  70.0,
        "relative_humidity_2m_mean": 55.0,
        "relative_humidity_2m_min":  30.0,
        "surface_pressure_mean": 1008.0,
        "wind_speed_10m_max": 15.0,
        "wind_gusts_10m_max": 22.0,
        "precipitation_sum": 0.0,
        "shortwave_radiation_sum": 12.0,
        "et0_fao_evapotranspiration": 3.0,
        "city_encoded": 0.0,
        "is_coastal": 0.0,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "month": 1.0,
        "month_sin": math.sin(2 * math.pi * 1 / 12),
        "month_cos": math.cos(2 * math.pi * 1 / 12),
        "day_of_year": 15.0,
        "doy_sin": math.sin(2 * math.pi * 15 / 365),
        "doy_cos": math.cos(2 * math.pi * 15 / 365),
        "season_code": 4.0,
        "qualifying_day": 0.0,
        "heatwave_lag1": 0.0,
    }

    # Default all 110 features to 0.0
    for name in FEATURE_NAMES:
        record[name] = 0.0

    record.update(scalar_overrides)

    # Fill lag features with current value
    for k in FEATURE_NAMES:
        if "_lag" in k:
            if "temperature_2m_max" in k:  record[k] = tmax
            elif "temperature_2m_min" in k: record[k] = tmin
            elif "temperature_2m_mean" in k: record[k] = tmean
            elif "tmax_departure" in k:  record[k] = tmax_departure
            elif "surface_pressure" in k: record[k] = 1008.0
            elif "wind_speed" in k:       record[k] = 15.0
            elif "precipitation" in k:    record[k] = 0.0
            elif "relative_humidity" in k: record[k] = 55.0

    # Fill rolling features
    for k in FEATURE_NAMES:
        if "roll" in k:
            if "temperature_2m_max" in k:  record[k] = roll7_mean
            elif "temperature_2m_min" in k: record[k] = tmin
            elif "temperature_2m_mean" in k: record[k] = tmean
            elif "surface_pressure" in k:  record[k] = 1008.0
            elif "wind_speed" in k:        record[k] = 15.0
            elif "precipitation" in k:     record[k] = 0.0
            elif "relative_humidity" in k: record[k] = 55.0

    # Trend features
    for k in FEATURE_NAMES:
        if "tmax_delta" in k or "tmax_slope" in k:
            record[k] = 0.0

    record["city_key"] = city_key
    record["date"]     = date
    return record


def _make_hot_record(**kwargs) -> Dict:
    """Record for a severe heatwave day that triggers multiple expert rules."""
    return _make_normal_record(
        city_key="delhi",
        date="2024-05-17",
        tmax=44.5,
        tmin=29.0,
        tmean=37.0,
        tmax_departure=4.8,
        tmax_departure_zscore=2.95,
        tmax_normal=39.7,
        roll7_mean=42.0,
        **kwargs,
    )


def _make_mumbai_record(**kwargs) -> Dict:
    rec = _make_normal_record(
        city_key="mumbai", date="2024-05-10",
        tmax=36.0, tmin=28.5, tmean=32.0,
        tmax_departure=3.5, tmax_departure_zscore=1.0,
        tmax_normal=32.5, roll7_mean=35.0, **kwargs,
    )
    rec["city_encoded"] = 4.0
    rec["is_coastal"]   = 1.0
    rec["latitude"]     = 19.0760
    rec["longitude"]    = 72.8777
    return rec


# ===========================================================================
# GROUP A — InputValidator
# ===========================================================================

class TestInputValidator(unittest.TestCase):
    """Tests for src.etl.validator.InputValidator"""

    def setUp(self):
        from src.etl import InputValidator
        self.validator = InputValidator()

    def test_A01_valid_record_passes(self):
        rec = _make_normal_record()
        result = self.validator.validate(rec)
        self.assertTrue(result.valid, f"Errors: {result.errors}")

    def test_A02_missing_city_key_errors(self):
        rec = _make_normal_record()
        del rec["city_key"]
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(any("city_key" in e or "required" in e.lower() for e in result.errors))

    def test_A03_missing_date_errors(self):
        rec = _make_normal_record()
        del rec["date"]
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)

    def test_A04_missing_temperature_2m_max_errors(self):
        rec = _make_normal_record()
        del rec["temperature_2m_max"]
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)

    def test_A05_invalid_city_errors(self):
        rec = _make_normal_record()
        rec["city_key"] = "hyderabad"
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(any("hyderabad" in e.lower() or "city" in e.lower() for e in result.errors))

    def test_A06_delhi_valid(self):
        rec = _make_normal_record(city_key="delhi")
        self.assertTrue(self.validator.validate(rec).valid)

    def test_A06_lucknow_valid(self):
        rec = _make_normal_record(city_key="lucknow")
        self.assertTrue(self.validator.validate(rec).valid)

    def test_A06_nagpur_valid(self):
        rec = _make_normal_record(city_key="nagpur")
        self.assertTrue(self.validator.validate(rec).valid)

    def test_A06_ahmedabad_valid(self):
        rec = _make_normal_record(city_key="ahmedabad")
        self.assertTrue(self.validator.validate(rec).valid)

    def test_A06_mumbai_valid(self):
        rec = _make_normal_record(city_key="mumbai")
        self.assertTrue(self.validator.validate(rec).valid)

    def test_A07_city_alias_new_delhi(self):
        from src.etl import normalise_city
        self.assertEqual(normalise_city("New Delhi"), "delhi")
        self.assertEqual(normalise_city("new_delhi"), "delhi")

    def test_A08_malformed_date_errors(self):
        rec = _make_normal_record()
        rec["date"] = "17/05/2024"
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(any("date" in e.lower() for e in result.errors))

    def test_A09_nan_value_errors(self):
        rec = _make_normal_record()
        rec["temperature_2m_max"] = float("nan")
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)

    def test_A10_non_numeric_value_errors(self):
        rec = _make_normal_record()
        rec["temperature_2m_max"] = "very hot"
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)

    def test_A11_out_of_range_produces_warning_not_error(self):
        rec = _make_normal_record()
        rec["temperature_2m_max"] = 60.0  # above 55°C plausibility bound
        result = self.validator.validate(rec)
        self.assertTrue(len(result.warnings) > 0)
        self.assertTrue(any("temperature_2m_max" in w for w in result.warnings))

    def test_A12_far_future_date_produces_warning(self):
        rec = _make_normal_record()
        rec["date"] = "2031-01-01"
        result = self.validator.validate(rec)
        self.assertTrue(len(result.warnings) > 0)

    def test_A13_series_input_accepted(self):
        rec = _make_normal_record()
        result = self.validator.validate(pd.Series(rec))
        self.assertTrue(result.valid, f"Errors: {result.errors}")

    def test_A14_validation_result_to_dict_structure(self):
        rec = _make_normal_record()
        d = self.validator.validate(rec).to_dict()
        self.assertIn("valid", d)
        self.assertIn("errors", d)
        self.assertIn("warnings", d)
        self.assertIsInstance(d["errors"], list)
        self.assertIsInstance(d["warnings"], list)

    def test_A15_batch_validation_per_row_errors(self):
        rec_good = _make_normal_record()
        rec_bad  = _make_normal_record()
        rec_bad["city_key"] = "invalid_city"
        df = pd.DataFrame([rec_good, rec_bad])
        result = self.validator.validate_batch(df)
        self.assertFalse(result.valid)
        self.assertTrue(any("1" in e for e in result.errors))

    def test_A16_batch_duplicate_city_date_warns(self):
        rec = _make_normal_record()
        df = pd.DataFrame([rec, rec])
        result = self.validator.validate_batch(df)
        self.assertTrue(any("duplicate" in w.lower() for w in result.warnings))


# ===========================================================================
# GROUP B — FeatureContractValidator
# ===========================================================================

class TestFeatureContractValidator(unittest.TestCase):
    """Tests for src.etl.validator.FeatureContractValidator"""

    def setUp(self):
        from src.etl import FeatureContractValidator
        self.validator = FeatureContractValidator()

    def test_B01_correct_110_features_dict_passes(self):
        rec = _make_normal_record()
        result = self.validator.validate(rec)
        self.assertTrue(result.valid, f"Errors: {result.errors}")

    def test_B02_correct_110_features_dataframe_passes(self):
        rec = _make_normal_record()
        df = pd.DataFrame([rec])[FEATURE_NAMES]
        result = self.validator.validate(df)
        self.assertTrue(result.valid, f"Errors: {result.errors}")

    def test_B03_missing_feature_errors(self):
        rec = _make_normal_record()
        del rec["qualifying_day"]
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(any("qualifying_day" in e for e in result.errors))

    def test_B04_missing_multiple_features_errors(self):
        rec = _make_normal_record()
        for k in FEATURE_NAMES[:5]:
            del rec[k]
        result = self.validator.validate(rec)
        self.assertFalse(result.valid)

    def test_B05_nan_in_feature_errors(self):
        rec = _make_normal_record()
        df  = pd.DataFrame([rec])[FEATURE_NAMES].copy()
        df.at[0, "temperature_2m_max"] = float("nan")
        result = self.validator.validate(df)
        self.assertFalse(result.valid)

    def test_B06_extra_unexpected_column_produces_warning(self):
        rec = _make_normal_record()
        rec["some_extra_column"] = 99.9
        result = self.validator.validate(rec)
        self.assertTrue(result.valid)
        self.assertTrue(any("some_extra_column" in w for w in result.warnings))

    def test_B07_wrong_column_order_errors(self):
        rec = _make_normal_record()
        df = pd.DataFrame([rec])[FEATURE_NAMES[::-1]]
        result = self.validator.validate(df, check_order=True)
        self.assertFalse(result.valid)
        self.assertTrue(any("order" in e.lower() for e in result.errors))

    def test_B08_feature_names_property_returns_110(self):
        self.assertEqual(len(self.validator.feature_names), 110)
        self.assertEqual(self.validator.feature_names, FEATURE_NAMES)


# ===========================================================================
# GROUP C — FeatureTransformer
# ===========================================================================

class TestFeatureTransformer(unittest.TestCase):
    """Tests for src.etl.transformer.FeatureTransformer"""

    def setUp(self):
        from src.etl import FeatureTransformer
        self.transformer = FeatureTransformer()

    def test_C01_transform_produces_correct_shape(self):
        rec = _make_normal_record()
        out = self.transformer.transform(rec)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(out.shape, (1, 110))
        self.assertEqual(list(out.columns), FEATURE_NAMES)

    def test_C02_all_columns_float64(self):
        rec = _make_normal_record()
        out = self.transformer.transform(rec)
        for col in out.columns:
            self.assertEqual(out[col].dtype, np.float64, f"{col} has dtype {out[col].dtype}")

    def test_C03_no_nan_in_output(self):
        rec = _make_normal_record()
        out = self.transformer.transform(rec)
        self.assertFalse(out.isna().any().any())

    def test_C04_column_order_matches_feature_list(self):
        rec = _make_normal_record()
        out = self.transformer.transform(rec)
        self.assertEqual(list(out.columns), FEATURE_NAMES)

    def test_C05_transform_with_metadata_attaches_city_date(self):
        rec = _make_normal_record(city_key="lucknow", date="2024-05-01")
        out = self.transformer.transform_with_metadata(rec)
        self.assertIn("city_key", out.columns)
        self.assertIn("date", out.columns)
        self.assertEqual(out["city_key"].iloc[0], "lucknow")

    def test_C06_missing_feature_raises_value_error(self):
        rec = _make_normal_record()
        del rec["qualifying_day"]
        with self.assertRaises(ValueError):
            self.transformer.transform(rec)

    def test_C07_dict_and_dataframe_produce_same_result(self):
        rec = _make_normal_record()
        out1 = self.transformer.transform(rec)
        df   = pd.DataFrame([rec])
        out2 = self.transformer.transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_C08_get_feature_count_returns_110(self):
        self.assertEqual(self.transformer.get_feature_count(), 110)

    def test_C09_feature_names_property_correct(self):
        self.assertEqual(self.transformer.feature_names, FEATURE_NAMES)


# ===========================================================================
# GROUP D — ETLPipeline (single record)
# ===========================================================================

class TestETLPipelineSingle(unittest.TestCase):
    """Tests for src.etl.pipeline.ETLPipeline.run()"""

    def setUp(self):
        from src.etl import ETLPipeline
        self.etl = ETLPipeline()

    def test_D01_valid_record_returns_valid(self):
        rec = _make_normal_record()
        res = self.etl.run(rec)
        self.assertTrue(res.valid, f"Errors: {res.validation.errors}")
        self.assertIsNotNone(res.feature_df)
        # 110 features + city_key + date = 112 columns
        self.assertEqual(res.feature_df.shape[1], 112)

    def test_D02_invalid_city_returns_invalid(self):
        rec = _make_normal_record()
        rec["city_key"] = "bangalore"
        res = self.etl.run(rec)
        self.assertFalse(res.valid)
        self.assertIsNone(res.feature_df)

    def test_D03_malformed_date_returns_invalid(self):
        rec = _make_normal_record()
        rec["date"] = "not-a-date"
        res = self.etl.run(rec)
        self.assertFalse(res.valid)

    def test_D04_nan_feature_returns_invalid(self):
        rec = _make_normal_record()
        rec["temperature_2m_max"] = float("nan")
        res = self.etl.run(rec)
        self.assertFalse(res.valid)

    def test_D05_missing_feature_returns_invalid(self):
        rec = _make_normal_record()
        del rec["tmax_departure_zscore"]
        res = self.etl.run(rec)
        self.assertFalse(res.valid)

    def test_D06_series_input_accepted(self):
        rec = _make_normal_record()
        res = self.etl.run(pd.Series(rec))
        self.assertTrue(res.valid, f"Errors: {res.validation.errors}")

    def test_D07_single_row_dataframe_accepted(self):
        rec = _make_normal_record()
        res = self.etl.run(pd.DataFrame([rec]))
        self.assertTrue(res.valid, f"Errors: {res.validation.errors}")

    def test_D08_multi_row_dataframe_rejected_by_run(self):
        rec = _make_normal_record()
        res = self.etl.run(pd.DataFrame([rec, rec]))
        self.assertFalse(res.valid)
        self.assertTrue(any("run_batch" in e.lower() or "1 row" in e.lower()
                            for e in res.validation.errors))

    def test_D09_etl_result_to_dict_serialisable(self):
        rec = _make_normal_record()
        res = self.etl.run(rec)
        d   = res.to_dict()
        self.assertIn("valid", d)
        self.assertIn("validation", d)
        json.dumps(d)  # must not raise

    def test_D10_metadata_contains_city_date(self):
        rec = _make_normal_record(city_key="nagpur", date="2024-05-20")
        res = self.etl.run(rec)
        self.assertEqual(res.metadata.get("city_key"), "nagpur")
        self.assertEqual(res.metadata.get("date"), "2024-05-20")


# ===========================================================================
# GROUP E — ETLPipeline (batch)
# ===========================================================================

class TestETLPipelineBatch(unittest.TestCase):
    """Tests for src.etl.pipeline.ETLPipeline.run_batch()"""

    def setUp(self):
        from src.etl import ETLPipeline
        self.etl = ETLPipeline()

    def test_E01_valid_batch_all_rows_pass(self):
        recs = [_make_normal_record(city_key=c) for c in ["delhi", "lucknow", "nagpur"]]
        res  = self.etl.run_batch(pd.DataFrame(recs))
        self.assertTrue(res.valid)
        self.assertIsNotNone(res.feature_df)
        self.assertEqual(res.feature_df.shape[0], 3)

    def test_E02_one_bad_row_noted_in_errors(self):
        rec_good = _make_normal_record()
        rec_bad  = _make_normal_record()
        rec_bad["city_key"] = "X"
        res = self.etl.run_batch(pd.DataFrame([rec_good, rec_bad]))
        self.assertIsNotNone(res.feature_df)
        self.assertEqual(res.feature_df.shape[0], 1)
        self.assertTrue(any("1" in e for e in res.validation.errors))

    def test_E03_all_bad_rows_feature_df_none(self):
        rec = _make_normal_record()
        rec["city_key"] = "invalid"
        res = self.etl.run_batch(pd.DataFrame([rec, rec]))
        self.assertFalse(res.valid)
        self.assertIsNone(res.feature_df)

    def test_E04_duplicate_city_date_warns(self):
        rec = _make_normal_record(city_key="delhi", date="2024-05-01")
        res = self.etl.run_batch(pd.DataFrame([rec, rec]))
        self.assertTrue(any("duplicate" in w.lower() for w in res.validation.warnings))

    def test_E05_empty_dataframe_errors(self):
        res = self.etl.run_batch(pd.DataFrame())
        self.assertFalse(res.valid)

    def test_E06_non_dataframe_input_errors(self):
        res = self.etl.run_batch({"not": "a dataframe"})
        self.assertFalse(res.valid)


# ===========================================================================
# GROUP F — Expert Rules (individual functions)
# ===========================================================================

class TestExpertRulesIndividual(unittest.TestCase):
    """Tests for individual rule evaluation functions in src.expert_rules.rules"""

    # RULE_01 — Extreme Temperature
    def test_F01_rule01_triggers_plains_45C(self):
        from src.expert_rules import evaluate_rule_01
        r = evaluate_rule_01({"temperature_2m_max": 45.0, "city_key": "delhi"})
        self.assertTrue(r.triggered)
        self.assertEqual(r.rule_id, "RULE_01")
        self.assertEqual(r.severity, "CRITICAL")

    def test_F02_rule01_no_trigger_below_45C_plains(self):
        from src.expert_rules import evaluate_rule_01
        self.assertFalse(
            evaluate_rule_01({"temperature_2m_max": 44.9, "city_key": "delhi"}).triggered)

    def test_F03_rule01_triggers_coastal_40C(self):
        from src.expert_rules import evaluate_rule_01
        self.assertTrue(
            evaluate_rule_01({"temperature_2m_max": 40.0, "city_key": "mumbai"}).triggered)

    def test_F04_rule01_no_trigger_below_40C_coastal(self):
        from src.expert_rules import evaluate_rule_01
        self.assertFalse(
            evaluate_rule_01({"temperature_2m_max": 39.9, "city_key": "mumbai"}).triggered)

    def test_F05_rule01_boundary_exactly_45C(self):
        from src.expert_rules import evaluate_rule_01
        self.assertTrue(
            evaluate_rule_01({"temperature_2m_max": 45.0, "city_key": "lucknow"}).triggered)

    def test_F06_rule01_missing_tmax_no_trigger(self):
        from src.expert_rules import evaluate_rule_01
        self.assertFalse(evaluate_rule_01({"city_key": "delhi"}).triggered)

    # RULE_02 — Persistent Heat
    def test_F07_rule02_triggers_both_thresholds(self):
        from src.expert_rules import evaluate_rule_02
        r = evaluate_rule_02({"temperature_2m_max": 41.0,
                               "temperature_2m_max_roll7_mean": 38.0})
        self.assertTrue(r.triggered)
        self.assertEqual(r.severity, "WARNING")

    def test_F08_rule02_no_trigger_roll7_below(self):
        from src.expert_rules import evaluate_rule_02
        self.assertFalse(evaluate_rule_02({"temperature_2m_max": 42.0,
                                           "temperature_2m_max_roll7_mean": 36.9}).triggered)

    def test_F09_rule02_no_trigger_tmax_below(self):
        from src.expert_rules import evaluate_rule_02
        self.assertFalse(evaluate_rule_02({"temperature_2m_max": 39.9,
                                           "temperature_2m_max_roll7_mean": 38.0}).triggered)

    def test_F10_rule02_missing_roll7_no_trigger(self):
        from src.expert_rules import evaluate_rule_02
        self.assertFalse(evaluate_rule_02({"temperature_2m_max": 43.0}).triggered)

    # RULE_03 — High Nighttime Temperature
    def test_F11_rule03_triggers_at_28C(self):
        from src.expert_rules import evaluate_rule_03
        r = evaluate_rule_03({"temperature_2m_min": 28.0})
        self.assertTrue(r.triggered)
        self.assertEqual(r.severity, "WARNING")

    def test_F12_rule03_no_trigger_below_28C(self):
        from src.expert_rules import evaluate_rule_03
        self.assertFalse(evaluate_rule_03({"temperature_2m_min": 27.9}).triggered)

    def test_F13_rule03_missing_tmin_no_trigger(self):
        from src.expert_rules import evaluate_rule_03
        self.assertFalse(evaluate_rule_03({}).triggered)

    # RULE_04 — Compounded Heat Stress
    def test_F14_rule04_triggers_all_three_met(self):
        from src.expert_rules import evaluate_rule_04
        r = evaluate_rule_04({"temperature_2m_max": 41.0, "tmax_departure": 5.0,
                               "tmax_departure_zscore": 2.0})
        self.assertTrue(r.triggered)
        self.assertEqual(r.severity, "WARNING")

    def test_F15_rule04_no_trigger_missing_departure(self):
        from src.expert_rules import evaluate_rule_04
        self.assertFalse(evaluate_rule_04({"temperature_2m_max": 42.0,
                                           "tmax_departure_zscore": 2.0}).triggered)

    def test_F16_rule04_no_trigger_low_zscore(self):
        from src.expert_rules import evaluate_rule_04
        self.assertFalse(evaluate_rule_04({"temperature_2m_max": 41.0,
                                           "tmax_departure": 5.0,
                                           "tmax_departure_zscore": 1.4}).triggered)

    def test_F17_rule04_boundary_exact_thresholds(self):
        from src.expert_rules import evaluate_rule_04
        self.assertTrue(evaluate_rule_04({"temperature_2m_max": 40.0,
                                          "tmax_departure": 4.5,
                                          "tmax_departure_zscore": 1.5}).triggered)

    # RULE_05 — Vulnerable Population Alert
    def test_F18_rule05_triggers_HIGH(self):
        from src.expert_rules import evaluate_rule_05
        self.assertTrue(evaluate_rule_05({}, risk_level="HIGH").triggered)

    def test_F19_rule05_triggers_EXTREME_critical(self):
        from src.expert_rules import evaluate_rule_05
        r = evaluate_rule_05({}, risk_level="EXTREME")
        self.assertTrue(r.triggered)
        self.assertEqual(r.severity, "CRITICAL")

    def test_F20_rule05_no_trigger_LOW(self):
        from src.expert_rules import evaluate_rule_05
        self.assertFalse(evaluate_rule_05({}, risk_level="LOW").triggered)

    def test_F21_rule05_no_trigger_MODERATE(self):
        from src.expert_rules import evaluate_rule_05
        self.assertFalse(evaluate_rule_05({}, risk_level="MODERATE").triggered)

    def test_F22_rule05_no_trigger_none_risk(self):
        from src.expert_rules import evaluate_rule_05
        self.assertFalse(evaluate_rule_05({}, risk_level=None).triggered)

    # RULE_06 — Outdoor Exposure Warning
    def test_F23_rule06_triggers_HIGH(self):
        from src.expert_rules import evaluate_rule_06
        self.assertTrue(evaluate_rule_06({}, risk_level="HIGH").triggered)

    def test_F24_rule06_triggers_EXTREME_critical(self):
        from src.expert_rules import evaluate_rule_06
        r = evaluate_rule_06({}, risk_level="EXTREME")
        self.assertTrue(r.triggered)
        self.assertEqual(r.severity, "CRITICAL")

    def test_F25_rule06_no_trigger_LOW(self):
        from src.expert_rules import evaluate_rule_06
        self.assertFalse(evaluate_rule_06({}, risk_level="LOW").triggered)

    def test_F26_rule06_no_trigger_none(self):
        from src.expert_rules import evaluate_rule_06
        self.assertFalse(evaluate_rule_06({}, risk_level=None).triggered)

    # RULE_07 — Hydration and Cooling Reminder
    def test_F27_rule07_triggers_via_probability(self):
        from src.expert_rules import evaluate_rule_07
        r = evaluate_rule_07({"temperature_2m_max": 20.0}, heatwave_probability=0.30)
        self.assertTrue(r.triggered)
        self.assertEqual(r.severity, "INFO")

    def test_F28_rule07_triggers_via_tmax(self):
        from src.expert_rules import evaluate_rule_07
        self.assertTrue(evaluate_rule_07({"temperature_2m_max": 35.0},
                                         heatwave_probability=0.05).triggered)

    def test_F29_rule07_no_trigger_low_prob_low_tmax(self):
        from src.expert_rules import evaluate_rule_07
        self.assertFalse(evaluate_rule_07({"temperature_2m_max": 20.0},
                                          heatwave_probability=0.10).triggered)

    def test_F30_rule07_triggers_tmax_only_no_prob(self):
        from src.expert_rules import evaluate_rule_07
        self.assertTrue(evaluate_rule_07({"temperature_2m_max": 36.0},
                                         heatwave_probability=None).triggered)

    def test_F31_rule07_boundary_exactly_35C(self):
        from src.expert_rules import evaluate_rule_07
        self.assertTrue(evaluate_rule_07({"temperature_2m_max": 35.0},
                                         heatwave_probability=0.0).triggered)

    # RuleResult structure
    def test_F32_rule_result_to_dict_when_triggered(self):
        from src.expert_rules import evaluate_rule_01
        r = evaluate_rule_01({"temperature_2m_max": 46.0, "city_key": "delhi"})
        d = r.to_dict()
        self.assertEqual(d["rule_id"], "RULE_01")
        self.assertTrue(d["triggered"])
        self.assertIn("message", d)
        self.assertIn("description", d)
        self.assertIn("severity", d)

    def test_F33_rule_result_to_dict_not_triggered_no_message(self):
        from src.expert_rules import evaluate_rule_01
        r = evaluate_rule_01({"temperature_2m_max": 20.0, "city_key": "delhi"})
        d = r.to_dict()
        self.assertFalse(d["triggered"])
        self.assertNotIn("message", d)

    def test_F34_triggered_message_nonempty(self):
        from src.expert_rules import evaluate_rule_01
        r = evaluate_rule_01({"temperature_2m_max": 46.0, "city_key": "delhi"})
        self.assertIsInstance(r.message, str)
        self.assertGreater(len(r.message), 0)

    def test_F35_not_triggered_message_empty(self):
        from src.expert_rules import evaluate_rule_01
        r = evaluate_rule_01({"temperature_2m_max": 20.0, "city_key": "delhi"})
        self.assertEqual(r.message, "")


# ===========================================================================
# GROUP G — ExpertRuleEngine
# ===========================================================================

class TestExpertRuleEngine(unittest.TestCase):
    """Tests for src.expert_rules.engine.ExpertRuleEngine"""

    def setUp(self):
        from src.expert_rules import ExpertRuleEngine
        self.engine = ExpertRuleEngine()

    def test_G01_evaluate_returns_seven_results(self):
        self.assertEqual(len(self.engine.evaluate(_make_normal_record())), 7)

    def test_G02_rule_ids_are_RULE_01_to_07(self):
        results = self.engine.evaluate(_make_normal_record())
        ids = [r.rule_id for r in results]
        self.assertEqual(ids, [f"RULE_0{i}" for i in range(1, 8)])

    def test_G03_hot_record_triggers_multiple_rules(self):
        rec = _make_hot_record()
        results = self.engine.evaluate(rec, risk_level="EXTREME",
                                       heatwave_probability=0.92)
        self.assertGreaterEqual(len(self.engine.get_triggered(results)), 4)

    def test_G04_cold_record_zero_triggers(self):
        rec = _make_normal_record(tmax=15.0, tmin=3.0, tmean=9.0)
        results = self.engine.evaluate(rec, risk_level="LOW",
                                       heatwave_probability=0.01)
        self.assertEqual(len(self.engine.get_triggered(results)), 0)

    def test_G05_get_triggered_all_triggered(self):
        rec = _make_hot_record()
        results = self.engine.evaluate(rec, risk_level="HIGH")
        for r in self.engine.get_triggered(results):
            self.assertTrue(r.triggered)

    def test_G06_triggered_count_matches_get_triggered(self):
        rec = _make_hot_record()
        results = self.engine.evaluate(rec, risk_level="EXTREME")
        self.assertEqual(self.engine.triggered_count(results),
                         len(self.engine.get_triggered(results)))

    def test_G07_highest_severity_critical_for_hot(self):
        rec = _make_hot_record()
        results = self.engine.evaluate(rec, risk_level="EXTREME")
        self.assertEqual(self.engine.highest_severity(results), "CRITICAL")

    def test_G08_highest_severity_none_for_cold(self):
        rec = _make_normal_record(tmax=15.0, tmin=3.0)
        results = self.engine.evaluate(rec, risk_level="LOW")
        self.assertIsNone(self.engine.highest_severity(results))

    def test_G09_to_dict_list_produces_7_dicts(self):
        results = self.engine.evaluate(_make_normal_record())
        dlist = self.engine.to_dict_list(results)
        self.assertEqual(len(dlist), 7)
        self.assertTrue(all(isinstance(d, dict) for d in dlist))

    def test_G10_summarise_structure(self):
        rec = _make_hot_record()
        summary = self.engine.summarise(self.engine.evaluate(rec, risk_level="HIGH"))
        for key in ("total_rules", "triggered_count", "highest_severity",
                    "triggered_ids", "disclaimer"):
            self.assertIn(key, summary)
        self.assertEqual(summary["total_rules"], 7)

    def test_G11_series_input_accepted(self):
        results = self.engine.evaluate(pd.Series(_make_normal_record()))
        self.assertEqual(len(results), 7)

    def test_G12_wrong_input_type_raises_typeerror(self):
        with self.assertRaises(TypeError):
            self.engine.evaluate([1, 2, 3])

    def test_G13_get_by_severity_filters_correctly(self):
        rec = _make_hot_record()
        results = self.engine.evaluate(rec, risk_level="EXTREME")
        critical = self.engine.get_by_severity(results, "CRITICAL")
        self.assertTrue(all(r.severity.upper() == "CRITICAL" for r in critical))

    def test_G14_disclaimer_present_and_mentions_imd(self):
        from src.expert_rules import EXPERT_RULES_DISCLAIMER
        self.assertIsInstance(EXPERT_RULES_DISCLAIMER, str)
        self.assertGreater(len(EXPERT_RULES_DISCLAIMER), 20)
        self.assertIn("IMD", EXPERT_RULES_DISCLAIMER)


# ===========================================================================
# GROUP H — ClimateGuardPipeline (full pipeline)
# ===========================================================================

class TestClimateGuardPipeline(unittest.TestCase):
    """Tests for src.integration.pipeline.ClimateGuardPipeline"""

    def setUp(self):
        self.pipeline = _get_pipeline()

    def test_H01_pipeline_instantiates(self):
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.predictor.threshold, 0.70)
        self.assertEqual(self.pipeline.predictor.n_features, 110)

    def test_H02_valid_record_returns_valid(self):
        res = self.pipeline.analyze(_make_normal_record())
        self.assertTrue(res.valid, f"Errors: {res.validation['errors']}")

    def test_H03_result_contains_all_required_keys(self):
        res = self.pipeline.analyze(_make_normal_record())
        d   = res.to_dict()
        for key in ("input", "validation", "prediction", "risk",
                    "recommendations", "expert_rules", "warnings", "metadata"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_H04_prediction_has_probability_and_label(self):
        res = self.pipeline.analyze(_make_normal_record())
        self.assertIsNotNone(res.prediction)
        self.assertIn("probability", res.prediction)
        self.assertIn("prediction", res.prediction)
        self.assertGreaterEqual(res.prediction["probability"], 0.0)
        self.assertLessEqual(res.prediction["probability"], 1.0)
        self.assertIn(res.prediction["prediction"], (0, 1))

    def test_H05_risk_has_valid_level(self):
        res = self.pipeline.analyze(_make_normal_record())
        self.assertIsNotNone(res.risk)
        self.assertIn(res.risk["level"], ("LOW", "MODERATE", "HIGH", "EXTREME"))

    def test_H06_recommendations_is_list(self):
        res = self.pipeline.analyze(_make_normal_record())
        self.assertIsInstance(res.recommendations, list)

    def test_H07_expert_rules_is_list_of_7_dicts(self):
        res = self.pipeline.analyze(_make_normal_record())
        self.assertIsInstance(res.expert_rules, list)
        self.assertEqual(len(res.expert_rules), 7)

    def test_H08_expert_rules_dicts_have_required_keys(self):
        res = self.pipeline.analyze(_make_normal_record())
        for r in res.expert_rules:
            self.assertIn("rule_id", r)
            self.assertIn("triggered", r)
            self.assertIn("severity", r)
            self.assertIn("description", r)

    def test_H09_cold_record_low_risk(self):
        rec = _make_normal_record(tmax=18.0, tmin=5.0, tmean=11.0)
        res = self.pipeline.analyze(rec)
        self.assertTrue(res.valid)
        self.assertEqual(res.risk["level"], "LOW")

    def test_H10_metadata_contains_threshold_and_features(self):
        res = self.pipeline.analyze(_make_normal_record())
        self.assertEqual(res.metadata["part1_threshold"], 0.70)
        self.assertEqual(res.metadata["part1_n_features"], 110)

    def test_H11_input_metadata_contains_city_date(self):
        rec = _make_normal_record(city_key="lucknow", date="2024-05-01")
        res = self.pipeline.analyze(rec)
        self.assertEqual(res.input_metadata.get("city_key"), "lucknow")
        self.assertEqual(res.input_metadata.get("date"), "2024-05-01")

    def test_H12_mumbai_record_runs(self):
        res = self.pipeline.analyze(_make_mumbai_record())
        self.assertTrue(res.valid, f"Errors: {res.validation['errors']}")

    def test_H13_series_input_accepted(self):
        res = self.pipeline.analyze(pd.Series(_make_normal_record()))
        self.assertTrue(res.valid)

    def test_H14_analyze_batch_returns_list(self):
        recs = [_make_normal_record(city_key=c) for c in ["delhi", "lucknow"]]
        results = self.pipeline.analyze_batch(pd.DataFrame(recs))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.valid for r in results))

    def test_H15_analyze_batch_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            self.pipeline.analyze_batch({"not": "a dataframe"})

    def test_H16_info_returns_expected_structure(self):
        info = self.pipeline.info()
        self.assertIn("pipeline", info)
        self.assertEqual(info["threshold"], 0.70)
        self.assertEqual(info["n_features"], 110)

    def test_H17_explanation_present_or_none(self):
        res = self.pipeline.analyze(_make_normal_record(), include_explanation=True)
        # Either a dict or None — must not raise
        self.assertTrue(res.explanation is None or isinstance(res.explanation, dict))

    def test_H18_explanation_false_suppresses(self):
        res = self.pipeline.analyze(_make_normal_record(), include_explanation=False)
        self.assertIsNone(res.explanation)


# ===========================================================================
# GROUP I — Error Handling
# ===========================================================================

class TestErrorHandling(unittest.TestCase):

    def setUp(self):
        self.pipeline = _get_pipeline()

    def test_I01_invalid_city_returns_invalid(self):
        rec = _make_normal_record()
        rec["city_key"] = "chennai"
        res = self.pipeline.analyze(rec)
        self.assertFalse(res.valid)
        self.assertIsNone(res.prediction)
        self.assertIsNone(res.risk)

    def test_I02_missing_city_key_returns_invalid(self):
        rec = _make_normal_record()
        del rec["city_key"]
        self.assertFalse(self.pipeline.analyze(rec).valid)

    def test_I03_missing_date_returns_invalid(self):
        rec = _make_normal_record()
        del rec["date"]
        self.assertFalse(self.pipeline.analyze(rec).valid)

    def test_I04_malformed_date_returns_invalid(self):
        rec = _make_normal_record()
        rec["date"] = "not-a-date"
        self.assertFalse(self.pipeline.analyze(rec).valid)

    def test_I05_nan_feature_returns_invalid(self):
        rec = _make_normal_record()
        rec["temperature_2m_max"] = float("nan")
        self.assertFalse(self.pipeline.analyze(rec).valid)

    def test_I06_missing_features_returns_invalid(self):
        rec = _make_normal_record()
        for k in FEATURE_NAMES[:10]:
            del rec[k]
        self.assertFalse(self.pipeline.analyze(rec).valid)

    def test_I07_invalid_result_has_errors(self):
        rec = _make_normal_record()
        rec["city_key"] = "invalid"
        res = self.pipeline.analyze(rec)
        self.assertGreater(len(res.validation["errors"]), 0)

    def test_I08_invalid_result_expert_rules_empty(self):
        rec = _make_normal_record()
        rec["city_key"] = "invalid"
        self.assertEqual(self.pipeline.analyze(rec).expert_rules, [])

    def test_I09_invalid_result_recommendations_none(self):
        rec = _make_normal_record()
        rec["city_key"] = "invalid"
        self.assertIsNone(self.pipeline.analyze(rec).recommendations)

    def test_I10_non_numeric_feature_returns_invalid(self):
        rec = _make_normal_record()
        rec["temperature_2m_max"] = "extreme"
        self.assertFalse(self.pipeline.analyze(rec).valid)

    def test_I11_valid_result_has_empty_errors(self):
        res = self.pipeline.analyze(_make_normal_record())
        self.assertTrue(res.valid)
        self.assertEqual(res.validation["errors"], [])


# ===========================================================================
# GROUP J — JSON Serialisation
# ===========================================================================

class TestJSONSerialisation(unittest.TestCase):

    def setUp(self):
        self.pipeline = _get_pipeline()

    def test_J01_valid_result_to_json_no_raise(self):
        s = self.pipeline.analyze(_make_normal_record()).to_json()
        self.assertIsInstance(s, str)

    def test_J02_invalid_result_to_json_no_raise(self):
        rec = _make_normal_record()
        rec["city_key"] = "invalid"
        s = self.pipeline.analyze(rec).to_json()
        self.assertIsInstance(s, str)

    def test_J03_json_round_trip_preserves_fields(self):
        res = self.pipeline.analyze(_make_normal_record())
        d   = json.loads(res.to_json())
        for key in ("prediction", "risk", "expert_rules", "validation", "metadata"):
            self.assertIn(key, d)

    def test_J04_probability_is_float_in_json(self):
        d = json.loads(self.pipeline.analyze(_make_normal_record()).to_json())
        self.assertIsInstance(d["prediction"]["probability"], float)

    def test_J05_expert_rules_in_json_is_list_of_7(self):
        d = json.loads(self.pipeline.analyze(_make_normal_record()).to_json())
        self.assertIsInstance(d["expert_rules"], list)
        self.assertEqual(len(d["expert_rules"]), 7)

    def test_J06_hot_record_json_has_triggered_rules(self):
        res = self.pipeline.analyze(_make_hot_record())
        d   = json.loads(res.to_json())
        self.assertGreater(
            len([r for r in d["expert_rules"] if r["triggered"]]), 0)

    def test_J07_disclaimer_is_string_in_json(self):
        d = json.loads(self.pipeline.analyze(_make_normal_record()).to_json())
        self.assertIsInstance(d["metadata"]["expert_rules_disclaimer"], str)


# ===========================================================================
# GROUP K — Real-Data Smoke Tests
# ===========================================================================

class TestRealDataSmoke(unittest.TestCase):
    """Smoke tests using actual rows from data/splits/temporal/X_test.csv."""

    def setUp(self):
        self.pipeline = _get_pipeline()
        self.X, self.meta = _get_test_data()

    def _row(self, idx: int) -> Dict:
        row = self.X.iloc[idx].to_dict()
        row["city_key"] = self.meta.iloc[idx]["city_key"]
        row["date"]     = self.meta.iloc[idx]["date"]
        return row

    def test_K01_first_row_runs_successfully(self):
        res = self.pipeline.analyze(self._row(0))
        self.assertTrue(res.valid, f"Errors: {res.validation['errors']}")
        self.assertIsNotNone(res.prediction)
        self.assertGreaterEqual(res.prediction["probability"], 0.0)
        self.assertLessEqual(res.prediction["probability"], 1.0)

    def test_K02_hot_delhi_row_triggers_expert_rules(self):
        # Row 1475: delhi 2024-05-17, Tmax=44.5°C
        res = self.pipeline.analyze(self._row(1475))
        self.assertTrue(res.valid, f"Row 1475 failed: {res.validation['errors']}")
        triggered = [r for r in res.expert_rules if r["triggered"]]
        self.assertGreater(len(triggered), 0,
                           "Expected ≥1 expert rule to trigger on extreme heat day")

    def test_K03_batch_5_rows_all_succeed(self):
        rows = [self._row(i) for i in range(5)]
        results = self.pipeline.analyze_batch(pd.DataFrame(rows))
        self.assertEqual(len(results), 5)
        for i, r in enumerate(results):
            self.assertTrue(r.valid, f"Row {i} failed: {r.validation['errors']}")

    def test_K04_risk_level_always_valid(self):
        valid_levels = {"LOW", "MODERATE", "HIGH", "EXTREME"}
        for i in range(10):
            res = self.pipeline.analyze(self._row(i))
            if res.valid:
                self.assertIn(res.risk["level"], valid_levels)

    def test_K05_probability_always_bounded(self):
        for i in range(20):
            res = self.pipeline.analyze(self._row(i))
            if res.valid:
                p = res.prediction["probability"]
                self.assertGreaterEqual(p, 0.0, f"Row {i}: prob {p} < 0")
                self.assertLessEqual(p, 1.0,    f"Row {i}: prob {p} > 1")

    def test_K06_threshold_unchanged_after_run(self):
        self.pipeline.analyze(self._row(0))
        self.assertEqual(self.pipeline.predictor.threshold, 0.70)

    def test_K07_feature_count_unchanged_after_run(self):
        self.pipeline.analyze(self._row(0))
        self.assertEqual(self.pipeline.predictor.n_features, 110)


# ===========================================================================
# Test runner (matching project's test_part2.py pattern)
# ===========================================================================

def run_tests() -> bool:
    """Run all Part 3 tests and return True if all pass."""
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    test_classes = [
        TestInputValidator,           # Group A
        TestFeatureContractValidator, # Group B
        TestFeatureTransformer,       # Group C
        TestETLPipelineSingle,        # Group D
        TestETLPipelineBatch,         # Group E
        TestExpertRulesIndividual,    # Group F
        TestExpertRuleEngine,         # Group G
        TestClimateGuardPipeline,     # Group H
        TestErrorHandling,            # Group I
        TestJSONSerialisation,        # Group J
        TestRealDataSmoke,            # Group K
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

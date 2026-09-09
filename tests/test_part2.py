"""
tests/test_part2.py
ClimateGuard Part 2 — Risk Assessment, Adaptation Recommendations, Explainability

Run from project root:
    python tests/test_part2.py
    python -m pytest tests/test_part2.py -v

Test groups
-----------
Group A: Risk probability validation and range checking
Group B: Risk threshold boundary values
Group C: Risk level generation
Group D: Adaptation recommendation generation
Group E: Adaptation recommendation categories
Group F: Model access via predictor
Group G: Feature-name consistency
Group H: Explainability execution
Group I: Invalid input handling
Group J: Part 1 → Part 2 integration
Group K: Real-data smoke test (uses actual test split row)

Requirements
------------
All 110 features must be available.
Real test split: data/splits/temporal/X_test.csv, meta_test.csv, y_test.csv
Model:  models/final/climateguard_final_model.joblib
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import ClimateGuardPredictor
from src.risk import RiskAssessor, RiskAssessmentResult, RiskLevel
from src.risk.risk_assessment import probability_to_risk_level, RISK_THRESHOLDS
from src.adaptation import AdaptationEngine, Recommendation
from src.explainability import ClimateGuardExplainer, ExplainabilityResult
from src.risk_engine import ClimateGuardRiskEngine, RiskEngineResult

# ---------------------------------------------------------------------------
# Shared fixtures — loaded once for the module
# ---------------------------------------------------------------------------

_predictor: ClimateGuardPredictor = None
_engine: ClimateGuardRiskEngine = None


def _get_predictor() -> ClimateGuardPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ClimateGuardPredictor()
    return _predictor


def _get_engine() -> ClimateGuardRiskEngine:
    global _engine
    if _engine is None:
        _engine = ClimateGuardRiskEngine(predictor=_get_predictor())
    return _engine


def _load_real_test_row(n: int = 1) -> pd.DataFrame:
    """Load n real rows from the held-out test split."""
    X_path    = PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv"
    meta_path = PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv"
    y_path    = PROJECT_ROOT / "data" / "splits" / "temporal" / "y_test.csv"

    X    = pd.read_csv(X_path,    nrows=n).reset_index(drop=True)
    meta = pd.read_csv(meta_path, nrows=n).reset_index(drop=True)
    # read y — force float so numeric operations work; handle nrows=1 scalar squeeze edge case
    y_df = pd.read_csv(y_path, nrows=n).reset_index(drop=True)
    if y_df.shape[1] == 1:
        y_vals = y_df.iloc[:, 0].astype(float).values
    else:
        y_vals = y_df.squeeze().astype(float).values

    # Ensure X feature columns are numeric (float64)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Attach metadata (pass-through columns, not model inputs)
    extras = pd.DataFrame({
        "city_key":                meta["city_key"].values,
        "date":                    meta["date"].values,
        "actual_heatwave_next_day": y_vals,
    })
    return pd.concat([X, extras], axis=1)


# ===========================================================================
# Group A — Risk probability validation and range checking
# ===========================================================================

class TestRiskProbabilityValidation(unittest.TestCase):
    """Group A: Validate probability inputs."""

    def setUp(self):
        self.assessor = RiskAssessor()

    def test_valid_probability_zero(self):
        """Probability 0.0 is accepted and returns LOW."""
        result = self.assessor.assess(0.0, 0)
        self.assertEqual(result.risk_level, "LOW")

    def test_valid_probability_one(self):
        """Probability 1.0 is accepted and returns EXTREME."""
        result = self.assessor.assess(1.0, 1)
        self.assertEqual(result.risk_level, "EXTREME")

    def test_valid_probability_midrange(self):
        """Mid-range probabilities are accepted."""
        result = self.assessor.assess(0.5, 0)
        self.assertIn(result.risk_level, ["LOW", "MODERATE", "HIGH", "EXTREME"])

    def test_probability_above_one_raises(self):
        """Probability > 1.0 raises ValueError."""
        with self.assertRaises(ValueError):
            self.assessor.assess(1.01, 1)

    def test_probability_below_zero_raises(self):
        """Probability < 0.0 raises ValueError."""
        with self.assertRaises(ValueError):
            self.assessor.assess(-0.01, 0)

    def test_probability_nan_raises(self):
        """NaN probability raises ValueError or TypeError."""
        with self.assertRaises((ValueError, TypeError)):
            self.assessor.assess(float("nan"), 0)

    def test_probability_string_raises(self):
        """String probability raises TypeError."""
        with self.assertRaises((TypeError, ValueError)):
            self.assessor.assess("0.5", 0)

    def test_probability_none_raises(self):
        """None probability raises TypeError."""
        with self.assertRaises((TypeError, AttributeError)):
            self.assessor.assess(None, 0)

    def test_result_preserves_probability(self):
        """The result must preserve the original probability exactly."""
        prob = 0.7234
        result = self.assessor.assess(prob, 1)
        self.assertAlmostEqual(result.heatwave_probability, prob, places=6)

    def test_result_preserves_prediction(self):
        """The result must preserve the original prediction label."""
        result = self.assessor.assess(0.8, 1)
        self.assertEqual(result.prediction, 1)
        result2 = self.assessor.assess(0.2, 0)
        self.assertEqual(result2.prediction, 0)


# ===========================================================================
# Group B — Risk threshold boundary values
# ===========================================================================

class TestRiskThresholdBoundaries(unittest.TestCase):
    """Group B: Exact boundary probabilities map to the correct risk levels."""

    def _assess(self, prob: float) -> str:
        assessor = RiskAssessor()
        return assessor.assess(prob, int(prob >= 0.70)).risk_level

    def test_0_0_is_low(self):
        self.assertEqual(self._assess(0.0), "LOW")

    def test_0_29_is_low(self):
        self.assertEqual(self._assess(0.29), "LOW")

    def test_0_30_is_moderate(self):
        """0.30 is the lower bound for MODERATE (inclusive)."""
        self.assertEqual(self._assess(0.30), "MODERATE")

    def test_0_59_is_moderate(self):
        self.assertEqual(self._assess(0.59), "MODERATE")

    def test_0_60_is_high(self):
        """0.60 is the lower bound for HIGH (inclusive)."""
        self.assertEqual(self._assess(0.60), "HIGH")

    def test_0_79_is_high(self):
        self.assertEqual(self._assess(0.79), "HIGH")

    def test_0_80_is_extreme(self):
        """0.80 is the lower bound for EXTREME (inclusive)."""
        self.assertEqual(self._assess(0.80), "EXTREME")

    def test_1_0_is_extreme(self):
        self.assertEqual(self._assess(1.0), "EXTREME")

    def test_determinism(self):
        """Same probability always produces same risk level."""
        assessor = RiskAssessor()
        for prob in [0.0, 0.29, 0.30, 0.59, 0.60, 0.79, 0.80, 1.0]:
            r1 = assessor.assess(prob, 0).risk_level
            r2 = assessor.assess(prob, 0).risk_level
            self.assertEqual(r1, r2, f"Non-deterministic at prob={prob}")

    def test_probability_to_risk_level_function(self):
        """probability_to_risk_level helper function matches RiskAssessor."""
        test_cases = [
            (0.0,  "LOW"),
            (0.29, "LOW"),
            (0.30, "MODERATE"),
            (0.59, "MODERATE"),
            (0.60, "HIGH"),
            (0.79, "HIGH"),
            (0.80, "EXTREME"),
            (1.0,  "EXTREME"),
        ]
        for prob, expected in test_cases:
            with self.subTest(prob=prob):
                self.assertEqual(str(probability_to_risk_level(prob)), expected)


# ===========================================================================
# Group C — Risk level generation
# ===========================================================================

class TestRiskLevelGeneration(unittest.TestCase):
    """Group C: RiskAssessmentResult structure and RiskLevel enum."""

    def setUp(self):
        self.assessor = RiskAssessor()

    def test_result_is_risk_assessment_result(self):
        """assess() returns a RiskAssessmentResult."""
        result = self.assessor.assess(0.5, 0)
        self.assertIsInstance(result, RiskAssessmentResult)

    def test_risk_level_is_string(self):
        """risk_level is a string."""
        result = self.assessor.assess(0.5, 0)
        self.assertIsInstance(result.risk_level, str)

    def test_risk_level_enum_values(self):
        """All four risk levels exist in RiskLevel enum."""
        self.assertIn("LOW",      [str(r) for r in RiskLevel])
        self.assertIn("MODERATE", [str(r) for r in RiskLevel])
        self.assertIn("HIGH",     [str(r) for r in RiskLevel])
        self.assertIn("EXTREME",  [str(r) for r in RiskLevel])

    def test_to_dict_contains_required_keys(self):
        """to_dict() returns required keys."""
        result = self.assessor.assess(0.75, 1, city="delhi", date="2024-05-20")
        d = result.to_dict()
        self.assertIn("heatwave_probability", d)
        self.assertIn("prediction", d)
        self.assertIn("risk_level", d)
        self.assertIn("city", d)
        self.assertIn("date", d)

    def test_to_dict_no_city_date(self):
        """to_dict() omits city and date when not provided."""
        result = self.assessor.assess(0.75, 1)
        d = result.to_dict()
        self.assertNotIn("city", d)
        self.assertNotIn("date", d)

    def test_all_four_levels_reachable(self):
        """Each of the four risk levels can be produced."""
        produced = {
            self.assessor.assess(0.10, 0).risk_level,
            self.assessor.assess(0.45, 0).risk_level,
            self.assessor.assess(0.70, 1).risk_level,
            self.assessor.assess(0.90, 1).risk_level,
        }
        self.assertEqual(produced, {"LOW", "MODERATE", "HIGH", "EXTREME"})


# ===========================================================================
# Group D — Adaptation recommendation generation
# ===========================================================================

class TestAdaptationRecommendationGeneration(unittest.TestCase):
    """Group D: AdaptationEngine generates recommendations."""

    def setUp(self):
        self.engine = AdaptationEngine()

    def test_low_returns_recommendations(self):
        recs = self.engine.recommend("LOW")
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)

    def test_moderate_returns_recommendations(self):
        recs = self.engine.recommend("MODERATE")
        self.assertGreater(len(recs), 0)

    def test_high_returns_recommendations(self):
        recs = self.engine.recommend("HIGH")
        self.assertGreater(len(recs), 0)

    def test_extreme_returns_recommendations(self):
        recs = self.engine.recommend("EXTREME")
        self.assertGreater(len(recs), 0)

    def test_higher_risk_more_or_equal_recommendations(self):
        """Higher risk levels should have >= recommendations than lower."""
        low    = len(self.engine.recommend("LOW"))
        high   = len(self.engine.recommend("HIGH"))
        extreme = len(self.engine.recommend("EXTREME"))
        self.assertGreaterEqual(high, low)
        self.assertGreaterEqual(extreme, high)

    def test_recommend_as_dict_structure(self):
        """recommend_as_dict() returns correct keys."""
        out = self.engine.recommend_as_dict("EXTREME")
        self.assertIn("risk_level", out)
        self.assertIn("disclaimer", out)
        self.assertIn("recommendations", out)
        self.assertEqual(out["risk_level"], "EXTREME")
        self.assertIsInstance(out["disclaimer"], str)
        self.assertGreater(len(out["disclaimer"]), 0)
        self.assertIsInstance(out["recommendations"], list)

    def test_disclaimer_present(self):
        """Disclaimer text is non-empty and present in all outputs."""
        for level in ["LOW", "MODERATE", "HIGH", "EXTREME"]:
            out = self.engine.recommend_as_dict(level)
            self.assertGreater(len(out["disclaimer"]), 20,
                               msg=f"Disclaimer too short for level {level}")

    def test_case_insensitive(self):
        """Risk level is case-insensitive."""
        recs_upper = self.engine.recommend("HIGH")
        recs_lower = self.engine.recommend("high")
        self.assertEqual(len(recs_upper), len(recs_lower))

    def test_invalid_level_raises(self):
        """Unknown risk level raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.recommend("CRITICAL")

    def test_determinism_recommendations(self):
        """Same risk level always produces same recommendations."""
        for level in ["LOW", "MODERATE", "HIGH", "EXTREME"]:
            r1 = self.engine.recommend(level)
            r2 = self.engine.recommend(level)
            self.assertEqual([r.category for r in r1], [r.category for r in r2])
            self.assertEqual([r.message  for r in r1], [r.message  for r in r2])


# ===========================================================================
# Group E — Adaptation recommendation categories
# ===========================================================================

class TestAdaptationCategories(unittest.TestCase):
    """Group E: Recommendations have valid categories and messages."""

    def setUp(self):
        self.engine = AdaptationEngine()

    def _all_recommendations(self) -> list:
        all_recs = []
        for level in ["LOW", "MODERATE", "HIGH", "EXTREME"]:
            all_recs.extend(self.engine.recommend(level))
        return all_recs

    def test_recommendations_are_recommendation_instances(self):
        """Each item is a Recommendation dataclass."""
        for level in ["LOW", "MODERATE", "HIGH", "EXTREME"]:
            for rec in self.engine.recommend(level):
                self.assertIsInstance(rec, Recommendation)

    def test_all_have_non_empty_category(self):
        """Every recommendation has a non-empty category string."""
        for rec in self._all_recommendations():
            self.assertIsInstance(rec.category, str)
            self.assertGreater(len(rec.category.strip()), 0)

    def test_all_have_non_empty_message(self):
        """Every recommendation has a non-empty message."""
        for rec in self._all_recommendations():
            self.assertIsInstance(rec.message, str)
            self.assertGreater(len(rec.message.strip()), 0)

    def test_to_dict_format(self):
        """to_dict() on Recommendation returns category and message keys."""
        rec = self.engine.recommend("HIGH")[0]
        d = rec.to_dict()
        self.assertIn("category", d)
        self.assertIn("message", d)

    def test_extreme_includes_emergency_preparedness(self):
        """EXTREME level must include an Emergency Preparedness recommendation."""
        categories = self.engine.get_categories("EXTREME")
        self.assertIn("Emergency Preparedness", categories)

    def test_all_levels_include_hydration(self):
        """All risk levels must include a Hydration recommendation."""
        for level in ["LOW", "MODERATE", "HIGH", "EXTREME"]:
            categories = self.engine.get_categories(level)
            self.assertIn("Hydration", categories, msg=f"Missing Hydration at {level}")

    def test_extreme_includes_outdoor_exposure(self):
        """EXTREME level must include an Outdoor Exposure recommendation."""
        categories = self.engine.get_categories("EXTREME")
        self.assertIn("Outdoor Exposure", categories)


# ===========================================================================
# Group F — Model access
# ===========================================================================

class TestModelAccess(unittest.TestCase):
    """Group F: ClimateGuardPredictor model access."""

    def setUp(self):
        self.predictor = _get_predictor()

    def test_model_attribute_exists(self):
        """predictor.model is set."""
        self.assertIsNotNone(self.predictor.model)

    def test_model_has_predict_proba(self):
        """predictor.model has predict_proba method."""
        self.assertTrue(hasattr(self.predictor.model, "predict_proba"))

    def test_model_n_features_is_110(self):
        """Model expects exactly 110 features."""
        self.assertEqual(self.predictor.model.n_features_in_, 110)

    def test_feature_names_length(self):
        """predictor.feature_names has 110 entries."""
        self.assertEqual(len(self.predictor.feature_names), 110)

    def test_feature_names_are_strings(self):
        """All feature names are non-empty strings."""
        for name in self.predictor.feature_names:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_threshold_is_0_70(self):
        """Decision threshold is 0.70."""
        self.assertAlmostEqual(self.predictor.threshold, 0.70, places=6)


# ===========================================================================
# Group G — Feature-name consistency
# ===========================================================================

class TestFeatureNameConsistency(unittest.TestCase):
    """Group G: 110 feature names are consistent across components."""

    def setUp(self):
        self.predictor = _get_predictor()

    def test_feature_names_match_feature_list_json(self):
        """feature_names matches the authoritative feature_list.json order."""
        import json as _json
        fl_path = PROJECT_ROOT / "models" / "final" / "feature_list.json"
        with open(fl_path, encoding="utf-8") as f:
            raw = _json.load(f)
        expected = [entry["name"] for entry in raw]
        self.assertEqual(self.predictor.feature_names, expected)

    def test_target_not_in_feature_names(self):
        """heatwave_next_day must NOT be in feature names."""
        self.assertNotIn("heatwave_next_day", self.predictor.feature_names)

    def test_get_feature_matrix_shape(self):
        """get_feature_matrix() returns (n, 110) DataFrame."""
        rows = _load_real_test_row(n=3)
        fm = self.predictor.get_feature_matrix(rows)
        self.assertEqual(fm.shape[0], 3)
        self.assertEqual(fm.shape[1], 110)

    def test_get_feature_matrix_column_order(self):
        """get_feature_matrix() columns match predictor.feature_names order."""
        rows = _load_real_test_row(n=1)
        fm = self.predictor.get_feature_matrix(rows)
        self.assertEqual(list(fm.columns), self.predictor.feature_names)

    def test_engine_explainer_uses_predictor_feature_names(self):
        """Explainer's feature names match predictor's feature names."""
        engine = _get_engine()
        self.assertEqual(
            engine.explainer._feature_names,
            self.predictor.feature_names,
        )


# ===========================================================================
# Group H — Explainability execution
# ===========================================================================

class TestExplainabilityExecution(unittest.TestCase):
    """Group H: ClimateGuardExplainer runs and returns valid output."""

    def setUp(self):
        self.predictor = _get_predictor()
        self.explainer = ClimateGuardExplainer(self.predictor)
        self.row = _load_real_test_row(n=1)

    def test_explain_returns_result(self):
        """explain() returns ExplainabilityResult."""
        result = self.explainer.explain(self.row)
        self.assertIsInstance(result, ExplainabilityResult)

    def test_explain_method_is_labelled(self):
        """method is either 'shap' or 'global_rf_importance'."""
        result = self.explainer.explain(self.row)
        self.assertIn(result.method, ["shap", "global_rf_importance"])

    def test_top_features_is_list(self):
        """top_features is a non-empty list."""
        result = self.explainer.explain(self.row, top_n=5)
        self.assertIsInstance(result.top_features, list)
        self.assertGreater(len(result.top_features), 0)

    def test_top_n_respected(self):
        """Number of top features does not exceed top_n."""
        for top_n in [1, 5, 10]:
            result = self.explainer.explain(self.row, top_n=top_n)
            self.assertLessEqual(len(result.top_features), top_n)

    def test_top_feature_has_required_keys(self):
        """Each feature entry has feature, value, contribution, direction."""
        result = self.explainer.explain(self.row)
        for feat in result.top_features:
            self.assertIn("feature", feat)
            self.assertIn("value", feat)
            self.assertIn("contribution", feat)
            self.assertIn("direction", feat)

    def test_feature_names_in_result_are_from_contract(self):
        """Feature names in explanation are a subset of the 110-name contract."""
        result = self.explainer.explain(self.row)
        valid_names = set(self.predictor.feature_names)
        for feat in result.top_features:
            self.assertIn(feat["feature"], valid_names)

    def test_to_dict_has_notes(self):
        """to_dict() includes causality and qualifying_day notes."""
        result = self.explainer.explain(self.row)
        d = result.to_dict()
        self.assertIn("notes", d)
        self.assertIn("causality", d["notes"])
        self.assertIn("qualifying_day", d["notes"])

    def test_shap_available_returns_bool(self):
        """shap_available() returns a bool."""
        val = ClimateGuardExplainer.shap_available()
        self.assertIsInstance(val, bool)

    def test_global_importance_explained_as_global(self):
        """If using global fallback, method is 'global_rf_importance'."""
        # Force global importance path by creating explainer with no SHAP
        import src.explainability.explainer as exp_module
        orig_shap = exp_module._SHAP_AVAILABLE
        exp_module._SHAP_AVAILABLE = False

        explainer_no_shap = ClimateGuardExplainer.__new__(ClimateGuardExplainer)
        explainer_no_shap._predictor = self.predictor
        explainer_no_shap._model = self.predictor.model
        explainer_no_shap._feature_names = self.predictor.feature_names
        explainer_no_shap._shap_explainer = None

        result = explainer_no_shap._explain_global_importance(
            self.predictor.get_feature_matrix(self.row),
            0, 0.1, 5
        )
        self.assertEqual(result.method, "global_rf_importance")
        for feat in result.top_features:
            self.assertEqual(feat["direction"], "global_importance")

        exp_module._SHAP_AVAILABLE = orig_shap

    def test_explain_global_returns_list(self):
        """explain_global() returns a list of feature dicts."""
        top = self.explainer.explain_global(top_n=5)
        self.assertIsInstance(top, list)
        self.assertGreater(len(top), 0)
        for item in top:
            self.assertIn("rank", item)
            self.assertIn("feature", item)
            self.assertIn("importance", item)

    def test_probability_and_prediction_in_result(self):
        """ExplainabilityResult stores prediction and probability."""
        row = self.row
        pred_result = self.predictor.predict(row)
        exp_result = self.explainer.explain(
            row,
            prediction=pred_result.prediction_label,
            probability=pred_result.prediction_probability,
        )
        self.assertEqual(exp_result.prediction, pred_result.prediction_label)
        self.assertAlmostEqual(exp_result.probability, pred_result.prediction_probability, places=6)


# ===========================================================================
# Group I — Invalid input handling
# ===========================================================================

class TestInvalidInputHandling(unittest.TestCase):
    """Group I: Error handling for bad inputs across all components."""

    def test_risk_assessor_invalid_probability_high(self):
        """RiskAssessor rejects probability > 1."""
        assessor = RiskAssessor()
        with self.assertRaises(ValueError):
            assessor.assess(1.5, 1)

    def test_risk_assessor_invalid_probability_negative(self):
        """RiskAssessor rejects probability < 0."""
        assessor = RiskAssessor()
        with self.assertRaises(ValueError):
            assessor.assess(-0.1, 0)

    def test_risk_assessor_string_input(self):
        """RiskAssessor rejects string probability."""
        assessor = RiskAssessor()
        with self.assertRaises((TypeError, ValueError)):
            assessor.assess("high", 1)

    def test_adaptation_engine_invalid_level(self):
        """AdaptationEngine rejects unknown risk levels."""
        engine = AdaptationEngine()
        with self.assertRaises(ValueError):
            engine.recommend("VERY_HIGH")

    def test_adaptation_engine_empty_string(self):
        """AdaptationEngine rejects empty string."""
        engine = AdaptationEngine()
        with self.assertRaises(ValueError):
            engine.recommend("")

    def test_explainer_invalid_top_n_zero(self):
        """ClimateGuardExplainer rejects top_n=0."""
        explainer = ClimateGuardExplainer(_get_predictor())
        row = _load_real_test_row(n=1)
        with self.assertRaises(ValueError):
            explainer.explain(row, top_n=0)

    def test_explainer_invalid_top_n_over_110(self):
        """ClimateGuardExplainer rejects top_n > 110."""
        explainer = ClimateGuardExplainer(_get_predictor())
        row = _load_real_test_row(n=1)
        with self.assertRaises(ValueError):
            explainer.explain(row, top_n=111)

    def test_engine_missing_features_raises(self):
        """ClimateGuardRiskEngine raises on missing feature columns."""
        engine = _get_engine()
        bad_df = pd.DataFrame({"bad_feature": [1.0]})
        with self.assertRaises((ValueError, KeyError)):
            engine.analyze(bad_df)

    def test_predictor_nan_raises(self):
        """ClimateGuardPredictor rejects NaN values."""
        predictor = _get_predictor()
        row = _load_real_test_row(n=1)
        row.iloc[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            predictor.predict(row)


# ===========================================================================
# Group J — Part 1 → Part 2 integration
# ===========================================================================

class TestPart1ToPart2Integration(unittest.TestCase):
    """Group J: End-to-end pipeline integration tests."""

    def setUp(self):
        self.engine = _get_engine()
        self.predictor = _get_predictor()
        self.row = _load_real_test_row(n=1)

    def test_analyze_returns_risk_engine_result(self):
        """analyze() returns RiskEngineResult."""
        result = self.engine.analyze(self.row)
        self.assertIsInstance(result, RiskEngineResult)

    def test_probability_in_range(self):
        """Probability from engine is in [0, 1]."""
        result = self.engine.analyze(self.row)
        self.assertGreaterEqual(result.heatwave_probability, 0.0)
        self.assertLessEqual(result.heatwave_probability, 1.0)

    def test_prediction_is_binary(self):
        """Prediction label is 0 or 1."""
        result = self.engine.analyze(self.row)
        self.assertIn(result.prediction, [0, 1])

    def test_risk_level_is_valid(self):
        """Risk level is one of the four valid strings."""
        result = self.engine.analyze(self.row)
        self.assertIn(result.risk_level, ["LOW", "MODERATE", "HIGH", "EXTREME"])

    def test_recommendations_present(self):
        """Recommendations are non-empty."""
        result = self.engine.analyze(self.row)
        self.assertIn("recommendations", result.recommendations)
        self.assertGreater(len(result.recommendations["recommendations"]), 0)

    def test_explanation_present(self):
        """Explanation dict is included by default."""
        result = self.engine.analyze(self.row, include_explanation=True)
        self.assertIsNotNone(result.explanation)
        self.assertIn("top_features", result.explanation)

    def test_explanation_absent_when_disabled(self):
        """Explanation is None when include_explanation=False."""
        result = self.engine.analyze(self.row, include_explanation=False)
        self.assertIsNone(result.explanation)

    def test_to_dict_structure(self):
        """to_dict() has all required keys."""
        result = self.engine.analyze(self.row)
        d = result.to_dict()
        self.assertIn("heatwave_probability", d)
        self.assertIn("prediction", d)
        self.assertIn("risk_level", d)
        self.assertIn("recommendations", d)

    def test_to_json_is_valid_json(self):
        """to_json() produces valid JSON."""
        import json as _json
        result = self.engine.analyze(self.row, include_explanation=False)
        json_str = result.to_json()
        parsed = _json.loads(json_str)
        self.assertIn("risk_level", parsed)

    def test_prediction_consistent_with_probability(self):
        """Prediction label is consistent with probability and threshold 0.70."""
        result = self.engine.analyze(self.row)
        if result.heatwave_probability >= 0.70:
            self.assertEqual(result.prediction, 1)
        else:
            self.assertEqual(result.prediction, 0)

    def test_batch_analyze_returns_list(self):
        """analyze_batch() on 3 rows returns a list of 3 results."""
        rows = _load_real_test_row(n=3)
        results = self.engine.analyze_batch(rows)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, RiskEngineResult)


# ===========================================================================
# Group K — Real-data smoke test
# ===========================================================================

class TestRealDataSmokeTest(unittest.TestCase):
    """
    Group K: Full pipeline smoke test using a real row from the test split.

    This test uses actual data from:
        data/splits/temporal/X_test.csv
        data/splits/temporal/meta_test.csv
        data/splits/temporal/y_test.csv

    It verifies the complete Part 1 → Part 2 chain:
        Real feature row
            → ClimateGuardPredictor (Part 1)
            → probability + prediction
            → RiskAssessor
            → risk level
            → AdaptationEngine
            → recommendations
            → ClimateGuardExplainer
            → feature explanation
            → RiskEngineResult (structured output)
    """

    def test_full_pipeline_with_real_data(self):
        """
        SMOKE TEST: Full Part 1 → Part 2 pipeline on a real test row.
        """
        print("\n" + "=" * 60)
        print("REAL-DATA SMOKE TEST")
        print("=" * 60)

        engine = _get_engine()

        # Load a real test row
        rows = _load_real_test_row(n=1)
        city = str(rows["city_key"].iloc[0]) if "city_key" in rows.columns else "unknown"
        date = str(rows["date"].iloc[0])      if "date"     in rows.columns else "unknown"
        actual = int(rows["actual_heatwave_next_day"].iloc[0])

        print(f"  City              : {city}")
        print(f"  Date (day T)      : {date}")
        print(f"  Actual next day   : {actual} ({'HEATWAVE' if actual else 'Normal'})")

        # Run full pipeline
        result = engine.analyze(rows, include_explanation=True, top_n=10)

        print(f"  Probability       : {result.heatwave_probability:.4f}")
        print(f"  Prediction        : {result.prediction}")
        print(f"  Risk Level        : {result.risk_level}")
        print(f"  Explanation method: {result.explanation.get('explanation_method', 'N/A')}")
        print(f"  Top features      : {len(result.explanation.get('top_features', []))}")
        print(f"  Recommendations   : {len(result.recommendations.get('recommendations', []))}")

        # ---- Assertions ----

        # Probability
        self.assertGreaterEqual(result.heatwave_probability, 0.0)
        self.assertLessEqual(result.heatwave_probability, 1.0)

        # Prediction binary
        self.assertIn(result.prediction, [0, 1])

        # Threshold consistency
        if result.heatwave_probability >= 0.70:
            self.assertEqual(result.prediction, 1)
        else:
            self.assertEqual(result.prediction, 0)

        # Risk level valid
        self.assertIn(result.risk_level, ["LOW", "MODERATE", "HIGH", "EXTREME"])

        # Recommendations non-empty
        recs = result.recommendations.get("recommendations", [])
        self.assertGreater(len(recs), 0)
        for rec in recs:
            self.assertIn("category", rec)
            self.assertIn("message", rec)
            self.assertGreater(len(rec["message"]), 0)

        # Explanation present
        self.assertIsNotNone(result.explanation)
        self.assertIn("top_features", result.explanation)
        top_feats = result.explanation["top_features"]
        self.assertGreater(len(top_feats), 0)
        for feat in top_feats:
            self.assertIn("feature", feat)
            self.assertIn("value", feat)
            self.assertIn("contribution", feat)
            self.assertIn("direction", feat)

        # Causality note present
        notes = result.explanation.get("notes", {})
        self.assertIn("causality", notes)
        self.assertIn("qualifying_day", notes)

        # Disclaimer in recommendations
        disclaimer = result.recommendations.get("disclaimer", "")
        self.assertGreater(len(disclaimer), 20)

        # Serialisable to JSON
        import json as _json
        json_str = result.to_json()
        parsed = _json.loads(json_str)
        self.assertIn("heatwave_probability", parsed)
        self.assertIn("risk_level", parsed)
        self.assertIn("recommendations", parsed)

        print(f"\n  [PASS] All smoke test assertions passed.")
        print(f"  City={city}, Date={date}")
        print(f"  prob={result.heatwave_probability:.4f}, pred={result.prediction}, "
              f"risk={result.risk_level}")
        print("=" * 60)

    def test_real_row_feature_count(self):
        """Real test row has exactly 110 feature columns."""
        rows = _load_real_test_row(n=1)
        predictor = _get_predictor()
        # Drop non-feature columns before counting
        feature_cols = [c for c in rows.columns if c in predictor.feature_names]
        self.assertEqual(len(feature_cols), 110)

    def test_real_row_no_nans_in_features(self):
        """Real test row has no NaN values in the 110 feature columns."""
        rows = _load_real_test_row(n=1)
        predictor = _get_predictor()
        feature_cols = [c for c in predictor.feature_names if c in rows.columns]
        nan_count = rows[feature_cols].isna().sum().sum()
        self.assertEqual(nan_count, 0)

    def test_multiple_real_rows_pipeline(self):
        """Full pipeline runs successfully on 5 consecutive real rows."""
        engine = _get_engine()
        rows = _load_real_test_row(n=5)
        results = engine.analyze_batch(rows, include_explanation=True, top_n=5)

        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            with self.subTest(row=i):
                self.assertIn(result.prediction, [0, 1])
                self.assertIn(result.risk_level, ["LOW", "MODERATE", "HIGH", "EXTREME"])
                self.assertGreater(
                    len(result.recommendations.get("recommendations", [])), 0
                )


# ===========================================================================
# Main runner
# ===========================================================================

def run_tests() -> bool:
    """Run all tests and return True if all pass."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestRiskProbabilityValidation,       # Group A
        TestRiskThresholdBoundaries,          # Group B
        TestRiskLevelGeneration,              # Group C
        TestAdaptationRecommendationGeneration, # Group D
        TestAdaptationCategories,             # Group E
        TestModelAccess,                      # Group F
        TestFeatureNameConsistency,           # Group G
        TestExplainabilityExecution,          # Group H
        TestInvalidInputHandling,             # Group I
        TestPart1ToPart2Integration,          # Group J
        TestRealDataSmokeTest,                # Group K
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

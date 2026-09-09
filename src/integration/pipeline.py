"""
ClimateGuard Integration Pipeline — pipeline.py
Part 3

ClimateGuardPipeline is the unified Part 3 orchestrator.

It connects all components into a single end-to-end pipeline:

    Incoming Climate Data (dict / DataFrame)
            ↓
    ETL / Validation   (src.etl.ETLPipeline)
            ↓
    Feature Contract   (110-feature check)
            ↓
    Part 1 ML Predictor (src.prediction.ClimateGuardPredictor)
            ↓
    Probability + Prediction
            ↓
    Part 2 Risk Engine  (src.risk_engine.ClimateGuardRiskEngine)
            ↓
    Risk Level + Recommendations + Explainability
            ↓
    Part 3 Expert Rules (src.expert_rules.ExpertRuleEngine)
            ↓
    ClimateGuardResult  (final structured output)

Design constraints
------------------
- Part 1 predictor is loaded ONCE per ClimateGuardPipeline instance.
- Part 2 engine is loaded ONCE per ClimateGuardPipeline instance.
- No ML model is trained or retrained.
- Threshold 0.70 is never altered.
- Feature list is never modified.
- The pipeline ALWAYS returns a ClimateGuardResult; it never raises on
  validation failure (the result carries the error).

Output structure (ClimateGuardResult.to_dict())
-------------------------------------------------
{
    "input":        {...},          # input metadata
    "validation":   {"valid": ..., "warnings": [], "errors": []},
    "prediction":   {"probability": ..., "prediction": ...},
    "risk":         {"level": ..., "score": ...},
    "recommendations": [...],
    "explanation":  {...} or null,
    "expert_rules": [...],
    "warnings":     [...],
    "metadata":     {...}
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Project root / path setup
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl import ETLPipeline, ETLResult
from src.prediction import ClimateGuardPredictor
from src.risk_engine import ClimateGuardRiskEngine
from src.expert_rules import ExpertRuleEngine, EXPERT_RULES_DISCLAIMER


# ---------------------------------------------------------------------------
# ClimateGuardResult
# ---------------------------------------------------------------------------

class ClimateGuardResult:
    """
    Final structured output of the ClimateGuard Part 3 pipeline.

    Attributes
    ----------
    valid : bool
        True if the full pipeline ran successfully.
    input_metadata : dict
        Extracted city/date information from the input record.
    validation : dict
        ETL validation outcome (valid, errors, warnings).
    prediction : dict or None
        Part 1 output: probability and prediction label.
        None if ETL validation failed.
    risk : dict or None
        Part 2 risk assessment: level and score (probability).
        None if Part 1 failed.
    recommendations : list or None
        Part 2 adaptation recommendations list.
        None if Part 2 failed.
    explanation : dict or None
        Part 2 explainability output.
        None if explanation was disabled or unavailable.
    expert_rules : list
        Part 3 expert rule results (always present, may be empty on failure).
    warnings : list
        Pipeline-level warnings (non-blocking issues across all stages).
    metadata : dict
        Pipeline configuration information.
    """

    def __init__(
        self,
        valid: bool,
        input_metadata: Dict,
        validation: Dict,
        prediction: Optional[Dict],
        risk: Optional[Dict],
        recommendations: Optional[List],
        explanation: Optional[Dict],
        expert_rules: List[Dict],
        warnings: List[str],
        metadata: Dict,
    ):
        self.valid            = valid
        self.input_metadata   = input_metadata
        self.validation       = validation
        self.prediction       = prediction
        self.risk             = risk
        self.recommendations  = recommendations
        self.explanation      = explanation
        self.expert_rules     = expert_rules
        self.warnings         = warnings
        self.metadata         = metadata

    def to_dict(self) -> Dict:
        """Return the full result as a plain dictionary."""
        return {
            "input":           self.input_metadata,
            "validation":      self.validation,
            "prediction":      self.prediction,
            "risk":            self.risk,
            "recommendations": self.recommendations,
            "explanation":     self.explanation,
            "expert_rules":    self.expert_rules,
            "warnings":        self.warnings,
            "metadata":        self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the full result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        city   = self.input_metadata.get("city_key", "unknown")
        date   = self.input_metadata.get("date",     "unknown")
        prob   = self.prediction.get("probability", "N/A") if self.prediction else "N/A"
        level  = self.risk.get("level", "N/A") if self.risk else "N/A"
        n_exp  = sum(1 for r in self.expert_rules if r.get("triggered"))
        return (
            f"ClimateGuardResult({status}, city={city}, date={date}, "
            f"prob={prob}, risk={level}, expert_rules_triggered={n_exp})"
        )


# ---------------------------------------------------------------------------
# ClimateGuardPipeline
# ---------------------------------------------------------------------------

class ClimateGuardPipeline:
    """
    End-to-end ClimateGuard integration pipeline (Part 3).

    Orchestrates ETL → Part 1 → Part 2 → Expert Rules into a single call.

    Parameters
    ----------
    predictor : ClimateGuardPredictor, optional
        Pre-loaded Part 1 predictor.  Created automatically if not supplied.
        Pass an existing instance to share model loading across calls.
    include_explanation : bool, optional
        Whether to request feature explainability from Part 2.  Default True.
    top_n : int, optional
        Number of top features to return in the explanation.  Default 10.

    Usage
    -----
        pipeline = ClimateGuardPipeline()

        # Analyse one record (dict with 110 features + city_key + date)
        result = pipeline.analyze(record)

        print(result.valid)
        print(result.risk["level"])
        print(result.to_json())

        # Batch analysis
        results = pipeline.analyze_batch(df)
    """

    def __init__(
        self,
        predictor: Optional[ClimateGuardPredictor] = None,
        include_explanation: bool = True,
        top_n: int = 10,
    ):
        # Load Part 1 predictor once
        self._predictor = predictor if predictor is not None else ClimateGuardPredictor()

        # Build Part 2 engine sharing the same predictor
        self._risk_engine = ClimateGuardRiskEngine(
            predictor=self._predictor,
            include_explanation=include_explanation,
            top_n=top_n,
        )

        # Part 3 components
        self._etl_pipeline  = ETLPipeline()
        self._expert_engine = ExpertRuleEngine()

        self._include_explanation = include_explanation
        self._top_n               = top_n

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        data: Union[Dict, pd.DataFrame, pd.Series],
        include_explanation: Optional[bool] = None,
        top_n: Optional[int] = None,
    ) -> ClimateGuardResult:
        """
        Run the full pipeline on a single record.

        Steps
        -----
        1. ETL validation
        2. Part 1 prediction (via ClimateGuardPredictor inside Part 2 engine)
        3. Part 2 risk assessment + recommendations + explainability
        4. Part 3 expert rules
        5. Assemble ClimateGuardResult

        Parameters
        ----------
        data : dict, pd.Series, or 1-row pd.DataFrame
            Must contain all 110 engineered feature columns plus
            'city_key' and 'date' metadata columns.
        include_explanation : bool, optional
            Override the default explanation setting for this call.
        top_n : int, optional
            Override the default top-N for this call.

        Returns
        -------
        ClimateGuardResult
            Always returned — check result.valid for pipeline success.
        """
        _incl  = self._include_explanation if include_explanation is None else include_explanation
        _top_n = self._top_n               if top_n               is None else top_n

        pipeline_warnings: List[str] = []

        # Build input metadata first (before ETL may reject)
        input_meta = self._extract_input_meta(data)

        # -------------------------------------------------------------------
        # Stage 1 — ETL
        # -------------------------------------------------------------------
        etl_result: ETLResult = self._etl_pipeline.run(data)
        pipeline_warnings.extend(etl_result.validation.warnings)

        if not etl_result.valid:
            return ClimateGuardResult(
                valid=False,
                input_metadata=input_meta,
                validation=etl_result.validation.to_dict(),
                prediction=None,
                risk=None,
                recommendations=None,
                explanation=None,
                expert_rules=[],
                warnings=pipeline_warnings,
                metadata=self._build_metadata(shap_available=self._risk_engine.shap_available),
            )

        feature_df = etl_result.feature_df

        # -------------------------------------------------------------------
        # Stage 2 + 3 — Part 2 engine (internally calls Part 1)
        # -------------------------------------------------------------------
        try:
            engine_result = self._risk_engine.analyze(
                features=feature_df,
                city=input_meta.get("city_key"),
                date=input_meta.get("date"),
                include_explanation=_incl,
                top_n=_top_n,
            )
        except Exception as exc:  # noqa: BLE001
            pipeline_warnings.append(f"Part 2 engine error: {exc}")
            return ClimateGuardResult(
                valid=False,
                input_metadata=input_meta,
                validation=etl_result.validation.to_dict(),
                prediction=None,
                risk=None,
                recommendations=None,
                explanation=None,
                expert_rules=[],
                warnings=pipeline_warnings,
                metadata=self._build_metadata(shap_available=self._risk_engine.shap_available),
            )

        # Extract prediction and risk from Part 2 result
        prediction_dict = {
            "probability": engine_result.heatwave_probability,
            "prediction":  engine_result.prediction,
        }
        risk_dict = {
            "level": engine_result.risk_level,
            "score": engine_result.heatwave_probability,
        }
        recommendations = (
            engine_result.recommendations.get("recommendations", [])
            if engine_result.recommendations else []
        )
        explanation = engine_result.explanation

        # -------------------------------------------------------------------
        # Stage 4 — Part 3 Expert Rules
        # -------------------------------------------------------------------
        raw_record = self._to_dict(data)
        rule_results = self._expert_engine.evaluate(
            data=raw_record,
            risk_level=engine_result.risk_level,
            heatwave_probability=engine_result.heatwave_probability,
        )
        expert_rules_dicts = self._expert_engine.to_dict_list(rule_results)

        # Collect any expert-rule-triggered warnings into the top-level list
        for rr in rule_results:
            if rr.triggered and rr.severity in ("WARNING", "CRITICAL"):
                pipeline_warnings.append(f"[{rr.rule_id}] {rr.name}: {rr.message}")

        # -------------------------------------------------------------------
        # Stage 5 — Assemble result
        # -------------------------------------------------------------------
        return ClimateGuardResult(
            valid=True,
            input_metadata=input_meta,
            validation=etl_result.validation.to_dict(),
            prediction=prediction_dict,
            risk=risk_dict,
            recommendations=recommendations,
            explanation=explanation,
            expert_rules=expert_rules_dicts,
            warnings=pipeline_warnings,
            metadata=self._build_metadata(shap_available=self._risk_engine.shap_available),
        )

    def analyze_batch(
        self,
        df: pd.DataFrame,
        include_explanation: Optional[bool] = None,
        top_n: Optional[int] = None,
    ) -> List[ClimateGuardResult]:
        """
        Run the full pipeline on each row of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            One row per city/day.  Must contain all 110 feature columns
            plus city_key and date.

        Returns
        -------
        list of ClimateGuardResult
            One result per row.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"analyze_batch() expects a pd.DataFrame; got {type(df).__name__}."
            )
        results = []
        for _, row in df.iterrows():
            result = self.analyze(
                row.to_dict(),
                include_explanation=include_explanation,
                top_n=top_n,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def predictor(self) -> ClimateGuardPredictor:
        """The underlying Part 1 predictor instance."""
        return self._predictor

    @property
    def risk_engine(self) -> ClimateGuardRiskEngine:
        """The underlying Part 2 risk engine instance."""
        return self._risk_engine

    @property
    def expert_engine(self) -> ExpertRuleEngine:
        """The Part 3 expert rule engine instance."""
        return self._expert_engine

    @property
    def shap_available(self) -> bool:
        """True if SHAP is installed."""
        return self._risk_engine.shap_available

    def info(self) -> Dict:
        """Return a summary of the pipeline configuration."""
        return {
            "pipeline": "ClimateGuardPipeline (Part 3)",
            "predictor": str(self._predictor),
            "shap_available": self.shap_available,
            "include_explanation": self._include_explanation,
            "top_n": self._top_n,
            "threshold": self._predictor.threshold,
            "n_features": self._predictor.n_features,
            "expert_rules": [
                "RULE_01 Extreme Temperature",
                "RULE_02 Persistent Heat",
                "RULE_03 High Nighttime Temperature",
                "RULE_04 Compounded Heat Stress",
                "RULE_05 Vulnerable Population Alert",
                "RULE_06 Outdoor Exposure Warning",
                "RULE_07 Hydration and Cooling Reminder",
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_input_meta(self, data: Any) -> Dict:
        """Extract city_key and date metadata from any input format."""
        meta: Dict = {}
        try:
            if isinstance(data, pd.Series):
                d = data.to_dict()
            elif isinstance(data, pd.DataFrame) and len(data) > 0:
                d = data.iloc[0].to_dict()
            elif isinstance(data, dict):
                d = data
            else:
                return meta
            if "city_key" in d:
                meta["city_key"] = str(d["city_key"])
            elif "city" in d:
                meta["city_key"] = str(d["city"])
            if "date" in d:
                meta["date"] = str(d["date"])
        except Exception:  # noqa: BLE001
            pass
        return meta

    def _to_dict(self, data: Any) -> Dict:
        """Convert any input to a flat dict."""
        if isinstance(data, dict):
            return data
        if isinstance(data, pd.Series):
            return data.to_dict()
        if isinstance(data, pd.DataFrame):
            if len(data) == 1:
                return data.iloc[0].to_dict()
            return data.iloc[0].to_dict()  # use first row
        return {}

    def _build_metadata(self, shap_available: bool = False) -> Dict:
        return {
            "pipeline_version":    "1.0.0",
            "part1_threshold":     self._predictor.threshold,
            "part1_n_features":    self._predictor.n_features,
            "shap_available":      shap_available,
            "explanation_enabled": self._include_explanation,
            "expert_rules_disclaimer": EXPERT_RULES_DISCLAIMER,
        }

    def __repr__(self) -> str:
        return (
            f"ClimateGuardPipeline("
            f"threshold={self._predictor.threshold}, "
            f"n_features={self._predictor.n_features}, "
            f"shap={self.shap_available}, "
            f"explain={self._include_explanation})"
        )

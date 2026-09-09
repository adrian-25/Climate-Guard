"""
ClimateGuard Risk Engine — engine.py
Part 2

ClimateGuardRiskEngine is the unified Part 2 interface.  It combines:

    1. Part 1 prediction (ClimateGuardPredictor)
    2. Risk Assessment (RiskAssessor)
    3. Adaptation Recommendations (AdaptationEngine)
    4. Model Explainability (ClimateGuardExplainer)

Architecture
------------
    ClimateGuardPredictor (Part 1)
            ↓
    probability + prediction_label
            ↓
    ┌───────────────────┬──────────────────────┐
    │  RiskAssessor     │  ClimateGuardExplainer│
    └──────────┬────────┴──────────────────────┘
               ↓
    risk_level
               ↓
    AdaptationEngine
               ↓
    RiskEngineResult (structured Part 2 output)

Output structure
----------------
    {
        "city": "delhi",
        "date": "2024-05-20",
        "heatwave_probability": 0.82,
        "prediction": 1,
        "risk_level": "EXTREME",
        "explanation": {
            "explanation_method": "shap",
            "top_features": [
                {
                    "feature": "qualifying_day",
                    "value": 1.0,
                    "contribution": 0.18,
                    "direction": "increases_risk"
                },
                ...
            ],
            "notes": {...}
        },
        "recommendations": {
            "risk_level": "EXTREME",
            "disclaimer": "...",
            "recommendations": [{"category": ..., "message": ...}, ...]
        }
    }

Usage
-----
    from src.risk_engine import ClimateGuardRiskEngine
    import pandas as pd

    engine = ClimateGuardRiskEngine()

    # Pass a feature row as a DataFrame (110 columns)
    result = engine.analyze(features_df)

    # Or pass with explicit city/date metadata
    result = engine.analyze(
        features_df,
        city="delhi",
        date="2024-05-20",
        top_n=10,
        include_explanation=True,
    )

    print(result.risk_level)          # "EXTREME"
    print(result.to_dict())           # full structured output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on path when running as a script
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.prediction import ClimateGuardPredictor
from src.risk import RiskAssessor
from src.adaptation import AdaptationEngine
from src.explainability import ClimateGuardExplainer


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class RiskEngineResult:
    """
    Structured Part 2 output combining all three components.

    Attributes
    ----------
    city : str or None
    date : str or None
    heatwave_probability : float
    prediction : int (0 or 1)
    risk_level : str ("LOW" | "MODERATE" | "HIGH" | "EXTREME")
    explanation : dict or None
        Explainability output.  None if include_explanation=False.
    recommendations : dict
        Adaptation recommendations dict with risk_level, disclaimer,
        and list of {category, message} dicts.
    """

    def __init__(
        self,
        city: Optional[str],
        date: Optional[str],
        heatwave_probability: float,
        prediction: int,
        risk_level: str,
        recommendations: Dict,
        explanation: Optional[Dict] = None,
    ):
        self.city = city
        self.date = date
        self.heatwave_probability = float(heatwave_probability)
        self.prediction = int(prediction)
        self.risk_level = str(risk_level)
        self.recommendations = recommendations
        self.explanation = explanation

    def to_dict(self) -> Dict:
        """Return the full result as a plain dictionary."""
        d: Dict = {
            "heatwave_probability": self.heatwave_probability,
            "prediction": self.prediction,
            "risk_level": self.risk_level,
        }
        if self.city is not None:
            d["city"] = self.city
        if self.date is not None:
            d["date"] = self.date
        if self.explanation is not None:
            d["explanation"] = self.explanation
        d["recommendations"] = self.recommendations
        return d

    def to_json(self, indent: int = 2) -> str:
        """Return the full result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:
        loc = ""
        if self.city:
            loc += f" city={self.city}"
        if self.date:
            loc += f" date={self.date}"
        label_str = "HEATWAVE" if self.prediction == 1 else "normal"
        return (
            f"RiskEngineResult({label_str},{loc} "
            f"prob={self.heatwave_probability:.4f} "
            f"risk={self.risk_level})"
        )


# ---------------------------------------------------------------------------
# ClimateGuardRiskEngine
# ---------------------------------------------------------------------------

class ClimateGuardRiskEngine:
    """
    Unified Part 2 interface for ClimateGuard.

    Combines Part 1 prediction with risk assessment, adaptation
    recommendations, and optional model explainability.

    Parameters
    ----------
    predictor : ClimateGuardPredictor, optional
        A pre-constructed predictor.  If not provided, a new instance is
        created using default paths (project-relative).  Pass an existing
        instance if you want to share a single model load across multiple
        calls.
    include_explanation : bool, optional
        Whether to include model explainability in every analyze() call.
        Default is True.  Can be overridden per-call.
    top_n : int, optional
        Default number of top features to return in explanations.
        Default is 10.  Can be overridden per-call.

    Usage
    -----
        from src.risk_engine import ClimateGuardRiskEngine

        engine = ClimateGuardRiskEngine()

        # Analyze a single feature row
        result = engine.analyze(features_df)

        print(result.risk_level)
        print(result.to_dict())
        print(result.to_json())
    """

    def __init__(
        self,
        predictor: Optional[ClimateGuardPredictor] = None,
        include_explanation: bool = True,
        top_n: int = 10,
    ):
        # Part 1 — prediction
        self._predictor = predictor if predictor is not None else ClimateGuardPredictor()

        # Part 2 components
        self._risk_assessor = RiskAssessor()
        self._adaptation_engine = AdaptationEngine()
        self._explainer = ClimateGuardExplainer(self._predictor)

        self._default_include_explanation = include_explanation
        self._default_top_n = top_n

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        features: Union[pd.DataFrame, dict],
        city: Optional[str] = None,
        date: Optional[str] = None,
        include_explanation: Optional[bool] = None,
        top_n: Optional[int] = None,
    ) -> RiskEngineResult:
        """
        Run the full Part 2 pipeline on a single feature row.

        Steps:
            1. Predict probability and label via ClimateGuardPredictor.
            2. Assess risk level via RiskAssessor.
            3. Generate adaptation recommendations via AdaptationEngine.
            4. (Optional) Generate feature explanation via ClimateGuardExplainer.

        Parameters
        ----------
        features : pd.DataFrame or dict
            One row of input features (110 columns, same format as
            ClimateGuardPredictor).
        city : str, optional
            City identifier override (e.g. "delhi").  If the features dict/
            DataFrame already contains 'city_key' or 'city', that value is
            used unless overridden here.
        date : str, optional
            Date string override.  If the features DataFrame contains 'date',
            that value is used unless overridden here.
        include_explanation : bool, optional
            Whether to compute and return the feature explanation.
            Defaults to the engine's constructor setting (True).
        top_n : int, optional
            Number of top features to include in the explanation.
            Defaults to the engine's constructor setting (10).

        Returns
        -------
        RiskEngineResult
            Full structured output with probability, prediction, risk level,
            recommendations, and optional explanation.

        Raises
        ------
        ValueError / TypeError
            Propagated from the predictor on invalid input.
        """
        _include = self._default_include_explanation if include_explanation is None else include_explanation
        _top_n   = self._default_top_n              if top_n              is None else top_n

        # Step 1 — Prediction (Part 1)
        pred_result = self._predictor.predict(features)

        # Use city/date from result metadata unless caller provided overrides
        effective_city = city if city is not None else pred_result.city
        effective_date = date if date is not None else pred_result.date

        probability = pred_result.prediction_probability
        prediction  = pred_result.prediction_label

        # Step 2 — Risk assessment
        risk_result = self._risk_assessor.assess(
            probability=probability,
            prediction=prediction,
            city=effective_city,
            date=effective_date,
        )
        risk_level = risk_result.risk_level

        # Step 3 — Adaptation recommendations
        recommendations = self._adaptation_engine.recommend_as_dict(risk_level)

        # Step 4 — Explainability (optional)
        explanation = None
        if _include:
            exp_result = self._explainer.explain(
                features=features,
                prediction=prediction,
                probability=probability,
                top_n=_top_n,
            )
            explanation = exp_result.to_dict()

        return RiskEngineResult(
            city=effective_city,
            date=effective_date,
            heatwave_probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            recommendations=recommendations,
            explanation=explanation,
        )

    def analyze_batch(
        self,
        features_df: pd.DataFrame,
        include_explanation: Optional[bool] = None,
        top_n: Optional[int] = None,
    ) -> List[RiskEngineResult]:
        """
        Run the full Part 2 pipeline on each row of a DataFrame.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with 110 feature columns plus optional city/date columns.
            Each row is processed independently.
        include_explanation : bool, optional
        top_n : int, optional

        Returns
        -------
        list of RiskEngineResult
            One result per row.
        """
        results = []
        for _, row in features_df.iterrows():
            # Convert each row to a single-row DataFrame to preserve column metadata.
            # row.to_frame().T can produce object-dtype columns for numeric values;
            # re-cast feature columns to float64 to satisfy the predictor's type check.
            row_df = row.to_frame().T.reset_index(drop=True)
            for col in self._predictor.feature_names:
                if col in row_df.columns:
                    row_df[col] = pd.to_numeric(row_df[col], errors="coerce")
            result = self.analyze(
                features=row_df,
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
    def explainer(self) -> ClimateGuardExplainer:
        """The explainability module instance."""
        return self._explainer

    @property
    def shap_available(self) -> bool:
        """True if SHAP is installed and available."""
        return ClimateGuardExplainer.shap_available()

    def info(self) -> Dict:
        """Return a summary of the engine configuration."""
        return {
            "predictor": str(self._predictor),
            "shap_available": self.shap_available,
            "default_include_explanation": self._default_include_explanation,
            "default_top_n": self._default_top_n,
            "risk_thresholds": [
                {"lower_inclusive": lb, "risk_level": str(rl)}
                for lb, rl in RiskAssessor.THRESHOLDS
            ],
            "adaptation_risk_levels": self._adaptation_engine.all_risk_levels(),
        }

    def __repr__(self) -> str:
        return (
            f"ClimateGuardRiskEngine("
            f"shap={self.shap_available}, "
            f"explain={self._default_include_explanation}, "
            f"top_n={self._default_top_n})"
        )

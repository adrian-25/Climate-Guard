"""
ClimateGuard Explainability Module
Part 2 — Risk, Adaptation, Explainability

Exposes ClimateGuardExplainer for downstream use.

Usage:
    from src.explainability import ClimateGuardExplainer
    from src.prediction import ClimateGuardPredictor

    predictor = ClimateGuardPredictor()
    explainer = ClimateGuardExplainer(predictor)

    # Explain a single row (DataFrame)
    explanation = explainer.explain(feature_row_df)
    print(explanation)
"""

from src.explainability.explainer import ClimateGuardExplainer, ExplainabilityResult

__all__ = ["ClimateGuardExplainer", "ExplainabilityResult"]

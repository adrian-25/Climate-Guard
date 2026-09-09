"""
ClimateGuard Risk Engine — Main Part 2 Interface
Part 2 — Risk, Adaptation, Explainability

Exposes ClimateGuardRiskEngine: the unified Part 2 interface that combines
risk assessment, adaptation recommendations, and model explainability.

Usage:
    from src.risk_engine import ClimateGuardRiskEngine

    engine = ClimateGuardRiskEngine()
    result = engine.analyze(features_df, city="delhi", date="2024-05-20")
    print(result)
"""

from src.risk_engine.engine import ClimateGuardRiskEngine, RiskEngineResult

__all__ = ["ClimateGuardRiskEngine", "RiskEngineResult"]

"""
ClimateGuard Risk Assessment Module
Part 2 — Risk, Adaptation, Explainability

Exposes RiskAssessor and RiskAssessmentResult for downstream use.

Usage:
    from src.risk import RiskAssessor
    assessor = RiskAssessor()
    result = assessor.assess(probability=0.82, prediction=1, city="delhi", date="2024-05-20")
"""

from src.risk.risk_assessment import RiskAssessor, RiskAssessmentResult, RiskLevel

__all__ = ["RiskAssessor", "RiskAssessmentResult", "RiskLevel"]

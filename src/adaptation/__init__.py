"""
ClimateGuard Adaptation Recommendation Module
Part 2 — Risk, Adaptation, Explainability

Exposes AdaptationEngine and Recommendation for downstream use.

Usage:
    from src.adaptation import AdaptationEngine
    engine = AdaptationEngine()
    recommendations = engine.recommend("HIGH")
"""

from src.adaptation.recommendations import AdaptationEngine, Recommendation

__all__ = ["AdaptationEngine", "Recommendation"]

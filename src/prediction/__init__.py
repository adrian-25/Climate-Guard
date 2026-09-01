"""
ClimateGuard Prediction Interface
Phase 15 — Part 1 output module

Exposes ClimateGuardPredictor for use by:
  Part 2 (Kshitij) — Risk, Adaptation, Explainability
  Part 3 (Pradnesh) — Expert Module, ETL, Integration

Usage:
    from src.prediction import ClimateGuardPredictor
    predictor = ClimateGuardPredictor()
    result = predictor.predict(features_dict)
"""

from src.prediction.predictor import ClimateGuardPredictor

__all__ = ["ClimateGuardPredictor"]

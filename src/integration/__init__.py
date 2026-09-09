"""
ClimateGuard Integration Module
Part 3 — End-to-End Pipeline Orchestration

Exposes ClimateGuardPipeline: the unified Part 3 orchestrator that connects
ETL validation, Part 1 prediction, Part 2 risk assessment, and Part 3 expert
rules into a single callable.

Usage:
    from src.integration import ClimateGuardPipeline

    pipeline = ClimateGuardPipeline()
    result = pipeline.analyze(feature_record)

    print(result.valid)
    print(result.risk["level"])
    print(result.to_json())
"""

from src.integration.pipeline import ClimateGuardPipeline, ClimateGuardResult

__all__ = ["ClimateGuardPipeline", "ClimateGuardResult"]

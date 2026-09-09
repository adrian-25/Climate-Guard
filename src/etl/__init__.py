"""
ClimateGuard ETL Module
Part 3 — Input Validation and Transformation

Exposes the ETL pipeline and its components for use by the integration layer.

Usage:
    from src.etl import ETLPipeline, InputValidator, FeatureContractValidator, FeatureTransformer

    pipeline = ETLPipeline()
    result = pipeline.run(record_dict)
    if result.valid:
        # result.feature_df is ready for ClimateGuardPredictor
        pass
"""

from src.etl.validator import (
    InputValidator,
    FeatureContractValidator,
    ValidationResult,
    VALID_CITY_KEYS,
    normalise_city,
)
from src.etl.transformer import FeatureTransformer
from src.etl.pipeline import ETLPipeline, ETLResult

__all__ = [
    "ETLPipeline",
    "ETLResult",
    "InputValidator",
    "FeatureContractValidator",
    "FeatureTransformer",
    "ValidationResult",
    "VALID_CITY_KEYS",
    "normalise_city",
]

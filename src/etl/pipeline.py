"""
ClimateGuard ETL — pipeline.py
Part 3

Provides ETLPipeline: a thin orchestrator that chains together the
InputValidator, FeatureContractValidator, and FeatureTransformer into a
single callable step.

The ETL pipeline is the first stage of the ClimateGuard Part 3 integration:

    Raw Input
        ↓
    ETLPipeline.run(data)
        ↓  (InputValidator)
        ↓  (FeatureContractValidator)
        ↓  (FeatureTransformer)
        ↓
    ETLResult
        - feature_df  : 110-column DataFrame ready for Part 1 predictor
        - validation  : ValidationResult (errors, warnings)
        - metadata    : dict (city_key, date, original_shape)

The pipeline ALWAYS returns an ETLResult — it never raises on validation
failure.  Callers must check ``result.valid`` before proceeding.

Design constraints
------------------
- Does NOT call the Part 1 predictor (that belongs in the integration layer).
- Does NOT re-implement feature engineering.
- Stops at validation if any blocking error is found (feature_df is None).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.etl.validator import InputValidator, FeatureContractValidator, ValidationResult
from src.etl.transformer import FeatureTransformer

# ---------------------------------------------------------------------------
# ETLResult
# ---------------------------------------------------------------------------

@dataclass
class ETLResult:
    """
    Result of the ETL pipeline run.

    Attributes
    ----------
    valid : bool
        True if the data passed all validation checks and was successfully
        transformed.  False if any blocking error occurred.
    validation : ValidationResult
        Combined validation outcome (errors + warnings from all stages).
    feature_df : pd.DataFrame or None
        The 110-column feature DataFrame ready for ClimateGuardPredictor.
        None if validation failed.
    metadata : dict
        Extracted metadata: city_key, date, original_shape.
    """
    valid: bool
    validation: ValidationResult
    feature_df: Optional[pd.DataFrame]
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "validation": self.validation.to_dict(),
            "feature_shape": list(self.feature_df.shape) if self.feature_df is not None else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        shape = str(self.feature_df.shape) if self.feature_df is not None else "None"
        return f"ETLResult({status}, feature_df={shape})"


# ---------------------------------------------------------------------------
# ETLPipeline
# ---------------------------------------------------------------------------

class ETLPipeline:
    """
    Validates and transforms incoming climate data into the 110-feature
    contract format expected by ClimateGuardPredictor.

    The pipeline runs three sequential stages:
        Stage 1 — InputValidator:         structural + range checks on raw record
        Stage 2 — FeatureContractValidator: 110-feature contract check
        Stage 3 — FeatureTransformer:     select, reorder, cast to float64

    If Stage 1 or 2 produces blocking errors, Stage 3 is skipped and
    ``ETLResult.feature_df`` is None.

    Parameters
    ----------
    feature_list_path : Path or str, optional
        Path to feature_list.json.  Defaults to models/final/feature_list.json.
    strict_range_check : bool, optional
        If True (default), range violations produce warnings (not errors).
        Range violations are never blocking — they are sanity guards only.

    Usage
    -----
        pipeline = ETLPipeline()

        # Single record
        result = pipeline.run(record_dict)
        if not result.valid:
            print(result.validation.errors)
        else:
            predictor.predict(result.feature_df)

        # Batch
        result = pipeline.run_batch(df)
    """

    def __init__(
        self,
        feature_list_path: Optional[Union[str, Path]] = None,
        strict_range_check: bool = True,
    ):
        self._input_validator = InputValidator()
        self._feature_validator = FeatureContractValidator(feature_list_path)
        self._transformer = FeatureTransformer(feature_list_path)
        self._strict_range_check = strict_range_check

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, data: Union[Dict, pd.DataFrame, pd.Series]) -> ETLResult:
        """
        Run the full ETL pipeline on a single record.

        Parameters
        ----------
        data : dict, pd.Series, or 1-row pd.DataFrame

        Returns
        -------
        ETLResult
        """
        combined_validation = ValidationResult()

        # Normalise to dict
        if isinstance(data, pd.Series):
            record = data.to_dict()
        elif isinstance(data, pd.DataFrame):
            if len(data) != 1:
                combined_validation.add_error(
                    f"run() accepts exactly 1 row; got {len(data)}. "
                    "Use run_batch() for multiple rows."
                )
                return ETLResult(
                    valid=False,
                    validation=combined_validation,
                    feature_df=None,
                    metadata={},
                )
            record = data.iloc[0].to_dict()
        elif isinstance(data, dict):
            record = data
        else:
            combined_validation.add_error(
                f"Input must be dict, pd.Series, or pd.DataFrame; got {type(data).__name__}."
            )
            return ETLResult(
                valid=False,
                validation=combined_validation,
                feature_df=None,
                metadata={},
            )

        # Extract metadata before validation
        metadata = self._extract_metadata(record)

        # Stage 1: Input validation
        input_result = self._input_validator.validate(record)
        combined_validation.merge(input_result)

        # Stage 2: Feature contract validation (only if stage 1 passed)
        if combined_validation.valid:
            contract_result = self._feature_validator.validate(record)
            combined_validation.merge(contract_result)

        # Stage 3: Transform (only if validation passed)
        feature_df = None
        if combined_validation.valid:
            try:
                feature_df = self._transformer.transform_with_metadata(record)
            except (ValueError, TypeError) as exc:
                combined_validation.add_error(
                    f"Transformation failed: {exc}"
                )

        return ETLResult(
            valid=combined_validation.valid,
            validation=combined_validation,
            feature_df=feature_df,
            metadata=metadata,
        )

    def run_batch(self, df: pd.DataFrame) -> ETLResult:
        """
        Run the ETL pipeline on a batch DataFrame.

        Each row is validated individually.  Rows that fail validation are
        excluded from the output feature_df and noted in the validation result.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        ETLResult
            feature_df contains only the rows that passed validation.
            If ALL rows fail, feature_df is None.
        """
        if not isinstance(df, pd.DataFrame):
            v = ValidationResult()
            v.add_error(
                f"run_batch() expects a pd.DataFrame; got {type(df).__name__}."
            )
            return ETLResult(valid=False, validation=v, feature_df=None, metadata={})

        if df.empty:
            v = ValidationResult()
            v.add_error("Input DataFrame is empty.")
            return ETLResult(valid=False, validation=v, feature_df=None, metadata={})

        combined = ValidationResult()
        good_rows = []

        for idx, row in df.iterrows():
            row_result = self.run(row.to_dict())
            for err in row_result.validation.errors:
                combined.add_error(f"Row {idx}: {err}")
            for warn in row_result.validation.warnings:
                combined.add_warning(f"Row {idx}: {warn}")
            if row_result.valid and row_result.feature_df is not None:
                good_rows.append(row_result.feature_df)

        # Batch-level duplicate check
        if "city_key" in df.columns and "date" in df.columns:
            dupes = df.duplicated(subset=["city_key", "date"], keep=False)
            if dupes.any():
                dupe_pairs = df[dupes][["city_key", "date"]].drop_duplicates()
                for _, dr in dupe_pairs.iterrows():
                    combined.add_warning(
                        f"Duplicate (city_key, date) pair: "
                        f"city={dr.get('city_key')}, date={dr.get('date')}."
                    )

        if not good_rows:
            if combined.valid:
                combined.add_error("No rows passed ETL validation.")
            return ETLResult(
                valid=False,
                validation=combined,
                feature_df=None,
                metadata={"original_shape": list(df.shape), "valid_rows": 0},
            )

        import pandas as _pd
        feature_df = _pd.concat(good_rows, ignore_index=True)
        metadata = {
            "original_shape": list(df.shape),
            "valid_rows": len(good_rows),
        }

        return ETLResult(
            valid=combined.valid,
            validation=combined,
            feature_df=feature_df,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metadata(self, record: Dict) -> Dict:
        """Extract metadata fields from a raw record."""
        meta = {}
        if "city_key" in record:
            meta["city_key"] = str(record["city_key"])
        elif "city" in record:
            meta["city_key"] = str(record["city"])
        if "date" in record:
            meta["date"] = str(record["date"])
        return meta

    def __repr__(self) -> str:
        return "ETLPipeline(stages=[InputValidator, FeatureContractValidator, FeatureTransformer])"

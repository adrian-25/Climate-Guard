"""
ClimateGuard ETL — validator.py
Part 3

Validates incoming climate/weather data records BEFORE they are passed to the
Part 1 ML predictor.  Validation is purely structural and range-based; it does
NOT reproduce or alter any Part 1 feature-engineering logic.

Validation stages
-----------------
1. Required field presence
2. City validity (must be one of the five supported city_key values)
3. Date format and calendar validity
4. Data type check / safe numeric coercion
5. Missing value detection
6. Numeric range sanity checks (climate variables)
7. Duplicate date detection (within a batch)
8. 110-feature contract check (when a pre-built feature row is supplied)

Output
------
All validators return a ``ValidationResult`` with:
    valid    : bool
    errors   : list[str]  — blocking issues; pipeline must NOT proceed
    warnings : list[str]  — non-blocking notices

The pipeline stops at the ETL layer if ``valid == False``.

IMPORTANT
---------
These range checks are project-defined sanity guards.
They are NOT official IMD or WMO data-quality standards.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent

FEATURE_LIST_PATH = PROJECT_ROOT / "models" / "final" / "feature_list.json"

# ---------------------------------------------------------------------------
# Supported cities
# ---------------------------------------------------------------------------

#: Canonical city_key values (lowercase, matching the project convention)
VALID_CITY_KEYS: frozenset = frozenset({
    "delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"
})

#: Human-readable city names accepted as input (mapped to city_key)
_CITY_NAME_MAP: Dict[str, str] = {
    # canonical keys
    "delhi":     "delhi",
    "lucknow":   "lucknow",
    "nagpur":    "nagpur",
    "ahmedabad": "ahmedabad",
    "mumbai":    "mumbai",
    # common alternates
    "new delhi": "delhi",
    "new_delhi": "delhi",
}


def normalise_city(value: Any) -> Optional[str]:
    """
    Normalise a city name/key to the canonical city_key string.

    Returns None if the value is not recognised.
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    return _CITY_NAME_MAP.get(v)


# ---------------------------------------------------------------------------
# Numeric range constraints — project-defined sanity checks
# ---------------------------------------------------------------------------
# These bounds are generous physical plausibility guards for Indian conditions.
# They are NOT official IMD thresholds.
# Sources: ERA5 historical range for the five cities, common meteorological sense.

_RANGE_CONSTRAINTS: Dict[str, tuple] = {
    # temperature (°C) — physical plausibility for Indian cities
    "temperature_2m_max":  (-5.0,  55.0),
    "temperature_2m_mean": (-5.0,  50.0),
    "temperature_2m_min":  (-10.0, 45.0),
    "apparent_temperature_max":  (-10.0, 65.0),
    "apparent_temperature_mean": (-10.0, 60.0),
    "apparent_temperature_min":  (-15.0, 55.0),
    # precipitation (mm)
    "precipitation_sum": (0.0, 500.0),
    # humidity (%)
    "relative_humidity_2m_max":  (0.0, 100.0),
    "relative_humidity_2m_mean": (0.0, 100.0),
    "relative_humidity_2m_min":  (0.0, 100.0),
    # surface pressure (hPa)
    "surface_pressure_mean": (900.0, 1050.0),
    # wind speed (km/h)
    "wind_speed_10m_max": (0.0, 200.0),
    "wind_gusts_10m_max": (0.0, 300.0),
    # radiation (MJ/m²)
    "shortwave_radiation_sum": (0.0, 40.0),
    # evapotranspiration (mm)
    "et0_fao_evapotranspiration": (0.0, 20.0),
    # tmax departure (°C) — generous range
    "tmax_departure": (-20.0, 25.0),
    # z-score — standard normal range
    "tmax_departure_zscore": (-10.0, 10.0),
    # climatological normal temperature
    "tmax_normal": (10.0, 50.0),
}

# Fields that are always required in raw input (before feature engineering)
_REQUIRED_RAW_FIELDS: List[str] = [
    "city_key",
    "date",
    "temperature_2m_max",
]

# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Result of a validation pass.

    Attributes
    ----------
    valid    : bool
        True only if no blocking errors were found.
    errors   : list[str]
        Blocking issues. Pipeline must NOT proceed if any errors are present.
    warnings : list[str]
        Non-blocking notices. Pipeline may proceed but should log them.
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "ValidationResult") -> None:
        """Absorb another ValidationResult into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> Dict:
        return {
            "valid":    self.valid,
            "errors":   list(self.errors),
            "warnings": list(self.warnings),
        }

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return (
            f"ValidationResult({status}, "
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )


# ---------------------------------------------------------------------------
# InputValidator
# ---------------------------------------------------------------------------

class InputValidator:
    """
    Validates a raw climate data record or dict before ETL transformation.

    Checks performed (in order):
    1. Required fields
    2. City validity
    3. Date format and calendar validity
    4. Data types (numeric coercibility)
    5. Missing / NaN values
    6. Numeric range sanity checks

    Usage
    -----
        validator = InputValidator()
        result = validator.validate(record_dict)
        if not result.valid:
            raise ValueError(result.errors)
    """

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        range_constraints: Optional[Dict[str, tuple]] = None,
    ):
        self._required_fields = required_fields if required_fields is not None else _REQUIRED_RAW_FIELDS
        self._range_constraints = range_constraints if range_constraints is not None else _RANGE_CONSTRAINTS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self, record: Union[Dict, pd.Series]) -> ValidationResult:
        """
        Validate a single raw record dict or pd.Series.

        Parameters
        ----------
        record : dict or pd.Series

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        if isinstance(record, pd.Series):
            record = record.to_dict()
        elif not isinstance(record, dict):
            result.add_error(
                f"Input must be a dict or pd.Series; got {type(record).__name__}."
            )
            return result

        self._check_required_fields(record, result)
        if not result.valid:
            # Can't proceed sensibly without required fields
            return result

        self._check_city(record, result)
        self._check_date(record, result)
        self._check_types_and_missing(record, result)
        self._check_numeric_ranges(record, result)

        return result

    def validate_batch(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame of records.

        Checks each row individually, then checks for duplicate (city, date) pairs.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        ValidationResult
            Combined result across all rows.
        """
        combined = ValidationResult()

        if not isinstance(df, pd.DataFrame):
            combined.add_error(
                f"validate_batch() expects a pd.DataFrame; got {type(df).__name__}."
            )
            return combined

        if df.empty:
            combined.add_error("Input DataFrame is empty.")
            return combined

        for idx, row in df.iterrows():
            row_result = self.validate(row.to_dict())
            for err in row_result.errors:
                combined.add_error(f"Row {idx}: {err}")
            for warn in row_result.warnings:
                combined.add_warning(f"Row {idx}: {warn}")

        # Check for duplicate (city_key, date) pairs
        if "city_key" in df.columns and "date" in df.columns:
            dupes = df.duplicated(subset=["city_key", "date"], keep=False)
            if dupes.any():
                dupe_rows = df[dupes][["city_key", "date"]].drop_duplicates()
                for _, dr in dupe_rows.iterrows():
                    combined.add_warning(
                        f"Duplicate (city_key, date) pair found: "
                        f"city={dr.get('city_key')}, date={dr.get('date')}."
                    )

        return combined

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_required_fields(self, record: Dict, result: ValidationResult) -> None:
        missing = [f for f in self._required_fields if f not in record]
        if missing:
            result.add_error(
                f"Missing required field(s): {missing}. "
                f"All of the following are required: {self._required_fields}."
            )

    def _check_city(self, record: Dict, result: ValidationResult) -> None:
        city_val = record.get("city_key") or record.get("city")
        if city_val is None:
            result.add_error(
                "City not provided. 'city_key' must be one of: "
                + ", ".join(sorted(VALID_CITY_KEYS)) + "."
            )
            return
        normalised = normalise_city(city_val)
        if normalised is None:
            result.add_error(
                f"Unrecognised city: '{city_val}'. "
                f"Supported city_key values: {sorted(VALID_CITY_KEYS)}. "
                "ClimateGuard covers Delhi, Lucknow, Nagpur, Ahmedabad, Mumbai only."
            )

    def _check_date(self, record: Dict, result: ValidationResult) -> None:
        date_val = record.get("date")
        if date_val is None:
            result.add_error("'date' field is missing.")
            return
        date_str = str(date_val).strip()
        # Try ISO format YYYY-MM-DD
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            result.add_error(
                f"Invalid date format: '{date_str}'. "
                "Expected ISO format YYYY-MM-DD (e.g. '2024-05-20')."
            )
            return
        # Plausibility: within ERA5 data range + near future
        if parsed.year < 1990 or parsed.year > 2030:
            result.add_warning(
                f"Date '{date_str}' is outside the expected range (1990–2030). "
                "The model was trained on 1990–2022 data."
            )

    def _check_types_and_missing(self, record: Dict, result: ValidationResult) -> None:
        """Check that numeric climate fields can be coerced to float."""
        non_numeric = []
        missing_vals = []

        climate_fields = [k for k in record if k not in {"city_key", "city", "date", "state", "region_type"}]

        for key in climate_fields:
            val = record[key]
            if val is None:
                missing_vals.append(key)
                continue
            # Check for float NaN
            try:
                fval = float(val)
                if math.isnan(fval):
                    missing_vals.append(key)
            except (TypeError, ValueError):
                non_numeric.append(key)

        if missing_vals:
            result.add_error(
                f"Missing or NaN values in {len(missing_vals)} field(s): "
                f"{missing_vals[:10]}"
                + (" ..." if len(missing_vals) > 10 else "")
                + ". The model cannot process missing values."
            )
        if non_numeric:
            result.add_error(
                f"Non-numeric values in {len(non_numeric)} field(s): "
                f"{non_numeric[:10]}"
                + (" ..." if len(non_numeric) > 10 else "")
                + ". All climate variables must be numeric."
            )

    def _check_numeric_ranges(self, record: Dict, result: ValidationResult) -> None:
        """Sanity-check known climate variables against physical plausibility bounds."""
        for field_name, (lo, hi) in self._range_constraints.items():
            if field_name not in record:
                continue
            val = record[field_name]
            if val is None:
                continue
            try:
                fval = float(val)
                if math.isnan(fval):
                    continue
            except (TypeError, ValueError):
                continue  # already caught in type check

            if not (lo <= fval <= hi):
                result.add_warning(
                    f"Value for '{field_name}' is {fval:.3f}, outside expected range "
                    f"[{lo}, {hi}]. "
                    "This is a project-defined sanity bound, not an official IMD threshold."
                )


# ---------------------------------------------------------------------------
# FeatureContractValidator
# ---------------------------------------------------------------------------

class FeatureContractValidator:
    """
    Validates a pre-built feature row against the 110-feature contract.

    This is a final gate check immediately before calling the Part 1 predictor.
    It verifies:
        - Exactly 110 feature columns present
        - Column names match feature_list.json exactly
        - Column order matches feature_list.json exactly
        - No NaN values
        - All columns are numeric

    Usage
    -----
        fcv = FeatureContractValidator()
        result = fcv.validate(feature_df)
    """

    def __init__(self, feature_list_path: Optional[Path] = None):
        fp = Path(feature_list_path) if feature_list_path else FEATURE_LIST_PATH
        if not fp.exists():
            raise FileNotFoundError(
                f"feature_list.json not found at {fp}. "
                "Ensure models/final/feature_list.json exists."
            )
        with open(fp, encoding="utf-8") as f:
            raw = json.load(f)
        self._feature_names: List[str] = [entry["name"] for entry in raw]
        self._n_features: int = len(self._feature_names)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def validate(
        self,
        data: Union[pd.DataFrame, Dict],
        check_order: bool = True,
    ) -> ValidationResult:
        """
        Validate a feature row or DataFrame against the 110-feature contract.

        Parameters
        ----------
        data : pd.DataFrame or dict
            The feature data to validate.
        check_order : bool
            If True (default), column order must match feature_list.json exactly.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            result.add_error(
                f"Feature data must be a dict or pd.DataFrame; got {type(data).__name__}."
            )
            return result

        # Drop known non-feature columns for counting purposes
        _passthrough = {"city", "city_key", "date", "state", "region_type",
                        "heatwave_next_day", "actual_heatwave_next_day",
                        "prediction_probability", "prediction_label"}
        feature_cols = [c for c in df.columns if c not in _passthrough]

        # Check count
        missing_features = [f for f in self._feature_names if f not in feature_cols]
        extra_features   = [c for c in feature_cols if c not in self._feature_names]

        if missing_features:
            result.add_error(
                f"Feature contract violation: {len(missing_features)} required feature(s) "
                f"are missing: {missing_features[:10]}"
                + (" ..." if len(missing_features) > 10 else "")
                + f". Expected exactly {self._n_features} features from feature_list.json."
            )

        if extra_features:
            result.add_warning(
                f"{len(extra_features)} unexpected column(s) present (will be ignored): "
                f"{extra_features[:10]}"
                + (" ..." if len(extra_features) > 10 else "")
            )

        if missing_features:
            return result  # Can't check order or values without all columns

        # Check column order
        if check_order:
            actual_order = [c for c in df.columns if c in self._feature_names]
            if actual_order != self._feature_names:
                result.add_error(
                    "Feature contract violation: column order does not match "
                    "feature_list.json. The model requires features in the exact "
                    "indexed order. Re-order the DataFrame columns accordingly."
                )

        # Check NaN values
        feat_df = df[self._feature_names]
        nan_counts = feat_df.isna().sum()
        nan_feats = list(nan_counts[nan_counts > 0].index)
        if nan_feats:
            result.add_error(
                f"Feature contract violation: NaN values in {len(nan_feats)} "
                f"feature(s): {nan_feats[:10]}"
                + (" ..." if len(nan_feats) > 10 else "")
                + ". The model cannot process missing values."
            )

        # Check numeric types
        non_numeric = [
            col for col in self._feature_names
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            result.add_error(
                f"Feature contract violation: non-numeric type in {len(non_numeric)} "
                f"feature column(s): {non_numeric[:10]}"
                + (" ..." if len(non_numeric) > 10 else "")
            )

        return result

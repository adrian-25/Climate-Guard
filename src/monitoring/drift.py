"""Feature-distribution drift checks for the deployed ClimateGuard model.

This module is deliberately monitoring-only: it never retrains, changes a
prediction, or changes the production Random Forest threshold.  It compares a
city's recent, valid model-input rows with a versioned reference distribution
created from the model's 1990--2022 training window.

The reference is stratified by city and calendar month to avoid treating normal
seasonality or fixed city metadata as drift.  Population Stability Index (PSI)
is used as a screening signal, not as evidence of a model-performance change.
Actual performance requires observed outcomes and is tracked separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

MIN_HISTORY_ROWS = 30
EPSILON = 1e-6
MONITORING_FEATURES = (
    "apparent_temperature_max",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "tmax_departure",
    "tmax_departure_zscore",
)


@dataclass(frozen=True)
class DriftAssessment:
    """A serialisable, read-only distribution-drift screening result."""

    status: str
    city: Optional[str]
    sample_size: int
    minimum_history_rows: int
    level: Optional[str]
    overall_psi: Optional[float]
    drifting_feature_count: int
    features: List[Dict[str, Any]]
    reference: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation for API responses."""
        return {
            "status": self.status,
            "city": self.city,
            "sample_size": self.sample_size,
            "minimum_history_rows": self.minimum_history_rows,
            "level": self.level,
            "overall_psi": self.overall_psi,
            "drifting_feature_count": self.drifting_feature_count,
            "features": self.features,
            "reference": self.reference,
            "message": self.message,
            "limitations": (
                "PSI screens for input-distribution shift only. It does not measure "
                "forecast accuracy, prove climate change, or trigger automatic retraining."
            ),
        }


class ClimateDriftMonitor:
    """Assess recent city-level model inputs against a frozen PSI baseline.

    The baseline is a small JSON artifact rather than a training dataset, so
    deployment does not rely on unavailable local training files.  Each feature
    is compared to the same city's reference values for the relevant calendar
    month.  At least 30 complete historical rows are required before returning
    an assessment.
    """

    def __init__(self, baseline: Mapping[str, Any]):
        self._baseline = dict(baseline)
        self._validate_baseline()
        self.minimum_history_rows = int(
            self._baseline.get("minimum_history_rows", MIN_HISTORY_ROWS)
        )
        self.features = tuple(self._baseline["monitoring_features"])

    @classmethod
    def from_file(cls, path: Path | str) -> "ClimateDriftMonitor":
        """Load a versioned baseline artifact without modifying it."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    def info(self) -> Dict[str, Any]:
        """Return public metadata without exposing all reference bin details."""
        return {
            "available": True,
            "baseline_version": self._baseline["baseline_version"],
            "reference_period": self._baseline["reference"]["period"],
            "reference_rows": self._baseline["reference"]["rows"],
            "monitoring_features": list(self.features),
            "minimum_history_rows": self.minimum_history_rows,
            "method": "Population Stability Index (PSI), city/month stratified",
        }

    def assess(self, rows: pd.DataFrame, city: Optional[str] = None) -> DriftAssessment:
        """Return a city/month-stratified PSI assessment for recent feature rows.

        ``rows`` must include a ``date`` column and all configured monitoring
        features.  Invalid or incomplete rows are excluded rather than filled,
        because imputation can hide a real distribution shift.
        """
        if not isinstance(rows, pd.DataFrame):
            raise TypeError("rows must be a pandas DataFrame")
        missing = [name for name in ("date", *self.features) if name not in rows.columns]
        if missing:
            raise ValueError(f"Missing drift-monitoring columns: {missing[:5]}")

        resolved_city = city or self._single_city(rows)
        if not resolved_city:
            raise ValueError("city is required when rows do not contain one city_key value")
        city_key = str(resolved_city).lower().strip()
        if city_key not in self._baseline["cities"]:
            raise ValueError(f"No drift baseline is available for city: {city_key}")

        prepared = rows.loc[:, ["date", *self.features]].copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        for feature in self.features:
            prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")
        prepared = prepared.dropna(subset=["date", *self.features]).copy()

        if len(prepared) < self.minimum_history_rows:
            return self._insufficient_history(city_key, len(prepared))

        prepared["month"] = prepared["date"].dt.month
        feature_results = [
            self._feature_assessment(prepared, city_key, feature) for feature in self.features
        ]
        comparable = [entry["psi"] for entry in feature_results if entry["psi"] is not None]
        overall = float(np.mean(comparable)) if comparable else None
        level = self._level(overall) if overall is not None else None
        drifting = sum(
            entry["level"] in {"drift_detected", "high_drift"} for entry in feature_results
        )
        return DriftAssessment(
            status="available",
            city=city_key,
            sample_size=len(prepared),
            minimum_history_rows=self.minimum_history_rows,
            level=level,
            overall_psi=round(overall, 4) if overall is not None else None,
            drifting_feature_count=drifting,
            features=feature_results,
            reference=self.info(),
            message=(
                "Input-distribution screening completed. Review individual features and "
                "observed-outcome performance before considering retraining."
            ),
        )

    def _feature_assessment(self, rows: pd.DataFrame, city: str, feature: str) -> Dict[str, Any]:
        weighted_psi = 0.0
        total_rows = 0
        missing_reference_months: List[int] = []
        for month, subset in rows.groupby("month", sort=False):
            baseline = self._baseline["cities"][city][str(int(month))][feature]
            observed = subset[feature].to_numpy(dtype=float)
            if len(observed) == 0:
                continue
            psi = self._psi(observed, baseline["bin_edges"], baseline["proportions"])
            weighted_psi += psi * len(observed)
            total_rows += len(observed)
        value = weighted_psi / total_rows if total_rows else None
        return {
            "feature": feature,
            "psi": round(float(value), 4) if value is not None else None,
            "level": self._level(value) if value is not None else "unavailable",
            "sample_size": total_rows,
            "missing_reference_months": missing_reference_months,
        }

    @staticmethod
    def _psi(observed: np.ndarray, bin_edges: Iterable[float], expected: Iterable[float]) -> float:
        edges = np.asarray(list(bin_edges), dtype=float)
        expected_arr = np.asarray(list(expected), dtype=float)
        observed_counts, _ = np.histogram(observed, bins=edges)
        observed_arr = observed_counts / max(1, observed_counts.sum())
        expected_arr = np.clip(expected_arr, EPSILON, None)
        observed_arr = np.clip(observed_arr, EPSILON, None)
        return float(np.sum((observed_arr - expected_arr) * np.log(observed_arr / expected_arr)))

    @staticmethod
    def _level(psi: Optional[float]) -> str:
        if psi is None:
            return "unavailable"
        if psi < 0.10:
            return "stable"
        if psi < 0.20:
            return "monitor"
        if psi < 0.25:
            return "drift_detected"
        return "high_drift"

    def _insufficient_history(self, city: str, sample_size: int) -> DriftAssessment:
        return DriftAssessment(
            status="insufficient_history",
            city=city,
            sample_size=sample_size,
            minimum_history_rows=self.minimum_history_rows,
            level=None,
            overall_psi=None,
            drifting_feature_count=0,
            features=[],
            reference=self.info(),
            message=(
                f"At least {self.minimum_history_rows} complete historical feature rows are "
                f"required; only {sample_size} are available."
            ),
        )

    @staticmethod
    def _single_city(rows: pd.DataFrame) -> Optional[str]:
        if "city_key" not in rows.columns:
            return None
        values = rows["city_key"].dropna().astype(str).unique()
        return str(values[0]) if len(values) == 1 else None

    def _validate_baseline(self) -> None:
        required = {"baseline_version", "reference", "monitoring_features", "cities"}
        missing = required.difference(self._baseline)
        if missing:
            raise ValueError(f"Invalid drift baseline; missing: {sorted(missing)}")
        if not self._baseline["monitoring_features"]:
            raise ValueError("Drift baseline has no monitoring features")

"""
ClimateGuard Prediction Interface — predictor.py
Phase 15

Provides ClimateGuardPredictor: a clean, importable Python class that wraps
the Phase 14 finalised Random Forest model.

Design principles
-----------------
- Load model and feature list once per instance (not per call).
- Validate inputs strictly — never silently fill or reorder missing features.
- Expose the raw model and feature list for Part 2 explainability use.
- Work via project-relative paths; no hard-coded absolute paths.
- No training, no threshold changes, no dataset modifications.

Prediction contract
-------------------
Input  : 110 features in the exact order from models/final/feature_list.json
Output : prediction_probability (float [0,1]),  prediction_label (0 or 1)
         prediction_label = 1  if  probability >= THRESHOLD (0.70)

Usage
-----
    from src.prediction import ClimateGuardPredictor

    predictor = ClimateGuardPredictor()

    # Single prediction from a dict
    result = predictor.predict({"apparent_temperature_max": 42.1, ...})

    # Single prediction from a 1-row DataFrame
    result = predictor.predict(df.iloc[[0]])

    # Batch prediction
    results_df = predictor.predict_batch(df)

    # Probability only
    prob = predictor.predict_probability(features)

    # Access underlying model for SHAP / explainability
    model        = predictor.model
    feature_names = predictor.feature_names
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — locked from Phase 14 (do not change)
# ---------------------------------------------------------------------------
THRESHOLD: float = 0.70
N_FEATURES: int = 110
TARGET_NAME: str = "heatwave_next_day"

# Resolve project root as the directory two levels above this file
# (this file lives at  <project_root>/src/prediction/predictor.py)
_THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent

DEFAULT_MODEL_PATH       = PROJECT_ROOT / "models" / "final" / "climateguard_final_model.joblib"
DEFAULT_FEATURE_LIST_PATH = PROJECT_ROOT / "models" / "final" / "feature_list.json"

# Pass-through identity columns that are NOT model features
_PASSTHROUGH_COLS = {"city", "city_key", "date", "state", "region_type",
                     "heatwave", "hw_event_id", "hw_event_start",
                     "hw_event_end", "hw_event_length", TARGET_NAME}


# ---------------------------------------------------------------------------
# PredictionResult — single-row result container
# ---------------------------------------------------------------------------

class PredictionResult:
    """
    Container for a single prediction result.

    Attributes
    ----------
    prediction_probability : float
        Model confidence that tomorrow is a heatwave day (range [0.0, 1.0]).
    prediction_label : int
        Binary decision: 1 = heatwave predicted for tomorrow, 0 = no heatwave.
        Applied threshold: 0.70.
    city : str or None
        City identifier if provided in the input.
    date : str or None
        Date string if provided in the input (represents day T; prediction is for T+1).
    threshold : float
        Threshold used for binarisation (always 0.70).
    """

    def __init__(
        self,
        prediction_probability: float,
        prediction_label: int,
        city: Optional[str] = None,
        date: Optional[str] = None,
    ):
        self.prediction_probability = float(prediction_probability)
        self.prediction_label = int(prediction_label)
        self.city = city
        self.date = date
        self.threshold = THRESHOLD

    def to_dict(self) -> Dict:
        """Return result as a plain dictionary."""
        d: Dict = {
            "prediction_probability": self.prediction_probability,
            "prediction_label": self.prediction_label,
            "threshold": self.threshold,
        }
        if self.city is not None:
            d["city"] = self.city
        if self.date is not None:
            d["date"] = self.date
        return d

    def __repr__(self) -> str:
        label_str = "HEATWAVE" if self.prediction_label == 1 else "normal"
        loc = ""
        if self.city:
            loc += f" city={self.city}"
        if self.date:
            loc += f" date={self.date}"
        return (
            f"PredictionResult({label_str},{loc} "
            f"prob={self.prediction_probability:.4f} "
            f"thresh={self.threshold})"
        )


# ---------------------------------------------------------------------------
# ClimateGuardPredictor
# ---------------------------------------------------------------------------

class ClimateGuardPredictor:
    """
    Wraps the Phase 14 Random Forest model for 1-day-ahead heatwave prediction.

    Loads the model and feature list once on construction.  All prediction
    calls reuse the loaded model — no file I/O on each call.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the joblib model file.
        Defaults to ``models/final/climateguard_final_model.joblib``
        relative to the project root.
    feature_list_path : str or Path, optional
        Path to the feature_list.json file.
        Defaults to ``models/final/feature_list.json``
        relative to the project root.

    Raises
    ------
    FileNotFoundError
        If the model or feature list file does not exist.
    ValueError
        If the loaded model does not expect exactly 110 features, or if
        the feature list length does not match.
    RuntimeError
        If the model cannot be loaded.

    Attributes
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        The loaded final model (read-only; do not retrain).
    feature_names : list of str
        Ordered list of 110 feature names exactly as the model expects them.
    threshold : float
        Decision threshold (0.70).
    n_features : int
        Expected feature count (110).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        feature_list_path: Optional[Union[str, Path]] = None,
    ):
        mp  = Path(model_path)        if model_path        else DEFAULT_MODEL_PATH
        flp = Path(feature_list_path) if feature_list_path else DEFAULT_FEATURE_LIST_PATH

        self._model_path        = mp
        self._feature_list_path = flp
        self.threshold          = THRESHOLD
        self.n_features         = N_FEATURES

        self._load_feature_list(flp)
        self._load_model(mp)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_feature_list(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Feature list not found: {path}\n"
                "Expected: models/final/feature_list.json  — has Phase 14 been completed?"
            )
        with open(path, encoding="utf-8") as f:
            raw: List[Dict] = json.load(f)

        if len(raw) != N_FEATURES:
            raise ValueError(
                f"Feature list has {len(raw)} entries; expected {N_FEATURES}."
            )

        # Validate that indices are 0-based sequential
        for i, entry in enumerate(raw):
            if entry["index"] != i:
                raise ValueError(
                    f"Feature list index mismatch at position {i}: "
                    f"got index {entry['index']}, expected {i}."
                )

        self.feature_names: List[str] = [entry["name"] for entry in raw]
        self._feature_dtypes: Dict[str, str] = {
            entry["name"]: entry.get("dtype", "float64") for entry in raw
        }

        # Sanity: target must not be in feature list
        if TARGET_NAME in self.feature_names:
            raise ValueError(
                f"Target column '{TARGET_NAME}' found in feature list. "
                "This indicates a data-leakage issue."
            )

    def _load_model(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Expected: models/final/climateguard_final_model.joblib  — "
                "has Phase 14 been completed?"
            )
        try:
            self.model = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from {path}: {exc}"
            ) from exc

        actual = getattr(self.model, "n_features_in_", None)
        if actual is None:
            raise RuntimeError(
                "Loaded object does not appear to be a fitted sklearn model "
                f"(no n_features_in_ attribute). File: {path}"
            )
        if actual != N_FEATURES:
            raise ValueError(
                f"Model expects {actual} features; contract requires {N_FEATURES}. "
                "Ensure models/final/climateguard_final_model.joblib is the Phase 14 artifact."
            )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_and_extract(
        self, data: Union[Dict, pd.DataFrame]
    ) -> tuple[np.ndarray, Optional[str], Optional[str]]:
        """
        Validate input, extract feature matrix and optional metadata.

        Returns
        -------
        X : np.ndarray, shape (n_rows, 110)
        city : str or None
        date : str or None
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = data.to_frame().T.reset_index(drop=True)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(
                f"Input must be a dict, pd.Series, or pd.DataFrame; got {type(data).__name__}."
            )

        # Extract optional metadata before feature validation
        city = str(df["city_key"].iloc[0]) if "city_key" in df.columns else (
               str(df["city"].iloc[0])     if "city"     in df.columns else None)
        date = str(df["date"].iloc[0])     if "date"     in df.columns else None

        # Check all required features are present
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature(s): {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
                + "\nAll 110 features from models/final/feature_list.json are required."
            )

        # Extract feature columns in the exact required order
        X_df = df[self.feature_names]

        # Check for NaN values
        nan_counts = X_df.isna().sum()
        nan_features = nan_counts[nan_counts > 0]
        if len(nan_features) > 0:
            raise ValueError(
                f"NaN values found in {len(nan_features)} feature(s): "
                f"{list(nan_features.index)[:5]}"
                + (" ..." if len(nan_features) > 5 else "")
                + "\nThe model cannot process missing values. "
                "Ensure all features are computed before calling predict."
            )

        # Check numeric types
        non_numeric = [
            col for col in self.feature_names
            if not pd.api.types.is_numeric_dtype(X_df[col])
        ]
        if non_numeric:
            raise ValueError(
                f"Non-numeric values in feature(s): {non_numeric[:5]}"
                + (" ..." if len(non_numeric) > 5 else "")
            )

        return X_df.values.astype(np.float64), city, date

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    def predict_probability(
        self, features: Union[Dict, pd.Series, pd.DataFrame]
    ) -> Union[float, np.ndarray]:
        """
        Return raw heatwave probability/probabilities (no threshold applied).

        Parameters
        ----------
        features : dict, pd.Series, or pd.DataFrame
            Input features. Must contain all 110 required columns.

        Returns
        -------
        float
            If input is a dict or single-row DataFrame/Series — scalar probability.
        np.ndarray of shape (n_rows,)
            If input is a multi-row DataFrame.
        """
        X, _, _ = self._validate_and_extract(features)
        probs = self.model.predict_proba(X)[:, 1]
        return float(probs[0]) if probs.shape[0] == 1 else probs

    def predict(
        self, features: Union[Dict, pd.Series, pd.DataFrame]
    ) -> PredictionResult:
        """
        Run a single prediction and return a PredictionResult.

        Applies threshold=0.70 to determine the binary label.

        Parameters
        ----------
        features : dict, pd.Series, or 1-row pd.DataFrame
            Input features. Must contain all 110 required columns.
            May optionally contain 'city', 'city_key', and/or 'date' columns
            which will be included in the result metadata (not passed to model).

        Returns
        -------
        PredictionResult
            .prediction_probability : float in [0, 1]
            .prediction_label       : 0 or 1
            .city                   : str or None
            .date                   : str or None
            .threshold              : 0.70

        Raises
        ------
        ValueError
            If input is multi-row. Use predict_batch() for multiple rows.
        """
        X, city, date = self._validate_and_extract(features)
        if X.shape[0] != 1:
            raise ValueError(
                f"predict() accepts exactly 1 row; got {X.shape[0]}. "
                "Use predict_batch() for multiple rows."
            )
        prob  = float(self.model.predict_proba(X)[:, 1][0])
        label = int(prob >= self.threshold)
        return PredictionResult(prob, label, city=city, date=date)

    def predict_batch(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run predictions on a batch DataFrame.

        Does NOT modify the original DataFrame in-place.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain all 110 required feature columns.
            May optionally contain identity columns (city, city_key, date, etc.)
            which are preserved in the output.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with all original columns preserved plus:
              - prediction_probability : float
              - prediction_label       : int (0 or 1)

        Raises
        ------
        ValueError / TypeError
            On invalid input (missing features, NaNs, wrong types).
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"predict_batch() requires a pd.DataFrame; got {type(data).__name__}."
            )
        if len(data) == 0:
            raise ValueError("predict_batch() received an empty DataFrame.")

        X, _, _ = self._validate_and_extract(data)
        probs  = self.model.predict_proba(X)[:, 1]
        labels = (probs >= self.threshold).astype(int)

        result = data.copy()
        result["prediction_probability"] = probs
        result["prediction_label"]       = labels
        return result

    # ------------------------------------------------------------------
    # Explainability access (for Part 2 / Kshitij)
    # ------------------------------------------------------------------

    @property
    def feature_array_names(self) -> List[str]:
        """
        Alias for feature_names — explicit list for SHAP column alignment.
        Returns the ordered list of 110 feature names exactly as the model expects.
        """
        return self.feature_names

    def get_feature_matrix(
        self, data: Union[Dict, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Return the validated, ordered feature DataFrame (110 columns) without
        running a prediction. Useful for SHAP or other explainability tools.

        Parameters
        ----------
        data : dict or pd.DataFrame
            Input containing all 110 required columns.

        Returns
        -------
        pd.DataFrame of shape (n_rows, 110) with columns in model order.
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        X, _, _ = self._validate_and_extract(data)
        return pd.DataFrame(X, columns=self.feature_names)

    # ------------------------------------------------------------------
    # Info / repr
    # ------------------------------------------------------------------

    def info(self) -> Dict:
        """Return a summary dictionary of the predictor configuration."""
        return {
            "model_type": type(self.model).__name__,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "target": TARGET_NAME,
            "model_path": str(self._model_path),
            "feature_list_path": str(self._feature_list_path),
            "model_params": self.model.get_params(),
        }

    def __repr__(self) -> str:
        return (
            f"ClimateGuardPredictor("
            f"model={type(self.model).__name__}, "
            f"n_features={self.n_features}, "
            f"threshold={self.threshold})"
        )

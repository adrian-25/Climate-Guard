"""
ClimateGuard Explainability — explainer.py
Part 2

Provides per-prediction feature importance explanations for the Part 1
Random Forest model.

Two explanation modes
---------------------
1. SHAP (preferred, requires `shap` package):
   Uses shap.TreeExplainer to compute per-prediction SHAP values.
   SHAP values represent each feature's marginal contribution to the
   model's predicted probability for this specific input.
   Direction is derived from the sign of the SHAP value.

2. Global RF Feature Importance (fallback, always available):
   Uses the Random Forest model's built-in `feature_importances_` array.
   This is the GLOBAL mean-decrease-in-impurity importance across all
   training trees — NOT a per-prediction explanation.

   ⚠️ IMPORTANT: Global feature importance is NOT the same as SHAP values.
   It shows which features were generally important to the model during
   training, but CANNOT tell you why the model made a specific prediction.
   The fallback explanation is clearly labelled with "method": "global_rf_importance"
   to prevent misinterpretation.

SHAP installation
-----------------
SHAP is an optional dependency.  Install it with:

    pip install shap

If shap is not installed, the explainer falls back to global RF importance
automatically.  No error is raised.

Scientific language contract
-----------------------------
- Explanations reflect model behaviour only.
- They do NOT prove causality.
- Feature contributions "contributed to the model prediction" — they did
  NOT "cause the heatwave".
- qualifying_day is correlated with the target by construction (same IMD
  threshold family).  It must not be described as an independently discovered
  causal feature.  This limitation is documented in the output.

Feature contract
-----------------
All 110 feature names are taken from ClimateGuardPredictor.feature_names.
Feature names and order are NEVER altered.

Usage
-----
    from src.prediction import ClimateGuardPredictor
    from src.explainability import ClimateGuardExplainer

    predictor = ClimateGuardPredictor()
    explainer = ClimateGuardExplainer(predictor)

    # Single-row DataFrame (110 features, same format as predictor)
    explanation = explainer.explain(row_df, top_n=10)

    print(explanation.method)          # "shap" or "global_rf_importance"
    print(explanation.top_features)    # list of dicts with feature/value/contribution/direction
    print(explanation.to_dict())

    # Check whether SHAP is available
    print(ClimateGuardExplainer.shap_available())
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional SHAP import — clean degradation if not installed
# ---------------------------------------------------------------------------
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _shap = None          # type: ignore[assignment]
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Qualifying-day limitation note
# ---------------------------------------------------------------------------

_QUALIFYING_DAY_NOTE = (
    "qualifying_day encodes the same IMD-inspired threshold criteria used to "
    "construct the heatwave label (Tmax >= 40 °C + departure >= 4.5 °C for plains; "
    "Tmax >= 37 °C for coastal cities).  A high contribution from qualifying_day "
    "reflects this feature engineering choice, NOT an independently discovered "
    "physical signal.  See docs/explainability.md for the full limitation."
)

_CAUSALITY_NOTE = (
    "Explanation reflects model behaviour only.  "
    "Feature contributions indicate what drove this model prediction — "
    "they do NOT imply causality or prove that these features caused a heatwave."
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ExplainabilityResult:
    """
    Container for a single-row explainability result.

    Attributes
    ----------
    prediction : int
        Binary prediction label (0 or 1) from Part 1.
    probability : float
        Heatwave probability from Part 1.
    method : str
        Explanation method used: "shap" or "global_rf_importance".
    top_features : list of dict
        Up to `top_n` features sorted by absolute contribution.
        Each dict has:
            feature     : str   — feature name
            value       : float — feature value for this row (NaN for global fallback)
            contribution: float — SHAP value or normalised global importance
            direction   : str   — "increases_risk", "decreases_risk", or "global_importance"
    n_top : int
        Number of features returned.
    shap_available : bool
        Whether SHAP was available at explanation time.
    qualifying_day_note : str
        Required disclaimer about qualifying_day.
    causality_note : str
        Required disclaimer about model behaviour vs causality.
    shap_base_value : float or None
        SHAP expected-value baseline (SHAP mode only; None otherwise).

    Notes
    -----
    When method == "global_rf_importance":
        - contributions are the model's mean-decrease-in-impurity importances
          (normalised to sum to 1.0 across all 110 features).
        - direction is always "global_importance" — NOT per-prediction direction.
        - feature values are the actual values from the input row.
        - This is a global model summary, NOT a local per-prediction explanation.
    """

    def __init__(
        self,
        prediction: int,
        probability: float,
        method: str,
        top_features: List[Dict],
        shap_available: bool,
        shap_base_value: Optional[float] = None,
    ):
        self.prediction = int(prediction)
        self.probability = float(probability)
        self.method = method
        self.top_features = top_features
        self.n_top = len(top_features)
        self.shap_available = shap_available
        self.shap_base_value = shap_base_value
        self.qualifying_day_note = _QUALIFYING_DAY_NOTE
        self.causality_note = _CAUSALITY_NOTE

    def to_dict(self) -> Dict:
        """Return as a plain dictionary suitable for JSON serialisation."""
        d: Dict = {
            "prediction": self.prediction,
            "probability": self.probability,
            "explanation_method": self.method,
            "shap_available": self.shap_available,
            "top_features": self.top_features,
        }
        if self.shap_base_value is not None:
            d["shap_base_value"] = self.shap_base_value
        d["notes"] = {
            "causality": self.causality_note,
            "qualifying_day": self.qualifying_day_note,
        }
        if self.method == "global_rf_importance":
            d["global_importance_warning"] = (
                "This explanation uses GLOBAL Random Forest feature importance "
                "computed over all training trees.  It reflects general model behaviour, "
                "NOT what drove this specific prediction.  "
                "Install the 'shap' package for per-prediction SHAP explanations."
            )
        return d

    def __repr__(self) -> str:
        return (
            f"ExplainabilityResult("
            f"method={self.method}, "
            f"n_top={self.n_top}, "
            f"prob={self.probability:.4f}, "
            f"pred={self.prediction})"
        )


# ---------------------------------------------------------------------------
# ClimateGuardExplainer
# ---------------------------------------------------------------------------

class ClimateGuardExplainer:
    """
    Produces feature-level explanations for ClimateGuard predictions.

    Wraps the ClimateGuardPredictor's underlying model to provide
    per-prediction SHAP explanations (or a clearly labelled global
    feature importance fallback when SHAP is not installed).

    Parameters
    ----------
    predictor : ClimateGuardPredictor
        The loaded Part 1 predictor.  The explainer reads predictor.model
        and predictor.feature_names.  It does NOT retrain or modify the model.

    Notes
    -----
    - The explainer is NOT a second model.  It reads the model from the
      provided predictor instance.
    - Feature names and order come from predictor.feature_names (110 names).
    - No features are renamed, reordered, added, or removed.
    - SHAP values are per-prediction local explanations.
    - Global RF importance is a training-time global summary — clearly
      distinguished from SHAP in all outputs.

    SHAP installation
    -----------------
    Install with:  pip install shap

    If not installed, the explainer falls back to global RF importance
    automatically.  No error is raised.
    """

    def __init__(self, predictor) -> None:
        """
        Parameters
        ----------
        predictor : ClimateGuardPredictor
            Must have .model (RandomForestClassifier) and .feature_names (list of 110 str).
        """
        # Validate predictor interface
        if not hasattr(predictor, "model") or not hasattr(predictor, "feature_names"):
            raise TypeError(
                "predictor must be a ClimateGuardPredictor instance with .model "
                "and .feature_names attributes."
            )
        if len(predictor.feature_names) != 110:
            raise ValueError(
                f"predictor.feature_names must have 110 entries; "
                f"got {len(predictor.feature_names)}."
            )

        self._predictor = predictor
        self._model = predictor.model
        self._feature_names: List[str] = predictor.feature_names

        # Pre-build the SHAP explainer once if available
        self._shap_explainer = None
        if _SHAP_AVAILABLE:
            try:
                self._shap_explainer = _shap.TreeExplainer(self._model)
            except Exception as exc:
                warnings.warn(
                    f"SHAP TreeExplainer could not be initialised ({exc}). "
                    "Falling back to global RF feature importance.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @staticmethod
    def shap_available() -> bool:
        """Return True if the shap package is importable."""
        return _SHAP_AVAILABLE

    def explain(
        self,
        features: Union[pd.DataFrame, dict],
        prediction: Optional[int] = None,
        probability: Optional[float] = None,
        top_n: int = 10,
    ) -> ExplainabilityResult:
        """
        Explain a single prediction.

        Parameters
        ----------
        features : pd.DataFrame or dict
            One row of input features (110 columns, same format as the predictor).
            If a dict is provided, it is converted to a single-row DataFrame.
        prediction : int, optional
            The Part 1 prediction label (0 or 1).  If None, the predictor is
            called to obtain it.
        probability : float, optional
            The Part 1 probability.  If None, the predictor is called to
            obtain it.
        top_n : int, optional
            Number of top features to return, sorted by absolute contribution.
            Defaults to 10.  Must be in [1, 110].

        Returns
        -------
        ExplainabilityResult
            Contains top features, method label, disclaimers, and optional
            SHAP base value.

        Raises
        ------
        ValueError
            If top_n is out of [1, 110] or if input has wrong shape.
        TypeError
            If features is not a DataFrame or dict.
        """
        if top_n < 1 or top_n > 110:
            raise ValueError(f"top_n must be in [1, 110]; got {top_n}.")

        # Normalise input to a validated single-row DataFrame
        X_df = self._predictor.get_feature_matrix(features)
        if X_df.shape[0] != 1:
            raise ValueError(
                f"explain() accepts exactly 1 row; got {X_df.shape[0]}. "
                "Call explain() once per row."
            )

        # Obtain prediction and probability if not supplied
        if probability is None or prediction is None:
            result = self._predictor.predict(features)
            probability = result.prediction_probability
            prediction = result.prediction_label

        # Choose explanation method
        if self._shap_explainer is not None:
            return self._explain_shap(X_df, prediction, probability, top_n)
        else:
            return self._explain_global_importance(X_df, prediction, probability, top_n)

    def explain_global(self, top_n: int = 20) -> List[Dict]:
        """
        Return the model's global feature importance (all features or top_n).

        This is the GLOBAL mean-decrease-in-impurity importance across all
        training trees.  It is NOT a per-prediction explanation.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to return.  Defaults to 20.

        Returns
        -------
        list of dict, each with:
            rank        : int
            feature     : str
            importance  : float (normalised, sums to 1.0 across all 110)
            note        : str
        """
        importances = self._model.feature_importances_  # shape (110,)
        idx_sorted = np.argsort(importances)[::-1][:top_n]

        result = []
        for rank, idx in enumerate(idx_sorted, start=1):
            result.append({
                "rank": rank,
                "feature": self._feature_names[idx],
                "importance": float(importances[idx]),
                "note": (
                    "Global mean-decrease-in-impurity importance from training. "
                    "NOT a per-prediction SHAP value."
                ),
            })
        return result

    # ------------------------------------------------------------------
    # Internal explanation methods
    # ------------------------------------------------------------------

    def _explain_shap(
        self,
        X_df: pd.DataFrame,
        prediction: int,
        probability: float,
        top_n: int,
    ) -> ExplainabilityResult:
        """Compute SHAP values for the positive class."""
        X_arr = X_df.values  # shape (1, 110)

        # shap_values is a list of 2 arrays [class0, class1] for binary classification
        shap_values = self._shap_explainer.shap_values(X_arr)

        # For binary RF: shap_values is [neg_class_array, pos_class_array]
        # Each array has shape (n_samples, n_features)
        if isinstance(shap_values, list):
            sv_pos = shap_values[1][0]  # positive class (heatwave), first row
        else:
            # Some SHAP versions return a single array for binary classification
            sv_pos = shap_values[0]

        # Get SHAP base value for the positive class
        try:
            if hasattr(self._shap_explainer, "expected_value"):
                ev = self._shap_explainer.expected_value
                base_val = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
            else:
                base_val = None
        except Exception:
            base_val = None

        # Sort by absolute SHAP value
        feature_values = X_arr[0]  # shape (110,)
        abs_sv = np.abs(sv_pos)
        idx_sorted = np.argsort(abs_sv)[::-1][:top_n]

        top_features = []
        for idx in idx_sorted:
            sv = float(sv_pos[idx])
            direction = (
                "increases_risk" if sv > 0
                else "decreases_risk" if sv < 0
                else "neutral"
            )
            top_features.append({
                "feature": self._feature_names[idx],
                "value": float(feature_values[idx]),
                "contribution": sv,
                "direction": direction,
            })

        return ExplainabilityResult(
            prediction=prediction,
            probability=probability,
            method="shap",
            top_features=top_features,
            shap_available=True,
            shap_base_value=base_val,
        )

    def _explain_global_importance(
        self,
        X_df: pd.DataFrame,
        prediction: int,
        probability: float,
        top_n: int,
    ) -> ExplainabilityResult:
        """
        Fall back to global RF feature importance.

        ⚠️ This is a GLOBAL model summary, NOT a per-prediction explanation.
        Every input row will produce the same ranked feature list.
        Clearly labelled in the output as "global_rf_importance".
        """
        importances = self._model.feature_importances_  # normalised, sums to ~1.0
        feature_values = X_df.values[0]  # actual values from this row
        idx_sorted = np.argsort(importances)[::-1][:top_n]

        top_features = []
        for idx in idx_sorted:
            top_features.append({
                "feature": self._feature_names[idx],
                "value": float(feature_values[idx]),
                "contribution": float(importances[idx]),
                "direction": "global_importance",
            })

        return ExplainabilityResult(
            prediction=prediction,
            probability=probability,
            method="global_rf_importance",
            top_features=top_features,
            shap_available=False,
            shap_base_value=None,
        )

    def __repr__(self) -> str:
        mode = "shap" if self._shap_explainer is not None else "global_rf_importance"
        return (
            f"ClimateGuardExplainer("
            f"method={mode}, "
            f"n_features={len(self._feature_names)})"
        )

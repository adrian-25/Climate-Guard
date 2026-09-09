"""
ClimateGuard ETL — transformer.py
Part 3

Transforms validated climate data into a format accepted by the Part 1
ClimateGuardPredictor.

Design principle
----------------
The transformer does NOT re-implement Part 1 feature engineering.
Its role is narrower:

    - Accept records that already contain the 110 engineered features
      (e.g. rows from the project's feature CSVs or pre-processed feeds).
    - Select, reorder, and cast those columns to match the exact feature
      contract expected by ClimateGuardPredictor.
    - Attach optional metadata columns (city_key, date) that are passed
      through but not fed to the model.

For users who provide raw weather observations without pre-built features,
the transformer provides a thin passthrough layer and refers them to the
project's feature_engineering.py pipeline (Phase 7 artifact) — the
transformer does NOT duplicate that logic.

Why not re-implement feature engineering here?
----------------------------------------------
Part 1's feature engineering is a 35-year, multi-stage pipeline that
requires historical lag/rolling data per city.  Re-implementing it in Part 3
would create two divergent implementations and risk silent disagreements.
The correct approach is to call the existing pipeline.

Feature contract reference
--------------------------
models/final/feature_list.json — 110 features, ordered by index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent
FEATURE_LIST_PATH = PROJECT_ROOT / "models" / "final" / "feature_list.json"

# Pass-through columns that are never model inputs
_PASSTHROUGH_COLS = frozenset({
    "city", "city_key", "date", "state", "region_type",
    "heatwave_next_day", "heatwave", "hw_event_id",
    "hw_event_start", "hw_event_end", "hw_event_length",
    "actual_heatwave_next_day", "prediction_probability",
    "prediction_label",
})


# ---------------------------------------------------------------------------
# FeatureTransformer
# ---------------------------------------------------------------------------

class FeatureTransformer:
    """
    Selects and reorders columns to match the 110-feature contract for the
    Part 1 ClimateGuardPredictor.

    This transformer assumes the input already contains the 110 engineered
    feature columns (produced by Phase 7 feature engineering or an equivalent
    pre-processing step).  It does NOT compute lag, rolling, or anomaly
    features from scratch.

    Parameters
    ----------
    feature_list_path : Path or str, optional
        Path to feature_list.json.  Defaults to models/final/feature_list.json.

    Usage
    -----
        transformer = FeatureTransformer()
        feature_df = transformer.transform(input_df)
        # feature_df has exactly 110 columns in the exact required order.
    """

    def __init__(self, feature_list_path: Optional[Union[str, Path]] = None):
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
        """Ordered list of 110 required feature names."""
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def transform(
        self,
        data: Union[pd.DataFrame, Dict],
    ) -> pd.DataFrame:
        """
        Select and reorder columns to produce a feature DataFrame with exactly
        110 columns in the exact order required by ClimateGuardPredictor.

        Parameters
        ----------
        data : pd.DataFrame or dict
            Input containing all 110 required feature columns.
            May contain additional pass-through columns (city_key, date, etc.)
            which are stripped from the output.

        Returns
        -------
        pd.DataFrame
            Shape (n_rows, 110).  Columns are exactly self.feature_names.
            All values are float64.

        Raises
        ------
        ValueError
            If any required feature column is missing or contains NaN values.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(
                f"Input must be a dict or pd.DataFrame; got {type(data).__name__}."
            )

        # Check all required feature columns are present
        missing = [f for f in self._feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot transform: {len(missing)} required feature column(s) missing: "
                f"{missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
                + f"\nAll {self._n_features} features from feature_list.json must be present."
            )

        # Select features in exact required order, cast to float64
        feature_df = df[self._feature_names].copy()

        try:
            feature_df = feature_df.astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not cast feature columns to float64: {exc}\n"
                "All 110 feature columns must be numeric."
            ) from exc

        # Final NaN check after cast
        nan_counts = feature_df.isna().sum()
        nan_feats = list(nan_counts[nan_counts > 0].index)
        if nan_feats:
            raise ValueError(
                f"NaN values remain after transformation in {len(nan_feats)} "
                f"feature(s): {nan_feats[:10]}"
                + (" ..." if len(nan_feats) > 10 else "")
                + ". The model cannot process missing values."
            )

        return feature_df.reset_index(drop=True)

    def transform_with_metadata(
        self,
        data: Union[pd.DataFrame, Dict],
    ) -> pd.DataFrame:
        """
        Like transform(), but re-attaches city_key and date columns to the
        output DataFrame for downstream metadata access.

        The extra columns are placed AFTER the 110 feature columns and are
        NOT passed to the model (ClimateGuardPredictor ignores them).

        Returns
        -------
        pd.DataFrame
            Shape (n_rows, 110 + n_meta_cols).
        """
        if isinstance(data, dict):
            df_in = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df_in = data.copy()
        else:
            raise TypeError(
                f"Input must be a dict or pd.DataFrame; got {type(data).__name__}."
            )

        feature_df = self.transform(df_in)

        # Re-attach available metadata columns
        meta_cols = [c for c in ("city_key", "date") if c in df_in.columns]
        if meta_cols:
            meta_df = df_in[meta_cols].reset_index(drop=True)
            return pd.concat([feature_df, meta_df], axis=1)
        return feature_df

    def get_feature_count(self) -> int:
        """Return the number of required features (always 110)."""
        return self._n_features

    def __repr__(self) -> str:
        return f"FeatureTransformer(n_features={self._n_features})"

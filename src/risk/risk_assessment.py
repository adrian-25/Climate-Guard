"""
ClimateGuard Risk Assessment — risk_assessment.py
Part 2

Maps the ML model's heatwave probability to an operational risk level.

Risk thresholds (project-defined operational categories)
---------------------------------------------------------
These thresholds were defined for this project to translate a continuous
heatwave probability into an actionable risk level.  They are NOT official
India Meteorological Department (IMD) risk categories.

    [0.00, 0.30)  → LOW
    [0.30, 0.60)  → MODERATE
    [0.60, 0.80)  → HIGH
    [0.80, 1.00]  → EXTREME

The mapping is deterministic and does not alter the underlying ML probability
or prediction label produced by Part 1 (ClimateGuardPredictor).

Design notes
------------
- The input probability is always the raw output of the Random Forest model.
- The Part 1 prediction label (threshold = 0.70) is preserved verbatim.
- Risk level is derived independently from probability; it is a communication
  aid and does NOT replace the ML prediction.
- Boundary values are handled inclusively on the lower bound:
    0.30 → MODERATE  (not LOW)
    0.60 → HIGH      (not MODERATE)
    0.80 → EXTREME   (not HIGH)

Usage
-----
    from src.risk import RiskAssessor

    assessor = RiskAssessor()

    result = assessor.assess(
        probability=0.82,
        prediction=1,
        city="delhi",
        date="2024-05-20",
    )

    print(result.risk_level)          # "EXTREME"
    print(result.to_dict())
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Risk level enum
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """
    Operational heatwave risk levels for ClimateGuard.

    These are project-defined categories based on the Random Forest model's
    probability output.  They are NOT official IMD risk categories.

    Values are ordered low→extreme; string comparisons work directly.
    """
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    EXTREME  = "EXTREME"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Risk thresholds — project-defined, document clearly
# ---------------------------------------------------------------------------

#: Lower bounds (inclusive) for each risk level.
#: Source: project-defined operational thresholds for ClimateGuard.
#: NOT official IMD categories.
RISK_THRESHOLDS = [
    (0.80, RiskLevel.EXTREME),
    (0.60, RiskLevel.HIGH),
    (0.30, RiskLevel.MODERATE),
    (0.00, RiskLevel.LOW),
]


def probability_to_risk_level(probability: float) -> RiskLevel:
    """
    Map a heatwave probability to a risk level.

    Parameters
    ----------
    probability : float
        ML-model heatwave probability in [0.0, 1.0].

    Returns
    -------
    RiskLevel
        Deterministic mapping:
            [0.00, 0.30) → LOW
            [0.30, 0.60) → MODERATE
            [0.60, 0.80) → HIGH
            [0.80, 1.00] → EXTREME

    Raises
    ------
    ValueError
        If probability is outside [0.0, 1.0] or is not a numeric type.
    TypeError
        If probability is not a numeric type.

    Notes
    -----
    Thresholds are project-defined for ClimateGuard.  They are NOT official
    IMD risk categories.
    """
    # Type check
    if not isinstance(probability, (int, float)):
        raise TypeError(
            f"probability must be a numeric type (int or float); "
            f"got {type(probability).__name__}."
        )

    prob = float(probability)

    # NaN check — float("nan") passes the range test but must be rejected
    if prob != prob:  # NaN is the only float that is not equal to itself
        raise ValueError(
            f"probability must be a valid finite number; got NaN. "
            "Ensure the value comes from the ClimateGuardPredictor."
        )

    # Range check — allow exactly 0.0 and 1.0
    if prob < 0.0 or prob > 1.0:
        raise ValueError(
            f"probability must be in [0.0, 1.0]; got {prob:.6f}. "
            "Ensure the value comes from the ClimateGuardPredictor."
        )

    for threshold, level in RISK_THRESHOLDS:
        if prob >= threshold:
            return level

    # Unreachable in normal operation (0.0 >= 0.00 always matches LOW)
    return RiskLevel.LOW  # pragma: no cover


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class RiskAssessmentResult:
    """
    Container for a single risk assessment result.

    Attributes
    ----------
    city : str or None
        City identifier (e.g. "delhi").  None if not provided.
    date : str or None
        Date string for the observed day T.  The prediction is for T+1.
        None if not provided.
    heatwave_probability : float
        Raw ML model probability from Part 1. Range [0.0, 1.0].
    prediction : int
        Binary prediction label from Part 1 (0 or 1, threshold = 0.70).
    risk_level : str
        One of: "LOW", "MODERATE", "HIGH", "EXTREME".
        Derived from heatwave_probability using project-defined thresholds.

    Notes
    -----
    risk_level is a project-defined operational category.
    It is NOT an official IMD risk category.
    """

    def __init__(
        self,
        heatwave_probability: float,
        prediction: int,
        risk_level: RiskLevel,
        city: Optional[str] = None,
        date: Optional[str] = None,
    ):
        self.heatwave_probability = float(heatwave_probability)
        self.prediction = int(prediction)
        self.risk_level = str(risk_level)
        self.city = city
        self.date = date

    def to_dict(self) -> dict:
        """Return the result as a plain dictionary."""
        d: dict = {
            "heatwave_probability": self.heatwave_probability,
            "prediction": self.prediction,
            "risk_level": self.risk_level,
        }
        if self.city is not None:
            d["city"] = self.city
        if self.date is not None:
            d["date"] = self.date
        return d

    def __repr__(self) -> str:
        loc = ""
        if self.city:
            loc += f" city={self.city}"
        if self.date:
            loc += f" date={self.date}"
        label_str = "HEATWAVE" if self.prediction == 1 else "normal"
        return (
            f"RiskAssessmentResult({label_str},{loc} "
            f"prob={self.heatwave_probability:.4f} "
            f"risk={self.risk_level})"
        )


# ---------------------------------------------------------------------------
# RiskAssessor
# ---------------------------------------------------------------------------

class RiskAssessor:
    """
    Converts Part 1 prediction outputs into an operational risk assessment.

    The assessor is stateless and does not load any model.  It consumes
    the probability and prediction label produced by ClimateGuardPredictor
    and adds a human-readable risk level.

    Risk level is determined solely by probability using project-defined
    thresholds.  The mapping is deterministic; identical inputs always
    produce identical outputs.

    Risk thresholds (project-defined, NOT official IMD categories):
        [0.00, 0.30) → LOW
        [0.30, 0.60) → MODERATE
        [0.60, 0.80) → HIGH
        [0.80, 1.00] → EXTREME

    Usage
    -----
        from src.risk import RiskAssessor

        assessor = RiskAssessor()

        result = assessor.assess(
            probability=0.82,
            prediction=1,
            city="delhi",
            date="2024-05-20",
        )
        print(result.risk_level)          # EXTREME
        print(result.to_dict())
    """

    # Expose the threshold map for inspection / documentation
    THRESHOLDS = RISK_THRESHOLDS

    def assess(
        self,
        probability: float,
        prediction: int,
        city: Optional[str] = None,
        date: Optional[str] = None,
    ) -> RiskAssessmentResult:
        """
        Assess the risk level from a Part 1 prediction.

        Parameters
        ----------
        probability : float
            Heatwave probability from ClimateGuardPredictor (range [0.0, 1.0]).
        prediction : int
            Binary prediction label from ClimateGuardPredictor (0 or 1).
        city : str, optional
            City identifier (e.g. "delhi").
        date : str, optional
            Date string for day T.

        Returns
        -------
        RiskAssessmentResult
            Contains the original probability, original prediction label,
            the derived risk level, and optional metadata.

        Raises
        ------
        ValueError
            If probability is outside [0.0, 1.0].
        TypeError
            If probability is not a numeric type.
        """
        risk_level = probability_to_risk_level(probability)

        return RiskAssessmentResult(
            heatwave_probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            city=city,
            date=date,
        )

    @staticmethod
    def get_threshold_table() -> list:
        """
        Return the project-defined risk threshold table.

        Returns
        -------
        list of dict
            Each entry has keys: lower_bound_inclusive, upper_bound_exclusive, risk_level.

        Notes
        -----
        These are project-defined operational thresholds, NOT official IMD categories.
        """
        table = []
        for i, (lower, level) in enumerate(RISK_THRESHOLDS):
            upper = RISK_THRESHOLDS[i - 1][0] if i > 0 else 1.01  # 1.01 = inclusive 1.0
            table.append({
                "lower_bound_inclusive": lower,
                "upper_bound_exclusive": upper,
                "risk_level": str(level),
            })
        return table

    def __repr__(self) -> str:
        return "RiskAssessor(stateless, project-defined thresholds)"

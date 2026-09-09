"""
ClimateGuard Adaptation Recommendation Engine — recommendations.py
Part 2

Generates practical heatwave adaptation recommendations based on the
operational risk level produced by the Risk Assessment module.

Design principles
-----------------
- Recommendations are risk-level specific, concise, actionable, and non-alarmist.
- They are written to be understandable by the general public.
- They are NOT official government guidance or medical instructions.
  This must be communicated to end users.
- Categories cover the main domains of heatwave preparedness:
    Hydration, Outdoor Exposure, Cooling, Vulnerable Populations,
    Workplace, Public Awareness, Emergency Preparedness.
- The recommendation set is deterministic: identical risk level always
  produces identical output.

IMPORTANT DISCLAIMER
--------------------
These recommendations are project-defined operational guidance for the
ClimateGuard system.  They are NOT official government heat action plan
instructions, official IMD advisories, or certified medical advice.
Always follow guidance from local authorities during heat emergencies.

Usage
-----
    from src.adaptation import AdaptationEngine

    engine = AdaptationEngine()
    recs = engine.recommend("EXTREME")

    for r in recs:
        print(r.category, ":", r.message)

    # Or get dict output
    out = engine.recommend_as_dict("HIGH")
    # {
    #   "risk_level": "HIGH",
    #   "disclaimer": "...",
    #   "recommendations": [{"category": ..., "message": ...}, ...]
    # }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Disclaimer — must accompany all recommendation outputs
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "These are project-defined operational recommendations for the ClimateGuard "
    "early-warning system.  They are NOT official government heat action plan "
    "instructions, official India Meteorological Department advisories, or certified "
    "medical advice.  Always follow guidance from your local authorities and health "
    "professionals during heat emergencies."
)


# ---------------------------------------------------------------------------
# Recommendation data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recommendation:
    """
    A single adaptation recommendation.

    Attributes
    ----------
    category : str
        High-level category label (e.g. "Hydration", "Outdoor Exposure").
    message : str
        Concise, actionable guidance text.
    """
    category: str
    message:  str

    def to_dict(self) -> Dict[str, str]:
        return {"category": self.category, "message": self.message}


# ---------------------------------------------------------------------------
# Recommendation sets — keyed by risk level string
# ---------------------------------------------------------------------------

_RECOMMENDATIONS: Dict[str, List[Recommendation]] = {

    "LOW": [
        Recommendation(
            category="Hydration",
            message=(
                "Maintain your normal daily fluid intake.  "
                "Drinking adequate water throughout the day is good practice "
                "regardless of heat conditions."
            ),
        ),
        Recommendation(
            category="Outdoor Exposure",
            message=(
                "No unusual restrictions are indicated.  "
                "If outdoors during midday, light clothing and a hat are sensible precautions."
            ),
        ),
        Recommendation(
            category="Public Awareness",
            message=(
                "Monitor local weather forecasts.  "
                "Conditions may change; stay informed if temperatures are rising."
            ),
        ),
    ],

    "MODERATE": [
        Recommendation(
            category="Hydration",
            message=(
                "Increase your fluid intake, particularly water.  "
                "Avoid excessive caffeine or alcohol, which can contribute to dehydration "
                "in warm conditions."
            ),
        ),
        Recommendation(
            category="Outdoor Exposure",
            message=(
                "Reduce prolonged outdoor activity during the hottest part of the day "
                "(typically 11:00–16:00).  "
                "Take breaks in the shade and limit strenuous exertion."
            ),
        ),
        Recommendation(
            category="Cooling",
            message=(
                "Use fans, damp cloths, or cool showers if you feel overheated.  "
                "Keep indoor spaces reasonably ventilated."
            ),
        ),
        Recommendation(
            category="Vulnerable Populations",
            message=(
                "Check in on elderly neighbours, young children, and anyone with a "
                "chronic health condition.  "
                "These groups are more susceptible to heat-related illness."
            ),
        ),
        Recommendation(
            category="Public Awareness",
            message=(
                "Follow local weather updates and any heat-related advisories from "
                "your municipal or state authorities."
            ),
        ),
    ],

    "HIGH": [
        Recommendation(
            category="Hydration",
            message=(
                "Drink water frequently throughout the day — do not wait until you feel "
                "thirsty.  "
                "Oral rehydration salts or electrolyte drinks may help if sweating heavily."
            ),
        ),
        Recommendation(
            category="Outdoor Exposure",
            message=(
                "Avoid unnecessary outdoor exposure during peak heat hours (11:00–17:00).  "
                "If outdoor work is essential, schedule the most strenuous tasks for early "
                "morning or evening and ensure regular rest breaks in a cool, shaded area."
            ),
        ),
        Recommendation(
            category="Cooling",
            message=(
                "Use air conditioning, cooling centres, or other means to keep body "
                "temperature comfortable.  "
                "Wet the skin and use fans to aid evaporative cooling.  "
                "Keep curtains or blinds closed on sun-facing windows to reduce indoor heat."
            ),
        ),
        Recommendation(
            category="Vulnerable Populations",
            message=(
                "Actively monitor elderly people, infants, pregnant women, and those with "
                "pre-existing conditions (heart, kidney, or respiratory disease).  "
                "Ensure they have access to cool spaces and sufficient fluids."
            ),
        ),
        Recommendation(
            category="Workplace",
            message=(
                "Employers with outdoor or poorly ventilated indoor work environments "
                "should provide additional rest breaks, shaded rest areas, and ready access "
                "to water.  "
                "Reschedule non-essential outdoor tasks where possible."
            ),
        ),
        Recommendation(
            category="Public Awareness",
            message=(
                "Heatwave risk is elevated.  Share information about cooling locations and "
                "heat-illness warning signs (dizziness, nausea, confusion) with family, "
                "colleagues, and community members."
            ),
        ),
    ],

    "EXTREME": [
        Recommendation(
            category="Hydration",
            message=(
                "Drink water continuously throughout the day.  "
                "Aim for at least 2–3 litres unless medically restricted.  "
                "Avoid alcohol, caffeinated drinks, and sugary beverages that increase fluid loss."
            ),
        ),
        Recommendation(
            category="Outdoor Exposure",
            message=(
                "Minimise all unnecessary outdoor activity.  "
                "If you must go outside, keep exposure to the absolute minimum and seek "
                "shade or cool environments immediately after.  "
                "This applies to children and adults alike."
            ),
        ),
        Recommendation(
            category="Cooling",
            message=(
                "Access air-conditioned spaces — home, public cooling centres, shopping "
                "malls, or community centres — for as long as possible during the hottest "
                "hours.  "
                "Apply cool, damp cloths to wrists, neck, and forehead.  "
                "Block direct sunlight on all windows."
            ),
        ),
        Recommendation(
            category="Vulnerable Populations",
            message=(
                "Take immediate and active steps to ensure elderly individuals, infants, "
                "and those with serious health conditions are in cool environments.  "
                "Check on them frequently — at least every few hours.  "
                "Do not leave anyone alone in a hot, unventilated space."
            ),
        ),
        Recommendation(
            category="Workplace",
            message=(
                "Suspend or significantly curtail outdoor and physically demanding work "
                "during peak heat hours.  "
                "Ensure workers have frequent rest in cool environments, unlimited access "
                "to water, and are trained to recognise heat-illness symptoms."
            ),
        ),
        Recommendation(
            category="Emergency Preparedness",
            message=(
                "Know the location of the nearest hospital or medical centre.  "
                "Be aware of heat-illness warning signs: heavy sweating, rapid heartbeat, "
                "confusion, weakness, or fainting.  "
                "Seek medical attention promptly if any of these occur.  "
                "Contact local emergency services if someone becomes unresponsive."
            ),
        ),
        Recommendation(
            category="Public Awareness",
            message=(
                "An extreme heatwave risk has been indicated.  "
                "Share this information with neighbours and community members, especially "
                "those who may be isolated.  "
                "Follow all official heat-emergency guidance from your local, state, and "
                "national authorities immediately."
            ),
        ),
    ],
}


# ---------------------------------------------------------------------------
# AdaptationEngine
# ---------------------------------------------------------------------------

class AdaptationEngine:
    """
    Generates heatwave adaptation recommendations for a given risk level.

    The engine is stateless: it returns a fixed set of recommendations for
    each risk level.  All outputs are deterministic.

    IMPORTANT: These recommendations are project-defined operational guidance.
    They are NOT official government heat action plan instructions, official
    IMD advisories, or medical advice.  The disclaimer attribute carries the
    full disclaimer text and must accompany public-facing output.

    Usage
    -----
        from src.adaptation import AdaptationEngine

        engine = AdaptationEngine()

        # Returns a list of Recommendation objects
        recs = engine.recommend("EXTREME")
        for r in recs:
            print(r.category, ":", r.message)

        # Returns a dict with risk_level, disclaimer, and list of dicts
        out = engine.recommend_as_dict("HIGH")
    """

    #: The disclaimer text that should accompany all public-facing output.
    DISCLAIMER = DISCLAIMER

    #: Valid risk level strings accepted by this engine.
    VALID_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "EXTREME"}

    def recommend(
        self,
        risk_level: Union[str, "RiskLevel"],
        include_disclaimer: bool = False,
    ) -> List[Recommendation]:
        """
        Return a list of Recommendation objects for the given risk level.

        Parameters
        ----------
        risk_level : str or RiskLevel
            One of: "LOW", "MODERATE", "HIGH", "EXTREME".
            Case-insensitive.
        include_disclaimer : bool, optional
            Ignored for this return type (disclaimer is in recommend_as_dict).

        Returns
        -------
        list of Recommendation
            Non-empty list; length varies by risk level.

        Raises
        ------
        ValueError
            If risk_level is not one of the four valid levels.
        """
        level = str(risk_level).upper().strip()
        self._validate_risk_level(level)
        return _RECOMMENDATIONS[level]

    def recommend_as_dict(
        self,
        risk_level: Union[str, "RiskLevel"],
    ) -> Dict:
        """
        Return recommendations as a dictionary suitable for JSON serialisation.

        Parameters
        ----------
        risk_level : str or RiskLevel
            One of: "LOW", "MODERATE", "HIGH", "EXTREME".

        Returns
        -------
        dict with keys:
            risk_level      : str
            disclaimer      : str
            recommendations : list of {"category": str, "message": str}

        Raises
        ------
        ValueError
            If risk_level is not one of the four valid levels.
        """
        level = str(risk_level).upper().strip()
        self._validate_risk_level(level)
        return {
            "risk_level": level,
            "disclaimer": DISCLAIMER,
            "recommendations": [r.to_dict() for r in _RECOMMENDATIONS[level]],
        }

    def get_categories(self, risk_level: Union[str, "RiskLevel"]) -> List[str]:
        """
        Return the list of category names for a given risk level.

        Parameters
        ----------
        risk_level : str or RiskLevel

        Returns
        -------
        list of str
        """
        level = str(risk_level).upper().strip()
        self._validate_risk_level(level)
        return [r.category for r in _RECOMMENDATIONS[level]]

    def all_risk_levels(self) -> List[str]:
        """Return all valid risk level strings in order low→extreme."""
        return ["LOW", "MODERATE", "HIGH", "EXTREME"]

    @staticmethod
    def _validate_risk_level(level: str) -> None:
        if level not in {"LOW", "MODERATE", "HIGH", "EXTREME"}:
            raise ValueError(
                f"Invalid risk level: '{level}'. "
                f"Expected one of: LOW, MODERATE, HIGH, EXTREME."
            )

    def __repr__(self) -> str:
        return (
            "AdaptationEngine("
            "levels=[LOW, MODERATE, HIGH, EXTREME], "
            "stateless, project-defined recommendations)"
        )

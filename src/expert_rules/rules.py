"""
ClimateGuard Expert Rules — rules.py
Part 3

Defines the seven deterministic expert rules that add domain-context
warnings on top of the ML prediction from Part 1 and the risk assessment
from Part 2.

Design principles
-----------------
- Rules are DETERMINISTIC: identical inputs always produce identical outputs.
- Rules are TRANSPARENT: every threshold is documented with its rationale.
- Rules are INDEPENDENT: each rule is separately testable and identifiable.
- Rules do NOT replace the ML prediction; they are contextual overlays.
- Rules do NOT re-implement Part 2 recommendation logic; they complement it.

IMPORTANT DISCLAIMER
--------------------
All thresholds in this file are project-defined operational thresholds.
They are NOT official India Meteorological Department (IMD) warning thresholds,
NOT official government heat action plan triggers, and NOT certified medical
thresholds.  ClimateGuard expert rules are supplementary contextual guidance
for research/educational purposes only.

Structure of each rule
-----------------------
    rule_id     : str  — unique identifier (e.g. "RULE_01")
    name        : str  — short human-readable name
    description : str  — what the rule checks and why
    severity    : str  — "INFO" | "WARNING" | "CRITICAL"
    message     : str  — user-facing guidance when triggered
    triggered   : bool — whether the rule fired for this input
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# RuleResult — result of a single rule evaluation
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    """
    Result of evaluating one expert rule.

    Attributes
    ----------
    rule_id     : str   Rule identifier (e.g. "RULE_01")
    name        : str   Short rule name
    triggered   : bool  True if the rule condition was met
    severity    : str   "INFO" | "WARNING" | "CRITICAL"
    message     : str   User-facing guidance (non-empty when triggered)
    description : str   Rule description (always present)
    threshold_note : str  Documents the threshold source
    """
    rule_id: str
    name: str
    triggered: bool
    severity: str
    message: str
    description: str
    threshold_note: str = ""

    def to_dict(self) -> Dict:
        d = {
            "rule_id":    self.rule_id,
            "name":       self.name,
            "triggered":  self.triggered,
            "severity":   self.severity,
            "description": self.description,
        }
        if self.triggered:
            d["message"] = self.message
        if self.threshold_note:
            d["threshold_note"] = self.threshold_note
        return d

    def __repr__(self) -> str:
        status = "TRIGGERED" if self.triggered else "not triggered"
        return f"RuleResult({self.rule_id} [{self.severity}] {status})"


# ---------------------------------------------------------------------------
# Helper: safe float extraction
# ---------------------------------------------------------------------------

def _get_float(data: Dict, key: str, default: Optional[float] = None) -> Optional[float]:
    """Extract a float from a dict, returning default if missing or unconvertible."""
    val = data.get(key, default)
    if val is None:
        return default
    try:
        fval = float(val)
        if fval != fval:  # NaN check
            return default
        return fval
    except (TypeError, ValueError):
        return default


# ===========================================================================
# RULE 01 — Extreme Temperature
# ===========================================================================

#: Project-defined threshold for extreme temperature alert.
#: Plains cities (Delhi, Lucknow, Nagpur, Ahmedabad): 45°C is the IMD-inspired
#: "absolute severe heatwave" threshold.  We use it here as an extreme alert.
#: Coastal city (Mumbai): 40°C is analogously extreme given its lower baseline.
#: Source: aligned with IMD-inspired thresholds documented in
#:         docs/heatwave_labeling_methodology.md.
#: NOT an official IMD warning threshold — project-defined operational rule.
EXTREME_TEMP_THRESHOLD_PLAINS: float = 45.0   # °C
EXTREME_TEMP_THRESHOLD_COASTAL: float = 40.0  # °C

_COASTAL_CITIES: frozenset = frozenset({"mumbai"})
_PLAINS_CITIES:  frozenset = frozenset({"delhi", "lucknow", "nagpur", "ahmedabad"})


def evaluate_rule_01(data: Dict) -> RuleResult:
    """
    RULE_01 — Extreme Temperature

    Fires when temperature_2m_max exceeds the extreme threshold for the city type.

    Thresholds (project-defined, NOT official IMD):
        Plains cities : temperature_2m_max >= 45°C
        Coastal cities: temperature_2m_max >= 40°C

    Parameters
    ----------
    data : dict
        Must contain 'temperature_2m_max'.  May contain 'city_key' or 'city'.
    """
    tmax = _get_float(data, "temperature_2m_max")
    city = str(data.get("city_key") or data.get("city") or "").lower().strip()

    triggered = False
    threshold = EXTREME_TEMP_THRESHOLD_PLAINS
    if city in _COASTAL_CITIES:
        threshold = EXTREME_TEMP_THRESHOLD_COASTAL

    if tmax is not None and tmax >= threshold:
        triggered = True

    return RuleResult(
        rule_id="RULE_01",
        name="Extreme Temperature",
        triggered=triggered,
        severity="CRITICAL",
        message=(
            f"Extreme temperature detected: {tmax:.1f}°C "
            f"(threshold {threshold:.0f}°C for this city type). "
            "This exceeds the severe heatwave temperature boundary. "
            "Immediate precautionary measures are strongly advised."
        ) if triggered else "",
        description=(
            "Fires when today's maximum temperature exceeds the extreme threshold "
            f"({EXTREME_TEMP_THRESHOLD_PLAINS}°C for plains cities, "
            f"{EXTREME_TEMP_THRESHOLD_COASTAL}°C for coastal cities). "
            "Based on IMD-inspired criteria documented in the project methodology."
        ),
        threshold_note=(
            f"Project-defined: plains={EXTREME_TEMP_THRESHOLD_PLAINS}°C, "
            f"coastal={EXTREME_TEMP_THRESHOLD_COASTAL}°C. "
            "NOT an official IMD warning threshold."
        ),
    )


# ===========================================================================
# RULE 02 — Persistent Heat
# ===========================================================================

#: Persistent heat fires when today's Tmax AND the 7-day rolling mean Tmax
#: both exceed project-defined threshold values.
#: Rationale: multiple consecutive hot days increase physiological heat stress
#: beyond what a single hot day produces.
#: Threshold values are project-defined; NOT official IMD.
PERSISTENT_HEAT_TMAX_THRESHOLD: float   = 40.0  # °C — today's max
PERSISTENT_HEAT_ROLL7_THRESHOLD: float  = 37.0  # °C — 7-day rolling mean


def evaluate_rule_02(data: Dict) -> RuleResult:
    """
    RULE_02 — Persistent Heat

    Fires when today's temperature_2m_max >= 40°C AND the 7-day rolling mean
    (temperature_2m_max_roll7_mean) >= 37°C.

    Thresholds are project-defined. NOT official IMD.
    """
    tmax        = _get_float(data, "temperature_2m_max")
    roll7_mean  = _get_float(data, "temperature_2m_max_roll7_mean")

    triggered = False
    if (
        tmax is not None
        and roll7_mean is not None
        and tmax >= PERSISTENT_HEAT_TMAX_THRESHOLD
        and roll7_mean >= PERSISTENT_HEAT_ROLL7_THRESHOLD
    ):
        triggered = True

    return RuleResult(
        rule_id="RULE_02",
        name="Persistent Heat",
        triggered=triggered,
        severity="WARNING",
        message=(
            f"Persistent heat detected: today's Tmax={tmax:.1f}°C, "
            f"7-day rolling mean Tmax={roll7_mean:.1f}°C. "
            "Multiple consecutive hot days compound physiological heat stress. "
            "Vulnerable individuals should take extra precautions."
        ) if triggered else "",
        description=(
            f"Fires when temperature_2m_max >= {PERSISTENT_HEAT_TMAX_THRESHOLD}°C "
            f"AND temperature_2m_max_roll7_mean >= {PERSISTENT_HEAT_ROLL7_THRESHOLD}°C. "
            "Sustained high temperatures increase cumulative heat exposure risk."
        ),
        threshold_note=(
            f"Project-defined: Tmax>={PERSISTENT_HEAT_TMAX_THRESHOLD}°C AND "
            f"roll7_mean>={PERSISTENT_HEAT_ROLL7_THRESHOLD}°C. "
            "NOT official IMD thresholds."
        ),
    )


# ===========================================================================
# RULE 03 — High Nighttime Temperature
# ===========================================================================

#: Rationale: when nighttime temperatures remain high, the body cannot recover
#: from daytime heat stress, increasing the cumulative physiological burden.
#: Project-defined; NOT official IMD.
NIGHTTIME_TMIN_THRESHOLD: float = 28.0  # °C — today's min temperature


def evaluate_rule_03(data: Dict) -> RuleResult:
    """
    RULE_03 — High Nighttime Temperature

    Fires when temperature_2m_min >= 28°C, indicating insufficient
    overnight thermal recovery.

    Threshold is project-defined. NOT official IMD.
    """
    tmin = _get_float(data, "temperature_2m_min")

    triggered = (tmin is not None and tmin >= NIGHTTIME_TMIN_THRESHOLD)

    return RuleResult(
        rule_id="RULE_03",
        name="High Nighttime Temperature",
        triggered=triggered,
        severity="WARNING",
        message=(
            f"High nighttime temperature detected: Tmin={tmin:.1f}°C "
            f"(threshold {NIGHTTIME_TMIN_THRESHOLD}°C). "
            "Reduced overnight cooling limits the body's ability to recover from "
            "daytime heat stress. Ensure sleeping areas are ventilated or cooled."
        ) if triggered else "",
        description=(
            f"Fires when temperature_2m_min >= {NIGHTTIME_TMIN_THRESHOLD}°C. "
            "High minimum temperatures prevent physiological recovery between hot days, "
            "compounding cumulative heat stress."
        ),
        threshold_note=(
            f"Project-defined: Tmin>={NIGHTTIME_TMIN_THRESHOLD}°C. "
            "NOT an official IMD threshold."
        ),
    )


# ===========================================================================
# RULE 04 — Compounded Heat Stress
# ===========================================================================

#: Fires when multiple heat-related indicators simultaneously signal
#: elevated stress: high Tmax + high departure from normal + high z-score.
#: Rationale: convergence of multiple indicators reduces false positives.
#: All thresholds project-defined.
COMPOUND_TMAX_THRESHOLD:      float = 40.0  # °C
COMPOUND_DEPARTURE_THRESHOLD: float = 4.5   # °C departure from climatological normal
COMPOUND_ZSCORE_THRESHOLD:    float = 1.5   # z-score of Tmax departure


def evaluate_rule_04(data: Dict) -> RuleResult:
    """
    RULE_04 — Compounded Heat Stress

    Fires when temperature_2m_max >= 40°C AND tmax_departure >= 4.5°C
    AND tmax_departure_zscore >= 1.5.

    Thresholds are project-defined. NOT official IMD.
    """
    tmax       = _get_float(data, "temperature_2m_max")
    departure  = _get_float(data, "tmax_departure")
    zscore     = _get_float(data, "tmax_departure_zscore")

    triggered = (
        tmax      is not None and tmax      >= COMPOUND_TMAX_THRESHOLD
        and departure is not None and departure >= COMPOUND_DEPARTURE_THRESHOLD
        and zscore    is not None and zscore    >= COMPOUND_ZSCORE_THRESHOLD
    )

    return RuleResult(
        rule_id="RULE_04",
        name="Compounded Heat Stress",
        triggered=triggered,
        severity="WARNING",
        message=(
            f"Compounded heat stress detected: Tmax={tmax:.1f}°C, "
            f"departure={departure:.2f}°C, z-score={zscore:.2f}. "
            "Multiple heat indicators are simultaneously elevated, suggesting "
            "an unusually severe heat episode relative to historical norms. "
            "Take all recommended precautions."
        ) if triggered else "",
        description=(
            f"Fires when temperature_2m_max >= {COMPOUND_TMAX_THRESHOLD}°C "
            f"AND tmax_departure >= {COMPOUND_DEPARTURE_THRESHOLD}°C "
            f"AND tmax_departure_zscore >= {COMPOUND_ZSCORE_THRESHOLD}. "
            "Convergence of absolute, relative, and anomaly indicators."
        ),
        threshold_note=(
            f"Project-defined: Tmax>={COMPOUND_TMAX_THRESHOLD}°C, "
            f"departure>={COMPOUND_DEPARTURE_THRESHOLD}°C, "
            f"z-score>={COMPOUND_ZSCORE_THRESHOLD}. "
            "NOT official IMD thresholds. Departure and z-score thresholds are "
            "aligned with the IMD-inspired qualifying_day criteria in "
            "docs/heatwave_labeling_methodology.md."
        ),
    )


# ===========================================================================
# RULE 05 — Vulnerable Population Alert
# ===========================================================================

#: Fires when ML risk level is HIGH or EXTREME.
#: Provides a targeted reminder about vulnerable populations.
#: Complements (does NOT duplicate) the Part 2 adaptation recommendations
#: which also include a Vulnerable Populations category.
_VULNERABLE_TRIGGER_LEVELS: frozenset = frozenset({"HIGH", "EXTREME"})


def evaluate_rule_05(data: Dict, risk_level: Optional[str] = None) -> RuleResult:
    """
    RULE_05 — Vulnerable Population Alert

    Fires when the Part 2 risk level is HIGH or EXTREME.

    Parameters
    ----------
    data       : dict  Feature record (used only to extract Tmax for message).
    risk_level : str   Risk level from Part 2 RiskAssessor ("LOW" / "MODERATE" /
                       "HIGH" / "EXTREME").  If None, rule does not trigger.
    """
    tmax = _get_float(data, "temperature_2m_max")
    rl = (risk_level or "").upper().strip()

    triggered = rl in _VULNERABLE_TRIGGER_LEVELS

    tmax_str = f" (Tmax={tmax:.1f}°C)" if tmax is not None else ""
    return RuleResult(
        rule_id="RULE_05",
        name="Vulnerable Population Alert",
        triggered=triggered,
        severity="CRITICAL" if rl == "EXTREME" else "WARNING",
        message=(
            f"Risk level is {rl}{tmax_str}. "
            "Elderly individuals, infants, pregnant women, and people with pre-existing "
            "health conditions (cardiovascular, respiratory, renal) face significantly "
            "elevated risk. Actively check on vulnerable people and ensure they have "
            "access to cool environments and adequate hydration."
        ) if triggered else "",
        description=(
            "Fires when Part 2 risk level is HIGH or EXTREME. "
            "Provides a targeted reminder that vulnerable populations need active "
            "monitoring and support beyond general public guidance."
        ),
        threshold_note=(
            "Triggers when risk_level in {HIGH, EXTREME}. "
            "Risk thresholds are project-defined (Part 2): "
            "HIGH=[0.60,0.80), EXTREME=[0.80,1.00]. NOT official IMD thresholds."
        ),
    )


# ===========================================================================
# RULE 06 — Outdoor Exposure Warning
# ===========================================================================

#: Fires when risk level is HIGH or EXTREME, reinforcing outdoor restriction guidance.
_OUTDOOR_TRIGGER_LEVELS: frozenset = frozenset({"HIGH", "EXTREME"})


def evaluate_rule_06(data: Dict, risk_level: Optional[str] = None) -> RuleResult:
    """
    RULE_06 — Outdoor Exposure Warning

    Fires when the Part 2 risk level is HIGH or EXTREME.

    Provides explicit outdoor exposure guidance with time-window specifics.

    Parameters
    ----------
    data       : dict  Feature record.
    risk_level : str   Part 2 risk level.
    """
    tmax = _get_float(data, "temperature_2m_max")
    rl   = (risk_level or "").upper().strip()

    triggered = rl in _OUTDOOR_TRIGGER_LEVELS

    tmax_str = f" (Tmax={tmax:.1f}°C)" if tmax is not None else ""
    return RuleResult(
        rule_id="RULE_06",
        name="Outdoor Exposure Warning",
        triggered=triggered,
        severity="CRITICAL" if rl == "EXTREME" else "WARNING",
        message=(
            f"Risk level is {rl}{tmax_str}. "
            "Avoid unnecessary outdoor exposure during peak heat hours (11:00–17:00 IST). "
            "If outdoor activity is unavoidable, seek shade regularly, drink water "
            "before feeling thirsty, and reduce physical exertion to a minimum."
        ) if triggered else "",
        description=(
            "Fires when Part 2 risk level is HIGH or EXTREME. "
            "Provides specific outdoor exposure guidance reinforcing Part 2 recommendations."
        ),
        threshold_note=(
            "Triggers when risk_level in {HIGH, EXTREME}. "
            "Project-defined. NOT official IMD guidance."
        ),
    )


# ===========================================================================
# RULE 07 — Hydration and Cooling Reminder
# ===========================================================================

#: Fires when the model probability exceeds 0.30 (MODERATE risk or above)
#: OR when temperature_2m_max >= 35°C.
#: Rationale: early hydration reminder at a lower threshold than heat alert.
HYDRATION_PROB_THRESHOLD: float = 0.30   # probability threshold
HYDRATION_TMAX_THRESHOLD: float = 35.0  # °C


def evaluate_rule_07(
    data: Dict,
    heatwave_probability: Optional[float] = None,
) -> RuleResult:
    """
    RULE_07 — Hydration and Cooling Reminder

    Fires when heatwave_probability >= 0.30 OR temperature_2m_max >= 35°C.

    This rule fires at a lower threshold than the main alert rules to
    encourage early preventive hydration behaviour.

    Parameters
    ----------
    data                : dict  Feature record.
    heatwave_probability: float Part 1 model probability. May be None.
    """
    tmax = _get_float(data, "temperature_2m_max")
    prob = heatwave_probability

    prob_trigger = (prob is not None and prob >= HYDRATION_PROB_THRESHOLD)
    tmax_trigger = (tmax is not None and tmax >= HYDRATION_TMAX_THRESHOLD)
    triggered    = prob_trigger or tmax_trigger

    reason_parts = []
    if prob_trigger:
        reason_parts.append(f"model probability={prob:.2f}")
    if tmax_trigger:
        reason_parts.append(f"Tmax={tmax:.1f}°C")
    reason = ", ".join(reason_parts)

    return RuleResult(
        rule_id="RULE_07",
        name="Hydration and Cooling Reminder",
        triggered=triggered,
        severity="INFO",
        message=(
            f"Elevated heat conditions ({reason}). "
            "Drink water regularly throughout the day — do not wait until you feel thirsty. "
            "Use fans, cool showers, or damp cloths if you feel overheated. "
            "Reduce strenuous activity during the warmest hours of the day."
        ) if triggered else "",
        description=(
            f"Fires when heatwave_probability >= {HYDRATION_PROB_THRESHOLD} "
            f"OR temperature_2m_max >= {HYDRATION_TMAX_THRESHOLD}°C. "
            "Provides early hydration and cooling guidance at a lower activation "
            "threshold than the main alert rules."
        ),
        threshold_note=(
            f"Project-defined: probability>={HYDRATION_PROB_THRESHOLD} "
            f"OR Tmax>={HYDRATION_TMAX_THRESHOLD}°C. "
            "NOT official IMD thresholds."
        ),
    )

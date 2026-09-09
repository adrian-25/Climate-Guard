"""
ClimateGuard Expert Rules Module
Part 3 — Deterministic Domain-Context Rules

Exposes the ExpertRuleEngine and rule evaluation functions.

Usage:
    from src.expert_rules import ExpertRuleEngine

    engine = ExpertRuleEngine()
    results = engine.evaluate(
        data=feature_dict,
        risk_level="HIGH",
        heatwave_probability=0.75,
    )
    for r in engine.get_triggered(results):
        print(r.rule_id, r.severity, r.message)

DISCLAIMER: Expert rules are project-defined contextual guidance.
They are NOT official IMD warnings or government heat action plan outputs.
"""

from src.expert_rules.rules import (
    RuleResult,
    evaluate_rule_01,
    evaluate_rule_02,
    evaluate_rule_03,
    evaluate_rule_04,
    evaluate_rule_05,
    evaluate_rule_06,
    evaluate_rule_07,
    EXTREME_TEMP_THRESHOLD_PLAINS,
    EXTREME_TEMP_THRESHOLD_COASTAL,
    PERSISTENT_HEAT_TMAX_THRESHOLD,
    PERSISTENT_HEAT_ROLL7_THRESHOLD,
    NIGHTTIME_TMIN_THRESHOLD,
    COMPOUND_TMAX_THRESHOLD,
    COMPOUND_DEPARTURE_THRESHOLD,
    COMPOUND_ZSCORE_THRESHOLD,
    HYDRATION_PROB_THRESHOLD,
    HYDRATION_TMAX_THRESHOLD,
)
from src.expert_rules.engine import ExpertRuleEngine, EXPERT_RULES_DISCLAIMER

__all__ = [
    "ExpertRuleEngine",
    "RuleResult",
    "EXPERT_RULES_DISCLAIMER",
    "evaluate_rule_01",
    "evaluate_rule_02",
    "evaluate_rule_03",
    "evaluate_rule_04",
    "evaluate_rule_05",
    "evaluate_rule_06",
    "evaluate_rule_07",
    "EXTREME_TEMP_THRESHOLD_PLAINS",
    "EXTREME_TEMP_THRESHOLD_COASTAL",
    "PERSISTENT_HEAT_TMAX_THRESHOLD",
    "PERSISTENT_HEAT_ROLL7_THRESHOLD",
    "NIGHTTIME_TMIN_THRESHOLD",
    "COMPOUND_TMAX_THRESHOLD",
    "COMPOUND_DEPARTURE_THRESHOLD",
    "COMPOUND_ZSCORE_THRESHOLD",
    "HYDRATION_PROB_THRESHOLD",
    "HYDRATION_TMAX_THRESHOLD",
]

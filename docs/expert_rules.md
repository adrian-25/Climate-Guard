# ClimateGuard Part 3 — Expert Rule Engine

**Module:** `src/expert_rules/`  
**Owner:** Pradnesh (Part 3)  
**Status:** Complete  
**Last updated:** 2026-09-10

---

## ⚠️ Important Disclaimer

**All expert rules in ClimateGuard are project-defined contextual guidance generated for research and educational purposes.**

They are:
- **NOT** official India Meteorological Department (IMD) warnings
- **NOT** official government heat action plan outputs
- **NOT** certified medical advice
- **NOT** certified public health guidance

All thresholds are project-defined operational thresholds. Always follow guidance from your local authorities and health professionals during heat emergencies.

---

## 1. Purpose

The Expert Rule Engine provides seven deterministic, explainable rules that add domain-context alerts on top of the ML prediction from Part 1 and the risk assessment from Part 2.

Rules are **contextual overlays** — they do not replace the ML model. They surface specific heat-risk conditions that the model probability alone may not communicate clearly.

Design principles:
- **Deterministic** — identical inputs always produce identical outputs
- **Transparent** — every threshold is documented with its rationale
- **Independent** — each rule is separately testable and identifiable
- **Non-duplicative** — rules do not reproduce Part 2 recommendation logic

---

## 2. Architecture

```
Feature record (dict / pd.Series)
    + optional risk_level from Part 2
    + optional heatwave_probability from Part 1
            ↓
ExpertRuleEngine.evaluate(...)
            ↓
List[RuleResult]   — 7 results, one per rule (RULE_01 → RULE_07)
```

The engine is **stateless**. Each `evaluate()` call is independent.

---

## 3. Module Files

| File | Class / Function | Description |
|---|---|---|
| `src/expert_rules/__init__.py` | — | Package exports, threshold constants |
| `src/expert_rules/rules.py` | `RuleResult` | Result dataclass |
| `src/expert_rules/rules.py` | `evaluate_rule_01` .. `evaluate_rule_07` | Individual rule functions |
| `src/expert_rules/engine.py` | `ExpertRuleEngine` | Rule orchestrator |

---

## 4. RuleResult

Every rule evaluation returns a `RuleResult` dataclass.

```python
@dataclass
class RuleResult:
    rule_id:        str    # "RULE_01" .. "RULE_07"
    name:           str    # Short human-readable name
    triggered:      bool   # True if the rule condition was met
    severity:       str    # "INFO" | "WARNING" | "CRITICAL"
    message:        str    # User-facing guidance (non-empty when triggered)
    description:    str    # Rule description (always present)
    threshold_note: str    # Documents the threshold source
```

`to_dict()` returns a plain dictionary. When `triggered=False`, `message` is omitted from the dict to reduce noise.

---

## 5. The Seven Expert Rules

### RULE_01 — Extreme Temperature

**Severity:** CRITICAL  
**Trigger condition:**

| City type | Threshold |
|---|---|
| Plains (Delhi, Lucknow, Nagpur, Ahmedabad) | `temperature_2m_max` ≥ 45.0°C |
| Coastal (Mumbai) | `temperature_2m_max` ≥ 40.0°C |

**Rationale:** 45°C is the IMD-inspired "absolute severe heatwave" override threshold used in the project's heatwave labelling methodology (see `docs/heatwave_labeling_methodology.md`). It is applied here as an extreme temperature alert. The coastal threshold of 40°C is analogously extreme for Mumbai's lower-baseline climate.

**Input required:** `temperature_2m_max`. Optionally `city_key` or `city` for city-type detection (defaults to plains if absent).

**Threshold note:** Project-defined. Aligned with IMD-inspired criteria. NOT official IMD severe-heatwave declaration threshold.

---

### RULE_02 — Persistent Heat

**Severity:** WARNING  
**Trigger condition:**  
`temperature_2m_max` ≥ 40.0°C **AND** `temperature_2m_max_roll7_mean` ≥ 37.0°C

**Rationale:** A single hot day is qualitatively different from multiple consecutive hot days. When the 7-day rolling mean Tmax exceeds 37°C, the body has been under sustained thermal stress without recovery. This rule captures multi-day heat accumulation that the single-day model probability does not fully represent.

**Input required:** `temperature_2m_max`, `temperature_2m_max_roll7_mean`.  
If either is missing, the rule does not trigger.

**Threshold note:** Project-defined. 40°C aligns with the IMD-inspired plains qualifying-day base threshold. 37°C for the rolling mean is a project-defined sustained-heat threshold. NOT official IMD.

---

### RULE_03 — High Nighttime Temperature

**Severity:** WARNING  
**Trigger condition:**  
`temperature_2m_min` ≥ 28.0°C

**Rationale:** Elevated nighttime temperatures prevent the body from recovering from daytime heat stress. When Tmin stays above 28°C, the cumulative physiological burden increases significantly — even if the following day's Tmax is not extreme. This is a known risk multiplier in public health heat research.

**Input required:** `temperature_2m_min`.  
If missing, the rule does not trigger.

**Threshold note:** Project-defined. 28°C is a conservative threshold for insufficient overnight thermal recovery in an Indian context. NOT official IMD.

---

### RULE_04 — Compounded Heat Stress

**Severity:** WARNING  
**Trigger condition:**  
`temperature_2m_max` ≥ 40.0°C **AND** `tmax_departure` ≥ 4.5°C **AND** `tmax_departure_zscore` ≥ 1.5

**Rationale:** Convergence of three independent heat indicators (absolute temperature, departure from climatological normal, and statistical anomaly) provides high-confidence signal of an unusually severe heat episode. The departure threshold (4.5°C) and z-score threshold (1.5) are aligned with the IMD-inspired qualifying-day criteria used in the project's heatwave labelling methodology, providing internal consistency.

**Input required:** `temperature_2m_max`, `tmax_departure`, `tmax_departure_zscore`.  
All three must be present; if any is missing, the rule does not trigger.

**Threshold note:** Project-defined. Departure threshold of 4.5°C and Tmax threshold of 40°C are aligned with the project's heatwave labelling criteria in `docs/heatwave_labeling_methodology.md`. The z-score threshold of 1.5 is project-defined. NOT official IMD.

---

### RULE_05 — Vulnerable Population Alert

**Severity:** CRITICAL (EXTREME risk) / WARNING (HIGH risk)  
**Trigger condition:**  
`risk_level` ∈ {"HIGH", "EXTREME"}

**Rationale:** When the Part 2 risk engine classifies conditions as HIGH or EXTREME, vulnerable populations (elderly, infants, pregnant women, and those with cardiovascular, respiratory, or renal conditions) face significantly elevated risk. This rule provides a targeted, explicit reminder to actively check on vulnerable individuals — beyond the general adaptation recommendations from Part 2.

**Input required:** `risk_level` parameter (from Part 2).  
If `risk_level` is `None` or not HIGH/EXTREME, the rule does not trigger.  
`temperature_2m_max` is used for the message (optional).

**Threshold note:** Triggers when `risk_level` ∈ {HIGH, EXTREME}. Part 2 risk thresholds are project-defined: HIGH=[0.60, 0.80), EXTREME=[0.80, 1.00]. NOT official IMD.

---

### RULE_06 — Outdoor Exposure Warning

**Severity:** CRITICAL (EXTREME risk) / WARNING (HIGH risk)  
**Trigger condition:**  
`risk_level` ∈ {"HIGH", "EXTREME"}

**Rationale:** When risk is HIGH or EXTREME, outdoor exposure during peak heat hours (11:00–17:00 IST) carries meaningful danger. This rule provides specific outdoor exposure guidance complementing the Part 2 adaptation recommendations.

**Input required:** `risk_level` parameter (from Part 2).  
`temperature_2m_max` is used for the message (optional).

**Threshold note:** Same as RULE_05. Triggers at HIGH/EXTREME risk_level. NOT official IMD guidance.

---

### RULE_07 — Hydration and Cooling Reminder

**Severity:** INFO  
**Trigger condition:**  
`heatwave_probability` ≥ 0.30 **OR** `temperature_2m_max` ≥ 35.0°C

**Rationale:** This is the earliest-firing rule, designed to encourage preventive hydration behaviour before conditions reach alert severity. The probability threshold of 0.30 corresponds to the bottom of MODERATE risk (Part 2 threshold). The temperature threshold of 35°C is a warm-but-not-extreme Indian summer day.

**Input required:** Either `heatwave_probability` or `temperature_2m_max` (or both).  
If both are absent, the rule does not trigger.

**Threshold note:** Project-defined. Probability 0.30 aligns with the bottom of the MODERATE risk bracket. Tmax 35°C is a project-defined early-warning temperature. NOT official IMD.

---

## 6. Trigger Summary Table

| Rule | ID | Severity | Key inputs | Trigger |
|---|---|---|---|---|
| Extreme Temperature | RULE_01 | CRITICAL | `temperature_2m_max`, `city_key` | Tmax ≥ 45°C (plains) / 40°C (coastal) |
| Persistent Heat | RULE_02 | WARNING | `temperature_2m_max`, `roll7_mean` | Tmax ≥ 40°C AND roll7 ≥ 37°C |
| High Nighttime Temp | RULE_03 | WARNING | `temperature_2m_min` | Tmin ≥ 28°C |
| Compounded Stress | RULE_04 | WARNING | `tmax`, `departure`, `zscore` | Tmax ≥ 40°C AND dep ≥ 4.5 AND z ≥ 1.5 |
| Vulnerable Population | RULE_05 | CRIT/WARN | `risk_level` | HIGH or EXTREME |
| Outdoor Exposure | RULE_06 | CRIT/WARN | `risk_level` | HIGH or EXTREME |
| Hydration Reminder | RULE_07 | INFO | `probability`, `temperature_2m_max` | prob ≥ 0.30 OR Tmax ≥ 35°C |

---

## 7. ExpertRuleEngine API

```python
from src.expert_rules import ExpertRuleEngine

engine = ExpertRuleEngine()

# Evaluate all 7 rules
results = engine.evaluate(
    data=feature_dict,          # dict or pd.Series
    risk_level="HIGH",           # optional: from Part 2
    heatwave_probability=0.72,   # optional: from Part 1
)

# All 7 results (triggered and not)
for r in results:
    print(r.rule_id, r.triggered, r.severity)

# Triggered rules only
triggered = engine.get_triggered(results)

# Count
count = engine.triggered_count(results)   # int

# Highest severity among triggered rules ("CRITICAL" > "WARNING" > "INFO" > None)
sev = engine.highest_severity(results)

# Filter by severity
critical = engine.get_by_severity(results, "CRITICAL")

# Convert to JSON-serialisable list of dicts
dicts = engine.to_dict_list(results)

# Summary dict
summary = engine.summarise(results)
# {total_rules, triggered_count, highest_severity, triggered_ids, disclaimer}
```

---

## 8. Example Output

For a hot Delhi day (Tmax=44.5°C, Tmin=29°C, departure=4.8°C, z=2.95, risk=EXTREME, prob=0.92):

```json
[
  {
    "rule_id": "RULE_01",
    "name": "Extreme Temperature",
    "triggered": true,
    "severity": "CRITICAL",
    "description": "Fires when today's maximum temperature exceeds...",
    "message": "Extreme temperature detected: 44.5°C (threshold 45°C for this city type)...",
    "threshold_note": "Project-defined: plains=45.0°C, coastal=40.0°C. NOT an official IMD warning threshold."
  },
  {
    "rule_id": "RULE_02",
    "name": "Persistent Heat",
    "triggered": true,
    "severity": "WARNING",
    "description": "Fires when temperature_2m_max >= 40.0°C AND roll7_mean >= 37.0°C...",
    "message": "Persistent heat detected: today Tmax=44.5°C, 7-day rolling mean=42.0°C..."
  },
  {
    "rule_id": "RULE_03",
    "name": "High Nighttime Temperature",
    "triggered": true,
    "severity": "WARNING",
    "message": "High nighttime temperature detected: Tmin=29.0°C..."
  },
  {
    "rule_id": "RULE_04",
    "name": "Compounded Heat Stress",
    "triggered": true,
    "severity": "WARNING",
    "message": "Compounded heat stress detected: Tmax=44.5°C, departure=4.80°C, z-score=2.95..."
  },
  {
    "rule_id": "RULE_05",
    "name": "Vulnerable Population Alert",
    "triggered": true,
    "severity": "CRITICAL",
    "message": "Risk level is EXTREME (Tmax=44.5°C). Elderly individuals, infants..."
  },
  {
    "rule_id": "RULE_06",
    "name": "Outdoor Exposure Warning",
    "triggered": true,
    "severity": "CRITICAL",
    "message": "Risk level is EXTREME (Tmax=44.5°C). Avoid unnecessary outdoor exposure..."
  },
  {
    "rule_id": "RULE_07",
    "name": "Hydration and Cooling Reminder",
    "triggered": true,
    "severity": "INFO",
    "message": "Elevated heat conditions (model probability=0.92, Tmax=44.5°C)..."
  }
]
```

For a cold January day (Tmax=18°C, risk=LOW, prob=0.00): all 7 rules `triggered=false`.

---

## 9. Integration with Part 2

Rules RULE_05 and RULE_06 consume `risk_level` from Part 2's `RiskAssessor`.  
Rule RULE_07 consumes `heatwave_probability` from Part 1's `ClimateGuardPredictor`.

The expert rules **do not replace** Part 2 recommendations. They are additive overlays that provide specific named alerts on top of the general recommendation set.

Part 2 produces structured recommendations by category (Hydration, Cooling, Outdoor Exposure, etc.).  
Part 3 expert rules produce named, severity-tagged alerts with specific threshold documentation.

---

## 10. Determinism Guarantee

Every expert rule is a pure function of its inputs. There is no internal state, no randomness, and no external I/O. Given identical inputs:
- `triggered` is always the same
- `severity` is always the same
- `message` is always the same (modulo float formatting)

This makes rules independently testable and auditable.

---

## 11. Limitations

1. **Temperature-focused.** Current rules are primarily based on temperature indicators. Humidity, wind, and radiation features are present in the 110-feature set but not yet used as primary rule triggers.

2. **Binary trigger.** Rules are either triggered or not. There is no partial or probabilistic triggering.

3. **Rules 05 and 06 require Part 2 context.** Without `risk_level`, these rules cannot fire — a cold-chain pipeline that omits Part 2 will not see Vulnerable Population or Outdoor Exposure alerts.

4. **City-type detection.** RULE_01 uses `city_key` to select plains vs coastal threshold. If `city_key` is absent, it defaults to the plains threshold (the more conservative choice).

5. **Not official warnings.** These rules must never be presented to end users as official IMD warnings or government heat action plan outputs.

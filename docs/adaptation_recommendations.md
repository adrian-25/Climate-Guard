# Adaptation Recommendations

**ClimateGuard — Part 2**  
**Module:** `src/adaptation/recommendations.py`  
**Owner:** Kshitij (Part 2 — Risk, Adaptation, Explainability)  
**Status:** Complete  
**Date:** 2026-09-10

---

## 1. Overview

The Adaptation Recommendation Engine generates practical, risk-level-appropriate heatwave preparedness guidance.  Recommendations are categorised by domain and scaled to the severity of the risk level produced by the Risk Assessment module.

**⚠️ DISCLAIMER: These recommendations are project-defined operational guidance for the ClimateGuard early-warning system.  They are NOT official government heat action plan instructions, official India Meteorological Department (IMD) advisories, or certified medical advice.  Always follow guidance from local authorities and health professionals during heat emergencies.**

---

## 2. Architecture Position

```
RiskAssessor → risk_level
                    ↓
            AdaptationEngine (this module)
                    ↓
            List[Recommendation] or dict
```

---

## 3. Recommendation Categories

| Category | Description |
|---|---|
| Hydration | Fluid intake guidance |
| Outdoor Exposure | Advice on limiting time outdoors during peak heat |
| Cooling | Cooling mechanisms — fans, AC, cooling centres |
| Vulnerable Populations | Specific guidance for high-risk groups |
| Workplace | Employer and worker precautions |
| Public Awareness | Information sharing and monitoring |
| Emergency Preparedness | Emergency contacts, heat-illness recognition |

Not all categories are present at all risk levels.  Higher risk levels include more categories.

---

## 4. Recommendation Sets by Risk Level

### LOW
- Hydration: Normal daily fluid intake.
- Outdoor Exposure: No unusual restrictions.
- Public Awareness: Monitor local forecasts.

### MODERATE
- Hydration: Increase fluid intake, avoid dehydrating beverages.
- Outdoor Exposure: Reduce prolonged midday outdoor activity.
- Cooling: Fans, cool showers, ventilation.
- Vulnerable Populations: Check on elderly and high-risk individuals.
- Public Awareness: Follow local heat advisories.

### HIGH
- Hydration: Drink frequently; consider oral rehydration salts.
- Outdoor Exposure: Avoid peak hours (11:00–17:00); schedule strenuous tasks for early morning.
- Cooling: Access air conditioning or cooling centres; keep sun-facing windows covered.
- Vulnerable Populations: Actively monitor and ensure access to cool environments.
- Workplace: Additional rest breaks, shaded areas, access to water; reschedule outdoor tasks.
- Public Awareness: Share heat-illness warning signs with community.

### EXTREME
- Hydration: Continuous intake, 2–3 litres daily; avoid alcohol and caffeinated drinks.
- Outdoor Exposure: Minimise all unnecessary outdoor activity for adults and children.
- Cooling: Access air-conditioned spaces for maximum possible time; apply cool cloths; block sunlight.
- Vulnerable Populations: Take immediate active steps; check every few hours; never leave alone in hot unventilated space.
- Workplace: Suspend or significantly curtail outdoor work during peak hours.
- Emergency Preparedness: Know hospital locations; recognise heat-illness signs (sweating, rapid heartbeat, confusion, fainting).
- Public Awareness: Share information with community; follow all official heat-emergency guidance.

---

## 5. Input / Output

### Input

| Parameter | Type | Description |
|---|---|---|
| `risk_level` | str | "LOW" \| "MODERATE" \| "HIGH" \| "EXTREME" (case-insensitive) |

### Output: `List[Recommendation]` (via `recommend()`)

Each `Recommendation` has:
- `category` : str
- `message`  : str

### Output: `dict` (via `recommend_as_dict()`)

```json
{
  "risk_level": "EXTREME",
  "disclaimer": "These are project-defined operational recommendations...",
  "recommendations": [
    {"category": "Hydration", "message": "..."},
    {"category": "Outdoor Exposure", "message": "..."}
  ]
}
```

---

## 6. Usage

```python
from src.adaptation import AdaptationEngine

engine = AdaptationEngine()

# List of Recommendation objects
recs = engine.recommend("HIGH")
for r in recs:
    print(r.category, ":", r.message)

# Dict for serialisation
out = engine.recommend_as_dict("EXTREME")
# out["disclaimer"] contains the full disclaimer text
# out["recommendations"] is a list of {"category", "message"} dicts

# Categories only
cats = engine.get_categories("HIGH")
# ["Hydration", "Outdoor Exposure", "Cooling", ...]
```

---

## 7. Error Handling

| Situation | Error |
|---|---|
| Unknown risk level string | `ValueError` |
| Empty string | `ValueError` |

Input is case-insensitive.  "high", "HIGH", "High" are all accepted.

---

## 8. Design Decisions

1. **Risk-level-appropriate scaling**: More severe risk levels include more categories and stronger language.

2. **Non-alarmist language**: All messages use measured, calm phrasing.  No catastrophising language.

3. **No medical diagnosis**: Messages do not diagnose illness or prescribe treatment.  They advise recognition and prompt medical attention.

4. **Disclaimer required**: The full disclaimer text must accompany any public-facing recommendation output.  It is included in `recommend_as_dict()`.

5. **Deterministic output**: Identical risk level always produces identical recommendations.

6. **Stateless**: The engine holds no state.  Suitable for concurrent calls.

---

## 9. Limitations

1. **Project-defined guidance only.** Recommendations are not validated by health authorities, the IMD, or government heat action plans.

2. **City-agnostic.** The same recommendations are given for all five cities, regardless of local infrastructure, humidity, or population density.  A real deployment should integrate local heat action plan guidance.

3. **No duration or trend factor.** Recommendations are based on the current-day risk level only.  Consecutive days of EXTREME risk would warrant stronger cumulative measures in a real system.

4. **No vulnerable-group targeting.** The same recommendations go to all users.  A real system might differentiate by user profile (age, health status, occupation).

---

## 10. Module Reference

| Symbol | Description |
|---|---|
| `Recommendation` | Frozen dataclass with `category` and `message` fields |
| `AdaptationEngine` | Main class; `recommend()`, `recommend_as_dict()`, `get_categories()` |
| `DISCLAIMER` | Module-level constant: the full disclaimer string |
| `_RECOMMENDATIONS` | Internal dict mapping risk level → list of Recommendation |

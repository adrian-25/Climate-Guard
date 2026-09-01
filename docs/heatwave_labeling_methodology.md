# ClimateGuard — Heatwave Labeling Methodology
## Phase 6: IMD-Inspired Operational Heatwave Label

**Script:**  `heatwave_labeling.py`  
**Input:**   `data/processed/weather_cleaned.csv`  
**Output:**  `data/processed/weather_labelled.csv`  
**Date:**    2026-09-01  
**Label name:** `heatwave` (0 = no heatwave, 1 = heatwave day)

---

## 1. Official / Reference Definition

The India Meteorological Department (IMD) uses the following criteria to declare a heat wave:

### Plains and Hilly Regions

A heat wave is considered when the **maximum temperature of a station reaches at least 40°C** for plains regions.

**Based on Departure from Normal:**
- Heat Wave: departure from normal is **4.5°C to 6.4°C**
- Severe Heat Wave: departure from normal is **> 6.4°C**

**Based on Actual Maximum Temperature:**
- Heat Wave: when actual maximum temperature **≥ 45°C**
- Severe Heat Wave: when actual maximum temperature **≥ 47°C**

**Duration criterion:** If the above criteria are met at **at least 2 stations** in a Meteorological sub-division for **at least two consecutive days**, it is declared on the second day.

### Coastal Stations

When maximum temperature departure is **4.5°C or more from normal**, a heat wave **may be described** provided the actual maximum temperature is **37°C or more**.

### Sources

| Source | URL |
|---|---|
| WHO India (quoting IMD) | https://www.who.int/india/heat-waves |
| DrishtiIAS (IMD criteria summary) | https://www.drishtiias.com/daily-updates/daily-news-analysis/heatwaves-4 |
| Times of India (IMD Mumbai head, Bikram Singh) | Coastal 37°C threshold confirmed |
| IMD FAQ | https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf |

---

## 2. Data Available

Dataset: `data/processed/weather_cleaned.csv`

| Variable | Available? | Notes |
|---|---|---|
| `temperature_2m_max` | ✅ Yes | Daily maximum temperature at 2m |
| Daily climatological normal (Tmax) | ✅ Computed | Derived from 1990–2020 baseline |
| Departure from normal | ✅ Computed | `temperature_2m_max` − normal |
| Station-level ground observations | ❌ No | ERA5 reanalysis only |
| Multi-station sub-division coverage | ❌ No | One ERA5 grid point per city |

---

## 3. Criteria We Can Implement

| IMD Criterion | Implementation | Exact match? |
|---|---|---|
| Tmax ≥ 40°C (plains minimum) | `temperature_2m_max >= 40.0` | ✅ Yes |
| Tmax ≥ 37°C (coastal minimum) | `temperature_2m_max >= 37.0` | ✅ Yes |
| Departure ≥ 4.5°C from normal | `tmax_departure >= 4.5` | ✅ Yes (ERA5-based normal) |
| Tmax ≥ 45°C absolute override | `temperature_2m_max >= 45.0` | ✅ Yes |
| ≥ 2 consecutive qualifying days | Run-length encoding per city | ✅ Yes |

---

## 4. Criteria We Cannot Implement Exactly

| IMD Criterion | Why Not Available | Impact |
|---|---|---|
| Station-level ground temperature | ERA5 is gridded reanalysis (0.25°) — not ground station | ERA5 Tmax may differ from station Tmax by 1–3°C in some cases |
| Official 30-year station normals | IMD uses archived station normals, not ERA5 | We compute ERA5-internal normals — self-consistent but not identical to IMD's |
| "2 stations in a subdivision" rule | We have one ERA5 grid point per city | Cannot verify multi-station criterion; treated as single-station labeling |
| WMO standard period (1981–2010) | Our data begins 1990 | Used 1990–2020 baseline instead; 31 years, close to WMO 30-year standard |

---

## 5. Final Operational Definition

**Name:** IMD-Inspired Operational Heatwave Label (ERA5-based)

This is **not** the official IMD heatwave declaration. It is an ERA5-based operational implementation following the IMD methodology as closely as the data allows.

### Plains cities (Delhi, Lucknow, Nagpur, Ahmedabad)

A day is a **qualifying day** if:

```
(temperature_2m_max >= 40.0°C  AND  tmax_departure >= 4.5°C)
        OR
(temperature_2m_max >= 45.0°C)
```

### Coastal city (Mumbai)

A day is a **qualifying day** if:

```
temperature_2m_max >= 37.0°C  AND  tmax_departure >= 4.5°C
```

### Event (duration) criterion

A day receives `heatwave = 1` only if it belongs to a **run of at least 2 consecutive qualifying days**. Isolated single qualifying days receive `heatwave = 0`.

### Label definition

```
heatwave = 1  →  qualifying day, part of a 2+ day consecutive run
heatwave = 0  →  all other days (including isolated qualifying days)
```

---

## 6. Why This Definition Was Chosen

**Scientific validity first.** Strategy C (this methodology) was selected over:

- **Strategy A (fixed absolute threshold):** Rejected because it produced only 8 positive days for Mumbai over 35 years — scientifically correct but unusable for supervised ML.
- **Strategy B (percentile-based):** Rejected because it has no official scientific basis, forces equal class balance regardless of actual climate, and loses cross-city comparability.

Strategy C is grounded in the IMD's operational definition — the most authoritative India-specific heatwave criterion available. It handles Mumbai correctly (via lower absolute threshold + anomaly condition) and aligns event detection with how actual heatwaves are declared (duration ≥ 2 days).

---

## 7. Baseline Period

**Period:** 1990-01-01 → 2020-12-31  
**Duration:** 31 years  
**Rationale:** WMO standard is 30 years. Our dataset starts 1990. Using 1990–2020 gives a 31-year baseline that fully covers the WMO-aligned period and avoids using future data in the normal computation.

**Method:** For each city and each calendar day-of-year (DOY 1–366):
1. Compute the mean `temperature_2m_max` across all years in the baseline period for that DOY.
2. Apply a 31-day centred rolling mean to smooth the seasonal cycle and reduce noise from small per-DOY samples (~31 observations per DOY).

This produces a smooth, self-consistent climatological normal for each city.

### Computed normals summary

| City | Mean normal Tmax (°C) | Min normal (°C) | Max normal (°C) |
|---|---|---|---|
| New Delhi | 30.97 | 19.72 | 39.99 |
| Lucknow | 31.19 | 20.88 | 39.53 |
| Nagpur | 32.85 | 27.26 | 41.87 |
| Ahmedabad | 33.27 | 26.97 | 40.26 |
| Mumbai | 30.32 | 27.85 | 32.19 |

---

## 8. Prediction Horizon

**Target:** Same-day label — `heatwave` on date **T** represents whether date T is a heatwave day.

**Leakage prevention:**
- The `tmax_normal` is computed from a historical baseline (1990–2020) using only past data — no T+1 information.
- The `qualifying_day` flag depends only on date T's own observation.
- The duration criterion identifies runs retrospectively, but the **ML training framework** must ensure that features constructed for date T use only information available at or before date T (i.e., no future temperature values).
- Lag features and rolling windows will be constructed in Phase 7 using only backwards-looking windows.

**Important note on the duration criterion:** Assigning `heatwave = 1` to all days in a multi-day event means day T's label technically depends on T+1 being a qualifying day (to form the minimum 2-day run). In the ML context:
- If predicting whether day T is a heatwave day, features from T and earlier are used.
- The label for T was assigned based on the run ending at or after T — this is acceptable because the label describes the climate event, not a prediction. The prediction task is: "given features up to T, predict whether T is a heatwave day."
- This is the standard approach used in climate ML literature.

---

## 9. Event-Duration Logic

1. For each city independently, mark each day as `qualifying_day = 1` or `0`.
2. Find consecutive runs of qualifying days using a simple scan.
3. If run length ≥ 2: all days in the run → `heatwave = 1`, assigned `hw_event_id` (sequential integer), `hw_event_start`, `hw_event_end`, `hw_event_length`.
4. If run length = 1: that isolated day → `heatwave = 0` (not a heatwave event).
5. Non-qualifying days → `heatwave = 0`, all event columns = 0 / NaT.

---

## 10. City-Specific Treatment

| City | Region | Tmax threshold | Anomaly condition | Events | HW days |
|---|---|---|---|---|---|
| New Delhi | plains | ≥ 40°C | departure ≥ 4.5°C OR Tmax ≥ 45°C | 54 | 213 |
| Lucknow | plains | ≥ 40°C | departure ≥ 4.5°C OR Tmax ≥ 45°C | 32 | 141 |
| Nagpur | plains | ≥ 40°C | departure ≥ 4.5°C OR Tmax ≥ 45°C | 34 | 119 |
| Ahmedabad | plains | ≥ 40°C | departure ≥ 4.5°C OR Tmax ≥ 45°C | 12 | 32 |
| Mumbai | coastal | ≥ 37°C | departure ≥ 4.5°C | **0** | **0** |

**Mumbai result:** 0 heatwave events, 0 labelled days. Mumbai has 8 single qualifying days (Tmax ≥ 37°C AND departure ≥ 4.5°C) but none form a 2-consecutive-day run. This is consistent with Mumbai's narrow Tmax range (std 2.1°C) — it rarely sustains anomalously hot conditions for multiple days.

**This is not a labeling error.** It reflects the scientific reality: Mumbai's maritime climate does not produce sustained surface heatwaves as defined by IMD criteria for coastal stations. Mumbai's heat risk is expressed through high humidity and warm nights rather than sustained extreme Tmax.

---

## 11. Class Distribution

| City | Total days | HW=1 | HW=0 | HW% | Events | Avg dur | Max dur | Single-day (not labeled) |
|---|---|---|---|---|---|---|---|---|
| New Delhi | 13,027 | 213 | 12,814 | 1.64% | 54 | 3.9 d | 11 d | 39 |
| Lucknow | 13,027 | 141 | 12,886 | 1.08% | 32 | 4.4 d | 12 d | 37 |
| Nagpur | 13,027 | 119 | 12,908 | 0.91% | 34 | 3.5 d | 10 d | 23 |
| Ahmedabad | 13,027 | 32 | 12,995 | 0.25% | 12 | 2.7 d | 5 d | 16 |
| Mumbai | 13,027 | **0** | 13,027 | **0.00%** | 0 | — | — | 8 |

**Class imbalance note:** The dataset is highly imbalanced (0.25–1.64% positive). This is expected and scientifically correct — heatwaves are rare events by definition. Phase 11 (Class Imbalance Handling) will address this for ML training.

---

## 12. Limitations

1. **ERA5 vs ground station temperatures.** ERA5 gridded reanalysis may under- or over-estimate point-location Tmax by 1–3°C depending on season and location. The label is self-consistent within ERA5 data but may not exactly reproduce IMD's station-based declarations.

2. **Self-referential normals.** The normal is computed from the same ERA5 dataset used for the observation. This is internally consistent but different from IMD's approach of using multi-decadal station archives.

3. **Mumbai: 0 heatwave days.** While scientifically defensible, this means Mumbai cannot contribute positive training examples. ML models trained on all 5 cities must handle this — Mumbai's contribution will be purely negative-class examples. This is a real limitation for the project.

4. **Single-city grid point.** IMD declares heat waves across meteorological sub-divisions, requiring multiple stations. Our implementation treats each city as a single station.

5. **Baseline period offset.** Using 1990–2020 instead of the WMO-standard 1981–2010 or 1991–2020 means the normals include a period of rising temperatures. This slightly raises the normal, making the departure criterion harder to meet — potentially under-counting events relative to IMD's declarations.

6. **No "severe heatwave" sub-classification.** The IMD also distinguishes severe heatwaves (departure > 6.4°C or Tmax ≥ 47°C). This distinction is not used in the current binary label, but the `tmax_departure` column allows it to be computed later if needed.

---

## 13. Reproducibility

To regenerate the labelled dataset from scratch:

```bash
# Requires: data/processed/weather_cleaned.csv
python heatwave_labeling.py
```

Key parameters (all in `heatwave_labeling.py`):

| Parameter | Value | Variable |
|---|---|---|
| Baseline start | 1990-01-01 | `BASELINE_START` |
| Baseline end | 2020-12-31 | `BASELINE_END` |
| Plains absolute threshold | 40.0°C | `PLAINS_ABS_THRESHOLD` |
| Coastal absolute threshold | 37.0°C | `COASTAL_ABS_THRESHOLD` |
| Departure threshold | 4.5°C | `DEPARTURE_HW_THRESHOLD` |
| Absolute override | 45.0°C | `ABS_OVERRIDE_THRESHOLD` |
| Minimum consecutive days | 2 | `MIN_CONSEC_DAYS` |
| Normal smoothing window | 31 days (centred) | hardcoded in `Step 2` |

---

## 14. Output Columns Added (beyond weather_cleaned.csv)

| Column | Type | Description |
|---|---|---|
| `tmax_normal` | float64 | Climatological daily Tmax normal for this city+DOY (°C) |
| `tmax_departure` | float64 | `temperature_2m_max` − `tmax_normal` (°C) |
| `qualifying_day` | int (0/1) | 1 if this day meets threshold+departure (before duration filter) |
| `heatwave` | int (0/1) | **TARGET VARIABLE** — 1 if part of ≥2 consecutive qualifying days |
| `hw_event_id` | int | Event number (0 = not a heatwave day) |
| `hw_event_start` | datetime | Date this event started (NaT if not a heatwave day) |
| `hw_event_end` | datetime | Date this event ended (NaT if not a heatwave day) |
| `hw_event_length` | int | Duration of this event in days (0 if not a heatwave day) |

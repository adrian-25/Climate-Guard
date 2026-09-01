# ClimateGuard — EDA Findings
## Phase 4: Exploratory Data Analysis Report

**Dataset:** `data/raw/all_cities_era5_raw.csv`  
**Source:** ERA5 via Open-Meteo Historical Weather API  
**Period:** 1990-01-01 → 2025-08-31  
**Analysis date:** 2026-09-01  
**Script:** `eda_climateguard.py`  
**Notebook:** `notebooks/01_eda.ipynb`  
**Plots:** `results/plots/EDA/` (30 plots)  

> All numbers in this document are computed directly from the master raw dataset.  
> No values are fabricated or estimated.

---

## 1. Dataset Overview

| Attribute | Value |
|---|---|
| Total rows | 65,135 |
| Total columns | 23 |
| Weather variables | 16 |
| Cities | 5 |
| Rows per city | 13,027 |
| Date range | 1990-01-01 → 2025-08-31 |
| Missing values | **0** |
| Missing dates (per city) | **0** |
| Duplicate city+date records | **0** |
| Physical-range violations | **0** |

---

## 2. Data Quality Confirmation

The dataset passed all validation checks:

- **65,135 rows** confirmed (5 × 13,027)
- All 5 expected cities present: Delhi, Lucknow, Nagpur, Ahmedabad, Mumbai
- All 16 weather variables populated for every row
- No NaN values anywhere in the dataset
- No duplicate dates within any city
- No physically impossible values (all within ERA5 plausible Indian climate bounds)
- All metadata (city name, coordinates, state, region_type) consistent within each city

---

## 3. City Comparison

| Metric | New Delhi | Lucknow | Nagpur | Ahmedabad | Mumbai |
|---|---|---|---|---|---|
| Region type | plains | plains | plains | plains | **coastal** |
| Mean Tmax (°C) | 31.02 | 31.26 | **32.85** | **33.28** | 30.38 |
| Max Tmax (°C) | 46.80 | 46.30 | **46.90** | 45.60 | 37.60 |
| Mean Tmin (°C) | 19.99 | 20.39 | 21.90 | 22.29 | **23.84** |
| Min Tmin (°C) | **2.50** | 3.10 | 7.20 | 5.90 | 13.10 |
| Mean RH (%) | 63.5 | 66.2 | 59.1 | 57.9 | **74.8** |
| Mean Precip (mm/day) | 1.91 | 2.66 | 3.14 | 2.01 | **5.33** |
| Total Precip (mm) | 24,822 | 34,708 | 40,876 | 26,173 | **69,423** |
| Mean Wind (km/h) | 14.2 | 14.4 | 12.7 | 13.7 | **17.4** |
| Mean Pressure (hPa) | 983.6 | 994.2 | 973.3 | 1002.2 | **1008.4** |
| Mean App Tmax (°C) | 32.89 | 33.63 | 35.39 | **35.68** | 34.39 |
| Mean Radiation (MJ/m²) | 18.85 | 18.59 | 18.58 | **19.34** | 18.76 |
| Mean ET₀ (mm/day) | 4.64 | 4.55 | 4.73 | **5.05** | 4.29 |

**Bold** = highest value in row.

---

## 4. Temperature Findings

### Distributions
- All plains cities show bimodal or broad Tmax distributions reflecting hot summer and cooler winter periods.
- Mumbai's Tmax distribution is narrow and near-Gaussian (std 2.1°C), centred around 30–32°C.
- Plains city Tmax standard deviations: Delhi 5.6°C, Lucknow 5.5°C, Nagpur 5.4°C, Ahmedabad 5.8°C.

### Absolute Extremes (from raw dataset)
| City | Hottest Day | Tmax (°C) | Coldest Day | Tmin (°C) |
|---|---|---|---|---|
| New Delhi | 1995-06-15 | **46.8** | — | 2.5 |
| Lucknow | 1995-06-16 | 46.3 | — | 3.1 |
| Nagpur | 2010-05-24 | **46.9** | — | 7.2 |
| Ahmedabad | 2016-05-19 | 45.6 | — | 5.9 |
| Mumbai | 2004-05-03 | 37.6 | — | 13.1 |

### Apparent Temperature
- Apparent Tmax consistently exceeds actual Tmax in Mumbai and during monsoon months due to humidity effects.
- Maximum apparent temperature observed: **50.9°C** (Mumbai, monsoon humidity amplification).
- Apparent Tmax mean for Ahmedabad (35.68°C) exceeds all other cities despite having lower actual Tmax than Nagpur, reflecting combined heat-humidity effect.

---

## 5. Seasonal Findings

### Hottest Months (Median Tmax)
| City | 1st | 2nd | 3rd |
|---|---|---|---|
| New Delhi | May (39.9°C) | Jun (38.5°C) | Apr (37.0°C) |
| Lucknow | May (39.5°C) | Apr (38.0°C) | Jun (37.8°C) |
| Nagpur | May (41.9°C) | Apr (40.1°C) | Jun (36.2°C) |
| Ahmedabad | May (40.2°C) | Apr (38.4°C) | Jun (38.0°C) |
| Mumbai | May (32.1°C) | Apr (32.0°C) | Mar (31.9°C) |

### Seasonal Patterns
- **Pre-Monsoon (Mar–May)** is the hottest season for all cities. May is the peak month.
- **Monsoon (Jun–Sep)** brings a sharp Tmax drop in plains cities (due to cloud cover and rainfall), but increases humidity. Lucknow Tmax drops from 39.5°C (May) to 37.8°C (Jun) to lower values through Aug.
- **Nagpur** is exceptional: median Tmax exceeds 40°C in both April and May — it has the longest pre-monsoon heat season.
- **Mumbai** shows almost no seasonal Tmax variation (range: ~28°C in winter to ~32°C in May). This is the most distinctive single characteristic of the coastal city.
- **Winter (Dec–Feb)**: Delhi and Lucknow can drop to Tmax ~20–22°C, while Mumbai rarely goes below 28°C Tmax.

---

## 6. Long-Term Trend Findings

> **Important limitation**: These are observed statistical trends from a single dataset over 35 years. They cannot by themselves establish causal attribution. Interpret as observational descriptions only.

### Annual Mean Tmin Trend (statistically significant, p<0.05 for all cities)
| City | Slope (°C/yr) | R² | Significance |
|---|---|---|---|
| New Delhi | +0.0331 | 0.379 | **p < 0.05** |
| Lucknow | +0.0359 | 0.521 | **p < 0.05** |
| Nagpur | +0.0302 | 0.504 | **p < 0.05** |
| Ahmedabad | +0.0400 | 0.549 | **p < 0.05** |
| Mumbai | +0.0346 | 0.608 | **p < 0.05** |

*Interpretation: Over the 1990–2025 period, observed annual mean Tmin has risen in all 5 cities. The strongest association is for Mumbai (R²=0.608) and Ahmedabad (R²=0.549).*

### Annual Mean Tmax Trend
| City | Slope (°C/yr) | R² | Significance |
|---|---|---|---|
| New Delhi | +0.0062 | 0.012 | not significant |
| Lucknow | +0.0137 | 0.081 | not significant |
| Nagpur | +0.0138 | 0.076 | not significant |
| Ahmedabad | +0.0075 | 0.024 | not significant |
| **Mumbai** | **+0.0281** | **0.633** | **p < 0.05** |

*Interpretation: Mumbai shows a statistically significant rising trend in annual mean Tmax. For plains cities, the Tmax trend is positive but not statistically significant at the annual mean level — annual variability is high.*

### Key observation
Tmin is rising more consistently and significantly than Tmax across all cities. This indicates warming nights, which has implications for heat stress accumulation (reduced nocturnal recovery).

---

## 7. Humidity Findings

| City | Mean RH (%) | Pre-Monsoon RH (%) | Monsoon RH (%) | Tmax–RH correlation (r) |
|---|---|---|---|---|
| New Delhi | 63.5 | ~45 | ~82 | −0.56 |
| Lucknow | 66.2 | ~47 | ~84 | −0.53 |
| Nagpur | 59.1 | ~34 | ~82 | **−0.70** |
| Ahmedabad | 57.9 | ~38 | ~74 | −0.33 |
| Mumbai | 74.8 | ~67 | ~83 | −0.52 |

- All cities show **negative correlation** between Tmax and RH — hotter days tend to be drier.
- **Nagpur** has the strongest negative Tmax–RH correlation (r = −0.73 for RH_max), reflecting the clearest hot-dry vs cool-moist regime separation.
- **Mumbai** has the highest mean RH (74.8%) and the narrowest RH range — its minimum daily RH never drops below 15% (vs 4% for Delhi in winter).
- **Ahmedabad** has the lowest mean RH (57.9%) reflecting its semi-arid climate.

---

## 8. Precipitation Findings

### Zero-Rain Frequency
| City | Zero-rain days | % of days |
|---|---|---|
| New Delhi | 8,464 | 65.0% |
| Lucknow | 8,185 | 62.8% |
| Nagpur | 8,140 | 62.5% |
| **Ahmedabad** | **9,586** | **73.6%** |
| Mumbai | 7,883 | 60.5% |

- Precipitation is highly zero-inflated in all cities.
- **Ahmedabad** has the most dry days (73.6%) — consistent with its semi-arid location.

### Rainfall Intensity (on rain days only)
- Mumbai: median 4.3 mm, p99 = ~140 mm — extreme monsoon events
- Ahmedabad: median 3.8 mm, p99 = ~108 mm
- Delhi: median 2.8 mm on rain days

### Monsoon Signature
- All cities show strong Jun–Sep monsoon precipitation peaks.
- Mumbai receives dramatically more monsoon rain than plains cities.
- Total period precipitation: Mumbai 69,423 mm vs Delhi 24,822 mm — a 2.8× difference.

---

## 9. Wind / Pressure / Radiation Findings

### Wind
- **Mumbai** has the highest mean wind speed (17.4 km/h) — consistent with maritime coastal exposure.
- **Nagpur** has the lowest mean wind speed (12.7 km/h).
- `wind_speed_10m_max` and `wind_gusts_10m_max` are highly correlated (r = +0.903).

### Surface Pressure
- Varies by altitude: Nagpur (elevation ~310m) mean 973.3 hPa vs Mumbai (sea level) mean 1008.4 hPa.
- Seasonal pressure pattern shows pre-monsoon low and winter high for all cities.
- Tmax–pressure correlation is negative and moderate for Delhi (r = −0.76) and Lucknow (r = −0.74), weaker for others.

### Shortwave Radiation
- **Ahmedabad** has the highest mean shortwave radiation (19.34 MJ/m²) — fewest cloud days.
- All cities show lower radiation during monsoon (Jun–Sep) due to cloud cover.
- Peak radiation occurs in **Apr–May** (pre-monsoon clear skies coinciding with the hottest period).

### Evapotranspiration (ET₀)
- **ET₀ is the strongest correlate of Tmax** across all cities (r ≈ +0.75 to +0.89).
- ET₀ is highest in **Ahmedabad** (mean 5.05 mm/day) — combined effect of high temperature, high radiation, and low humidity.
- ET₀ and shortwave radiation are strongly coupled (r = +0.901).

---

## 10. Correlation Findings

### Perfect / Near-Perfect Correlations (multicollinearity risk)
| Pair | r |
|---|---|
| `precipitation_sum` ↔ `rain_sum` | **+1.000** — identical |
| `temperature_2m_min` ↔ `apparent_temperature_mean` | +0.975 |
| `temperature_2m_min` ↔ `apparent_temperature_min` | +0.975 |
| `relative_humidity_2m_min` ↔ `relative_humidity_2m_mean` | +0.934 |
| `temperature_2m_mean` ↔ `apparent_temperature_max` | +0.965 |
| `wind_speed_10m_max` ↔ `wind_gusts_10m_max` | +0.903 |

### Strong Positive Correlations
- All temperature variables (actual and apparent) are strongly inter-correlated (r = +0.73 to +0.97).
- `shortwave_radiation_sum` ↔ `et0_fao_evapotranspiration`: r = +0.901

### Strong Negative Correlations
- `relative_humidity_2m_mean` ↔ `et0_fao_evapotranspiration`: r = −0.797
- `relative_humidity_2m_max` ↔ `et0_fao_evapotranspiration`: r = −0.765
- `relative_humidity_2m_mean` ↔ `shortwave_radiation_sum`: r = −0.706

### Implications for Feature Engineering (Phase 7)
- Dropping `rain_sum` is safe (identical to `precipitation_sum`).
- Evaluate whether to retain both humidity max and mean (r = +0.928).
- Temperature cluster has high multicollinearity — dimensionality reduction or careful feature selection will be needed.
- `et0_fao_evapotranspiration` integrates temperature, radiation, and humidity — strong candidate as a derived feature.

---

## 11. Preliminary Extreme-Temperature Findings

> **These are PRELIMINARY threshold analyses. Final heatwave labels = Phase 6 (IMD criteria).**

### Threshold-Exceedance Summary
| City | Threshold | Exceed days | % of days | Hottest date | Hottest T (°C) | Max consecutive run |
|---|---|---|---|---|---|---|
| New Delhi | ≥40°C | 1,223 | 9.39% | 1995-06-15 | 46.8 | 37 days |
| Lucknow | ≥40°C | 1,061 | 8.14% | 1995-06-16 | 46.3 | 33 days |
| **Nagpur** | ≥40°C | **1,761** | **13.52%** | 2010-05-24 | **46.9** | **56 days** |
| Ahmedabad | ≥40°C | 1,117 | 8.57% | 2016-05-19 | 45.6 | 37 days |
| Mumbai | ≥37°C | **8** | **0.06%** | 2004-05-03 | 37.6 | 1 day |

### Key Observations
- **Nagpur** has the most extreme-heat days (1,761, 13.52%) and the longest consecutive run (56 days) — it is climatically the most heatwave-prone city in this dataset.
- **Mumbai** essentially never experiences threshold-level heat even at the lower 37°C threshold — only 8 days in 35 years.
- Delhi and Lucknow saw their all-time highs during **June 1995** — the same multi-day heat event visible across both cities.
- Extreme days concentrate in **April–June** for all plains cities.
- There is no clear uniform increasing trend in annual exceedance counts for plains cities — year-to-year variability is high.

---

## 12. Coastal vs Plains Comparison

### Key Differences
| Metric | Plains (avg 4 cities) | Mumbai (coastal) |
|---|---|---|
| Mean Tmax (°C) | 32.10 | **30.38** (−1.72°C) |
| Tmax std (°C) | 5.77 | **2.11** (−3.66°C) |
| Mean Tmin (°C) | 21.14 | **23.84** (+2.70°C) |
| Mean RH (%) | 61.68 | **74.77** (+13.09%) |
| Mean Precip (mm/day) | 2.43 | **5.33** (×2.2) |
| Mean Wind (km/h) | 13.74 | **17.42** (+3.68 km/h) |
| Mean Pressure (hPa) | 988.32 | **1008.36** (+20 hPa) |

### Observed Coastal Characteristics (from data)
1. **Much narrower temperature range.** Mumbai Tmax std = 2.1°C vs plains ~5.8°C. The thermal buffering effect of the Arabian Sea is clearly visible in the data.
2. **Higher minimum temperatures.** Mumbai never goes below 13.1°C (Tmin), while Delhi drops to 2.5°C.
3. **Higher baseline humidity.** Mumbai mean RH = 74.8%, never dropping below 15%.
4. **Higher precipitation.** Mumbai total = 69,423 mm vs Delhi 24,822 mm over the same period.
5. **Higher wind speed.** Consistent with coastal maritime exposure.
6. **No extreme heat days.** Mumbai is protected from the pre-monsoon heat spikes that affect plains cities.

> **Limitation**: These differences reflect both coastal geography AND latitude differences. Mumbai is at 19°N vs Delhi at 28°N. The dataset alone cannot fully disentangle coastal effect from latitude effect.

---

## 13. Limitations

1. **ERA5 reanalysis, not ground observations.** Data is model-derived at 0.25° (~28 km) resolution. Local urban heat islands, sub-grid topography, and station-specific microclimates are not captured.

2. **Trend descriptions are observational, not causal.** The 35-year trends documented here (particularly for Tmin) are consistent with published Indian climate literature, but this dataset alone cannot establish causal attribution to any specific forcing.

3. **Heatwave labels not yet defined.** The preliminary threshold analysis uses simple fixed absolute thresholds. IMD heatwave criteria involve multiple conditions (absolute threshold AND anomaly from 30-year normal). Phase 6 will define proper labels.

4. **Only 5 cities.** Results describe these specific locations. Generalization to other Indian cities requires caution.

5. **`precipitation_sum` = `rain_sum`.** Snowfall contribution is zero for these locations. Only one of these columns is needed downstream.

6. **No urban heat island correction.** ERA5 data uses land-surface model representations and does not correct for urban microclimate.

7. **Period ends 2025-08-31.** Data is not available for Sep–Dec 2025.

8. **Latitude confound in coastal comparison.** Mumbai (19°N) is 9–10° further south than Delhi (28°N). Some climate differences attributed to "coastal" may partly reflect latitude.

---

## 14. Recommendations for Phase 5 (Data Cleaning)

Based on EDA findings, the following actions are recommended for Phase 5:

| Priority | Action | Justification |
|---|---|---|
| 🔴 Required | **Confirm dataset integrity** before cleaning begins | Re-run final validation to ensure raw file unchanged |
| 🟡 Recommended | **Drop `rain_sum`** from the processed dataset | Identical to `precipitation_sum` (r = 1.000) |
| 🟡 Recommended | **Retain all temperature variables** | High inter-correlation expected; feature selection = Phase 7 |
| 🟡 Recommended | **Retain `wind_gusts_10m_max`** for now | Correlated with wind_speed but may capture different extremes |
| 🟢 Optional | **Create date-derived columns** (month, year, DOY, season) | Needed for temporal feature engineering in Phase 7 |
| ⚠️ Do NOT | **Remove statistical outliers** | Extreme Tmax days are potential heatwave labels — not noise |
| ⚠️ Do NOT | **Impute any values** | Zero missing values — no imputation needed |
| ⚠️ Do NOT | **Modify `data/raw/`** | All cleaning goes to `data/processed/` |

### For Phase 6 (Heatwave Labeling)
- Use city-specific 30-year climatological normals (calculated from 1990–2020 baseline) for anomaly-based labeling.
- Mumbai's threshold should be reconsidered (37°C yielded only 8 events in 35 years — may need lower threshold or anomaly-only approach).
- Nagpur's long consecutive runs (up to 56 days at ≥40°C) suggest duration-based labeling will be important.

# ClimateGuard — Data Dictionary
## ERA5 Five-City Daily Weather Dataset

**File:** `data/raw/all_cities_era5_raw.csv`  
**Source:** Open-Meteo Historical Weather API — ERA5 model (0.25° resolution)  
**Period:** 1990-01-01 → 2025-08-31  
**Cities:** New Delhi, Lucknow, Nagpur, Ahmedabad, Mumbai  
**Total rows:** 65,135 (5 cities × 13,027 days)  
**Total columns:** 23  
**Missing values:** 0  
**Duplicate records:** 0  

---

## Meta Columns (Non-Weather)

| Column | Description | Type | Example |
|---|---|---|---|
| `city` | Full city name | string | `New Delhi` |
| `city_key` | Lowercase key identifier | string | `delhi` |
| `latitude` | City centre latitude (decimal degrees) | float64 | `28.6139` |
| `longitude` | City centre longitude (decimal degrees) | float64 | `77.2090` |
| `region_type` | Geographic region type | string | `plains` / `coastal` |
| `state` | Indian state name | string | `Delhi` |
| `date` | Calendar date (daily frequency) | datetime64 | `1990-01-01` |

### City Metadata

| city_key | city | state | region_type | latitude | longitude |
|---|---|---|---|---|---|
| delhi | New Delhi | Delhi | plains | 28.6139 | 77.2090 |
| lucknow | Lucknow | Uttar Pradesh | plains | 26.8467 | 80.9462 |
| nagpur | Nagpur | Maharashtra | plains | 21.1458 | 79.0882 |
| ahmedabad | Ahmedabad | Gujarat | plains | 23.0225 | 72.5714 |
| mumbai | Mumbai | Maharashtra | coastal | 19.0760 | 72.8777 |

---

## Weather Variables

### Temperature (6 variables)

| Variable | Description | Unit | Type | Min | Max | Mean | Median | Std | Missing | HW Relevant |
|---|---|---|---|---|---|---|---|---|---|---|
| `temperature_2m_max` | Daily maximum air temperature at 2 m above surface | °C | float64 | 10.20 | 46.90 | 31.76 | 31.60 | 5.29 | 0% | ✅ Yes |
| `temperature_2m_min` | Daily minimum air temperature at 2 m above surface | °C | float64 | 2.50 | 35.50 | 21.68 | 23.70 | 6.03 | 0% | ✅ Yes |
| `temperature_2m_mean` | Daily mean air temperature at 2 m above surface | °C | float64 | 5.90 | 40.10 | 26.27 | 27.20 | 5.45 | 0% | ✅ Yes |
| `apparent_temperature_max` | Daily maximum apparent (feels-like) temperature | °C | float64 | 8.20 | 50.90 | 34.40 | 35.30 | 6.66 | 0% | ✅ Yes |
| `apparent_temperature_min` | Daily minimum apparent (feels-like) temperature | °C | float64 | -0.50 | 39.00 | 23.47 | 25.80 | 8.32 | 0% | ✅ Yes |
| `apparent_temperature_mean` | Daily mean apparent (feels-like) temperature | °C | float64 | 3.80 | 42.80 | 28.42 | 30.50 | 7.29 | 0% | ✅ Yes |

**Notes:**
- Apparent temperature (feels-like) combines temperature, humidity, and wind speed into a heat-index-like measure.
- Apparent Tmax reaches 50.9°C (Mumbai, monsoon) — significantly higher than actual Tmax due to high humidity.

---

### Precipitation (2 variables)

| Variable | Description | Unit | Type | Min | Max | Mean | Median | Std | Missing | HW Relevant |
|---|---|---|---|---|---|---|---|---|---|---|
| `precipitation_sum` | Total daily precipitation (rain + snow water equivalent) | mm | float64 | 0.00 | 317.30 | 3.01 | 0.00 | 10.11 | 0% | ✅ Yes |
| `rain_sum` | Daily liquid rain total | mm | float64 | 0.00 | 317.30 | 3.01 | 0.00 | 10.11 | 0% | ⬜ No |

**Notes:**
- `precipitation_sum` and `rain_sum` are perfectly correlated (r = +1.000) in this dataset. Snow contribution is negligible for these tropical/semi-arid locations.
- Highly zero-inflated: 60–74% of days have zero rainfall.
- Maximum 317.3 mm recorded in Mumbai (monsoon day).

---

### Wind (2 variables)

| Variable | Description | Unit | Type | Min | Max | Mean | Median | Std | Missing | HW Relevant |
|---|---|---|---|---|---|---|---|---|---|---|
| `wind_speed_10m_max` | Maximum daily wind speed at 10 m above surface | km/h | float64 | 4.30 | 53.90 | 14.48 | 13.80 | 4.83 | 0% | ✅ Yes |
| `wind_gusts_10m_max` | Maximum daily wind gusts at 10 m above surface | km/h | float64 | 11.50 | 99.40 | 29.31 | 28.10 | 8.61 | 0% | ⬜ No |

**Notes:**
- `wind_speed_10m_max` and `wind_gusts_10m_max` are highly correlated (r = +0.903).
- Wind can reduce apparent heat stress and is relevant to felt temperature.

---

### Relative Humidity (3 variables)

| Variable | Description | Unit | Type | Min | Max | Mean | Median | Std | Missing | HW Relevant |
|---|---|---|---|---|---|---|---|---|---|---|
| `relative_humidity_2m_max` | Daily maximum relative humidity at 2 m | % | int64 | 20 | 100 | 80.23 | 85 | 15.56 | 0% | ✅ Yes |
| `relative_humidity_2m_min` | Daily minimum relative humidity at 2 m | % | int64 | 4 | 95 | 45.43 | 42 | 20.01 | 0% | ✅ Yes |
| `relative_humidity_2m_mean` | Daily mean relative humidity at 2 m | % | int64 | 14 | 99 | 64.30 | 66 | 17.54 | 0% | ✅ Yes |

**Notes:**
- Humidity is a critical heatwave amplifier. High humidity prevents evaporative cooling and raises heat stress.
- Mumbai minimum RH never drops below 15% (vs 4% for Delhi). Plains cities experience very dry pre-monsoon periods.

---

### Surface Pressure (1 variable)

| Variable | Description | Unit | Type | Min | Max | Mean | Median | Std | Missing | HW Relevant |
|---|---|---|---|---|---|---|---|---|---|---|
| `surface_pressure_mean` | Daily mean surface (station) pressure | hPa | float64 | 957.90 | 1018.10 | 992.33 | 993.70 | 13.82 | 0% | ⬜ No |

**Notes:**
- Surface pressure varies substantially across cities due to altitude differences (Nagpur at ~310m has lower pressure than Mumbai at sea level).
- Nagpur: mean 973.3 hPa; Mumbai: mean 1008.4 hPa; difference reflects ~350m altitude difference.

---

### Radiation & Evapotranspiration (2 variables)

| Variable | Description | Unit | Type | Min | Max | Mean | Median | Std | Missing | HW Relevant |
|---|---|---|---|---|---|---|---|---|---|---|
| `shortwave_radiation_sum` | Daily sum of downward shortwave solar radiation at surface | MJ/m² | float64 | 1.14 | 29.42 | 18.82 | 18.83 | 5.37 | 0% | ✅ Yes |
| `et0_fao_evapotranspiration` | Daily FAO reference evapotranspiration (water stress proxy) | mm | float64 | 0.38 | 12.27 | 4.65 | 4.37 | 1.83 | 0% | ✅ Yes |

**Notes:**
- ET₀ is the strongest single correlate of Tmax across all cities (r ≈ +0.75 to +0.89).
- ET₀ is strongly correlated with shortwave radiation (r = +0.901).
- High ET₀ indicates intense evaporative demand — a proxy for heat stress on vegetation and humans.

---

## Summary: Variables Likely Relevant for Heatwave Prediction

| Variable | Relevance |
|---|---|
| `temperature_2m_max` | **Primary signal** — main heatwave indicator |
| `temperature_2m_min` | Nighttime recovery failure is a heatwave amplifier |
| `temperature_2m_mean` | Contextual thermal load |
| `apparent_temperature_max` | Human heat stress indicator (includes humidity + wind) |
| `apparent_temperature_mean` | Cumulative felt heat |
| `relative_humidity_2m_mean` | Modifies heat stress severity |
| `relative_humidity_2m_min` | Dry conditions relevant for fire/evaporative risk |
| `precipitation_sum` | Absence of rain precedes heatwaves |
| `wind_speed_10m_max` | Wind-driven cooling effect |
| `shortwave_radiation_sum` | Solar heating driver |
| `et0_fao_evapotranspiration` | Integrated heat+dryness stress metric |

Variables **not recommended as primary features** (but retained in raw data):
- `rain_sum` — duplicate of `precipitation_sum`
- `wind_gusts_10m_max` — largely redundant with `wind_speed_10m_max`
- `surface_pressure_mean` — low Tmax correlation in most cities
- `apparent_temperature_min` — less informative than max/mean

> ⚠️ Final feature selection occurs in **Phase 7 — Feature Engineering**. No variables should be removed from the raw dataset.

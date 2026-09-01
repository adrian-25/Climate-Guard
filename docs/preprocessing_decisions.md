# ClimateGuard — Preprocessing Decisions
## Phase 5: Data Cleaning

**Input:**  `data/raw/all_cities_era5_raw.csv`  
**Output:** `data/processed/weather_cleaned.csv`  
**Script:** `data_cleaning.py`  
**Date:**   2026-09-01  
**Result:**  105 checks passed / 0 failed — DATA CLEANING STATUS: **PASS**

---

## Raw Dataset Integrity

| Check | Result |
|---|---|
| MD5 before cleaning | `71d25a015e2c6a8015a155785b8d7cd0` |
| MD5 after cleaning | `71d25a015e2c6a8015a155785b8d7cd0` |
| Raw file modified? | **NO** |
| `rain_sum` still in raw? | **YES** (raw untouched) |

The raw master dataset was opened read-only and was never written to. The cleaned output was saved to a separate file in `data/processed/`.

---

## Shape Change

| | Rows | Columns |
|---|---|---|
| Raw dataset | 65,135 | 23 |
| Cleaned dataset | **65,135** | **22** |
| Rows removed | 0 | — |
| Columns removed | — | 1 (`rain_sum`) |

---

## What Was Changed

### 1. `rain_sum` — REMOVED

**Column removed:** `rain_sum`

**Evidence from EDA:**
- Pearson correlation with `precipitation_sum`: r = **+1.000**
- Maximum absolute difference between the two columns: **0.0** (confirmed by `data_cleaning.py` Check 9)
- The two series are byte-for-byte identical for all 65,135 rows

**Reason:**
For these five Indian cities (tropical/semi-arid, no snowfall contribution), total precipitation = liquid rain. ERA5 returned identical values for both variables across the entire 1990–2025 period.

Retaining `rain_sum` alongside `precipitation_sum` would introduce perfect multicollinearity into any downstream model — a guaranteed source of instability in linear models and a source of redundant splits in tree-based models. It adds zero information.

**Decision:** `rain_sum` is removed from the processed dataset. `precipitation_sum` is retained.

---

## What Was NOT Changed

### 2. No rows removed

The EDA found:
- 0 physically invalid values
- 0 missing values
- 0 duplicate records
- 0 date gaps

There was no basis for row removal. All 65,135 rows are retained.

### 3. No imputation performed

**Reason:** The dataset contains 0 missing values across all 16 weather variables. Imputation would fabricate data where none is needed and is explicitly not required.

### 4. Extreme temperature observations retained — NOT treated as outliers

**Reason:** ClimateGuard's core purpose is heatwave risk prediction. The highest temperatures in the dataset are the primary signal the system must learn to detect and predict.

EDA confirmed:
- All values are within physical bounds for Indian climate (no impossible readings)
- Statistical IQR outliers in Tmax correspond to genuine extreme heat days (e.g., Delhi 46.8°C on 1995-06-15, Nagpur 46.9°C on 2010-05-24)
- These observations are exactly what Phase 6 heatwave labeling will identify as positive examples

Removing them would destroy the dataset's ability to support heatwave prediction. They are retained in full.

### 5. `wind_gusts_10m_max` retained

EDA showed `wind_speed_10m_max` ↔ `wind_gusts_10m_max` correlation of r = +0.903. Despite high correlation, this variable was **not removed**. Reasons:
- High correlation ≠ zero additional information
- Gusts capture instantaneous extremes not reflected in max sustained wind
- Feature selection belongs to Phase 7 — not Phase 5

The same logic applies to all other correlated variable pairs (temperature cluster, humidity trio).

### 6. No lag features, rolling features, or derived variables created

These belong to Phase 7 — Feature Engineering. Phase 5 is strictly structural cleaning.

### 7. No heatwave labels created

Heatwave labeling (IMD criteria, city-specific normals, duration conditions) belongs to Phase 6.

### 8. Column ordering and sort order

- Columns are ordered: meta (6) → date (1) → weather variables (15)
- Rows are sorted by `city_key` (alphabetical), then `date` (ascending)
- Sort order is a convenience — it has no effect on data values

---

## Final Dataset Specification

**File:** `data/processed/weather_cleaned.csv`

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `city` | string | Full city name |
| 2 | `city_key` | string | Lowercase city identifier |
| 3 | `latitude` | float64 | City latitude (decimal degrees) |
| 4 | `longitude` | float64 | City longitude (decimal degrees) |
| 5 | `region_type` | string | `plains` or `coastal` |
| 6 | `state` | string | Indian state |
| 7 | `date` | datetime64 | Calendar date |
| 8 | `temperature_2m_max` | float64 | Daily max temperature (°C) |
| 9 | `temperature_2m_min` | float64 | Daily min temperature (°C) |
| 10 | `temperature_2m_mean` | float64 | Daily mean temperature (°C) |
| 11 | `apparent_temperature_max` | float64 | Daily max apparent temperature (°C) |
| 12 | `apparent_temperature_min` | float64 | Daily min apparent temperature (°C) |
| 13 | `apparent_temperature_mean` | float64 | Daily mean apparent temperature (°C) |
| 14 | `precipitation_sum` | float64 | Daily total precipitation (mm) |
| 15 | `wind_speed_10m_max` | float64 | Daily max wind speed at 10m (km/h) |
| 16 | `wind_gusts_10m_max` | float64 | Daily max wind gusts at 10m (km/h) |
| 17 | `relative_humidity_2m_max` | int64 | Daily max relative humidity (%) |
| 18 | `relative_humidity_2m_min` | int64 | Daily min relative humidity (%) |
| 19 | `relative_humidity_2m_mean` | int64 | Daily mean relative humidity (%) |
| 20 | `surface_pressure_mean` | float64 | Daily mean surface pressure (hPa) |
| 21 | `shortwave_radiation_sum` | float64 | Daily solar radiation sum (MJ/m²) |
| 22 | `et0_fao_evapotranspiration` | float64 | Daily FAO reference ET₀ (mm) |

---

## Validation Results (Post-Save)

All checks run on the saved `weather_cleaned.csv` file after re-loading from disk:

| Check | Result |
|---|---|
| Row count = 65,135 | ✅ PASS |
| Column count = 22 | ✅ PASS |
| `rain_sum` absent | ✅ PASS |
| `precipitation_sum` present | ✅ PASS |
| Missing values = 0 | ✅ PASS |
| city+date duplicates = 0 | ✅ PASS |
| All 5 cities present | ✅ PASS |
| Date range 1990-01-01 → 2025-08-31 | ✅ PASS |
| All 15 weather variables within physical ranges | ✅ PASS (15/15) |
| Raw dataset MD5 unchanged | ✅ PASS |

**Total checks: 105 passed / 0 failed**

---

## Files Created

| File | Description |
|---|---|
| `data/processed/weather_cleaned.csv` | Cleaned ML-ready base dataset (8.54 MB) |
| `results/phase5_cleaning_log.txt` | Full machine-readable cleaning log |
| `docs/preprocessing_decisions.md` | This document |
| `data_cleaning.py` | Reproducible cleaning script |

# Final ML Dataset — ClimateGuard Phase 8

**Created:** 2026-09-01  
**Script:** `build_ml_dataset.py`  
**Upstream input:** `data/features/climateguard_features.csv` (Phase 7 output, MD5: `fdb559545ef4a0155fbb5c8a813c9eb8`)

---

## 1. Input Dataset

| Property | Value |
|---|---|
| File | `data/features/climateguard_features.csv` |
| Shape | 65,095 rows x 121 columns |
| Phase | 7 (Feature Engineering) |
| Status | Read-only — not modified by Phase 8 |
| MD5 (verified) | `fdb559545ef4a0155fbb5c8a813c9eb8` |

The Phase 7 output contains all engineered features for all 5 cities over the date range **1990-01-08 to 2025-08-30** (40 rows dropped from the Phase 6 labelled dataset: 7 head + 1 tail per city x 5 cities).

---

## 2. Baseline Feature Definition

**File:** `data/features/ml_baseline.csv`  
**Feature count:** 29  
**Source:** `results/phase7_feature_groups.json` -> `baseline_features`

The baseline feature set represents **current-day, non-temporal information only**. It contains no lags, no rolling windows, and no trend features. It is used in Phase 13 as the non-temporal comparison model.

| Group | Features | Count |
|---|---|---|
| Group 1 — Current weather | `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `apparent_temperature_max`, `apparent_temperature_min`, `apparent_temperature_mean`, `precipitation_sum`, `wind_speed_10m_max`, `wind_gusts_10m_max`, `relative_humidity_2m_max`, `relative_humidity_2m_min`, `relative_humidity_2m_mean`, `surface_pressure_mean`, `shortwave_radiation_sum`, `et0_fao_evapotranspiration`, `tmax_normal`, `tmax_departure`, `qualifying_day` | 18 |
| Group 6 — Calendar | `month`, `day_of_year`, `season_code`, `month_sin`, `month_cos`, `doy_sin`, `doy_cos` | 7 |
| Group 7 — City | `city_encoded`, `is_coastal`, `latitude`, `longitude` | 4 |

**Total: 29 features**

---

## 3. Temporal Feature Definition

**File:** `data/features/ml_temporal.csv`  
**Feature count:** 110  
**Source:** `results/phase7_feature_groups.json` -> `temporal_features`

The temporal feature set extends the baseline with lag, rolling, trend, and anomaly features that give the model access to recent weather history.

| Group | Description | Count |
|---|---|---|
| Group 1 — Current weather | 15 weather variables + tmax_normal + tmax_departure + qualifying_day | 18 |
| Group 2 — Lag features | T-1/T-2/T-3/T-7 lags for 7 key variables + tmax_departure lags + heatwave_lag1 | 33 |
| Group 3 — Rolling features | 3-day and 7-day rolling mean/max/min for 7 key variables (shift(1).rolling(N)) | 42 |
| Group 4 — Trend features | tmax_delta_1d/3d/7d + tmax_slope_3d/7d | 5 |
| Group 5 — Anomaly features | tmax_departure_zscore (30-day trailing z-score) | 1 |
| Group 6 — Calendar features | month, day_of_year, season_code, sin/cos encodings | 7 |
| Group 7 — City features | city_encoded, is_coastal, latitude, longitude | 4 |

**Total: 110 features**

---

## 4. Target Definition

```
TARGET: heatwave_next_day

heatwave_next_day(T) = heatwave(T+1)
```

The model uses features available on date T to predict whether the *following* day (T+1) is a heatwave day.

- Type: `float64`, binary values `{0.0, 1.0}`
- Computed per-city via `groupby('city_key')['heatwave'].shift(-1)` in Phase 7
- Verified to equal `heatwave(T+1)` exactly for all 5 cities (0 mismatches)
- Appears as the **last column** in both output CSVs
- **Excluded from all feature lists** (verified programmatically in Phase 8)

---

## 5. Number of Features

| Dataset | Features | Target | ID columns | Total columns |
|---|---|---|---|---|
| ml_baseline.csv | 29 | 1 | 10 | 40 |
| ml_temporal.csv | 110 | 1 | 10 | 121 |

ID columns (present for traceability, not used as ML features):
`city`, `city_key`, `date`, `state`, `region_type`, `heatwave`, `hw_event_id`, `hw_event_start`, `hw_event_end`, `hw_event_length`

---

## 6. Number of Rows

| Stage | Rows | Notes |
|---|---|---|
| weather_labelled.csv (Phase 6) | 65,135 | 5 cities x 13,027 days |
| climateguard_features.csv (Phase 7) | 65,095 | Minus 40 (7 head + 1 tail per city x 5) |
| ml_baseline / ml_temporal (Phase 8) | **65,080** | Minus 15 (3 zscore-NaN rows per city x 5) |

Per city: **13,016 rows** each.  
Date range per city: **1990-01-11 to 2025-08-30**.

---

## 7. City Distribution

| City | city_key | Region | Rows |
|---|---|---|---|
| New Delhi | delhi | plains | 13,016 |
| Lucknow | lucknow | plains | 13,016 |
| Nagpur | nagpur | plains | 13,016 |
| Ahmedabad | ahmedabad | plains | 13,016 |
| Mumbai | mumbai | coastal | 13,016 |
| **Total** | | | **65,080** |

---

## 8. Class Distribution

### Overall (identical for both datasets)

| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 64,575 | 99.22% |
| Heatwave (1) | 505 | 0.78% |
| **Total** | **65,080** | |

**Imbalance ratio: 1:128** (128 normal days per heatwave day)

### Per City

| City | Total | Positive | Negative | Positive % |
|---|---|---|---|---|
| Delhi | 13,016 | 213 | 12,803 | 1.64% |
| Lucknow | 13,016 | 141 | 12,875 | 1.08% |
| Nagpur | 13,016 | 119 | 12,897 | 0.91% |
| Ahmedabad | 13,016 | 32 | 12,984 | 0.25% |
| **Mumbai** | 13,016 | **0** | 13,016 | **0.00%** |

Mumbai has zero positive heatwave events. See Section 12 for the full rationale.

---

## 9. Missing Value Handling

### hw_event_start / hw_event_end

- 64,590 NaN values each (99.2% of rows)
- These are NaN for all non-heatwave rows by construction from Phase 6
- They are **passthrough metadata**, not ML features
- No action taken — NaN is the correct value for non-event days

### tmax_departure_zscore (resolved)

**Root cause:** In Phase 7, the zscore was computed using:
```python
dep_past.rolling(window=30, min_periods=10).std()
```
Phase 7 dropped the first 7 rows per city. The first 3 rows remaining in the Phase 7 output (original rows 7, 8, 9 of each city) had only 7, 8, and 9 prior observations — all below `min_periods=10` — so `rolling().std()` returned NaN.

**Resolution:** Phase 8 drops these 3 rows per city (15 rows total). These rows are not imputed or zero-filled — they are removed because the zscore is genuinely undefined for them. No artificial values were introduced.

**Final state:** 0 NaN values in any feature column of either ML dataset (verified).

---

## 10. Leakage Prevention

### Rules verified

| Rule | Status |
|---|---|
| All lag features have lag >= 1 | PASS |
| Rolling features use shift(1).rolling(N) — window is [T-N, ..., T-1] | PASS |
| Delta features compare T vs T-N — T is current-day (allowed as Group 1) | PASS |
| No `_lead`, `_next`, `_t1`, `t_plus` patterns in any feature name | PASS |
| Target (heatwave_next_day) not in any feature list | PASS |
| heatwave(T) not used directly — only heatwave_lag1 = heatwave(T-1) | PASS |
| No weather variable from T+1 or later | PASS |
| City boundaries respected: all shift/rolling applied inside per-city groupby | PASS |

### Audit results

- Phase 7 leakage audit: **PASSED** (110 features, 0 issues)
- Phase 8 leakage audit: **PASSED** (0 issues on final column sets)
- Full per-feature audit table: `results/phase8_feature_audit.csv`

Audit table columns: `feature`, `in_baseline`, `in_temporal`, `feature_group`, `time_reference`, `dtype`, `role`, `leakage_status`

---

## 11. Why Baseline and Temporal Datasets Are Maintained Separately

The project includes a controlled experiment in Phase 13:

> **Does providing a model with weather history (lags, rolling windows, trends) improve next-day heatwave prediction over using only current-day observations?**

To answer this cleanly:
- `ml_baseline.csv` — current-day information only (Groups 1, 6, 7)
- `ml_temporal.csv` — full temporal feature set (all 7 groups)

Both datasets have identical rows, identical targets, and identical identifier columns. They differ only in their feature columns. This design permits a fair, controlled comparison on exactly the same observations.

The feature group registry (`results/phase7_feature_groups.json`) allows Phase 13 to reconstruct each feature matrix without duplicating data storage.

---

## 12. Mumbai's Zero-Positive Status

Mumbai has **zero heatwave days** across 35 years (1990–2025) under the IMD-inspired coastal threshold:

```
qualifying_day = 1  if:
    temperature_2m_max >= 37.0 C  AND  tmax_departure >= 4.5 C

heatwave = 1  if:  qualifying_day=1 for >= 2 consecutive days
```

This result is scientifically correct:

- Mumbai's maritime climate produces a very narrow Tmax range (std 2.1°C vs ~5.8°C for plains cities)
- The 37°C threshold + departure requirement is rarely met simultaneously in a coastal environment
- EDA confirmed only 8 qualifying days in 35 years for Mumbai, none forming a consecutive run of 2+
- The Times of India (quoted in Phase 6 methodology) confirms the IMD Mumbai head uses a 37°C coastal threshold

**Mumbai is kept in both ML datasets.** Its 13,016 rows of negative examples represent conditions that do not produce heatwaves under the operational definition — information a classifier can use. The decision on whether to include or exclude Mumbai from supervised training is deferred to Phase 10 (Baseline Models), where its effect on model behaviour can be measured empirically.

**Do not add artificial positives for Mumbai.**

---

## 13. Limitations

| Limitation | Detail |
|---|---|
| ERA5 reanalysis, not IMD station data | Systematic warm/cool biases possible vs. observed surface temperatures |
| IMD-inspired label, not official IMD | Heatwave labels are derived from ERA5 data using IMD-inspired thresholds, not official IMD observations |
| No teleconnection features | El Nino, IOD, MJO indices (known heatwave precursors) are not included |
| No spatial features | City identity encoded as integer; no geographic distance, terrain, or wind-direction features |
| qualifying_day in feature set | Strongly correlated with target; may inflate apparent model performance — review before final evaluation |
| Severe class imbalance (1:128) | Standard accuracy is misleading; precision, recall, F1, and AUC required; addressed in Phase 11 |
| Mumbai zero positives | Model trained on all 5 cities will never see a Mumbai heatwave; Mumbai evaluation limited to true-negative performance |
| No external validation | Dataset covers one geographic region; generalization to other Indian cities is untested |

---

## Files Produced by Phase 8

| File | Shape | Size | Description |
|---|---|---|---|
| `data/features/ml_baseline.csv` | 65,080 x 40 | 16.66 MB | Baseline ML dataset |
| `data/features/ml_temporal.csv` | 65,080 x 121 | 59.22 MB | Temporal ML dataset |
| `results/phase8_report.txt` | — | — | Full validation run log (23 checks, all PASSED) |
| `results/phase8_feature_audit.csv` | 121 rows | — | Per-feature audit table |
| `build_ml_dataset.py` | — | — | Reproducible script |

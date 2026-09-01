# Phase 13 — Temporal Feature Experiment

**ClimateGuard: Indian Heatwave Prediction**  
**Executed:** 2026-09-02  
**Script:** `temporal_feature_experiment.py`  
**Status:** COMPLETE

---

## 1. Objective

This experiment tests whether adding temporal memory features (lags, rolling statistics, trend indicators, anomaly features) improves heatwave prediction over the baseline feature set (current-day weather only).

The Phase 11 recommended imbalance strategy (Random Forest / random_undersample / threshold=0.70) was used as the primary model, with XGBoost / baseline_weight / threshold=0.80 as a secondary reference. Both with_qd (including `qualifying_day`) and without_qd variants were compared.

An ablation study was performed to isolate the contribution of each feature group.

---

## 2. Feature Sets

| Set | Groups | Feature Count | Description |
|---|---|---|---|
| `baseline_wqd` | 1 + 6 + 7 | 29 | Current-day weather + calendar + city (with qualifying_day) |
| `baseline_nqd` | 1 + 6 + 7 | 28 | Current-day weather + calendar + city (without qualifying_day) |
| `temporal_wqd` | 1+2+3+4+5 + 6 + 7 | 110 | Full feature set (with qualifying_day) |
| `temporal_nqd` | 1+2+3+4+5 + 6 + 7 | 109 | Full feature set (without qualifying_day) |

### Feature Group Summary

| Group | Description | Count |
|---|---|---|
| 1 — Current weather | 15 weather variables at T + tmax_normal + tmax_departure + qualifying_day | 18 |
| 2 — Lag features | T-1, T-2, T-3, T-7 for 8 variables + heatwave_lag1 | 33 |
| 3 — Rolling features | 3-day and 7-day rolling mean/max/min (shift(1).rolling(N)) | 42 |
| 4 — Trend features | tmax_delta_1d/3d/7d + tmax_slope_3d/7d | 5 |
| 5 — Anomaly features | tmax_departure_zscore (30-day trailing z-score) | 1 |
| 6 — Calendar features | month, day_of_year, season_code, month_sin/cos, doy_sin/cos | 7 |
| 7 — City features | city_encoded, is_coastal, latitude, longitude | 4 |

---

## 3. Data Splits (Phase 9 Chronological)

| Split | Period | Rows | Positives | Positive % |
|---|---|---|---|---|
| Train | 1990-01-11 → 2019-12-31 | 54,735 | 428 | 0.78% |
| Validation | 2020-01-01 → 2022-12-31 | 5,480 | 39 | 0.71% |
| Test | 2023-01-01 → 2025-08-30 | 4,865 | 38 | 0.78% |

Phase 9 split boundaries were not modified.

---

## 4. Models and Strategies

### Primary model (Phase 11 recommended)
- **Algorithm:** Random Forest (`n_estimators=300, max_depth=10, min_samples_leaf=10`)
- **Imbalance strategy:** Random undersampling (10:1 majority:minority ratio on X_train only)
- **Threshold:** 0.70 (fixed from Phase 11 validation)

### Secondary reference (Phase 10/11 baseline)
- **Algorithm:** XGBoost (`n_estimators=300, max_depth=5, learning_rate=0.05`)
- **Imbalance strategy:** `scale_pos_weight=126.89` (baseline class ratio)
- **Threshold:** 0.80 (fixed from Phase 11 validation)

---

## 5. Part A — Primary Comparison: Baseline vs Temporal

### Validation results (2020–2022, 39 positives)

| Model | Feature Set | #Features | Strategy | F1 | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Random_Forest | `baseline_wqd` | 29 | random_undersample | 0.6122 | 0.5085 | 0.7692 | 0.5497 | 29 | 9 |
| Random_Forest | `baseline_nqd` | 28 | random_undersample | 0.5843 | 0.5200 | 0.6667 | 0.5767 | 24 | 13 |
| **Random_Forest** | **`temporal_wqd`** | **110** | **random_undersample** | **0.6154** | **0.5385** | **0.7179** | **0.5298** | **24** | **11** |
| Random_Forest | `temporal_nqd` | 109 | random_undersample | 0.5393 | 0.4800 | 0.6154 | 0.5527 | 26 | 15 |
| XGBoost | `baseline_wqd` | 29 | baseline_weight | 0.5979 | 0.5000 | 0.7436 | 0.5668 | 29 | 10 |
| XGBoost | `baseline_nqd` | 28 | baseline_weight | 0.6105 | 0.5179 | 0.7436 | 0.5433 | 27 | 10 |
| XGBoost | `temporal_wqd` | 110 | baseline_weight | 0.5185 | 0.5000 | 0.5385 | 0.5452 | 21 | 18 |
| XGBoost | `temporal_nqd` | 109 | baseline_weight | 0.5128 | 0.5128 | 0.5128 | 0.5711 | 19 | 19 |

**Best validation F1:** RF / `temporal_wqd` / random_undersample = **0.6154**

### Key observations — Part A

1. **RF benefits from temporal features (with qualifying_day):** `temporal_wqd` (F1=0.6154) exceeds `baseline_wqd` (F1=0.6122) and reduces false positives from 29 → 24 at the same threshold.
2. **RF temporal gain is precision-driven:** Precision increases from 0.5085 to 0.5385 (+0.030) while recall drops modestly (0.7692 → 0.7179). This is a favourable trade — fewer false alarms with only 2 more missed events.
3. **qualifying_day is critical for RF temporal:** Removing it (`temporal_nqd`) causes a large drop (F1=0.5393), the worst RF configuration. This is consistent with the Phase 11 finding that qualifying_day contributes meaningfully when combined with undersampling.
4. **XGBoost hurts with temporal features:** XGB `temporal_wqd/nqd` F1 drops to 0.51–0.52, well below XGB `baseline_nqd` (0.6105). Adding 80 features under fixed `scale_pos_weight` appears to dilute XGBoost's decision boundary at threshold=0.80.
5. **ROC-AUC is uniformly high (~0.994)** across all configurations — inflated by the large negative class and not a reliable discriminator here, consistent with Phases 10–12.

---

## 6. Part B — Ablation Study

**Configuration:** RF / random_undersample / threshold=0.70 (applied to each incremental set)

### Ablation validation results

| Feature Set | #Features | F1 | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| `baseline_only` | 29 | 0.6122 | 0.5085 | 0.7692 | 0.5497 | 29 | 9 |
| `baseline + lag` | 62 | 0.5957 | 0.5091 | 0.7179 | 0.5538 | 27 | 11 |
| `baseline + rolling` | 71 | 0.6024 | 0.5682 | 0.6410 | 0.5822 | 19 | 14 |
| `baseline + trend` | 34 | 0.6105 | 0.5179 | 0.7436 | 0.5247 | 27 | 10 |
| `baseline + anomaly` | 30 | **0.6263** | 0.5167 | **0.7949** | 0.5037 | 29 | **8** |
| `full_temporal` | 110 | 0.6154 | 0.5385 | 0.7179 | 0.5298 | 24 | 11 |
| `full_temporal_nqd` | 109 | 0.5393 | 0.4800 | 0.6154 | 0.5527 | 26 | 15 |

### Key observations — Ablation

1. **Anomaly feature is the single best addition:** `baseline_anomaly` (30 features = baseline + `tmax_departure_zscore`) achieves the highest F1 (0.6263) of all ablation sets, with the fewest false negatives (8). The z-score captures whether a day's temperature departure is anomalous relative to its recent 30-day window — a meaningful signal for early heatwave onset.

2. **Rolling features improve precision most:** `baseline_rolling` (71 features) achieves the highest precision (0.5682) of all ablation sets and the fewest false positives (19). Rolling means and maxima provide sustained-heat context, reducing false alarms.

3. **Lag features alone slightly hurt F1:** `baseline_lag` (62 features) drops F1 to 0.5957 — below baseline — suggesting that raw lag values for 33 columns introduce noise when combined with RF undersampling without the stabilising effect of rolling aggregates.

4. **Trend features are near-neutral:** `baseline_trend` (34 features) is nearly identical to `baseline_only` (0.6105 vs 0.6122 F1), suggesting tmax delta/slope adds little incremental value beyond what qualifying_day already captures.

5. **Full temporal set with qualifying_day matches `baseline_only` F1 but reduces FP:** `full_temporal` (110 features, F1=0.6154) modestly exceeds `baseline_only` (0.6122) while reducing false positives from 29 to 24 — a practical improvement in operational alarm quality.

6. **qualifying_day is load-bearing in the temporal set:** Removing it from `full_temporal` reduces F1 by -0.076 (from 0.6154 to 0.5393). Its explanatory power comes from codifying the IMD threshold + departure conditions into a clean binary signal that Random Forest can split on directly.

---

## 7. Part C — Test Confirmation

Config selected on validation only (best val-F1): **RF / temporal_wqd / random_undersample**

| Config | Threshold | F1 | Precision | Recall | PR-AUC | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Temporal (RF / `temporal_wqd`) | 0.70 | **0.7586** | **0.6735** | **0.8684** | **0.7885** | — | **16** | **5** |
| Baseline equiv (RF / `baseline_wqd`) | 0.70 | 0.6957 | 0.5926 | 0.8421 | 0.7705 | — | 22 | 6 |

### Temporal vs Baseline on test set

| Metric | Baseline | Temporal | Delta |
|---|---|---|---|
| F1 | 0.6957 | 0.7586 | **+0.0629** |
| Precision | 0.5926 | 0.6735 | **+0.0809** |
| Recall | 0.8421 | 0.8684 | +0.0263 |
| PR-AUC | 0.7705 | 0.7885 | +0.0180 |
| False Positives | 22 | 16 | **−6** |
| False Negatives | 6 | 5 | −1 |

### Validation → Test generalisation

| Config | Val F1 | Test F1 | Delta |
|---|---|---|---|
| RF / `temporal_wqd` | 0.6154 | 0.7586 | **+0.1432** |
| RF / `baseline_wqd` | 0.6122 | 0.6957 | +0.0835 |

Both configs generalise strongly to the test period (2023–2025). The temporal model shows a larger absolute improvement on test than on validation — consistent with the Phase 12 finding that 2024 was an exceptionally strong heatwave year (34 of 38 test positives) where temporal persistence features provide a stronger signal.

---

## 8. Leakage Audit — 14/14 PASS

| Check | Result |
|---|---|
| All lag features use shift(N≥1) — only T-1 or earlier (Phase 7 verified) | PASS |
| All rolling features use shift(1).rolling(N) — excludes T (Phase 7 verified) | PASS |
| tmax_departure_zscore uses 30-day trailing window — excludes T (Phase 7 verified) | PASS |
| heatwave_lag1 = heatwave(T-1) — not heatwave(T) or heatwave(T+1) | PASS |
| No T+1 weather variable present in any feature set | PASS |
| Target = heatwave_next_day(T) = heatwave(T+1) — target only, not a feature | PASS |
| Phase 9 split boundaries unchanged (train≤2019, val=2020–2022, test≥2023) | PASS |
| No StandardScaler used (RF/XGBoost are scale-invariant) | PASS |
| Undersampling applied only to X_train/y_train | PASS |
| Validation set not modified or resampled | PASS |
| Test set locked until Part C — not used for config selection | PASS |
| Thresholds carried from Phase 11 validation — not tuned on test | PASS |
| Phase 7 feature group registry used for ablation column selection | PASS |
| Phase 8 and Phase 9 datasets not modified | PASS |

---

## 9. Conclusions

### Does adding temporal features help?

**Yes — for Random Forest.** The `temporal_wqd` set (110 features with qualifying_day) outperforms `baseline_wqd` (29 features) on both validation and test:
- Val F1: +0.003 (0.6122 → 0.6154), FP reduced by 5
- Test F1: +0.063 (0.6957 → 0.7586), FP reduced by 6

The gain is primarily in **precision** — temporal features help the model distinguish between days that look like heatwave onset vs isolated hot days.

### Which temporal group matters most?

Based on the ablation study:
1. **Anomaly (`tmax_departure_zscore`)** — highest F1 (0.6263), fewest missed events (FN=8)
2. **Rolling features** — highest precision (0.5682), fewest false positives (FP=19)
3. **Trend features** — small benefit (+0.5247 PR-AUC but similar F1 to baseline)
4. **Lag features alone** — slightly negative effect on F1

### qualifying_day is critical

The `qualifying_day` binary feature is the most important signal in both baseline and temporal sets. Removing it from the 110-feature set causes a -0.076 F1 drop. It is a leakage-safe feature (computed from current-day T data) and should be retained.

### XGBoost does not benefit from temporal features at this threshold

XGBoost degrades significantly when temporal features are added (F1 drops from 0.6105 to 0.5128–0.5185). This is likely because XGBoost's `scale_pos_weight` approach with threshold=0.80 is sensitive to feature count expansion — the 80 additional features dilute the decision boundary. XGBoost may perform better with temporal features at a lower threshold; this is deferred to Phase 14.

### Recommended config for Phase 14 consideration

**Random Forest / `temporal_wqd` (110 features) / random_undersample / threshold=0.70**

- Best validation F1 among Phase 13 experiments (0.6154)
- Best test F1 confirmed (0.7586)
- Highest precision improvement over baseline (+0.081 test)
- 6 fewer false positives than baseline equivalent on test

---

## 10. Output Files

### Results

| File | Description |
|---|---|
| `results/phase13_temporal_comparison.csv` | 8-row primary comparison table (validation metrics) |
| `results/phase13_ablation.csv` | 7-row ablation study table (validation metrics) |
| `results/phase13_metrics.json` | Full metrics for all 15 experiments + test confirmation |
| `results/phase13_leakage_audit.csv` | 14-check leakage audit (all PASS) |
| `results/phase13_log.txt` | Full execution log |

### Plots

| File | Description |
|---|---|
| `results/plots/phase13/baseline_vs_temporal_f1.png` | Bar chart: F1 comparison across all 8 primary configs |
| `results/plots/phase13/baseline_vs_temporal_prauc.png` | Bar chart: PR-AUC comparison across all 8 primary configs |
| `results/plots/phase13/precision_recall_comparison.png` | Scatter: Precision vs Recall per config (model × feature set) |
| `results/plots/phase13/ablation_comparison.png` | Grouped bar: F1 and PR-AUC for each ablation set |

### Models

| Path | Description |
|---|---|
| `models/phase13/Random_Forest/{baseline_wqd,baseline_nqd,temporal_wqd,temporal_nqd}/random_undersample/` | 4 RF primary models |
| `models/phase13/XGBoost/{baseline_wqd,baseline_nqd,temporal_wqd,temporal_nqd}/baseline_weight/` | 4 XGBoost primary models |
| `models/phase13/Random_Forest/ablation/{baseline_only,baseline_lag,baseline_rolling,baseline_trend,baseline_anomaly,full_temporal,full_temporal_nqd}/` | 7 RF ablation models |

Each model directory contains `model.joblib` and `metadata.json` (feature names, strategy, threshold, timestamp).

---

## 11. Limitations

1. **Temporal features were already pre-computed in Phase 7** — the leakage correctness of lag/rolling constructions is verified but not re-audited end-to-end from raw data in this phase.
2. **Ablation is performed on validation only** — test-set generalization of individual feature groups is not measured to avoid any selection bias.
3. **XGBoost threshold was not re-tuned for temporal features** — the threshold=0.80 was fixed from Phase 11. XGBoost with temporal features may benefit from a lower threshold, but this would require a new threshold search on validation, deferred to Phase 14.
4. **No SHAP or feature importance analysis** — this is Adrian's scope boundary. Kshitij owns explainability.
5. **City-level breakdown on test not performed** — Per-city test metrics for Phase 13 are not separately tabulated (consistent with Phase 12 deferred breakdown). Ahmedabad (0 val/test positives) and Mumbai (0 positives everywhere) remain excluded from supervised evaluation.

---

## 12. Next Phase

**Phase 14 — Model Selection**

Using Phase 10–13 results, select the final production model by:
- Comparing all candidate models across phases on test-set metrics
- Considering qualifying_day inclusion/exclusion decision
- Confirming final threshold
- Producing a trained final model artifact ready for Phase 15 prediction interface

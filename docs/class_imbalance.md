# Class Imbalance Handling — ClimateGuard Phase 11

**Created:** 2026-09-01  
**Script:** `train_imbalance_models.py`  
**Split used:** Phase 9 chronological split (train 1990–2019, val 2020–2022, test 2023–2025)  
**Test set status:** Held out — NOT used in Phase 11

---

## 1. Why the Dataset Is Imbalanced

ClimateGuard predicts heatwave occurrence one day in advance across five Indian cities.
Heatwaves are, by definition, rare events.  The label `heatwave_next_day` derives from
the IMD-Inspired Operational Heatwave Label (ERA5-based), which requires:

- Plains cities: Tmax >= 40 °C **and** departure from normal >= 4.5 °C (or Tmax >= 45 °C absolute override)
- Coastal city (Mumbai): Tmax >= 37 °C and departure >= 4.5 °C
- Duration filter: at least **2 consecutive qualifying days** to form an event

These conditions are meteorologically strict.  Heatwave days cluster in the pre-monsoon
months (April–June) and are geographically concentrated in the plains cities.
Mumbai has zero qualifying events in the entire 1990–2025 record under the coastal definition.
Ahmedabad, while a plains city, has extremely few events and all fall in the training window.

The result is a severe class imbalance that cannot be engineered away without changing
the scientific definition or fabricating data — neither of which is acceptable.

---

## 2. Original Class Distribution

### Training set (1990-01-11 to 2019-12-31)

| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 54,307 | 99.22% |
| Heatwave (1) | 428 | 0.78% |
| **Imbalance ratio** | **1 : 126.9** | |

### Validation set (2020-01-01 to 2022-12-31)

| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 5,441 | 99.29% |
| Heatwave (1) | 39 | 0.71% |

### Per-city positives (validation)

| City | Positives | Notes |
|---|---|---|
| Delhi | 28 | Most positives — reliable city-level metrics |
| Lucknow | 7 | Sparse — city metrics are noisy |
| Nagpur | 4 | Very sparse — city metrics are unreliable |
| Ahmedabad | 0 | N/A — all 32 positives fall in training window |
| Mumbai | 0 | N/A — zero positives in any split (coastal definition) |

---

## 3. Strategies Tested

Six strategies were investigated.  All training-set manipulations were applied
**only** to `X_train / y_train`.  Validation and test data were never modified.

### Strategy 1 — Phase 10 Baseline Class Weighting (reference)

The Phase 10 models used `class_weight='balanced'` (LR, RF) and
`scale_pos_weight = 54307/428 = 126.89` (XGBoost).  These configurations were
re-run in Phase 11 as the reference baseline.  No changes to training data.

### Strategy 2 — Stronger Class Weights

Positive-class weights were doubled relative to the auto-balanced value:
- LR / RF: `{0: 1.0, 1: 254}` (2× the ratio)
- XGBoost: not repeated here — covered by the SPW grid

Purpose: determine whether more aggressive upweighting of the minority class
improves precision-recall balance.

### Strategy 3 — XGBoost scale_pos_weight Grid

Four explicit positive-class weight values were tested:

| SPW | Meaning |
|---|---|
| 64 | Under-weights positives relative to the ratio (more conservative) |
| 128 | Approximately equal to the natural ratio (≈ baseline) |
| 256 | 2× the ratio (aggressive upweighting) |
| 512 | 4× the ratio (very aggressive) |

Purpose: identify the sweet spot between recall and false-positive burden.

### Strategy 4 — Random Oversampling

Minority-class training rows were resampled **with replacement** to match the
majority class size (54,307 positive rows in the modified training set).
`scale_pos_weight=1` and `class_weight=None` were used alongside oversampling
to avoid double-counting the rebalancing.

**Applies only to X_train / y_train.  Validation and test are unchanged.**

### Strategy 5 — Random Undersampling

Majority-class training rows were randomly downsampled **without replacement** to
10× the minority class size (4,280 negative rows, giving a 1:10 ratio).
This reduces training time significantly but discards information.

**Applies only to X_train / y_train.  Validation and test are unchanged.**

Final training set size after undersampling: 428 positives + 4,280 negatives = 4,708 rows.

### Strategy 6 — SMOTE

**imbalanced-learn** was not available in this environment, so SMOTE was skipped.
The strategy slot was preserved in the experiment loop and the results columns show
`smote_skipped`.  The numbers reported for `smote_skipped` rows are the models trained
on the original unmodified training data (equivalent to no resampling) — they should
be read as "what would the model look like without SMOTE" rather than a true SMOTE result.

**Suitability note:**  SMOTE creates synthetic samples by interpolating between
minority-class neighbours in feature space.  For temporal weather data there are
two concerns: (a) synthetic samples mix feature vectors from different dates and
cities, potentially creating meteorologically implausible combinations; (b) the
heatwave class is temporally clustered (events last 2–12 days), so nearest-neighbour
interpolation may generate samples that resemble the middle of an event rather than
the onset — which is what the model must detect.  If imbalanced-learn is installed
in future, SMOTE should be applied **only to X_train/y_train** (post-split) so that
the chronological split is not violated.

---

## 4. Training-Only Resampling Rules

The following rules were enforced without exception:

| Rule | Verification |
|---|---|
| StandardScaler fitted only on X_train | PASS |
| StandardScaler transform-only on X_val | PASS |
| Oversampling applied only to X_train / y_train | PASS |
| Undersampling applied only to X_train / y_train | PASS |
| SMOTE applied only to X_train / y_train | PASS (skipped due to missing library) |
| Validation data not resampled | PASS |
| Test data not loaded | PASS |
| Thresholds chosen only from validation predictions | PASS |
| Phase 9 split boundaries unchanged | PASS |
| No T+1 weather features | PASS |
| Target absent from feature matrices | PASS |

**Overall leakage audit: PASSED**

---

## 5. Threshold Analysis

Because the positive class is rare (0.78%), the default 0.5 classification threshold
is suboptimal.  A model trained to upweight the minority class generates higher
predicted probabilities for borderline examples; using 0.5 as the cutoff often
produces too many false positives.  Conversely, raising the threshold too high
collapses recall.

The following thresholds were evaluated on validation predictions:
`0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90`

For each experiment, two rows are recorded in `phase11_imbalance_comparison.csv`:

1. Default threshold (0.50)
2. Best-F1 threshold (the threshold on the validation set that maximises F1, with
   precision as a tiebreaker)

Full threshold-by-threshold breakdowns (precision, recall, F1, PR-AUC, TP, FP, TN, FN)
for every experiment are in `results/phase11_threshold_analysis.csv`.

### Optimal threshold observations

| Model / Strategy | Opt Threshold | F1 at opt |
|---|---|---|
| LR with_qd / baseline | 0.90 | 0.5197 |
| LR without_qd / baseline | 0.90 | 0.4882 |
| RF with_qd / baseline | 0.70 | 0.5664 |
| RF without_qd / baseline | 0.90 | 0.5500 |
| XGBoost with_qd / baseline | 0.80 | 0.5979 |
| XGBoost without_qd / baseline | 0.80 | 0.6105 |
| RF with_qd / random_undersample | **0.70** | **0.6122** |
| RF without_qd / random_undersample | 0.70 | 0.5843 |

Logistic Regression models systematically benefit from very high thresholds (0.90)
because they assign high probabilities to many borderline examples.
Tree-based models benefit from moderately elevated thresholds (0.70–0.80).

---

## 6. Validation Results

### 6a. Default threshold (0.50) — selected rows

| Model | Feature Set | Strategy | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| LR | with_qd | baseline_weight | 0.1689 | 0.9744 | 0.2879 | **0.6356** | 187 | 1 |
| LR | without_qd | baseline_weight | 0.1625 | 1.0000 | 0.2796 | 0.6216 | 201 | 0 |
| RF | with_qd | baseline_weight | 0.3505 | 0.8718 | 0.5000 | 0.5535 | 63 | 5 |
| RF | without_qd | baseline_weight | 0.3301 | 0.8718 | 0.4789 | 0.5325 | 69 | 5 |
| XGB | with_qd | baseline_weight | 0.3974 | 0.7949 | 0.5299 | 0.5668 | 47 | 8 |
| XGB | without_qd | baseline_weight | 0.4051 | 0.8205 | 0.5424 | 0.5433 | 47 | 7 |
| XGB | with_qd | **spw_64** | **0.4545** | 0.7692 | **0.5714** | 0.5111 | **36** | 9 |
| XGB | without_qd | spw_64 | 0.4265 | 0.7436 | 0.5421 | 0.5223 | 39 | 10 |
| XGB | with_qd | spw_128 | 0.3896 | 0.7692 | 0.5172 | 0.5535 | 47 | 9 |
| XGB | without_qd | spw_512 | 0.3750 | 0.8462 | 0.5197 | 0.5842 | 55 | 6 |
| RF | with_qd | random_undersample | 0.4177 | 0.8462 | 0.5593 | 0.5497 | 46 | 6 |
| RF | without_qd | random_undersample | 0.4024 | 0.8462 | 0.5455 | 0.5767 | 49 | 6 |
| XGB | with_qd | random_oversample | 0.3750 | 0.7692 | 0.5042 | 0.5119 | 50 | 9 |

**Best F1 at default threshold: XGBoost / with_qd / spw_64 — F1 = 0.5714, P = 0.4545, FP = 36**

### 6b. Threshold-optimised — selected rows

| Model | Feature Set | Strategy | Opt Thresh | Precision | Recall | F1 | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **RF** | **with_qd** | **random_undersample** | **0.70** | **0.5085** | **0.7692** | **0.6122** | 0.5497 | **29** | 9 |
| RF | without_qd | random_undersample | 0.70 | 0.5200 | 0.6667 | 0.5843 | 0.5767 | 24 | 13 |
| RF | with_qd | smote_skipped | 0.20 | 0.5179 | 0.7436 | 0.6105 | 0.5951 | 27 | 10 |
| RF | without_qd | smote_skipped | 0.15 | 0.4848 | 0.8205 | 0.6095 | 0.6048 | 34 | 7 |
| XGB | without_qd | baseline_weight | 0.80 | 0.5179 | 0.7436 | 0.6105 | 0.5433 | 27 | 10 |
| XGB | with_qd | baseline_weight | 0.80 | 0.5000 | 0.7436 | 0.5979 | 0.5668 | 29 | 10 |
| XGB | without_qd | smote_skipped | 0.15 | 0.5000 | 0.7692 | 0.6061 | 0.5526 | 30 | 9 |
| XGB | with_qd | spw_128 | 0.80 | 0.5000 | 0.7436 | 0.5979 | 0.5535 | 29 | 10 |

**Best F1 threshold-optimised: RF / with_qd / random_undersample at thresh=0.70 — F1 = 0.6122, P = 0.5085, R = 0.7692, FP = 29**

The next three best entries (RF/smote_skipped/0.20, XGB/baseline/0.80, XGB/smote_skipped/0.15)
all produce F1 between 0.606 and 0.611 — within 0.006 of the leader.

---

## 7. False-Positive / False-Negative Trade-off

For an operational heatwave warning system:

- A **false positive** (alarm raised, no heatwave) erodes public trust and causes
  unnecessary preparedness costs.
- A **false negative** (heatwave missed, no alarm) has direct public-health consequences.

The table below shows the FP/FN profile of the top strategies on validation:

| Strategy | FP at opt thresh | FN at opt thresh | Interpretation |
|---|---|---|---|
| RF / random_undersample / 0.70 | 29 | 9 | 29 unnecessary alarms per 39 real events |
| RF / smote_skipped / 0.20 | 27 | 10 | Slightly fewer FP, one more missed event |
| XGB / baseline / 0.80 | 27–29 | 10 | Similar FP burden, same missed events |
| LR / baseline / 0.50 | 187–201 | 0–1 | Catches almost everything — 5× the false alarms |
| XGB / spw_64 / 0.50 | 36 | 9 | Good precision/FP balance, slightly less recall |

**No strategy simultaneously minimises both FP and FN.**  The choice depends on the
operational preference: a public health system may accept higher FP to minimise missed
events; an advisory system serving agriculture or electricity planning may need lower FP
to maintain credibility.

For the candidate recommendation below, a moderate-recall, moderate-precision balance
is preferred: catching ~77% of events while producing ~29 false alarms over a 3-year
validation period (roughly one false alarm per 38 days).

---

## 8. Qualifying-Day Findings

`qualifying_day(T)` = 1 when today's Tmax meets the heatwave threshold and departure
criterion.  It is **leakage-safe** (derived from T, not T+1) but is a strong precursor
to the next-day heatwave label.

### Effect across strategies (F1 and PR-AUC, default threshold)

| Strategy | F1 with_qd | F1 without_qd | Delta F1 | PR-AUC with_qd | PR-AUC without_qd | Delta PR-AUC |
|---|---|---|---|---|---|---|
| baseline_weight (XGB) | 0.5299 | 0.5424 | −0.0125 | 0.5668 | 0.5433 | +0.0235 |
| spw_64 (XGB) | 0.5714 | 0.5421 | **+0.0293** | 0.5111 | 0.5223 | −0.0112 |
| spw_128 (XGB) | 0.5172 | 0.5000 | +0.0172 | 0.5535 | 0.5654 | −0.0119 |
| random_undersample (RF) | 0.5593 | 0.5455 | +0.0138 | 0.5497 | 0.5767 | −0.0270 |
| random_oversample (RF) | 0.4690 | 0.4698 | −0.0008 | 0.5542 | 0.5974 | −0.0432 |

**Pattern:** For XGBoost with reduced SPW (spw_64), `qualifying_day` provides a meaningful
F1 improvement (+0.029).  For Random Forest with random undersampling, the with_qd
variant is slightly better on F1 but the without_qd variant has a higher PR-AUC.

The qualifying_day feature consistently provides F1 benefit for the best-performing strategy
(RF/random_undersample/with_qd at threshold 0.70, F1=0.6122 vs F1=0.5843 without_qd).

**Conclusion from Phase 11:** `qualifying_day` is beneficial when combined with
undersampling and an elevated threshold.  The final decision on inclusion in the
production feature set is deferred to Phase 14 (Model Selection).

---

## 9. Mumbai Limitation

Mumbai has **zero positive heatwave events** in any split (train, val, test).

This is scientifically correct.  Mumbai's maritime climate produces a narrow daily
temperature range (Tmax std ≈ 2.1 °C vs ≈ 5.8 °C for plains cities).  The IMD coastal
definition requires Tmax >= 37 °C and departure >= 4.5 °C simultaneously — a combination
that does not occur in the ERA5 record.

**No synthetic positives were created.**  Mumbai contributed 3× 1,096 = 3,288 negative-class
validation rows.  Its presence in training teaches models that maritime-climate feature
patterns are associated with non-heatwave outcomes.

For all city-level positive-class metrics in Mumbai rows, results are reported as:
`N/A — no positive ground-truth examples`

---

## 10. Ahmedabad Limitation

Ahmedabad has 32 positive heatwave days in the training set but **zero positives in the
validation or test splits**.  All 32 events occurred before 2020.

This is not a data error — it reflects real climate variability.  Ahmedabad's heatwave
events in the ERA5 record are concentrated in specific years (e.g., 2010, 2015, 2016)
that fall within the training period.

For all city-level positive-class metrics in Ahmedabad validation/test rows, results are
reported as: `N/A — no positive ground-truth examples`

The absence of Ahmedabad validation positives means the model has not been evaluated on
Ahmedabad's positive class.  This is a genuine limitation that Phase 12 (full model
evaluation on the test set) will not fully resolve — the test period (2023–2025) also
has zero Ahmedabad positives.

---

## 11. Recommended Strategy

**Recommended candidate strategy: Random Forest / with_qd / random_undersample / threshold = 0.70**

| Metric | Value |
|---|---|
| Validation F1 | 0.6122 |
| Validation Precision | 0.5085 |
| Validation Recall | 0.7692 |
| Validation PR-AUC | 0.5497 |
| Validation ROC-AUC | 0.9946 |
| TP | 30 |
| FP | 29 |
| TN | 5,412 |
| FN | 9 |
| Predicted positives | 59 |
| Actual positives | 39 |

**Rationale:**

1. **Highest F1 overall** (0.6122) — best balance of precision and recall
2. **Elevated precision** (0.5085) — over 50% of raised alarms correspond to real heatwave days
3. **Acceptable recall** (0.7692) — captures ~77% of actual heatwave days
4. **Low false-positive burden** — 29 FP over 3 years is operationally manageable
5. **Interpretable** — Random Forest with undersampling and an explicit threshold is
   fully explainable (feature importance available, no synthetic data artefacts)
6. **SMOTE-free** — avoids the meteorological-plausibility concerns of interpolated
   weather states

**Close alternatives worth considering in Phase 14:**

- XGBoost / without_qd / baseline_weight / threshold=0.80 (F1=0.6105, FP=27)
  — marginally fewer FP, same F1 bracket, good interpretability via feature importance
- RF / without_qd / smote_skipped / threshold=0.15 (F1=0.6095, FP=34)
  — equivalent F1 but more FP and relies on a very low threshold

The final production model is **NOT** selected in Phase 11.
Phase 14 (Model Selection) will compare baseline (Phase 10) and imbalance-tuned (Phase 11)
candidates after Phase 12 evaluation and Phase 13 temporal feature experiment.

---

## 12. Limitations

1. **Validation set is small.** 39 positives over 3 years is statistically fragile.
   Small changes in recall (1–2 correct predictions) shift F1 significantly.
   Rankings between close strategies should be interpreted cautiously.

2. **SMOTE not tested.** `imbalanced-learn` was not available.  The `smote_skipped`
   rows in the comparison CSV represent the baseline model, not SMOTE.  If the library
   is installed, re-running Phase 11 with SMOTE enabled is straightforward.

3. **Undersampling discards 93% of training data.** The 1:10 ratio preserves enough
   majority-class examples for temporal diversity but risks losing rare climatic
   conditions.  The positive result (RF/undersample achieves the best F1) should
   be verified on the test set in Phase 12.

4. **Ahmedabad and Mumbai have no validation positives.** Model calibration for those
   cities cannot be assessed from Phase 11 results.

5. **Hyperparameters are fixed at Phase 10 baseline values.** No hyperparameter search
   was performed.  Phase 11 isolates the effect of imbalance strategy only.
   Hyperparameter tuning is deferred to Phase 12/14.

6. **Temporal feature set not explored here.** Phase 11 uses the baseline feature set
   (29 features, Groups 1+6+7) only.  Phase 13 will investigate whether adding temporal
   features (lags, rolling statistics, trends — Groups 2–5) changes the imbalance strategy
   recommendation.

7. **City-level metrics for Nagpur (4 positives) and Lucknow (7 positives) are noisy.**
   Single-event swings in these cities produce large metric changes; per-city results
   should be read as directional, not definitive.

---

## Artifacts

| Path | Contents |
|---|---|
| `results/phase11_imbalance_comparison.csv` | 60 rows (30 experiments × default + opt threshold) |
| `results/phase11_threshold_analysis.csv` | 420 rows (30 experiments × 14 thresholds) |
| `results/phase11_metrics.json` | Full metrics + city breakdown for all 30 experiments |
| `results/phase11_log.txt` | Full execution log |
| `results/phase11_confusion_matrices/` | 60 confusion matrix PNGs |
| `results/plots/phase11/precision_threshold.png` | Precision vs threshold (XGBoost) |
| `results/plots/phase11/recall_threshold.png` | Recall vs threshold (XGBoost) |
| `results/plots/phase11/f1_threshold.png` | F1 vs threshold (XGBoost) |
| `results/plots/phase11/pr_curve.png` | Precision-Recall curves for all experiments |
| `models/phase11/<model>/<feat_set>/<strategy>/model.joblib` | Trained models (30 total) |
| `models/phase11/<model>/<feat_set>/<strategy>/metadata.json` | Strategy metadata (30 total) |

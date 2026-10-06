# Model Card — ClimateGuard DL Comparison Models

> This card covers the GRU and LSTM sequence models trained as a research
> comparison against the production Random Forest. These models are **not**
> deployed in production. The production model card is at `MODEL_CARD.md`.

---

## Model details

| Attribute         | GRU raw-seq                  | LSTM raw-seq                 | GRU feat110-seq              |
|-------------------|------------------------------|------------------------------|------------------------------|
| Architecture      | 2-layer GRU + city embedding | 2-layer LSTM + city embedding| 2-layer GRU + city embedding |
| Hidden size       | 64                           | 64                           | 64                           |
| Sequence length   | 14 days                      | 14 days                      | 14 days                      |
| Input features    | 21 raw + calendar            | 21 raw + calendar            | 104 engineered               |
| Seeds / ensemble  | 5 (ensemble mean prob)       | 5 (ensemble mean prob)       | 3 (ensemble mean prob)       |
| Framework         | PyTorch 2.5.1+cpu            | PyTorch 2.5.1+cpu            | PyTorch 2.5.1+cpu            |
| Training device   | CPU                          | CPU                          | CPU                          |

### City embedding

Each city is encoded as a learned embedding (dimension 4) concatenated to the
final GRU/LSTM hidden state before the classification head. This allows the
model to learn city-specific biases without leaking geographic metadata into
the sequence features.

---

## Intended use

**Primary use:** Academic comparison of sequence-model vs tree-model approaches
for next-day heatwave prediction in Indian cities.

**Out-of-scope use:**
- Operational or public heatwave warnings. Use IMD advisories for operational
  decision-making.
- Predictions for cities outside the training distribution (European cities,
  cities not in `config/cities.json` India section).
- Any deployment that would replace, shadow, or override the production RF model
  without explicit owner approval and a versioned model registry entry.

---

## Training data

| Split      | Period         | Rows   | Heatwave days | Cities                                    |
|------------|----------------|--------|---------------|-------------------------------------------|
| Train      | 1990–2019      | 54,735 | 428           | Ahmedabad, Delhi, Lucknow, Mumbai, Nagpur |
| Validation | 2020–2022      | 5,480  | 38            | Same 5 cities                             |
| Test       | 2023–2025      | 4,865  | 38            | Same 5 cities                             |

Source: ERA5-Land reanalysis via Open-Meteo historical API.
Labels: IMD-inspired proxy (Tmax ≥ 40 °C and ≥ 4.5 °C above the 30-year normal
for plains cities; 45 °C absolute threshold where applicable). Labels are derived
from ERA5-Land, not official IMD observations.

**Important:** Ahmedabad has zero heatwave days in the test period (all 32
events fall in the training window). Mumbai has zero heatwave days across all
periods (maritime climate). Per-city metrics for these cities are undefined.

---

## Features

### raw-seq (21 features)

Raw ERA5-Land variables plus calendar encoding. `qualifying_day` and
`heatwave_lag1` are **excluded** (they encode part of the label or near-future
information). `city_encoded`, `is_coastal`, `latitude`, `longitude` are
**excluded** (geographic metadata handled by city embedding).

Features include: `temperature_2m_max/min/mean`, `apparent_temperature_max/min/mean`,
`tmax_departure`, `et0_fao_evapotranspiration`, `relative_humidity_2m_max/min/mean`,
`precipitation_sum`, `wind_speed_10m_max`, `wind_gusts_10m_max`,
`surface_pressure_mean`, `shortwave_radiation_sum`, `month_sin`, `month_cos`,
`doy_sin`, `doy_cos`.

### feat110-seq (104 features)

The full 110-feature engineered set minus the 6 excluded columns above.

---

## Evaluation

Test set: India 2023–2025, 4,865 rows, 38 positives.

### Ensemble results

| Model             | F1     | Precision | Recall | PR-AUC | ROC-AUC |
|-------------------|--------|-----------|--------|--------|---------|
| RF production     | 0.6947 | 0.5789    | 0.8684 | 0.8339 | 0.9979  |
| RF fair baseline  | 0.6535 | 0.5238    | 0.8684 | 0.7605 | 0.9974  |
| GRU raw-seq ens   | 0.7586 | 0.6735    | 0.8684 | 0.8414 | 0.9981  |
| LSTM raw-seq ens  | 0.7234 | 0.6071    | 0.8947 | 0.8664 | 0.9984  |
| GRU feat110 ens   | 0.7273 | 0.6400    | 0.8421 | 0.8579 | 0.9981  |

Thresholds are calibrated on the validation set (val-optimal F1). The production
RF uses a fixed 0.70 threshold.

### Bootstrap confidence intervals (F1 difference, 1,000 resamples, 14-day blocks)

| DL model        | vs RF-prod CI      | vs RF-fair CI      | Verdict                        |
|-----------------|--------------------|--------------------|--------------------------------|
| GRU raw-seq ens | [-0.080, +0.197]   | [-0.035, +0.240]   | Competitive; CI includes zero  |
| LSTM raw-seq ens| [-0.114, +0.165]   | [-0.068, +0.205]   | Competitive; CI includes zero  |
| GRU feat110 ens | [-0.083, +0.167]   | [-0.057, +0.224]   | Competitive; CI includes zero  |

The wide CIs reflect the small number of positive test events (38 total across
5 cities). No DL model statistically outperforms either RF baseline.

---

## Explainability

Integrated Gradients (Captum) applied to GRU raw-seq seed 0 on the 10
highest-risk test windows.

**Top 10 features by mean absolute IG attribution:**

| Rank | Feature                      | Mean |IG| |
|------|------------------------------|-----------|
| 1    | tmax_departure               | 0.0162    |
| 2    | temperature_2m_max           | 0.0077    |
| 3    | temperature_2m_mean          | 0.0064    |
| 4    | relative_humidity_2m_mean    | 0.0052    |
| 5    | et0_fao_evapotranspiration   | 0.0049    |

Spearman rank correlation between DL and RF importance rankings on the 21
features common to both models: ρ = 0.687 (p = 0.001). Both models agree that
temperature departure from normal and evapotranspiration are the most diagnostic
signals.

---

## Ethical considerations

- **Not an official warning system.** These models are trained on ERA5 reanalysis
  data, not official IMD observations. They must not be presented as authoritative
  heatwave forecasts or used to replace IMD advisories.
- **Underserved cities.** Ahmedabad and Mumbai have very few or zero test
  positives. Model performance for these cities cannot be meaningfully assessed.
- **Vulnerable populations.** Heatwave warnings have life-safety implications.
  Any operational deployment requires human expert review, connection to official
  meteorological services, and compliance with local public-health guidelines.
- **Label proxy.** The IMD-inspired label is a research proxy derived from
  ERA5-Land, not official observations. Real heatwave onset and end dates may
  differ.

---

## Caveats and recommendations

- Ensemble mean probability reduces variance but does not eliminate it. Single-seed
  results (reported in `dl_vs_rf.json`) show seed-to-seed F1 variation of ±0.04.
- The 14-day sequence length was chosen to capture multi-day heat build-up. Longer
  sequences may improve performance but require more context history at inference time.
- The model has not been evaluated on out-of-distribution years (post-2025) or on
  cities not in the training set.
- Before any production consideration, the DL module would require: independent
  validation on official IMD observations, evaluation on additional positive events
  (the 38-event test set is underpowered), and a model registry entry with explicit
  owner approval.

---

## Artifact inventory

| File                               | Description                          |
|------------------------------------|--------------------------------------|
| `dl/artifacts/gru_raw_seq_seed*/`  | GRU raw-seq checkpoints (seeds 0–4)  |
| `dl/artifacts/lstm_raw_seq_seed*/` | LSTM raw-seq checkpoints (seeds 0–4) |
| `dl/artifacts/gru_feat110_seq_seed*/` | GRU feat110 checkpoints (seeds 0–2)|
| `dl/artifacts/rf_fair.joblib`      | RF-fair baseline (joblib)            |
| `dl/artifacts/rf_fair_info.json`   | RF-fair training metadata            |
| `dl/results/dl_vs_rf.json`         | Full metrics + bootstrap CIs (JSON)  |
| `dl/results/dl_vs_rf.csv`          | Same, CSV format                     |
| `dl/results/dl_vs_rf.md`           | Human-readable comparison table      |
| `dl/results/pr_curves.png`         | PR curve comparison chart            |
| `dl/results/per_city_f1.png`       | Per-city F1 grouped bar chart        |
| `dl/results/calibration.png`       | Calibration comparison chart         |
| `dl/results/dl_feature_importance_ig.png` | IG attribution bar chart      |
| `dl/results/dl_ig_heatmap.png`     | IG attribution heatmap (time × feat) |
| `dl/results/explain_comparison_note.txt` | Computed RF vs DL comparison   |

---

*Card version: 1.0.0 — generated October 2026.*

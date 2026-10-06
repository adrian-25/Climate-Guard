# Deep Learning vs Random Forest — India Test Set (2023–2025)

**38 positive heatwave days out of 4 865 test rows (0.78%).**
Ahmedabad and Mumbai: 0 test positives — per-city metrics undefined (n/a).
RF-production was trained on 1990–2022 (train+val combined). All other models use train only (≤ 2019-12-31) for fairness.

| model                 | training_window        | features                                           | threshold_rule      |     F1 |   Precision |   Recall |   PR-AUC |   ROC-AUC |   Brier |    ECE |
|:----------------------|:-----------------------|:---------------------------------------------------|:--------------------|-------:|------------:|---------:|---------:|----------:|--------:|-------:|
| RF production         | 1990–2022 (train+val)  | 110 features (incl. qualifying_day)                | Fixed 0.70          | 0.6947 |      0.5789 |   0.8684 |   0.8339 |    0.9979 |  0.0083 | 0.0208 |
| RF fair               | 1990–2019 (train only) | 104 features (no qualifying_day, no heatwave_lag1) | Val-optimal (0.86)  | 0.6535 |      0.5238 |   0.8684 |   0.7605 |    0.9974 |  0.0112 | 0.0195 |
| GRU raw-seq ens       | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (≈0.93) | 0.7586 |      0.6735 |   0.8684 |   0.8414 |    0.9981 |  0.009  | 0.0136 |
| LSTM raw-seq ens      | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (≈0.90) | 0.7234 |      0.6071 |   0.8947 |   0.8664 |    0.9984 |  0.0082 | 0.0124 |
| GRU feat110 ens       | 1990–2019 (train only) | 104 feat (no qualifying_day)                       | Val-optimal (≈0.84) | 0.7273 |      0.64   |   0.8421 |   0.8579 |    0.9981 |  0.0066 | 0.0105 |
| GRU raw-seq focal ens | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (≈0.76) | 0.7529 |      0.6809 |   0.8421 |   0.8264 |    0.9977 |  0.0132 | 0.0368 |
| GRU raw-seq seed0     | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (0.94)  | 0.7416 |      0.6471 |   0.8684 |   0.8331 |    0.9978 |  0.0104 | 0.0144 |
| GRU raw-seq seed1     | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (0.94)  | 0.7556 |      0.6538 |   0.8947 |   0.867  |    0.9985 |  0.009  | 0.0141 |
| GRU raw-seq seed2     | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (0.94)  | 0.7442 |      0.6667 |   0.8421 |   0.8064 |    0.9974 |  0.0099 | 0.0131 |
| GRU raw-seq seed3     | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (0.93)  | 0.7805 |      0.7273 |   0.8421 |   0.8524 |    0.9983 |  0.0076 | 0.0118 |
| GRU raw-seq seed4     | 1990–2019 (train only) | 21 raw+cal (no qualifying_day)                     | Val-optimal (0.90)  | 0.6667 |      0.541  |   0.8684 |   0.8424 |    0.9981 |  0.0104 | 0.0147 |

## Notes
- **qualifying_day** excluded from DL and RF-fair: same threshold conditions as the heatwave label (structural correlation).
- **heatwave_lag1** excluded: derived from the label column; would leak label-correlated information across the sequence boundary.
- DL probabilities are not calibrated the same way as the RF. Threshold comparisons are approximate.
- With only 38 test positives, confidence intervals are wide; see bootstrap CIs in dl_vs_rf.json.
- The RF remains the production model for all predictions.

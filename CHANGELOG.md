# Changelog

## Unreleased

- Product: alert subscriptions, 7-day outlook, PWA/export/print/share tools.
- Operations: Docker, Render blueprint, CI coverage/audit/Docker build.
- Credibility: evaluation artifact, baseline comparisons, calibration, and live outcome tracking.
- DL: added a fixed gamma=2 focal-loss GRU ablation and calibration diagnostics (Brier, ECE,
  cross-seed probability spread). The focal ablation did not beat the weighted-BCE GRU and
  remains research-only.
- Live data: handle temporary Open-Meteo rate limits with a five-minute request cooldown,
  clearly-labelled stale-cache fallback, automatic browser retry, and an optional server-only
  `OPEN_METEO_API_KEY` for the dedicated provider endpoint.

## Phase 8 — DL comparison module

- Research: GRU and LSTM sequence models trained on India data (1990–2019) with
  leakage-safe windowed datasets (context prefix, no NaN padding, 4865 test rows).
- RF-fair baseline (train-only, no qualifying_day) for controlled comparison.
- Bootstrap CIs (14-day blocks, 1000 resamples) vs both RF baselines — all include
  zero; all results described as competitive with no statistically significant difference.
- Integrated Gradients attribution; Spearman rho=0.687 DL vs RF feature rankings.
- Read-only /api/dl/comparison and /api/dl/figure/{name} routes added to app.py.
- Performance page: DL section (India only, hidden by default, graceful 503 handling).
- 307 tests pass (278 existing + 26 DL + 3 new /api/dl contract tests).
- dl/README_DL.md: method, commands, results table, CI findings, limitations.
- dl/MODEL_CARD_DL.md: full model card with training data, evaluation, explainability,
  ethical considerations, and artifact inventory.

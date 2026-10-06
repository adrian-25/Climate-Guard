# ClimateGuard contributor guide

## Structure

- `app.py`: FastAPI application and existing API routes.
- `live_data.py`: Open-Meteo live feature construction and prediction.
- `src/`: model, risk, ETL, explainability and recommendation code.
- `web/`: plain HTML, CSS and JavaScript frontend.
- `data/` and `models/final/`: validated training artifacts; do not overwrite.
- `tests/`: automated backend tests.

## Local development

Run the dashboard with `uvicorn app:app --host 0.0.0.0 --port 8001`.
Run the existing interface tests with `python tests/test_prediction_interface.py` and the full pytest suite with `python -m pytest` after installing development dependencies.

## Conventions

- Use Python type hints, small additive functions, and clear docstrings.
- Keep frontend JavaScript dependency-free except for already pinned map/chart libraries.
- Preserve existing endpoint response shapes, element IDs, and CSS classes. Add new interfaces instead of changing old ones.
- Run formatting, linting, focused tests, and relevant browser checks before each commit.

## Safety rules

- Never overwrite the trained model or training data. New models and datasets must be versioned; the current model remains the default pending user approval.
- Put secrets only in environment variables and document blank values in `.env.example`. Never log or commit them.
- Do not commit files over 100 MB, caches, backups, `.env`, or generated local databases.
- Do not force-push. Fetch and compare the remote before the final push.
- Use mocked external services in tests; no real network calls in unit tests.
- Present IMD and live-weather context accurately: ClimateGuard is a research model, not an official warning system.

## Phase log

- Phase 1 completed foundation increments: model registry, explicit CORS, request IDs,
  structured request logs, lint/pre-commit configuration, mocked live-data tests, and API
  contract tests. Rate limits protect the expensive prediction, explanation, and live routes.
  Latest focused coverage run: 51% total (65% app module; 33% live-data module).
- Phase 2 completed credibility increments: a reproducible frozen-model evaluation artifact,
  temporal folds and calibration data, explicit persistence and IMD-style proxy baselines,
  data audit documentation, and an append-only SQLite live-forecast ledger with scheduled
  Open-Meteo observed-temperature proxy reconciliation. The dashboard reports insufficient
  live evidence honestly until matched outcomes accumulate; it never calls this an IMD warning.
- Phase 3 completed opt-in alerting: SQLite subscriptions with double opt-in and one-click
  unsubscribe, data-file message templates, dry-run Resend/SMTP seams, a live-refresh scheduler,
  deduplication, quiet hours, daily caps and retry backoff. Hindi and Marathi templates are
  machine-drafted and require native-speaker review before production use.
- Phases 4–6 completed product/release increments: the versioned city registry, 7-day live
  weather context, PWA/static caching, print/export/share controls, Docker/Compose/Render,
  CI quality gates, deployment guidance, model card, contribution guide and changelog. License
  intentionally remains pending explicit owner selection (MIT is recommended).
- Phase 7 QA: 274 tests passed at 82% total measured coverage; maintained-surface Ruff and
  Black checks passed. Desktop/mobile screenshots and live/historical browser flows passed
  without console errors. Docker Compose configuration validated, but a Docker image build
  could not run because Docker Desktop's daemon was unavailable on this host.
- Phase 8 DL comparison module: GRU and LSTM sequence models trained on the same India
  training window (1990–2019) with leakage-safe windowed datasets (every test row gets a real
  prediction via context prefix — no NaN padding). RF-fair baseline trained for a controlled
  comparison. Bootstrap CIs (14-day blocks, 1000 resamples) vs both production RF and RF-fair
  all include zero — competitive, no statistically significant difference. Feature attribution
  via Integrated Gradients; Spearman rho=0.687 between DL/RF importance rankings. Additive
  /api/dl/comparison and /api/dl/figure/{name} read-only routes in app.py. DL section added
  to Performance page (India only, hidden for Europe, handles 503 gracefully). 307 tests pass;
  Ruff + Black clean. dl/README_DL.md and dl/MODEL_CARD_DL.md document method and results.
  DL module does not affect production RF, model_registry.json, or any existing route.
- DL follow-up: a fixed gamma=2 focal-loss GRU ablation (three seeds) was trained and evaluated
  with the same leakage-safe protocol. It did not improve the weighted-BCE GRU (F1 0.7529 vs
  0.7586; PR-AUC 0.8264 vs 0.8414), and its ECE was higher (0.0368 vs 0.0136). DL comparison
  artifacts and the Performance page now report Brier, ECE, and cross-seed probability spread.

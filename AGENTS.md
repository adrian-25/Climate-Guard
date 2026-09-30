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

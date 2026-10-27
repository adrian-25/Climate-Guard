# Input drift monitoring

ClimateGuard now screens the distribution of recent India live-model inputs against a frozen,
versioned reference built from the production model's 1990-01-11 to 2022-12-31 train and
validation window. The reference artifact is
`models/monitoring/v1/drift_baseline.json`.

## Method

- At least 30 complete historical feature rows are required per city.
- Only observed/current rows are used; forecast rows are never included in the calculation.
- Twelve weather-derived model inputs are monitored with Population Stability Index (PSI).
- Each input is compared with the same city and calendar-month reference distribution, limiting
  normal city and seasonal differences from being reported as drift.
- Overall PSI is the mean of the monitored-feature PSI values. Levels are `stable` (<0.10),
  `monitor` (<0.20), `drift_detected` (<0.25), and `high_drift` (>=0.25).

## API

- `GET /api/drift/status` returns baseline availability and reference metadata.
- `GET /api/drift/{city}` returns the latest in-memory assessment. Run that city's live forecast
  first; this endpoint never makes a weather-provider request.
- Live forecast responses also include a `drift` object, so the dashboard can show the status.

## Interpretation and limits

PSI is a screening signal for input-distribution change. It does **not** measure predictive
accuracy, establish a climate trend, prove model bias, or trigger retraining. Review it with the
live observed-outcome record and data-quality checks before deciding to retrain. The monitor is
available for the India model only; Europe remains separately versioned.

To produce a new reference only after an approved model/training-data update, run:

```powershell
python scripts/build_drift_baseline.py
```

That command creates the monitoring artifact only. It never overwrites the production model or
training data.

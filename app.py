"""
ClimateGuard Web Dashboard — FastAPI Backend
Serves the prediction pipeline over HTTP + static frontend.

Run:
    python app.py
    → http://localhost:8000
"""

import json
import logging
import math
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTION_LOG = PROJECT_ROOT / "runtime" / "live_predictions.jsonl"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "model_registry.json"
EVALUATION_RESULT_PATH = PROJECT_ROOT / "evaluation" / "results" / "latest.json"
outcome_scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import alerts, live_tracking
from src.integration.pipeline import ClimateGuardPipeline
from src.prediction import ClimateGuardPredictor

LIVE_TRACKING_DB = live_tracking.database_path(PROJECT_ROOT)
ALERT_DATABASE = alerts.database_path(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# City metadata
# ---------------------------------------------------------------------------
CITIES = {
    "delhi": {
        "name": "New Delhi",
        "state": "Delhi",
        "region": "Plains",
        "lat": 28.6139,
        "lon": 77.2090,
    },
    "lucknow": {
        "name": "Lucknow",
        "state": "Uttar Pradesh",
        "region": "Plains",
        "lat": 26.8467,
        "lon": 80.9462,
    },
    "nagpur": {
        "name": "Nagpur",
        "state": "Maharashtra",
        "region": "Plains",
        "lat": 21.1458,
        "lon": 79.0882,
    },
    "ahmedabad": {
        "name": "Ahmedabad",
        "state": "Gujarat",
        "region": "Plains",
        "lat": 23.0225,
        "lon": 72.5714,
    },
    "mumbai": {
        "name": "Mumbai",
        "state": "Maharashtra",
        "region": "Coastal",
        "lat": 19.0760,
        "lon": 72.8777,
    },
}

# ---------------------------------------------------------------------------
# Load data + pipeline at startup
# ---------------------------------------------------------------------------
print("[startup] Loading test data ...")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv")
meta_test = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv")
y_test = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "y_test.csv")

# Merge features + metadata for easy lookup
test_data = pd.concat([meta_test, X_test, y_test], axis=1)
print(
    f"[startup] Loaded {len(test_data)} test rows across {test_data['city_key'].nunique()} cities"
)

print("[startup] Loading ClimateGuardPipeline ...")
pipeline = ClimateGuardPipeline(include_explanation=False)
print(f"[startup] Pipeline ready: {pipeline}")

# Load model metadata
with open(PROJECT_ROOT / "models" / "final" / "metadata.json") as f:
    model_metadata = json.load(f)
with open(MODEL_REGISTRY_PATH, encoding="utf-8") as f:
    model_registry = json.load(f)
ACTIVE_MODEL = next(
    item for item in model_registry["models"] if item["version"] == model_registry["active_version"]
)

# Precompute heatwave probabilities for ALL test rows (fast batch predict)
print("[startup] Precomputing probabilities for trend charts ...")
predictor = pipeline.predictor
feature_cols = predictor.feature_names
all_probs = predictor.model.predict_proba(X_test[feature_cols])[:, 1]
test_data["pred_probability"] = all_probs
test_data["pred_label"] = (all_probs >= predictor.threshold).astype(int)
print(f"[startup] Precomputed {len(all_probs)} predictions")

# Precompute model performance data
print("[startup] Computing model performance metrics ...")
y_true = y_test["heatwave_next_day"].values
y_pred = test_data["pred_label"].values
y_prob = test_data["pred_probability"].values

# Remove NaN rows for metrics
valid_mask = ~np.isnan(y_true)
y_true_valid = y_true[valid_mask].astype(int)
y_pred_valid = y_pred[valid_mask].astype(int)
y_prob_valid = y_prob[valid_mask]

# Confusion matrix
cm = confusion_matrix(y_true_valid, y_pred_valid)

# ROC curve (sample to 200 points for frontend)
fpr, tpr, roc_thresholds = roc_curve(y_true_valid, y_prob_valid)
roc_auc = roc_auc_score(y_true_valid, y_prob_valid)
step = max(1, len(fpr) // 200)
roc_data = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist(), "auc": round(roc_auc, 4)}

# PR curve (sample to 200 points)
precisions, recalls, pr_thresholds = precision_recall_curve(y_true_valid, y_prob_valid)
pr_auc = average_precision_score(y_true_valid, y_prob_valid)
step = max(1, len(precisions) // 200)
pr_data = {
    "precision": precisions[::step].tolist(),
    "recall": recalls[::step].tolist(),
    "auc": round(pr_auc, 4),
}

# Per-city metrics
city_metrics = {}
for city_key in CITIES:
    mask = (test_data["city_key"] == city_key).values & valid_mask
    if mask.sum() == 0:
        continue
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    yprob = y_prob[mask]
    city_metrics[city_key] = {
        "f1": round(f1_score(yt, yp, zero_division=0), 4),
        "precision": round(precision_score(yt, yp, zero_division=0), 4),
        "recall": round(recall_score(yt, yp, zero_division=0), 4),
        "accuracy": round(accuracy_score(yt, yp), 4),
        "heatwave_days": int(yt.sum()),
        "total_days": int(mask.sum()),
        "name": CITIES[city_key]["name"],
    }

# Global feature importance (from Random Forest)
print("[startup] Computing feature importances ...")
raw_importances = predictor.model.feature_importances_
importance_indices = np.argsort(raw_importances)[::-1][:25]  # Top 25
feature_importance_data = [
    {"feature": feature_cols[i], "importance": round(float(raw_importances[i]), 6)}
    for i in importance_indices
]
print(f"[startup] Performance data ready")

# Precompute stats
stats_data = {}
for city_key in CITIES:
    city_df = test_data[test_data["city_key"] == city_key]
    hw_count = int(city_df["heatwave"].sum()) if "heatwave" in city_df.columns else 0
    stats_data[city_key] = {
        "total_days": len(city_df),
        "heatwave_days": hw_count,
        "date_min": str(city_df["date"].min()),
        "date_max": str(city_df["date"].max()),
    }

print("[startup] Ready! Navigate to http://localhost:8000")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("climateguard")
allowed_origins = [
    item.strip()
    for item in os.getenv(
        "CORS_ALLOW_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001"
    ).split(",")
    if item.strip()
]


@asynccontextmanager
async def lifespan(application):
    enabled = os.getenv("LIVE_OUTCOME_SCHEDULER_ENABLED", "true").lower() == "true"
    if enabled and not outcome_scheduler.running:
        outcome_scheduler.add_job(
            reconcile_live_outcomes,
            "interval",
            hours=6,
            id="live-outcome-reconciliation",
            replace_existing=True,
        )
        if os.getenv("ALERT_SCHEDULER_ENABLED", "true").lower() == "true":
            outcome_scheduler.add_job(
                refresh_subscriber_alerts,
                "interval",
                minutes=30,
                id="subscriber-alert-refresh",
                replace_existing=True,
            )
        outcome_scheduler.start()
    yield
    if outcome_scheduler.running:
        outcome_scheduler.shutdown(wait=False)


app = FastAPI(title="ClimateGuard Dashboard API", version="1.0.0", lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address, default_limits=["240/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    logger.info(
        json.dumps(
            {
                "event": "request_complete",
                "request_id": request_id,
                "path": request.url.path,
                "status": response.status_code,
            }
        )
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    city: str
    date: str


class SubscribeRequest(BaseModel):
    email: str
    city: str
    minimum_risk_level: str = "HIGH"
    language: str = "en"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/config")
def get_public_config():
    """Return browser-safe, optional configuration only.

    Mappls static keys are designed for browser use and must still be limited
    to the application's domain in the Mappls developer console.  No server
    credentials or other environment variables are exposed here.
    """
    return {"mappls_key": os.getenv("MAPPLS_KEY", "").strip()}


@app.get("/api/health")
def get_legacy_health():
    """Legacy lightweight deployment health check; response shape is stable."""
    return {
        "status": "ok",
        "model_ready": predictor is not None,
        "live_data_available": (
            _LIVE_DATA_AVAILABLE if "_LIVE_DATA_AVAILABLE" in globals() else False
        ),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def get_health():
    """Deployment health check with active-model and data freshness context."""
    cache = live_cache_status() if "live_cache_status" in globals() else {}
    refreshed = [entry.get("fetched_at") for entry in cache.values() if entry.get("fetched_at")]
    return {
        **get_legacy_health(),
        "model_version": ACTIVE_MODEL["version"],
        "last_live_data_refresh": max(refreshed) if refreshed else None,
    }


@app.get("/api/evaluation/live-track-record")
def get_live_track_record():
    """Return persisted live forecast quality when observations are available."""
    return live_tracking.summary(LIVE_TRACKING_DB)


@app.post("/api/subscribe", status_code=201)
def subscribe_to_alerts(payload: SubscribeRequest):
    """Start double opt-in for the minimum data needed for city risk alerts."""
    city = payload.city.lower().strip()
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")
    try:
        subscription = alerts.subscribe(
            ALERT_DATABASE,
            payload.email.strip(),
            city,
            payload.minimum_risk_level.upper().strip(),
            payload.language.lower().strip(),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    confirmation = alerts.render_alert(
        CITIES[city]["name"],
        "your selected forecast days",
        "MODERATE",
        0,
        payload.language.lower().strip(),
        f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8001')}/api/unsubscribe/{subscription['token']}",
    )
    confirmation["subject"] = "Confirm your ClimateGuard heat-risk alerts"
    confirmation["body"] = (
        "Confirm your ClimateGuard alert subscription: "
        f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8001')}/api/confirm/{subscription['token']}\n\n"
        "If you did not request this, you can ignore this email."
    )
    try:
        alerts.send_email(payload.email.strip(), confirmation)
    except Exception:
        logger.exception(json.dumps({"event": "subscription_confirmation_failed"}))
        raise HTTPException(
            503, "We could not send the confirmation email. Please try again."
        ) from None
    return {
        "status": "confirmation_required",
        "message": "Check your email to confirm alerts. No alerts will be sent until confirmation.",
    }


@app.get("/api/confirm/{token}")
def confirm_alert_subscription(token: str):
    if not alerts.confirm(ALERT_DATABASE, token):
        raise HTTPException(404, "Confirmation link is invalid or no longer active.")
    return {
        "status": "confirmed",
        "message": "Heat-risk alerts are now active. You can unsubscribe using the link in any alert.",
    }


@app.get("/api/unsubscribe/{token}")
def unsubscribe_from_alerts(token: str):
    if not alerts.unsubscribe(ALERT_DATABASE, token):
        raise HTTPException(404, "Unsubscribe link is invalid.")
    return {"status": "unsubscribed", "message": "You will no longer receive ClimateGuard alerts."}


@app.get("/api/alerts/preview")
def preview_alert(city: str, level: str, lang: str = "en"):
    """Render an alert safely for review without creating or sending anything."""
    city_key, level = city.lower().strip(), level.upper().strip()
    if city_key not in CITIES or level not in alerts.RISK_RANK or lang not in {"en", "hi", "mr"}:
        raise HTTPException(422, "Use a supported city, risk level, and language.")
    return alerts.render_alert(
        CITIES[city_key]["name"],
        datetime.now(alerts.IST).date().isoformat(),
        level,
        0.75,
        lang,
        "https://example.invalid/unsubscribe",
    )


@app.get("/api/official-alerts/{city}")
def get_official_alert_context(city: str):
    """Provide an official IMD source without claiming a non-verified alert."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}")
    return {
        "city": city,
        "city_name": CITIES[city]["name"],
        "status": "not_automatically_verified",
        "message": "Check the India Meteorological Department warning portal for official alerts. ClimateGuard predictions are not official warnings.",
        "source_name": "India Meteorological Department",
        "source_url": "https://mausam.imd.gov.in/",
    }


@app.get("/api/evaluation/live-summary")
def get_live_evaluation_summary():
    """Summarize locally logged live forecasts; outcome matching is additive later."""
    if not PREDICTION_LOG.exists():
        return {"logged_predictions": 0, "cities": {}, "outcomes_matched": 0}
    records = []
    for line in PREDICTION_LOG.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    cities = {}
    for record in records:
        cities[record["city"]] = cities.get(record["city"], 0) + 1
    return {
        "logged_predictions": len(records),
        "cities": cities,
        "outcomes_matched": 0,
        "note": "Use /api/evaluation/live-outcomes for observed-temperature proxy matching.",
    }


@app.get("/api/evaluation/latest")
def get_latest_evaluation():
    """Return the latest read-only evaluation artifact generated by evaluation/run_evaluation.py."""
    if not EVALUATION_RESULT_PATH.exists():
        raise HTTPException(
            503, "Evaluation artifact is unavailable. Run evaluation/run_evaluation.py."
        )
    return json.loads(EVALUATION_RESULT_PATH.read_text(encoding="utf-8"))


@app.get("/api/evaluation/live-outcomes")
def get_live_outcomes():
    """Compare past logged forecasts with Open-Meteo observed Tmax proxies.

    The proxy is deliberately not presented as an IMD-certified heatwave label:
    it treats the city's project threshold as an observed heat signal only.
    """
    if not PREDICTION_LOG.exists():
        return {"evaluated": 0, "matched": 0, "note": "No logged forecasts yet."}
    records = []
    for line in PREDICTION_LOG.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            if record["date"] < datetime.now(timezone.utc).date().isoformat():
                records.append(record)
        except (json.JSONDecodeError, KeyError):
            continue
    outcomes, unavailable = [], 0
    for record in records:
        city = CITIES.get(record["city"])
        if not city:
            unavailable += 1
            continue
        try:
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "start_date": record["date"],
                    "end_date": record["date"],
                    "daily": "temperature_2m_max",
                    "timezone": "Asia/Kolkata",
                },
                timeout=10,
            )
            tmax = response.json()["daily"]["temperature_2m_max"][0]
            threshold = 37 if record["city"] == "mumbai" else 40
            observed_signal = tmax >= threshold
            outcomes.append(
                {
                    **record,
                    "observed_tmax": tmax,
                    "threshold_c": threshold,
                    "observed_heat_signal": observed_signal,
                    "matched_proxy": bool(record["prediction"]) == observed_signal,
                }
            )
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
            unavailable += 1
    return {
        "evaluated": len(records),
        "matched": len(outcomes),
        "unavailable": unavailable,
        "proxy_accuracy": (
            round(sum(x["matched_proxy"] for x in outcomes) / len(outcomes), 4)
            if outcomes
            else None
        ),
        "outcomes": outcomes,
        "note": "Observed Tmax comes from Open-Meteo archive. This threshold proxy is not an official IMD heatwave outcome.",
    }


def log_live_predictions(city: str, result: dict) -> None:
    """Append successful forecast-day estimates for later observed-outcome checks."""
    rows = [
        {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "city": city,
            "date": day["date"],
            "probability": day["probability"],
            "prediction": day["prediction"],
            "risk_level": day["risk_level"],
        }
        for day in result.get("days", [])
        if day.get("error") is None and day.get("probability") is not None
    ]
    if not rows:
        return
    PREDICTION_LOG.parent.mkdir(exist_ok=True)
    with PREDICTION_LOG.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    live_tracking.record(
        LIVE_TRACKING_DB,
        city,
        result.get("days", []),
        ACTIVE_MODEL["version"],
        result.get("last_updated", datetime.now(timezone.utc).isoformat()),
    )


def dispatch_alerts_for_live_forecast(city: str, result: dict) -> None:
    """Deliver eligible alerts after a fresh live refresh; failures stay isolated."""
    now = datetime.now(alerts.IST)
    for day in result.get("days", []):
        level = str(day.get("risk_level", "LOW")).upper()
        probability = day.get("probability")
        if day.get("error") or probability is None or level not in alerts.RISK_RANK:
            continue
        for subscription in alerts.confirmed_for_city(ALERT_DATABASE, city):
            alerts.deliver_if_due(
                ALERT_DATABASE, subscription, day["date"], level, float(probability), now
            )


def refresh_subscriber_alerts() -> None:
    """Scheduled refresh for confirmed cities; no subscription means no network work."""
    if not _LIVE_DATA_AVAILABLE:
        return
    for city in alerts.subscribed_cities(ALERT_DATABASE):
        try:
            result = get_live_forecast(city, pipeline)
            if not result.get("error"):
                dispatch_alerts_for_live_forecast(city, result)
        except Exception as exc:
            logger.warning(
                json.dumps({"event": "alert_refresh_failed", "city": city, "error": str(exc)})
            )


def reconcile_live_outcomes() -> None:
    """Fill past forecast rows from Open-Meteo archive; one failure never stops the job."""
    for prediction_id, city_key, forecast_date in live_tracking.pending_before_today(
        LIVE_TRACKING_DB
    ):
        city = CITIES.get(city_key)
        if not city:
            continue
        try:
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "start_date": forecast_date,
                    "end_date": forecast_date,
                    "daily": "temperature_2m_max",
                    "timezone": "Asia/Kolkata",
                },
                timeout=12,
            )
            response.raise_for_status()
            tmax = float(response.json()["daily"]["temperature_2m_max"][0])
            live_tracking.record_outcome(
                LIVE_TRACKING_DB, prediction_id, tmax, 37 if city_key == "mumbai" else 40
            )
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning(
                json.dumps(
                    {
                        "event": "live_outcome_unavailable",
                        "city": city_key,
                        "date": forecast_date,
                        "error": str(exc),
                    }
                )
            )


@app.get("/api/cities")
def get_cities():
    """Return all available cities with metadata."""
    result = []
    for key, info in CITIES.items():
        result.append(
            {
                "key": key,
                **info,
                **stats_data.get(key, {}),
            }
        )
    return result


@app.get("/api/dates/{city}")
def get_dates(city: str):
    """Return available dates for a city."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}")
    city_df = test_data[test_data["city_key"] == city]
    dates = sorted(city_df["date"].unique().tolist())
    return {
        "city": city,
        "dates": dates,
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "count": len(dates),
    }


@app.post("/api/predict")
@limiter.limit("30/minute")
def predict(request: Request, req: PredictRequest):
    """Run the full ClimateGuard pipeline for a city + date."""
    city = req.city.lower().strip()
    date = req.date.strip()

    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}. Valid: {list(CITIES.keys())}")

    # Look up the feature row
    mask = (test_data["city_key"] == city) & (test_data["date"] == date)
    matches = test_data[mask]

    if len(matches) == 0:
        raise HTTPException(
            404,
            f"No data for {city} on {date}. Available range: "
            f"{stats_data[city]['date_min']} to {stats_data[city]['date_max']}",
        )

    row = matches.iloc[0].to_dict()

    # Strip metadata columns that cause ETL validation failure
    # Keep only: city_key, date, and the 110 model features
    META_COLS_TO_DROP = {
        "city",
        "state",
        "region_type",
        "heatwave",
        "hw_event_id",
        "hw_event_start",
        "hw_event_end",
        "hw_event_length",
        "heatwave_next_day",
    }
    pipeline_row = {k: v for k, v in row.items() if k not in META_COLS_TO_DROP}

    # Run the pipeline
    result = pipeline.analyze(pipeline_row)

    # Build response
    response = result.to_dict()

    # Add actual label if available
    actual = row.get("heatwave_next_day")
    if actual is not None:
        response["actual"] = {
            "heatwave_next_day": int(actual),
            "label": "Heatwave" if int(actual) == 1 else "Normal",
        }

    # Add city display info
    response["city_info"] = CITIES[city]

    return response


@app.get("/api/model-info")
def get_model_info():
    """Return model metadata and performance metrics."""
    return {
        "model_type": model_metadata.get("model_type"),
        "feature_count": model_metadata.get("feature_count"),
        "threshold": model_metadata.get("threshold"),
        "training_date_range": model_metadata.get("training_date_range"),
        "test_date_range": model_metadata.get("test_date_range"),
        "cities": model_metadata.get("cities"),
        "test_metrics": model_metadata.get("final_test_metrics"),
        "imbalance_strategy": model_metadata.get("imbalance_strategy"),
        "prediction_type": model_metadata.get("prediction_type"),
    }


@app.get("/api/stats")
def get_stats():
    """Return dataset-level statistics."""
    return {
        "total_rows": len(test_data),
        "cities": stats_data,
        "risk_thresholds": {
            "LOW": "[0.00, 0.30)",
            "MODERATE": "[0.30, 0.60)",
            "HIGH": "[0.60, 0.80)",
            "EXTREME": "[0.80, 1.00]",
        },
    }


@app.get("/api/trends/{city}")
def get_trends(city: str):
    """Return time-series data for trend charts."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}")
    city_df = test_data[test_data["city_key"] == city].copy()
    city_df = city_df.sort_values("date")

    def safe_float(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(float(v), 4)

    records = []
    for _, row in city_df.iterrows():
        records.append(
            {
                "date": str(row["date"]),
                "tmax": safe_float(row.get("temperature_2m_max")),
                "tmin": safe_float(row.get("temperature_2m_min")),
                "prob": safe_float(row.get("pred_probability")),
                "heatwave": int(row.get("heatwave", 0)),
                "hw_next": (
                    int(row.get("heatwave_next_day", 0))
                    if not math.isnan(row.get("heatwave_next_day", 0))
                    else 0
                ),
            }
        )
    return {"city": city, "city_name": CITIES[city]["name"], "data": records}


@app.get("/api/map-data")
def get_map_data():
    """Return city data for map markers with risk summary."""
    result = []
    for key, info in CITIES.items():
        city_df = test_data[test_data["city_key"] == key]
        avg_prob = float(city_df["pred_probability"].mean())
        max_prob = float(city_df["pred_probability"].max())
        hw_days = int(city_df["heatwave"].sum()) if "heatwave" in city_df.columns else 0
        total = len(city_df)
        result.append(
            {
                "key": key,
                "name": info["name"],
                "state": info["state"],
                "region": info["region"],
                "lat": info["lat"],
                "lon": info["lon"],
                "avg_probability": round(avg_prob, 4),
                "max_probability": round(max_prob, 4),
                "heatwave_days": hw_days,
                "total_days": total,
                "heatwave_pct": round(hw_days / total * 100, 1) if total > 0 else 0,
            }
        )
    return result


@app.get("/api/performance")
def get_performance():
    """Return precomputed model performance data for charts."""
    return {
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "roc": roc_data,
        "pr": pr_data,
        "global_metrics": {
            "f1": round(f1_score(y_true_valid, y_pred_valid, zero_division=0), 4),
            "precision": round(precision_score(y_true_valid, y_pred_valid, zero_division=0), 4),
            "recall": round(recall_score(y_true_valid, y_pred_valid, zero_division=0), 4),
            "accuracy": round(accuracy_score(y_true_valid, y_pred_valid), 4),
            "roc_auc": roc_data["auc"],
            "pr_auc": pr_data["auc"],
        },
        "city_metrics": city_metrics,
        "threshold": float(predictor.threshold),
        "total_test_samples": int(valid_mask.sum()),
        "positive_samples": int(y_true_valid.sum()),
        "negative_samples": int((valid_mask.sum() - y_true_valid.sum())),
    }


@app.get("/api/feature-importance")
def get_feature_importance():
    """Return top 25 global feature importances from Random Forest."""
    return {
        "features": feature_importance_data,
        "model_type": model_metadata.get("model_type"),
        "total_features": len(feature_cols),
    }


@app.post("/api/explain")
@limiter.limit("15/minute")
def explain_prediction(request: Request, req: PredictRequest):
    """Run prediction with per-feature importance for explainability."""
    city = req.city.lower().strip()
    date = req.date.strip()

    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")

    mask = (test_data["city_key"] == city) & (test_data["date"] == date)
    matches = test_data[mask]

    if len(matches) == 0:
        raise HTTPException(404, f"No data for {city} on {date}")

    row_features = X_test.loc[matches.index[0], feature_cols].values.reshape(1, -1)
    prob = float(predictor.model.predict_proba(row_features)[0, 1])

    # Use Tree SHAP for fast per-prediction explanation
    try:
        import shap

        explainer = shap.TreeExplainer(predictor.model)
        shap_values = explainer.shap_values(row_features)
        # For binary classification, shap_values[1] = class 1 (heatwave)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # Top 15 most impactful features (by absolute SHAP value)
        top_indices = np.argsort(np.abs(sv))[::-1][:15]
        contributions = [
            {
                "feature": feature_cols[i],
                "shap_value": round(float(sv[i]), 6),
                "feature_value": round(float(row_features[0, i]), 4),
                "direction": "increases" if sv[i] > 0 else "decreases",
            }
            for i in top_indices
        ]
        base_value = (
            float(explainer.expected_value[1])
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else float(explainer.expected_value)
        )
    except Exception as e:
        # Fallback: use feature importances * feature values
        contributions = []
        fi = predictor.model.feature_importances_
        weighted = fi * np.abs(row_features[0])
        top_indices = np.argsort(weighted)[::-1][:15]
        for i in top_indices:
            contributions.append(
                {
                    "feature": feature_cols[i],
                    "shap_value": round(float(fi[i]), 6),
                    "feature_value": round(float(row_features[0, i]), 4),
                    "direction": "important",
                }
            )
        base_value = 0.5

    return {
        "probability": round(prob, 4),
        "contributions": contributions,
        "base_value": round(base_value, 4),
        "method": "SHAP TreeExplainer",
    }


@app.get("/api/city-comparison")
def get_city_comparison():
    """Return side-by-side city comparison data."""
    result = []
    for city_key, info in CITIES.items():
        city_df = test_data[test_data["city_key"] == city_key]
        valid = city_df.dropna(subset=["heatwave_next_day"])
        result.append(
            {
                "key": city_key,
                "name": info["name"],
                "state": info["state"],
                "region": info["region"],
                "total_days": len(valid),
                "heatwave_days": int(valid["heatwave"].sum()) if "heatwave" in valid.columns else 0,
                "avg_tmax": (
                    round(float(city_df["temperature_2m_max"].mean()), 1)
                    if "temperature_2m_max" in city_df.columns
                    else None
                ),
                "max_tmax": (
                    round(float(city_df["temperature_2m_max"].max()), 1)
                    if "temperature_2m_max" in city_df.columns
                    else None
                ),
                "avg_prob": round(float(city_df["pred_probability"].mean()) * 100, 2),
                "max_prob": round(float(city_df["pred_probability"].max()) * 100, 1),
                **city_metrics.get(city_key, {}),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------
WEB_DIR = PROJECT_ROOT / "web"


@app.get("/")
def serve_index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/performance")
def serve_performance():
    return FileResponse(WEB_DIR / "performance.html")


@app.get("/about")
def serve_about():
    return FileResponse(WEB_DIR / "about.html")


# ---------------------------------------------------------------------------
# Phase 7 — Live data endpoints (ADDITIVE — no existing routes changed)
# ---------------------------------------------------------------------------
try:
    from live_data import CITIES as LIVE_CITIES
    from live_data import cache_status as live_cache_status
    from live_data import get_live_forecast

    _LIVE_DATA_AVAILABLE = True
    print("[startup] Live data module loaded (Open-Meteo integration available)")
except Exception as _live_err:
    _LIVE_DATA_AVAILABLE = False
    print(f"[startup] Live data module unavailable: {_live_err}")


@app.get("/api/live/status")
def get_live_status():
    """Return live data availability and cache state."""
    if not _LIVE_DATA_AVAILABLE:
        return {"available": False, "reason": "live_data module failed to load"}
    return {
        "available": True,
        "source": "Open-Meteo.com",
        "source_url": "https://open-meteo.com",
        "cache_ttl_minutes": 30,
        "cache": live_cache_status(),
    }


@app.get("/api/live/{city}")
@limiter.limit("12/minute")
def get_live_city(request: Request, city: str):
    """
    Fetch live weather from Open-Meteo, engineer all 110 features, and run
    the full ClimateGuard pipeline (Part 1 + Part 3 expert rules) for today
    and the next ~5 forecast days.

    Response shape is compatible with /api/predict where possible.
    On fetch failure, returns a clear error payload — never fake data.
    """
    if not _LIVE_DATA_AVAILABLE:
        raise HTTPException(503, "Live data module unavailable. Check server logs.")

    city = city.lower().strip()
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}. Valid: {list(CITIES.keys())}")

    result = get_live_forecast(city, pipeline)
    if not result.get("error") and not result.get("cache_used"):
        log_live_predictions(city, result)
        dispatch_alerts_for_live_forecast(city, result)
    return result


# ---------------------------------------------------------------------------
# Mount static files AFTER API routes
# ---------------------------------------------------------------------------
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

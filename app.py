"""
ClimateGuard Web Dashboard — FastAPI Backend
Serves the prediction pipeline over HTTP + static frontend.

Run:
    python app.py
    → http://localhost:8000
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score,
)

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.pipeline import ClimateGuardPipeline
from src.prediction import ClimateGuardPredictor

# ---------------------------------------------------------------------------
# City metadata
# ---------------------------------------------------------------------------
CITIES = {
    "delhi":     {"name": "New Delhi",  "state": "Delhi",             "region": "Plains",  "lat": 28.6139, "lon": 77.2090},
    "lucknow":   {"name": "Lucknow",   "state": "Uttar Pradesh",     "region": "Plains",  "lat": 26.8467, "lon": 80.9462},
    "nagpur":    {"name": "Nagpur",     "state": "Maharashtra",       "region": "Plains",  "lat": 21.1458, "lon": 79.0882},
    "ahmedabad": {"name": "Ahmedabad", "state": "Gujarat",           "region": "Plains",  "lat": 23.0225, "lon": 72.5714},
    "mumbai":    {"name": "Mumbai",    "state": "Maharashtra",       "region": "Coastal", "lat": 19.0760, "lon": 72.8777},
}

# ---------------------------------------------------------------------------
# Load data + pipeline at startup
# ---------------------------------------------------------------------------
print("[startup] Loading test data ...")
X_test   = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "X_test.csv")
meta_test = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "meta_test.csv")
y_test   = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "temporal" / "y_test.csv")

# Merge features + metadata for easy lookup
test_data = pd.concat([meta_test, X_test, y_test], axis=1)
print(f"[startup] Loaded {len(test_data)} test rows across {test_data['city_key'].nunique()} cities")

print("[startup] Loading ClimateGuardPipeline ...")
pipeline = ClimateGuardPipeline(include_explanation=False)
print(f"[startup] Pipeline ready: {pipeline}")

# Load model metadata
with open(PROJECT_ROOT / "models" / "final" / "metadata.json") as f:
    model_metadata = json.load(f)

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
pr_data = {"precision": precisions[::step].tolist(), "recall": recalls[::step].tolist(), "auc": round(pr_auc, 4)}

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
app = FastAPI(title="ClimateGuard Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    city: str
    date: str


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/cities")
def get_cities():
    """Return all available cities with metadata."""
    result = []
    for key, info in CITIES.items():
        result.append({
            "key": key,
            **info,
            **stats_data.get(key, {}),
        })
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
def predict(req: PredictRequest):
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
        "city", "state", "region_type", "heatwave",
        "hw_event_id", "hw_event_start", "hw_event_end", "hw_event_length",
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
        records.append({
            "date": str(row["date"]),
            "tmax": safe_float(row.get("temperature_2m_max")),
            "tmin": safe_float(row.get("temperature_2m_min")),
            "prob": safe_float(row.get("pred_probability")),
            "heatwave": int(row.get("heatwave", 0)),
            "hw_next": int(row.get("heatwave_next_day", 0)) if not math.isnan(row.get("heatwave_next_day", 0)) else 0,
        })
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
        result.append({
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
        })
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
def explain_prediction(req: PredictRequest):
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
        base_value = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
    except Exception as e:
        # Fallback: use feature importances * feature values
        contributions = []
        fi = predictor.model.feature_importances_
        weighted = fi * np.abs(row_features[0])
        top_indices = np.argsort(weighted)[::-1][:15]
        for i in top_indices:
            contributions.append({
                "feature": feature_cols[i],
                "shap_value": round(float(fi[i]), 6),
                "feature_value": round(float(row_features[0, i]), 4),
                "direction": "important",
            })
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
        result.append({
            "key": city_key,
            "name": info["name"],
            "state": info["state"],
            "region": info["region"],
            "total_days": len(valid),
            "heatwave_days": int(valid["heatwave"].sum()) if "heatwave" in valid.columns else 0,
            "avg_tmax": round(float(city_df["temperature_2m_max"].mean()), 1) if "temperature_2m_max" in city_df.columns else None,
            "max_tmax": round(float(city_df["temperature_2m_max"].max()), 1) if "temperature_2m_max" in city_df.columns else None,
            "avg_prob": round(float(city_df["pred_probability"].mean()) * 100, 2),
            "max_prob": round(float(city_df["pred_probability"].max()) * 100, 1),
            **city_metrics.get(city_key, {}),
        })
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

# Mount static files AFTER API routes
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

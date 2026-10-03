"""Reproducible, separately versioned Europe data and model pipeline.

This module deliberately does not read or write the India training artifacts.
It uses the project's shared feature transformation, but trains a distinct
``europe-v1`` model against a documented, relative heatwave definition.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering import build_features_for_city

EUROPE_DIR = ROOT / "data" / "europe"
RAW_DIR = EUROPE_DIR / "raw"
SPLIT_DIR = EUROPE_DIR / "splits"
RESULTS_DIR = ROOT / "evaluation" / "europe"
MODEL_DIR = ROOT / "models" / "europe-v1"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "1991-01-01"
END_DATE = "2026-10-02"  # latest complete day when this regional dataset was prepared
BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"
MODEL_TEST_END = "2025-12-31"  # never evaluate an incomplete calendar year as the release test set
WARM_MONTHS = {5, 6, 7, 8, 9}
ABSOLUTE_FLOOR_C = 30.0
PERCENTILE = 0.90
WINDOW_RADIUS_DAYS = 15
MIN_DURATION_DAYS = 3

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "relative_humidity_2m_max", "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

# Coordinates/elevation verified against Open-Meteo geocoding on 2026-10-03.
# Monaco uses the city-level 43 m result rather than Open-Meteo's 9999 m polygon entry.
EUROPE_CITIES: dict[str, dict[str, Any]] = {
    "madrid": {"name": "Madrid", "state": "Spain", "country": "Spain", "region": "Interior", "region_type": "interior", "lat": 40.4165, "lon": -3.7026, "elevation": 665, "timezone": "Europe/Madrid"},
    "seville": {"name": "Seville", "state": "Spain", "country": "Spain", "region": "Mediterranean", "region_type": "mediterranean", "lat": 37.3828, "lon": -5.9732, "elevation": 16, "timezone": "Europe/Madrid"},
    "valencia": {"name": "Valencia", "state": "Spain", "country": "Spain", "region": "Mediterranean", "region_type": "mediterranean", "lat": 39.4739, "lon": -0.3797, "elevation": 15, "timezone": "Europe/Madrid"},
    "barcelona": {"name": "Barcelona", "state": "Spain", "country": "Spain", "region": "Mediterranean", "region_type": "mediterranean", "lat": 41.3888, "lon": 2.1590, "elevation": 15, "timezone": "Europe/Madrid"},
    "zaragoza": {"name": "Zaragoza", "state": "Spain", "country": "Spain", "region": "Interior", "region_type": "interior", "lat": 41.6561, "lon": -0.8773, "elevation": 214, "timezone": "Europe/Madrid"},
    "bilbao": {"name": "Bilbao", "state": "Spain", "country": "Spain", "region": "Atlantic", "region_type": "atlantic", "lat": 43.2627, "lon": -2.9253, "elevation": 20, "timezone": "Europe/Madrid"},
    "lisbon": {"name": "Lisbon", "state": "Portugal", "country": "Portugal", "region": "Atlantic", "region_type": "atlantic", "lat": 38.7251, "lon": -9.1498, "elevation": 68, "timezone": "Europe/Lisbon"},
    "porto": {"name": "Porto", "state": "Portugal", "country": "Portugal", "region": "Atlantic", "region_type": "atlantic", "lat": 41.1485, "lon": -8.6110, "elevation": 91, "timezone": "Europe/Lisbon"},
    "evora": {"name": "Évora", "state": "Portugal", "country": "Portugal", "region": "Interior", "region_type": "interior", "lat": 38.5659, "lon": -7.9040, "elevation": 242, "timezone": "Europe/Lisbon"},
    "andorra_la_vella": {"name": "Andorra la Vella", "state": "Andorra", "country": "Andorra", "region": "Mountain", "region_type": "mountain", "lat": 42.5078, "lon": 1.5211, "elevation": 1037, "timezone": "Europe/Andorra", "grid_note": "Underlying ERA5-Land cell is shared with nearby La Seu d'Urgell; Open-Meteo elevation adjustment still yields distinct temperatures."},
    "monaco": {"name": "Monaco", "state": "Monaco", "country": "Monaco", "region": "Mediterranean", "region_type": "mediterranean", "lat": 43.7300, "lon": 7.4200, "elevation": 43, "timezone": "Europe/Monaco"},
}
EUROPE_CITY_ORDER = list(EUROPE_CITIES)


def _circular_distance(left: np.ndarray, right: int) -> np.ndarray:
    """Circular day-of-year distance, with leap day retained consistently."""
    delta = np.abs(left - right)
    return np.minimum(delta, 366 - delta)


def _normal_and_thresholds(city_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return 31-day mean normals and 90th-percentile thresholds by day of year."""
    baseline = city_df.loc[
        (city_df["date"] >= BASELINE_START) & (city_df["date"] <= BASELINE_END),
        ["date", "temperature_2m_max"],
    ].copy()
    if baseline.empty:
        raise ValueError("Europe baseline data is unavailable")
    baseline["doy"] = pd.to_datetime(baseline["date"]).dt.dayofyear
    values = baseline["temperature_2m_max"].to_numpy(dtype=float)
    doys = baseline["doy"].to_numpy(dtype=int)
    normals, thresholds = {}, {}
    for day in range(1, 367):
        window = values[_circular_distance(doys, day) <= WINDOW_RADIUS_DAYS]
        normals[day] = float(np.mean(window))
        thresholds[day] = float(np.quantile(window, PERCENTILE))
    return (
        pd.to_datetime(city_df["date"]).dt.dayofyear.map(normals),
        pd.to_datetime(city_df["date"]).dt.dayofyear.map(thresholds),
    )


def label_europe_city(city_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the non-official Europe relative heatwave definition to one city."""
    df = city_df.copy().sort_values("date").reset_index(drop=True)
    normals, thresholds = _normal_and_thresholds(df)
    df["tmax_normal"] = normals.astype(float)
    df["tmax_departure"] = df["temperature_2m_max"] - df["tmax_normal"]
    df["tmax_p90"] = thresholds.astype(float)
    dates = pd.to_datetime(df["date"])
    df["qualifying_day"] = (
        (df["temperature_2m_max"] > df["tmax_p90"])
        & (df["temperature_2m_max"] >= ABSOLUTE_FLOOR_C)
        & dates.dt.month.isin(WARM_MONTHS)
    ).astype(int)
    heatwave = np.zeros(len(df), dtype=int)
    event_id = np.zeros(len(df), dtype=int)
    event, start = 0, 0
    qualifying = df["qualifying_day"].to_numpy()
    while start < len(df):
        if qualifying[start]:
            end = start + 1
            while end < len(df) and qualifying[end]:
                end += 1
            if end - start >= MIN_DURATION_DAYS:
                event += 1
                heatwave[start:end] = 1
                event_id[start:end] = event
            start = end
        else:
            start += 1
    df["heatwave"] = heatwave
    df["hw_event_id"] = event_id
    return df


def _download_city(city_key: str, city: dict[str, Any]) -> pd.DataFrame:
    """Download one complete archive response once, then reuse the local gzip cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{city_key}_era5_land.csv.gz"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])
    params = {
        "latitude": city["lat"], "longitude": city["lon"],
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES), "timezone": city["timezone"],
        "models": "era5_land",
    }
    response = None
    for attempt in range(1, 6):
        response = requests.get(ARCHIVE_URL, params=params, timeout=180)
        if response.status_code != 429:
            response.raise_for_status()
            break
        time.sleep(15 * attempt)
    else:
        assert response is not None
        response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily")
    if not daily or any(column not in daily for column in DAILY_VARIABLES):
        raise ValueError(f"Open-Meteo did not return the Europe daily data contract for {city_key}")
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df.insert(0, "city", city["name"])
    df.insert(1, "city_key", city_key)
    df.insert(2, "latitude", city["lat"])
    df.insert(3, "longitude", city["lon"])
    df.insert(4, "region_type", city["region_type"])
    df.insert(5, "state", city["state"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(cache_path, index=False, compression="gzip")
    return df


def download_europe_raw() -> pd.DataFrame:
    """Return the deterministic locally cached archive dataset for all Europe cities."""
    frames = []
    for key, city in EUROPE_CITIES.items():
        cache_path = RAW_DIR / f"{key}_era5_land.csv.gz"
        frames.append(_download_city(key, city))
        if not cache_path.exists():  # defensive guard; successful downloads always cache
            raise RuntimeError(f"Europe raw cache was not created for {key}")
        time.sleep(3)
    combined = pd.concat(frames, ignore_index=True)
    metadata_path = EUROPE_DIR / "city_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(EUROPE_CITIES, indent=2), encoding="utf-8")
    return combined


def prepare_europe_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Label and engineer Europe data using the shared India feature function."""
    labelled = pd.concat(
        [label_europe_city(group) for _, group in raw.groupby("city_key", sort=False)],
        ignore_index=True,
    )
    counts: dict[str, dict[str, int]] = {}
    for city_key, group in labelled.groupby("city_key"):
        counts[city_key] = {
            "all": int(group["heatwave"].sum()),
            "train": int(group.loc[group["date"] <= "2021-12-31", "heatwave"].sum()),
            "validation": int(group.loc[(group["date"] >= "2022-01-01") & (group["date"] <= "2022-12-31"), "heatwave"].sum()),
            "test": int(group.loc[(group["date"] >= "2023-01-01") & (group["date"] <= MODEL_TEST_END), "heatwave"].sum()),
        }
    features = pd.concat(
        [
            build_features_for_city(
                group,
                city_order=EUROPE_CITY_ORDER,
                coastal_region_types=("atlantic", "mediterranean"),
            )
            for _, group in labelled.groupby("city_key", sort=False)
        ],
        ignore_index=True,
    )
    features = features.sort_values(["city_key", "date"]).reset_index(drop=True)
    index_in_city = features.groupby("city_key").cumcount()
    final_index = features.groupby("city_key")["city_key"].transform("size") - 1
    features = features[(index_in_city >= 7) & (index_in_city < final_index)].copy()
    features = features[features["date"] <= MODEL_TEST_END].reset_index(drop=True)
    return features, counts


def save_europe_normals(raw: pd.DataFrame) -> None:
    """Persist only derived day-of-year values required for local live inference."""
    tables: dict[str, dict[str, dict[str, float]]] = {}
    for city_key, group in raw.groupby("city_key", sort=False):
        normals, thresholds = _normal_and_thresholds(group)
        doys = pd.to_datetime(group["date"]).dt.dayofyear
        tables[city_key] = {
            "tmax_normal": {str(day): round(float(normals[doys == day].iloc[0]), 6) for day in range(1, 367)},
            "tmax_p90": {str(day): round(float(thresholds[doys == day].iloc[0]), 6) for day in range(1, 367)},
        }
    (EUROPE_DIR / "city_tmax_normals.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    """Metrics which remain honest for single-class slices."""
    has_both = len(np.unique(y_true)) == 2
    return {
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4) if has_both else None,
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4) if has_both else None,
    }


def _curve_payload(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Frontend-compatible curves and calibration values for a binary test set."""
    if len(np.unique(y_true)) != 2:
        return {"roc": {"fpr": [], "tpr": [], "auc": None}, "pr": {"precision": [], "recall": [], "auc": None}, "calibration": {"predicted": [], "observed": [], "brier": None}}
    fpr, tpr, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(roc_auc_score(y_true, probability)), 4)},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist(), "auc": round(float(average_precision_score(y_true, probability)), 4)},
        "calibration": {"predicted": predicted.tolist(), "observed": observed.tolist(), "brier": round(float(brier_score_loss(y_true, probability)), 4)},
    }


def train_europe_model(features: pd.DataFrame, label_counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Train Europe-v1, choose a threshold only on validation, and save release artifacts."""
    raw_feature_list = json.loads((ROOT / "models" / "final" / "feature_list.json").read_text(encoding="utf-8"))
    # feature_list.json stores [{index, name, dtype}]; extract just the names
    if raw_feature_list and isinstance(raw_feature_list[0], dict):
        feature_names = [entry["name"] for entry in raw_feature_list]
    else:
        feature_names = list(raw_feature_list)
    missing = set(feature_names) - set(features.columns)
    if missing:
        raise ValueError(f"Shared feature contract is incomplete: {sorted(missing)}")
    train = features[features["date"] <= "2021-12-31"].copy()
    validation = features[(features["date"] >= "2022-01-01") & (features["date"] <= "2022-12-31")].copy()
    test = features[(features["date"] >= "2023-01-01") & (features["date"] <= MODEL_TEST_END)].copy()
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    model.fit(train[feature_names], train["heatwave_next_day"].astype(int))
    val_y = validation["heatwave_next_day"].astype(int).to_numpy()
    val_prob = model.predict_proba(validation[feature_names])[:, 1]
    candidates = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        pred = (val_prob >= threshold).astype(int)
        precision = precision_score(val_y, pred, zero_division=0)
        recall = recall_score(val_y, pred, zero_division=0)
        candidates.append((threshold, precision, recall, fbeta_score(val_y, pred, beta=2, zero_division=0)))
    # Recall-first selection with a modest precision guardrail, using F2 as the tiebreaker.
    eligible = [row for row in candidates if row[1] >= 0.25]
    threshold, _, _, _ = max(eligible or candidates, key=lambda row: (row[2], row[3], row[1], -row[0]))
    test_y = test["heatwave_next_day"].astype(int).to_numpy()
    test_prob = model.predict_proba(test[feature_names])[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
    per_city = {}
    for city_key, group in test.groupby("city_key"):
        y = group["heatwave_next_day"].astype(int).to_numpy()
        prob = model.predict_proba(group[feature_names])[:, 1]
        per_city[city_key] = {
            **_metric_summary(y, (prob >= threshold).astype(int), prob),
            "name": EUROPE_CITIES[city_key]["name"],
            "country": EUROPE_CITIES[city_key]["country"],
            "heatwave_days": int(y.sum()), "total_days": int(len(y)),
        }
    persistence = test["heatwave_lag1"].fillna(0).astype(int).to_numpy()
    percentile_rule = test["qualifying_day"].astype(int).to_numpy()
    result = {
        "version": "europe-v1", "definition": {
            "summary": "Non-official research label: May–September days at or above 30°C and above the city-specific 90th percentile in a 31-day calendar window, retained only in runs of at least three days.",
            "baseline": "1991–2020", "percentile": PERCENTILE, "window_radius_days": WINDOW_RADIUS_DAYS,
            "minimum_duration_days": MIN_DURATION_DAYS, "warm_months": sorted(WARM_MONTHS), "absolute_floor_c": ABSOLUTE_FLOOR_C,
        },
        "threshold": round(float(threshold), 2), "threshold_selection": "Validation-only; maximize recall among thresholds with precision >= 0.25, then maximize precision.",
        "splits": {"train_through": "2021-12-31", "validation": "2022", "test": "2023-01-01 to 2025-12-31"},
        "city_order": [k for k in EUROPE_CITY_ORDER if k in {ck for ck in label_counts}],
        "label_counts": label_counts, "global_metrics": _metric_summary(test_y, test_pred, test_prob),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "total_test_samples": int(len(test_y)), "positive_samples": int(test_y.sum()), "negative_samples": int((test_y == 0).sum()),
        "city_metrics": per_city, "baselines": {
            "persistence": _metric_summary(test_y, persistence, persistence.astype(float)),
            "city_p90_qualifying_day": _metric_summary(test_y, percentile_rule, percentile_rule.astype(float)),
        },
        **_curve_payload(test_y, test_prob),
        "limitations": "These relative, ERA5-Land/Open-Meteo labels are research labels, not national warning criteria. City and country results are not comparable to India’s IMD-inspired model or to official services.",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "feature_list.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    test.to_csv(SPLIT_DIR / "test_data.csv.gz", index=False, compression="gzip")
    (RESULTS_DIR / "performance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run() -> dict[str, Any]:
    """Build/reuse raw cache, create Europe labels/features, and train europe-v1."""
    raw = download_europe_raw()
    save_europe_normals(raw)
    features, counts = prepare_europe_features(raw)
    return train_europe_model(features, counts)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

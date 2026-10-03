"""Read-only Europe-v1 model service used by additive FastAPI routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.cities import CITIES
from src.europe_live import europe_rules

ROOT = Path(__file__).resolve().parents[1]


def is_europe_city(city_key: str) -> bool:
    return CITIES.get(city_key, {}).get("dataset_region") == "europe"


class EuropeService:
    """A small immutable adapter around the separately versioned Europe artifacts."""

    def __init__(self) -> None:
        model_dir = ROOT / "models" / "europe-v1"
        self.model = joblib.load(model_dir / "model.joblib")
        self.feature_names = json.loads((model_dir / "feature_list.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.test_data = pd.read_csv(ROOT / "data" / "europe" / "splits" / "test_data.csv.gz")
        self.test_data["date"] = self.test_data["date"].astype(str)
        self.test_data["pred_probability"] = self.model.predict_proba(self.test_data[self.feature_names])[:, 1]
        self.test_data["pred_label"] = (self.test_data["pred_probability"] >= self.threshold).astype(int)

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @staticmethod
    def risk_level(probability: float) -> str:
        if probability >= 0.8:
            return "EXTREME"
        if probability >= 0.6:
            return "HIGH"
        if probability >= 0.3:
            return "MODERATE"
        return "LOW"

    def dates(self, city: str) -> dict[str, Any]:
        dates = sorted(self.test_data.loc[self.test_data["city_key"] == city, "date"].unique().tolist())
        return {"city": city, "dates": dates, "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None, "count": len(dates)}

    def predict(self, city: str, date: str) -> dict[str, Any]:
        matches = self.test_data[(self.test_data["city_key"] == city) & (self.test_data["date"] == date)]
        if matches.empty:
            date_info = self.dates(city)
            raise KeyError(f"No data for {city} on {date}. Available range: {date_info['date_min']} to {date_info['date_max']}")
        row = matches.iloc[0]
        probability = float(row["pred_probability"])
        prediction = int(probability >= self.threshold)
        rules = europe_rules(row, probability)
        risk = self.risk_level(probability)
        recommendations = [
            {"category": "Hydration", "message": "Drink water regularly and seek a cool space if you feel unwell."},
            {"category": "Outdoor exposure", "message": "Move strenuous outdoor activity to cooler hours and rest in shade."},
            {"category": "Vulnerable people", "message": "Check in on older people, children, and anyone with health conditions."},
            {"category": "Official guidance", "message": "Check the relevant national meteorological service and local authority guidance."},
        ] if risk in {"HIGH", "EXTREME"} else [
            {"category": "Preparedness", "message": "Monitor local forecasts, keep water available, and plan for hotter periods."}
        ]
        return {
            "input": {"city_key": city, "date": date},
            "prediction": {"probability": round(probability, 4), "prediction": prediction, "threshold": self.threshold},
            "risk": {"level": risk}, "warnings": [
                "Europe-v1 uses a relative research heatwave definition, not an official national warning criterion.",
            ],
            "expert_rules": rules, "recommendations": recommendations,
            "actual": {"heatwave_next_day": int(row["heatwave_next_day"]), "label": "Heatwave" if int(row["heatwave_next_day"]) else "Normal"},
            "city_info": CITIES[city],
        }

    def trends(self, city: str) -> dict[str, Any]:
        city_data = self.test_data[self.test_data["city_key"] == city].sort_values("date")
        records = [{
            "date": str(row.date), "tmax": round(float(row.temperature_2m_max), 4),
            "tmin": round(float(row.temperature_2m_min), 4), "prob": round(float(row.pred_probability), 4),
            "heatwave": int(row.heatwave), "hw_next": int(row.heatwave_next_day),
        } for row in city_data.itertuples()]
        return {"city": city, "city_name": CITIES[city]["name"], "data": records}

    def map_data(self) -> list[dict[str, Any]]:
        rows = []
        for key, info in CITIES.items():
            if not is_europe_city(key):
                continue
            city = self.test_data[self.test_data["city_key"] == key]
            rows.append({"key": key, "name": info["name"], "state": info["state"], "country": info["country"], "region": info["region"], "lat": info["lat"], "lon": info["lon"], "avg_probability": round(float(city.pred_probability.mean()), 4), "max_probability": round(float(city.pred_probability.max()), 4), "heatwave_days": int(city.heatwave.sum()), "total_days": int(len(city)), "heatwave_pct": round(float(city.heatwave.mean()) * 100, 1)})
        return rows


def load_europe_service() -> EuropeService | None:
    """Leave the India application usable if optional Europe artifacts are absent."""
    try:
        return EuropeService()
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None

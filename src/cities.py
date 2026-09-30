"""Versioned city registry; only active-model cities are exposed."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITY_REGISTRY = json.loads((PROJECT_ROOT / "config" / "cities.json").read_text(encoding="utf-8"))
# Keep internal rollout metadata out of legacy endpoint payloads.
CITIES = {
    key: {field: value for field, value in city.items() if field != "model_status"}
    for key, city in CITY_REGISTRY.items()
}
CITY_ORDER = list(CITIES)

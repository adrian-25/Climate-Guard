"""Versioned city registry; only active-model cities are exposed."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITIES = json.loads((PROJECT_ROOT / "config" / "cities.json").read_text(encoding="utf-8"))
CITY_ORDER = list(CITIES)

"""Build the small, versioned PSI reference artifact used at runtime.

Run only when an explicitly approved replacement model/training window exists.
It reads the existing train/validation splits and writes a *new* monitoring
artifact; it never changes the training data or production model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.drift import MIN_HISTORY_ROWS, MONITORING_FEATURES

SPLITS = ROOT / "data" / "splits" / "temporal"
OUTPUT = ROOT / "models" / "monitoring" / "v1" / "drift_baseline.json"
N_BINS = 10


def _edges_and_proportions(values: pd.Series) -> tuple[list[float], list[float]]:
    """Create monotonic quantile bins and their reference proportions."""
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot build a drift baseline from empty values")
    quantiles = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(quantiles)
    if len(edges) < 2:
        center = float(values[0])
        edges = np.array([center - 0.5, center + 0.5])
    else:
        # Keep JSON standards-compliant while still catching every finite input.
        edges[0] = -1e308
        edges[-1] = 1e308
    counts, _ = np.histogram(values, bins=edges)
    return edges.tolist(), (counts / counts.sum()).tolist()


def _load_reference() -> pd.DataFrame:
    frames = []
    for split in ("train", "val"):
        features = pd.read_csv(SPLITS / f"X_{split}.csv")
        metadata = pd.read_csv(SPLITS / f"meta_{split}.csv")
        frames.append(pd.concat([metadata[["city_key", "date"]], features], axis=1))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    reference = _load_reference()
    reference["date"] = pd.to_datetime(reference["date"], errors="raise")
    cities: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for city, city_rows in reference.groupby("city_key", sort=True):
        cities[str(city)] = {}
        for month in range(1, 13):
            month_rows = city_rows[city_rows["date"].dt.month == month]
            cities[str(city)][str(month)] = {}
            for feature in MONITORING_FEATURES:
                edges, proportions = _edges_and_proportions(month_rows[feature])
                cities[str(city)][str(month)][feature] = {
                    "bin_edges": edges,
                    "proportions": proportions,
                }

    payload = {
        "baseline_version": "v1",
        "reference": {
            "source": "temporal train + validation splits",
            "period": "1990-01-11 to 2022-12-31",
            "rows": int(len(reference)),
            "cities": sorted(cities),
            "seasonal_stratification": "city_key + calendar month",
        },
        "minimum_history_rows": MIN_HISTORY_ROWS,
        "monitoring_features": list(MONITORING_FEATURES),
        "cities": cities,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

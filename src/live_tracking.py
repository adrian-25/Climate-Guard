"""SQLite persistence and transparent proxy scoring for live forecasts."""

import os
import sqlite3
from datetime import date
from pathlib import Path

from sklearn.metrics import precision_score, recall_score


def database_path(root: Path) -> Path:
    return Path(os.getenv("DATABASE_PATH", str(root / "runtime" / "climateguard.sqlite3")))


def initialise(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(live_predictions)")}
        if columns and "id" not in columns:
            # The first local prototype de-duplicated forecasts. Preserve its
            # rows while upgrading to append-only event storage.
            connection.execute("ALTER TABLE live_predictions RENAME TO live_predictions_legacy")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS live_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL, forecast_date TEXT NOT NULL, forecast_horizon INTEGER NOT NULL,
                probability REAL NOT NULL, prediction INTEGER NOT NULL, risk_level TEXT NOT NULL,
                model_version TEXT NOT NULL, generated_at TEXT NOT NULL,
                observed_tmax REAL, observed_heat_signal INTEGER
            )
        """)
        if columns and "id" not in columns:
            connection.execute("""
                INSERT INTO live_predictions(
                    city, forecast_date, forecast_horizon, probability, prediction,
                    risk_level, model_version, generated_at, observed_tmax, observed_heat_signal
                )
                SELECT city, forecast_date, forecast_horizon, probability, prediction,
                    risk_level, model_version, generated_at, observed_tmax, observed_heat_signal
                FROM live_predictions_legacy
            """)
            connection.execute("DROP TABLE live_predictions_legacy")


def record(path: Path, city: str, days: list[dict], model_version: str, generated_at: str) -> None:
    initialise(path)
    rows = []
    for horizon, day in enumerate(days):
        if day.get("error") or day.get("probability") is None:
            continue
        rows.append(
            (
                city,
                day["date"],
                horizon,
                day["probability"],
                day["prediction"],
                day["risk_level"],
                model_version,
                generated_at,
            )
        )
    with sqlite3.connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO live_predictions(city, forecast_date, forecast_horizon, probability, prediction, risk_level, model_version, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            rows,
        )


def summary(path: Path) -> dict:
    initialise(path)
    with sqlite3.connect(path) as connection:
        all_rows = connection.execute("SELECT COUNT(*) FROM live_predictions").fetchone()[0]
        matched = connection.execute(
            "SELECT prediction, observed_heat_signal FROM live_predictions WHERE observed_heat_signal IS NOT NULL"
        ).fetchall()
    if not matched:
        return {"logged_predictions": all_rows, "matched_outcomes": 0, "status": "not_enough_data"}
    predictions, outcomes = zip(*matched)
    return {
        "logged_predictions": all_rows,
        "matched_outcomes": len(matched),
        "status": "available",
        "precision": round(float(precision_score(outcomes, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(outcomes, predictions, zero_division=0)), 4),
        "note": "Outcomes use an Open-Meteo observed-temperature proxy, not an official IMD heatwave label.",
    }


def pending_before_today(path: Path) -> list[tuple]:
    initialise(path)
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT id, city, forecast_date FROM live_predictions WHERE observed_heat_signal IS NULL AND forecast_date < ?",
            (date.today().isoformat(),),
        ).fetchall()


def record_outcome(path: Path, prediction_id: int, tmax: float, threshold: float) -> None:
    initialise(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE live_predictions SET observed_tmax=?, observed_heat_signal=? WHERE id=?",
            (tmax, int(tmax >= threshold), prediction_id),
        )

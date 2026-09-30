import sqlite3
from pathlib import Path

from src import live_tracking


def test_record_and_empty_summary(tmp_path: Path):
    path = tmp_path / "tracking.sqlite3"
    live_tracking.record(
        path,
        "delhi",
        [{"date": "2030-01-01", "probability": 0.8, "prediction": 1, "risk_level": "EXTREME"}],
        "v1",
        "2030-01-01T00:00:00Z",
    )
    summary = live_tracking.summary(path)
    assert summary == {"logged_predictions": 1, "matched_outcomes": 0, "status": "not_enough_data"}


def test_outcome_reconciliation_updates_summary(tmp_path: Path):
    path = tmp_path / "tracking.sqlite3"
    days = [{"date": "2020-01-01", "probability": 0.8, "prediction": 1, "risk_level": "EXTREME"}]
    live_tracking.record(path, "delhi", days, "v1", "2020-01-01T00:00:00Z")
    prediction_id = live_tracking.pending_before_today(path)[0][0]
    live_tracking.record_outcome(path, prediction_id, 42, 40)
    summary = live_tracking.summary(path)
    assert summary["status"] == "available"
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0


def test_forecasts_are_append_only(tmp_path: Path):
    path = tmp_path / "tracking.sqlite3"
    days = [{"date": "2030-01-01", "probability": 0.8, "prediction": 1, "risk_level": "EXTREME"}]
    live_tracking.record(path, "delhi", days, "v1", "2030-01-01T00:00:00Z")
    live_tracking.record(path, "delhi", days, "v1", "2030-01-01T06:00:00Z")
    assert live_tracking.summary(path)["logged_predictions"] == 2


def test_legacy_tracking_rows_are_preserved_during_migration(tmp_path: Path):
    path = tmp_path / "tracking.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE live_predictions (
                city TEXT, forecast_date TEXT, forecast_horizon INTEGER,
                probability REAL, prediction INTEGER, risk_level TEXT,
                model_version TEXT, generated_at TEXT,
                observed_tmax REAL, observed_heat_signal INTEGER,
                PRIMARY KEY (city, forecast_date, model_version)
            )""")
        connection.execute(
            "INSERT INTO live_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("delhi", "2020-01-01", 1, 0.8, 1, "EXTREME", "v1", "2020-01-01T00:00:00Z", None, None),
        )

    live_tracking.initialise(path)
    assert live_tracking.summary(path)["logged_predictions"] == 1

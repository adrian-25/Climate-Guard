"""Network-free tests for the read-only PSI drift monitor."""

from pathlib import Path

import pandas as pd
import pytest

from src.monitoring import ClimateDriftMonitor

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "models" / "monitoring" / "v1" / "drift_baseline.json"


@pytest.fixture(scope="module")
def monitor() -> ClimateDriftMonitor:
    return ClimateDriftMonitor.from_file(BASELINE)


@pytest.fixture(scope="module")
def delhi_history(monitor: ClimateDriftMonitor) -> pd.DataFrame:
    features = pd.read_csv(ROOT / "data" / "splits" / "temporal" / "X_test.csv")
    metadata = pd.read_csv(ROOT / "data" / "splits" / "temporal" / "meta_test.csv")
    rows = pd.concat([metadata[["city_key", "date"]], features], axis=1)
    return rows[rows["city_key"] == "delhi"].head(monitor.minimum_history_rows + 5).copy()


def test_baseline_metadata_is_available(monitor: ClimateDriftMonitor):
    info = monitor.info()
    assert info["available"] is True
    assert info["minimum_history_rows"] == 30
    assert len(info["monitoring_features"]) == 12


def test_insufficient_history_is_explicit(
    monitor: ClimateDriftMonitor, delhi_history: pd.DataFrame
):
    result = monitor.assess(delhi_history.head(29), city="delhi").to_dict()
    assert result["status"] == "insufficient_history"
    assert result["sample_size"] == 29
    assert result["overall_psi"] is None


def test_assessment_returns_feature_level_psi(
    monitor: ClimateDriftMonitor, delhi_history: pd.DataFrame
):
    result = monitor.assess(delhi_history, city="delhi").to_dict()
    assert result["status"] == "available"
    assert result["sample_size"] >= 30
    assert result["level"] in {"stable", "monitor", "drift_detected", "high_drift"}
    assert len(result["features"]) == len(monitor.features)
    assert all(entry["psi"] is not None for entry in result["features"])


def test_extreme_input_shift_is_flagged(monitor: ClimateDriftMonitor, delhi_history: pd.DataFrame):
    shifted = delhi_history.copy()
    shifted["temperature_2m_max"] = 100.0
    result = monitor.assess(shifted, city="delhi").to_dict()
    temperature = next(
        item for item in result["features"] if item["feature"] == "temperature_2m_max"
    )
    assert temperature["level"] == "high_drift"


def test_missing_required_feature_is_rejected(
    monitor: ClimateDriftMonitor, delhi_history: pd.DataFrame
):
    with pytest.raises(ValueError, match="Missing drift-monitoring columns"):
        monitor.assess(delhi_history.drop(columns=["temperature_2m_max"]), city="delhi")

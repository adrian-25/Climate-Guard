"""Read-only API contract tests; model/data files are never modified."""

from fastapi.testclient import TestClient

import app
from src import alerts

client = TestClient(app.app)


def test_health_exposes_active_model_and_request_id():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1.0.0"
    assert response.headers["x-request-id"]


def test_legacy_health_response_shape_is_preserved():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert set(response.json()) == {"status", "model_ready", "live_data_available", "checked_at"}


def test_dashboard_read_endpoints_succeed():
    for path in (
        "/api/config",
        "/api/cities",
        "/api/model-info",
        "/api/stats",
        "/api/map-data",
        "/api/performance",
        "/api/feature-importance",
        "/api/city-comparison",
    ):
        response = client.get(path)
        assert response.status_code == 200, path


def test_city_endpoints_validate_unknown_city():
    for path in (
        "/api/dates/not-a-city",
        "/api/trends/not-a-city",
        "/api/official-alerts/not-a-city",
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert "detail" in response.json()


def test_predict_validates_unknown_city_without_changing_error_shape():
    response = client.post("/api/predict", json={"city": "not-a-city", "date": "2024-01-01"})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_trends_return_city_data():
    response = client.get("/api/trends/delhi")
    assert response.status_code == 200
    assert response.json()["city"] == "delhi"
    assert response.json()["data"]


def test_latest_evaluation_is_read_only_and_available():
    response = client.get("/api/evaluation/latest")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1.0.0"


def test_live_track_record_has_clear_insufficient_data_state():
    response = client.get("/api/evaluation/live-track-record")
    assert response.status_code == 200
    assert response.json()["status"] in {"not_enough_data", "available"}


def test_browser_weather_fallback_rejects_unknown_city_and_invalid_payload():
    unknown = client.post("/api/live/not-a-city/browser-weather", json={"daily": {}})
    assert unknown.status_code == 400

    invalid = client.post("/api/live/delhi/browser-weather", json={"daily": {}})
    assert invalid.status_code == 200
    assert invalid.json()["error_type"] == "client_weather_invalid"


def test_subscription_endpoints_require_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "ALERT_DATABASE", tmp_path / "alerts.sqlite3")
    response = client.post(
        "/api/subscribe",
        json={
            "email": "person@example.com",
            "city": "delhi",
            "minimum_risk_level": "HIGH",
            "language": "en",
        },
    )
    assert response.status_code == 201
    subscription = alerts.confirmed_for_city(app.ALERT_DATABASE, "delhi")
    assert subscription == []
    token = alerts.subscribe(app.ALERT_DATABASE, "other@example.com", "delhi", "HIGH", "en")[
        "token"
    ]
    assert client.get(f"/api/confirm/{token}").json()["status"] == "confirmed"
    assert client.get("/api/alerts/preview?city=delhi&level=HIGH&lang=mr").status_code == 200


# ---------------------------------------------------------------------------
# /api/dl routes — contract tests (work without torch installed)
# ---------------------------------------------------------------------------


def test_dl_comparison_returns_data_or_503():
    """Endpoint must return the DL vs RF JSON or a clear 503 if not generated."""
    response = client.get("/api/dl/comparison")
    assert response.status_code in (200, 503), f"Unexpected status: {response.status_code}"
    if response.status_code == 200:
        body = response.json()
        assert "rows" in body, "Response must contain 'rows' key"
        assert "summary" in body, "Response must contain 'summary' key"
        assert isinstance(body["rows"], list)
    else:
        body = response.json()
        assert "detail" in body
        assert body["detail"]["status"] == "not_generated"


def test_dl_figure_rejects_unknown_name():
    """Unknown figure names must return 404 with a detail message."""
    response = client.get("/api/dl/figure/nonexistent_figure")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_dl_figure_valid_names_return_image_or_503():
    """Valid figure names must return image/png or 503 if not yet generated."""
    valid_names = [
        "pr_curves",
        "per_city_f1",
        "calibration",
        "dl_feature_importance_ig",
        "dl_ig_heatmap",
    ]
    for name in valid_names:
        response = client.get(f"/api/dl/figure/{name}")
        assert response.status_code in (200, 503), f"{name}: unexpected {response.status_code}"
        if response.status_code == 200:
            assert response.headers["content-type"] == "image/png"

"""Read-only API contract tests; model/data files are never modified."""
from fastapi.testclient import TestClient

import app

client = TestClient(app.app)


def test_health_exposes_active_model_and_request_id():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1.0.0"
    assert response.headers["x-request-id"]


def test_dashboard_read_endpoints_succeed():
    for path in ("/api/config", "/api/cities", "/api/model-info", "/api/stats", "/api/map-data", "/api/performance", "/api/feature-importance", "/api/city-comparison"):
        response = client.get(path)
        assert response.status_code == 200, path


def test_city_endpoints_validate_unknown_city():
    for path in ("/api/dates/not-a-city", "/api/trends/not-a-city", "/api/official-alerts/not-a-city"):
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

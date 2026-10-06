from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "agro-risk-model"
    assert data["model_version"] == "1.0"
    assert data["model_type"] == "risk-scoring"
    assert data["status"] == "ready"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "farm_id": "FARM-001",
            "region": "Krasnodar",
            "crop_type": "wheat",
            "area_ha": 500,
            "temperature_avg": 22,
            "precipitation_mm": 300,
            "payment_delay_days": 0,
            "previous_defaults": 0,
            "debt": 1_000_000,
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["farm_id"] == "FARM-001"
    assert data["risk_score"] == 0.1
    assert data["risk_level"] == "low"
    assert data["recommendation"] == "Стандартное рассмотрение"
    assert data["model_version"] == "1.0"
    assert "request_id" in data


def test_pydantic_validation():
    response = client.post(
        "/predict",
        json={
            "farm_id": "FARM-002",
            "region": "Krasnodar",
            "crop_type": "wheat",
            "area_ha": -100,
            "temperature_avg": 22,
            "precipitation_mm": 300,
            "payment_delay_days": 0,
            "previous_defaults": 0,
            "debt": 1_000_000,
        },
    )

    assert response.status_code == 422


def test_unknown_region():
    response = client.post(
        "/predict",
        json={
            "farm_id": "FARM-003",
            "region": "Moscow",
            "crop_type": "wheat",
            "area_ha": 500,
            "temperature_avg": 22,
            "precipitation_mm": 300,
            "payment_delay_days": 0,
            "previous_defaults": 0,
            "debt": 1_000_000,
        },
    )

    assert response.status_code == 400


def test_get_prediction():
    create_response = client.post(
        "/predict",
        json={
            "farm_id": "FARM-004",
            "region": "Rostov",
            "crop_type": "sunflower",
            "area_ha": 800,
            "temperature_avg": 24,
            "precipitation_mm": 50,
            "payment_delay_days": 45,
            "previous_defaults": 0,
            "debt": 1_000_000,
        },
    )

    request_id = create_response.json()["request_id"]

    response = client.get(
        f"/predictions/{request_id}"
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == request_id


def test_prediction_not_found():
    response = client.get(
        "/predictions/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404


def test_limit():
    response = client.get(
        "/predictions?limit=2"
    )

    assert response.status_code == 200
    assert len(response.json()) <= 2


def test_risk_level_filter():
    client.post(
        "/predict",
        json={
            "farm_id": "FARM-HIGH",
            "region": "Stavropol",
            "crop_type": "corn",
            "area_ha": 1200,
            "temperature_avg": 26,
            "precipitation_mm": 70,
            "payment_delay_days": 60,
            "previous_defaults": 1,
            "debt": 7_000_000,
        },
    )

    response = client.get(
        "/predictions?risk_level=high"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) >= 1
    assert all(
        item["risk_level"] == "high"
        for item in data
    )


def test_limit_validation():
    response = client.get(
        "/predictions?limit=-5"
    )

    assert response.status_code == 422
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import main


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {
            "model": "auto",
            "choices": [{
                "message": {"content": "Это демонстрационное объяснение риска."},
                "finish_reason": "stop",
            }],
        }

    def json(self):
        return self._payload


class FakeLLMClient:
    async def post(self, path, json=None):
        return FakeResponse()

    async def aclose(self):
        return None


@pytest.fixture

def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "test.db"))
    # Keep tests deterministic and fast while preserving the async worker path.
    with TestClient(main.app) as test_client:
        test_client.app.state.llm_client = FakeLLMClient()
        yield test_client


def farm(farm_id="FARM-001", **overrides):
    payload = {
        "farm_id": farm_id,
        "region": "Krasnodar",
        "crop_type": "wheat",
        "area_ha": 500,
        "temperature_avg": 22,
        "precipitation_mm": 300,
        "payment_delay_days": 0,
        "previous_defaults": 0,
        "debt": 1_000_000,
    }
    payload.update(overrides)
    return payload


def test_health_and_middleware_headers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("X-Request-ID")
    assert float(response.headers["X-Process-Time"]) >= 0


def test_client_request_id_is_reused(client):
    response = client.post("/predict", json=farm(), headers={"X-Request-ID": "lab-demo-1"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "lab-demo-1"
    assert response.json()["request_id"] == "lab-demo-1"
    assert client.get("/predictions/lab-demo-1").status_code == 200


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "1.0"


def test_predict_persists_to_sqlite(client):
    response = client.post("/predict", json=farm())
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] == 0.1
    assert data["risk_level"] == "low"
    assert client.get("/predictions").json()[0]["request_id"] == data["request_id"]


def test_pydantic_validation_error_has_request_id(client):
    response = client.post("/predict", json=farm(area_ha=-100))
    assert response.status_code == 422
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_unknown_region(client):
    assert client.post("/predict", json=farm(region="Moscow")).status_code == 400


def test_batch_predict_and_conflict_lookup(client):
    response = client.post("/batch-predict", json={"items": [farm("F-1"), farm("F-2", debt=6_000_000)]}, headers={"X-Request-ID": "batch-1"})
    assert response.status_code == 200
    assert len(response.json()["items"]) == 2
    assert client.get("/predictions/batch-1").status_code == 409
    assert len(client.get("/requests/batch-1/predictions").json()) == 2


def test_batch_too_large(client):
    response = client.post("/batch-predict", json={"items": [farm(f"F-{n}") for n in range(9)]})
    assert response.status_code == 422


def test_prediction_not_found(client):
    assert client.get("/predictions/not-here").status_code == 404


def test_dynamic_queued_predict(client):
    response = client.post("/predict-queued", json=farm("F-QUEUE"))
    assert response.status_code == 200
    assert response.json()["farm_id"] == "F-QUEUE"
    assert client.get(f"/predictions/{response.json()['request_id']}").status_code == 200


def test_queue_status(client):
    response = client.get("/queue-status")
    assert response.status_code == 200
    assert response.json()["queue_max_size"] == main.QUEUE_MAX_SIZE
    assert response.json()["max_batch_size"] == main.MAX_BATCH_SIZE


def test_weekday_pagination_and_validation(client):
    client.post("/predict", json=farm())
    # The test row was created today; use today's English weekday.
    from datetime import datetime, timezone
    weekday = datetime.now(timezone.utc).strftime("%A").lower()
    response = client.get(f"/predictions/by-weekday?weekday={weekday}&page=1&per_page=5")
    assert response.status_code == 200
    assert response.json()["meta"]["total_items"] >= 1
    assert client.get("/predictions/by-weekday?weekday=funday").status_code == 400
    assert client.get(f"/predictions/by-weekday?weekday={weekday}&page=-1").status_code == 422


def test_llm_chat(client):
    response = client.post("/llm/chat", json={"prompt": "Что означает высокий риск?"})
    assert response.status_code == 200
    assert response.json()["answer"]
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_predict_with_llm_keeps_deterministic_score(client):
    response = client.post("/predict-with-llm", json=farm("F-LLM", payment_delay_days=45))
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] == 0.4
    assert data["risk_level"] == "medium"
    assert data["explanation"]


def test_llm_provider_error_mapping(client):
    class BadClient:
        async def post(self, path, json=None):
            return FakeResponse(status_code=502, payload={"error": "bad gateway"})
        async def aclose(self):
            return None
    client.app.state.llm_client = BadClient()
    response = client.post("/llm/chat", json={"prompt": "test"})
    assert response.status_code == 502
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_inference_timeout_returns_504(client, monkeypatch):
    monkeypatch.setattr(main, "INFERENCE_TIMEOUT_SECONDS", 0.001)
    response = client.post("/predict", json=farm("F-TIMEOUT"))
    assert response.status_code == 504
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_backpressure_returns_429_when_queue_is_full(client):
    class FullQueue:
        def full(self):
            return True
    client.app.state.inference_queue = FullQueue()
    response = client.post("/predict-queued", json=farm("F-BACKPRESSURE"))
    assert response.status_code == 429
    assert response.json()["request_id"] == response.headers["X-Request-ID"]

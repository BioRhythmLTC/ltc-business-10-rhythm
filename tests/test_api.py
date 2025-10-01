import os
from fastapi.testclient import TestClient

from service.main import app
from service import main as main_mod


def _mock_model_manager(monkeypatch):
    class FakeMM:
        device = "cpu"
        _loaded = True

        def ensure_loaded(self):
            self._loaded = True

        def predict(self, text: str):
            if text == "__raise__":
                raise RuntimeError("boom")
            return [
                {"start_index": 0, "end_index": min(5, len(text)), "entity": "B-TYPE"}
            ]

        def predict_batch(self, texts):
            out = []
            for t in texts:
                if t == "__raise__":
                    raise RuntimeError("boom")
                out.append([
                    {"start_index": 0, "end_index": min(5, len(t)), "entity": "B-TYPE"}
                ])
            return out

    fake = FakeMM()
    monkeypatch.setattr(main_mod, "model_manager", fake, raising=True)
    return fake


def test_health(monkeypatch):
    _mock_model_manager(monkeypatch)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "device" in data and "model_loaded" in data


def test_root_default_404(monkeypatch):
    _mock_model_manager(monkeypatch)
    # Force ROOT_PUBLIC=false → 404
    monkeypatch.setattr(main_mod, "ROOT_PUBLIC", False, raising=False)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 404


def test_root_public(monkeypatch):
    _mock_model_manager(monkeypatch)
    monkeypatch.setattr(main_mod, "ROOT_PUBLIC", True, raising=False)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body.get("name") == "X5 NER Service"


def test_predict_empty_input(monkeypatch):
    _mock_model_manager(monkeypatch)
    client = TestClient(app)
    r = client.post("/api/predict", json={"input": "   "})
    assert r.status_code == 200
    assert r.json() == []


def test_predict_ok(monkeypatch):
    _mock_model_manager(monkeypatch)
    client = TestClient(app)
    r = client.post("/api/predict", json={"input": "кока кола 0.5л"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body and set(body[0].keys()) == {"start_index", "end_index", "entity"}


def test_predict_batch_ok(monkeypatch):
    _mock_model_manager(monkeypatch)
    client = TestClient(app)
    r = client.post("/api/predict_batch", json={"inputs": ["a", "b"]})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 2
    assert all(isinstance(x, list) for x in body)


def test_predict_fail_safe_true(monkeypatch):
    _mock_model_manager(monkeypatch)
    monkeypatch.setattr(main_mod, "FAIL_SAFE", True, raising=False)
    client = TestClient(app)

    r = client.post("/api/predict", json={"input": "__raise__"})
    assert r.status_code == 200
    assert r.json() == []

    r2 = client.post("/api/predict_batch", json={"inputs": ["__raise__"]})
    assert r2.status_code == 200
    assert r2.json() == [[]]


def test_predict_fail_safe_false(monkeypatch):
    _mock_model_manager(monkeypatch)
    monkeypatch.setattr(main_mod, "FAIL_SAFE", False, raising=False)
    client = TestClient(app)

    r = client.post("/api/predict", json={"input": "__raise__"})
    assert r.status_code == 500

    r2 = client.post("/api/predict_batch", json={"inputs": ["__raise__"]})
    # Batch endpoint falls back to per-item and returns empty lists for failures
    assert r2.status_code == 200
    assert r2.json() == [[]]


def test_cache_endpoints(monkeypatch):
    _mock_model_manager(monkeypatch)
    client = TestClient(app)

    # Initially, stats should be present
    r = client.get("/cache/stats")
    assert r.status_code == 200
    stats = r.json()
    assert "enabled" in stats and "current_size" in stats

    # Clear cache
    rc = client.delete("/cache/clear")
    assert rc.status_code == 200
    assert rc.json().get("status") == "cache_cleared"


def test_warmup(monkeypatch):
    _mock_model_manager(monkeypatch)
    client = TestClient(app)
    r = client.post("/warmup")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "warmed_up"


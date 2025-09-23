from fastapi.testclient import TestClient

from service.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_predict_monkeypatch(monkeypatch):
    client = TestClient(app)

    def fake_predict(text: str):
        return [{"start_index": 0, "end_index": 1, "entity": "B-TYPE"}]

    # monkeypatch the internal predict function used by endpoints
    from service import main as m

    monkeypatch.setattr(m, "predict_api_spans", fake_predict)

    r = client.post("/api/predict", json={"input": "x"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    r2 = client.post("/api/predict_batch", json={"inputs": ["x", "y"]})
    assert r2.status_code == 200
    body = r2.json()
    assert isinstance(body, list) and len(body) == 2

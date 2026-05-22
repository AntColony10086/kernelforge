from fastapi.testclient import TestClient

from local_gateway.server import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "gateway": "local_gateway"}

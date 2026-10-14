"""HTTP + WebSocket smoke test (needs fastapi and httpx): python -m pytest tests/test_api.py"""
from fastapi.testclient import TestClient

from backend.main import app


def test_api_flow():
    with TestClient(app) as c:
        assert c.post("/reset").status_code == 200
        assert c.get("/api/status").json()["cursor"] == 14
        assert c.post("/inject-fault", json={"fault": "bogus"}).status_code == 400
        with c.websocket_connect("/ws/alerts") as ws:
            c.post("/inject-fault", json={"fault": "duplicate_rows"})
            r = c.post("/run-day").json()
            assert r["alert"]["cause"] == "duplicate_rows"
            msg = ws.receive_json()
            assert msg["type"] == "alert"
        assert len(c.get("/alerts").json()) == 1
        assert len(c.get("/alerts/1").json()["trace"]) >= 2
        assert c.get("/metrics?feature=row_count").json()["points"]
        assert c.get("/metrics?feature=nope").status_code == 404
        assert "detection" in c.get("/eval-results").json()
        assert c.get("/").status_code == 200

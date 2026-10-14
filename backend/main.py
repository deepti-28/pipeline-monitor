"""FastAPI app: REST + WebSocket, and it serves the dashboard from /frontend.

Run:  uvicorn backend.main:app --reload
"""
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline
from .config import FAULTS, RESULTS, ROOT

FRONTEND = ROOT / "frontend"


class FaultRequest(BaseModel):
    fault: str | None = None


class StatusRequest(BaseModel):
    status: str


class Hub:
    """Tracks open WebSocket clients and broadcasts new alerts."""

    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)


hub = Hub()


def _ensure_eval():
    path = RESULTS / "eval_results.json"
    if not path.exists():
        from .evaluate import run_eval, save
        save(run_eval())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(pipeline.ensure_ready)
    threading.Thread(target=_ensure_eval, daemon=True).start()
    yield


app = FastAPI(title="Pipeline Health Monitor", lifespan=lifespan)


@app.get("/api/status")
def get_status():
    return pipeline.status()


@app.post("/run-day")
async def run_day():
    result = await run_in_threadpool(pipeline.run_next_day)
    if result.get("alert"):
        await hub.broadcast({"type": "alert", "alert": result["alert"]})
    await hub.broadcast({"type": "status", "status": result["status"]})
    return result


@app.post("/inject-fault")
def inject_fault(req: FaultRequest):
    if req.fault is not None and req.fault not in FAULTS:
        raise HTTPException(400, f"fault must be one of {FAULTS}")
    return pipeline.queue_fault(req.fault)


@app.post("/reset")
async def reset():
    await run_in_threadpool(pipeline.reset)
    await hub.broadcast({"type": "reset"})
    return pipeline.status()


@app.get("/features")
def get_features():
    return pipeline.features()


@app.get("/metrics")
def get_metrics(feature: str = Query("row_count", description="e.g. row_count, fare_amount.mean")):
    if feature not in pipeline.features():
        raise HTTPException(404, f"unknown feature {feature!r}")
    return {"feature": feature, "points": pipeline.metric_series(feature)}


@app.get("/alerts")
def get_alerts():
    return pipeline.list_alerts()


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: int):
    a = pipeline.get_alert(alert_id)
    if not a:
        raise HTTPException(404, "alert not found")
    return a


@app.post("/alerts/{alert_id}/status")
def set_alert_status(alert_id: int, req: StatusRequest):
    try:
        a = pipeline.set_alert_status(alert_id, req.status)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not a:
        raise HTTPException(404, "alert not found")
    return a


@app.get("/eval-results")
async def eval_results():
    path = RESULTS / "eval_results.json"
    if not path.exists():
        await run_in_threadpool(_ensure_eval)
    res = json.loads(path.read_text())
    res.pop("cases", None)  # keep the payload small; full cases stay in results/
    return res


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; clients only listen
    except WebSocketDisconnect:
        hub.disconnect(ws)


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
